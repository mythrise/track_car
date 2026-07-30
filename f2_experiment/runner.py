"""Verifiable callback runner for the frozen two-arm F2 smoke contract.

This module does not load a backbone, construct a real optimizer, or launch
training.  It owns only the ordered paired recurrence and exposes narrow
callbacks for model-specific work.  Targets are deliberately separated from
the observation passed to the feature and head callbacks.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any, Literal

import torch

from .controller import (
    DEFAULT_CONFIG,
    ActionFilterConfig,
    ActionFilterController,
    ActionFilterState,
)
from .evaluation import G6Update, G7Update
from .model import ACTION_MAX_ABS, AP2_HORIZON, AP2Prediction
from .support import ARCHITECTURE_LOCK, F2ContractError, continues_sequence


S_CTRL = "S-CTRL"
S_SELF = "S-SELF"
ARM_ORDER = (S_CTRL, S_SELF)
SMOKE_ROWS = 256
SMOKE_UPDATES = 128
GRAD_ACCUM = 2
SMOKE_WARMUP_UPDATES = 16

ArmName = Literal["S-CTRL", "S-SELF"]
BranchName = Literal["branch1", "branch2"]
PrevSource = Literal["logged", "self"]


class RunnerContractError(F2ContractError):
    """Raised when the paired smoke runner must fail closed."""


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RunnerContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RunnerContractError(f"{label} is nonfinite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RunnerContractError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise RunnerContractError(f"{label} must be nonnegative")
    return result


def _finite_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise RunnerContractError(f"{label} must be a floating torch.Tensor")
    if bool((~torch.isfinite(value)).detach().any().cpu().item()):
        raise RunnerContractError(f"{label} is nonfinite")
    return value


def _scalar_tensor(value: Any, label: str) -> torch.Tensor:
    tensor = _finite_tensor(value, label)
    if tensor.numel() != 1:
        raise RunnerContractError(f"{label} must contain exactly one scalar")
    return tensor.reshape(())


def _tensor_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def _flatten_floats(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return (_finite_float(value, label),)
    if isinstance(value, torch.Tensor):
        tensor = _finite_tensor(value, label)
        values = tensor.detach().cpu().reshape(-1).tolist()
        if not values:
            raise RunnerContractError(f"{label} has zero support")
        return tuple(_finite_float(item, label) for item in values)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        flattened: list[float] = []
        for index, item in enumerate(value):
            flattened.extend(_flatten_floats(item, f"{label}[{index}]"))
        if not flattened:
            raise RunnerContractError(f"{label} has zero support")
        return tuple(flattened)
    raise RunnerContractError(f"{label} must be numeric telemetry")


def _scalar_float(value: Any, label: str) -> float:
    flattened = _flatten_floats(value, label)
    if len(flattened) != 1:
        raise RunnerContractError(f"{label} must contain exactly one scalar")
    return flattened[0]


@dataclass(frozen=True)
class RunnerRow:
    """One ordered smoke row with expert targets kept outside observation."""

    original_row_index: int
    sequence_id: str
    frame_idx: int
    mirrored: bool
    logged_prev_action: tuple[float, float, float]
    target_actions: torch.Tensor
    observation: Any
    aux_targets: Any = None

    def __post_init__(self) -> None:
        _nonnegative_int(self.original_row_index, "original_row_index")
        if not isinstance(self.sequence_id, str) or not self.sequence_id:
            raise RunnerContractError("sequence_id must be a nonempty string")
        _nonnegative_int(self.frame_idx, "frame_idx")
        if not isinstance(self.mirrored, bool):
            raise RunnerContractError("mirrored must be boolean")
        if (
            not isinstance(self.logged_prev_action, tuple)
            or len(self.logged_prev_action) != 3
        ):
            raise RunnerContractError(
                "logged_prev_action must be a three-axis tuple"
            )
        for axis, raw_value in enumerate(self.logged_prev_action):
            value = _finite_float(raw_value, f"logged_prev_action[{axis}]")
            if abs(value) > ACTION_MAX_ABS:
                raise RunnerContractError(
                    "logged_prev_action lies outside the frozen domain"
                )
        target = _finite_tensor(self.target_actions, "target_actions")
        if target.shape != (AP2_HORIZON, 3):
            raise RunnerContractError(
                f"target_actions must have shape ({AP2_HORIZON},3)"
            )

    def sequence_mapping(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "frame_idx": self.frame_idx,
            "mirrored": self.mirrored,
        }


@dataclass(frozen=True)
class RowEvent:
    arm: ArmName
    row_position: int
    original_row_index: int
    u_pre: int
    reset: bool
    reset_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HeadEvent:
    arm: ArmName
    row_position: int
    original_row_index: int
    u_pre: int
    branch: BranchName
    prev_source: PrevSource


@dataclass(frozen=True)
class FeatureForwardResult:
    value: Any
    reference_tensor: torch.Tensor


@dataclass(frozen=True)
class AuxForwardResult:
    loss: torch.Tensor


@dataclass(frozen=True)
class HeadForwardResult:
    prediction: AP2Prediction
    g7_telemetry: Mapping[str, Any] | Any


@dataclass(frozen=True)
class BackwardEvent:
    arm: ArmName
    row_position: int
    original_row_index: int
    u_pre: int
    unscaled_loss: torch.Tensor
    scaled_loss: torch.Tensor
    grad_accum: int = GRAD_ACCUM


@dataclass(frozen=True)
class OptimizerUpdateEvent:
    arm: ArmName
    u_pre: int
    row_positions: tuple[int, int]
    original_row_indices: tuple[int, int]
    row_loss_values: tuple[float, float]
    mean_loss: float
    grad_accum: int = GRAD_ACCUM


@dataclass(frozen=True)
class ArmCallbacks:
    """Model-specific callbacks; the runner owns all recurrence decisions."""

    checkpoint_state: Mapping[str, torch.Tensor]
    feature_forward: Callable[[Any, RowEvent], FeatureForwardResult]
    aux_forward: Callable[[Any, Any, RowEvent], AuxForwardResult]
    head_forward: Callable[[Any, torch.Tensor, HeadEvent], HeadForwardResult]
    track_loss: Callable[[AP2Prediction, torch.Tensor, HeadEvent], torch.Tensor]
    backward: Callable[[BackwardEvent], None]
    optimizer_step: Callable[[OptimizerUpdateEvent], None]
    audit_counters: Callable[[], Mapping[str, int]] | None = None


@dataclass(frozen=True)
class RunnerG7Update:
    """Full runner-side G7 point plus the evaluator-compatible projection."""

    u_pre: int
    per_method_over_base: Mapping[str, tuple[float, ...]]
    total_method_over_base: tuple[float, ...]
    abs_tanh_method_scales: Mapping[str, float]
    r_prev: tuple[float, ...]
    abs_tanh_s_prev: float
    head_observations: int

    def gate_update(self) -> G7Update:
        return G7Update(
            u_pre=self.u_pre,
            per_method_over_base=self.per_method_over_base,
            total_method_over_base=self.total_method_over_base,
            abs_tanh_method_scales=self.abs_tanh_method_scales,
            r_prev=self.r_prev,
            abs_tanh_s_prev=self.abs_tanh_s_prev,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "u_pre": self.u_pre,
            "per_method_over_base": {
                name: list(values)
                for name, values in self.per_method_over_base.items()
            },
            "total_method_over_base": list(self.total_method_over_base),
            "abs_tanh_method_scales": dict(self.abs_tanh_method_scales),
            "r_prev": list(self.r_prev),
            "abs_tanh_s_prev": self.abs_tanh_s_prev,
            "head_observations": self.head_observations,
        }


@dataclass(frozen=True)
class K0ControllerReceipt:
    """Per-row G9 hook with controller-owned k0 clamp telemetry."""

    arm: ArmName
    row_position: int
    original_row_index: int
    u_pre: int
    reset: bool
    reset_reasons: tuple[str, ...]
    branch2_prev_source: PrevSource
    raw_k0_fy: tuple[float, float] | None
    post_safety_clamp: tuple[float, float, float] | None
    rate_limited: tuple[float, float, float] | None
    filtered_sent: tuple[float, float, float] | None
    self_prev_after_fy: tuple[float, float]
    reconstruction_error: float | None
    range_violation_count: int
    range_observation_count: int
    synchronized_nonfinite_reset: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "row_position": self.row_position,
            "original_row_index": self.original_row_index,
            "u_pre": self.u_pre,
            "reset": self.reset,
            "reset_reasons": list(self.reset_reasons),
            "branch2_prev_source": self.branch2_prev_source,
            "raw_k0_fy": None if self.raw_k0_fy is None else list(self.raw_k0_fy),
            "post_safety_clamp": (
                None
                if self.post_safety_clamp is None
                else list(self.post_safety_clamp)
            ),
            "rate_limited": (
                None if self.rate_limited is None else list(self.rate_limited)
            ),
            "filtered_sent": (
                None if self.filtered_sent is None else list(self.filtered_sent)
            ),
            "self_prev_after_fy": list(self.self_prev_after_fy),
            "reconstruction_error": self.reconstruction_error,
            "range_violation_count": self.range_violation_count,
            "range_observation_count": self.range_observation_count,
            "synchronized_nonfinite_reset": self.synchronized_nonfinite_reset,
        }


@dataclass(frozen=True)
class RunnerTelemetryHooks:
    """Pre-step G6 producer and optional passive G7/G9 observers."""

    g6_update: Callable[[OptimizerUpdateEvent], G6Update]
    on_g7_update: Callable[[ArmName, RunnerG7Update], None] | None = None
    on_g9_transition: Callable[[K0ControllerReceipt], None] | None = None


@dataclass(frozen=True)
class NonfiniteResetReceipt:
    arm: ArmName
    row_position: int
    original_row_index: int
    u_pre: int
    logged_prev_fy: tuple[float, float]
    controller_state_after_reset: ActionFilterState
    self_prev_after_reset: torch.Tensor
    nonfinite_reset_count: int
    synchronized: bool = True


class RunnerNonfiniteActionError(RunnerContractError):
    """Fail-closed nonfinite action carrying synchronized-reset evidence."""

    def __init__(self, receipt: NonfiniteResetReceipt) -> None:
        super().__init__(
            "CTRL_NONFINITE: branch2 k0 synchronized controller/self reset"
        )
        self.receipt = receipt


@dataclass(frozen=True)
class ArmCounts:
    rows: int
    feature_forwards: int
    aux_forwards: int
    head_forwards: int
    track_loss_calls: int
    backward_calls: int
    optimizer_steps: int
    controller_steps: int
    static_resets: int
    nonfinite_resets: int
    branch1_logged_rows: int
    branch2_logged_rows: int
    branch2_self_rows: int
    g6_updates: int
    g7_updates: int
    g9_transitions: int
    expert_future_leak_count: int
    self_state_expert_overwrite_count: int

    def to_dict(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class CountReceipt:
    checkpoint_init_sha256: str
    arms: Mapping[ArmName, ArmCounts]
    expected_static_resets: int
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": "f2_paired_runner_count_receipt",
            "architecture_lock": ARCHITECTURE_LOCK,
            "checkpoint_init_sha256": self.checkpoint_init_sha256,
            "rows_per_arm": SMOKE_ROWS,
            "optimizer_updates_per_arm": SMOKE_UPDATES,
            "grad_accum": GRAD_ACCUM,
            "warmup": "u_pre<16",
            "loss": "L_aux+0.5*L1+0.5*L2",
            "expected_static_resets": self.expected_static_resets,
            "arms": {
                arm: counts.to_dict() for arm, counts in self.arms.items()
            },
            "passed": self.passed,
            "status": "PASS" if self.passed else "FAIL",
            "decision": "GO" if self.passed else "STOP",
        }


@dataclass(frozen=True)
class G9Telemetry:
    expected_static_resets: int
    observed_static_resets: int
    nonfinite_reset_count: int
    range_violation_count: int
    range_observation_count: int
    reconstruction_errors: tuple[float, ...]
    first_quartile_self_errors: tuple[float, ...]
    last_quartile_self_errors: tuple[float, ...]
    transitions: tuple[K0ControllerReceipt, ...]

    def gate_kwargs(self) -> dict[str, Any]:
        return {
            "expected_static_resets": self.expected_static_resets,
            "observed_static_resets": self.observed_static_resets,
            "nonfinite_reset_count": self.nonfinite_reset_count,
            "range_violation_count": self.range_violation_count,
            "range_observation_count": self.range_observation_count,
            "reconstruction_errors": self.reconstruction_errors,
            "first_quartile_self_errors": self.first_quartile_self_errors,
            "last_quartile_self_errors": self.last_quartile_self_errors,
        }


@dataclass(frozen=True)
class ArmRunResult:
    arm: ArmName
    counts: ArmCounts
    g6_updates: tuple[G6Update, ...]
    g7_updates: tuple[RunnerG7Update, ...]
    g9: G9Telemetry
    row_losses: tuple[float, ...]
    branch2_sources: tuple[PrevSource, ...]


@dataclass(frozen=True)
class PairedRunResult:
    checkpoint_init_sha256: str
    count_receipt: CountReceipt
    arms: Mapping[ArmName, ArmRunResult]
    static_reset_original_indices: tuple[int, ...]


def checkpoint_init_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a tensor state deterministically without torch serialization."""

    if not isinstance(state, Mapping) or not state:
        raise RunnerContractError("checkpoint state must be a nonempty mapping")
    digest = hashlib.sha256(b"f2-checkpoint-init-v1\0")
    for name in sorted(state):
        if not isinstance(name, str) or not name:
            raise RunnerContractError("checkpoint state names must be nonempty")
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise RunnerContractError(f"checkpoint state {name!r} is not a tensor")
        if tensor.is_sparse or tensor.is_quantized:
            raise RunnerContractError(
                f"checkpoint state {name!r} has unsupported storage"
            )
        if tensor.is_floating_point() or tensor.is_complex():
            if bool((~torch.isfinite(tensor)).detach().any().cpu().item()):
                raise RunnerContractError(
                    f"checkpoint state {name!r} is nonfinite"
                )
        contiguous = tensor.detach().cpu().contiguous().reshape(-1)
        raw = contiguous.view(torch.uint8).numpy().tobytes()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_rows(rows: Sequence[RunnerRow]) -> tuple[RunnerRow, ...]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise RunnerContractError("rows must be an ordered sequence")
    frozen = tuple(rows)
    if len(frozen) != SMOKE_ROWS:
        raise RunnerContractError(
            f"paired smoke requires exactly {SMOKE_ROWS} rows"
        )
    if any(not isinstance(row, RunnerRow) for row in frozen):
        raise RunnerContractError("every row must be a RunnerRow")
    original_indices = tuple(row.original_row_index for row in frozen)
    if any(right <= left for left, right in zip(original_indices, original_indices[1:])):
        raise RunnerContractError(
            "rows must be strictly ordered by original_row_index"
        )
    return frozen


def _validate_index_set(values: Set[int], label: str) -> frozenset[int]:
    if not isinstance(values, Set):
        raise RunnerContractError(f"{label} must be a set")
    result = frozenset(_nonnegative_int(value, label) for value in values)
    return result


def _build_reset_plan(
    rows: tuple[RunnerRow, ...],
    strafe_reset_original_indices: frozenset[int],
) -> tuple[tuple[str, ...], ...]:
    plan: list[tuple[str, ...]] = []
    previous: RunnerRow | None = None
    for row in rows:
        reasons: list[str] = []
        if previous is None:
            reasons.append("stream_first")
        elif not continues_sequence(
            previous.sequence_mapping(), row.sequence_mapping()
        ):
            reasons.append("sequence_discontinuity")
        if row.original_row_index in strafe_reset_original_indices:
            reasons.append("strafe_reset")
        plan.append(tuple(reasons))
        previous = row
    return tuple(plan)


def _telemetry_mapping(value: Any, label: str) -> dict[str, Any]:
    if hasattr(value, "as_dict"):
        value = value.as_dict()
    if not isinstance(value, Mapping):
        raise RunnerContractError(f"{label} must be mapping-like telemetry")
    return dict(value)


def _aggregate_g7(
    u_pre: int,
    observations: Sequence[Mapping[str, Any] | Any],
) -> RunnerG7Update:
    if len(observations) != GRAD_ACCUM * 2:
        raise RunnerContractError("G7 update must aggregate four head forwards")
    per_method: dict[str, list[float]] = {}
    total_values: list[float] = []
    r_prev_values: list[float] = []
    scale_snapshots: list[dict[str, float]] = []
    prev_scales: list[float] = []

    for index, raw_observation in enumerate(observations):
        telemetry = _telemetry_mapping(raw_observation, f"G7[{u_pre}][{index}]")
        required = {
            "per_method_over_base",
            "total_method_over_base",
            "abs_tanh_method_scales",
            "r_prev",
            "abs_tanh_s_prev",
        }
        missing = required - set(telemetry)
        if missing:
            raise RunnerContractError(
                f"G7[{u_pre}][{index}] is missing {sorted(missing)!r}"
            )
        ratios = telemetry["per_method_over_base"]
        scales = telemetry["abs_tanh_method_scales"]
        if not isinstance(ratios, Mapping) or not ratios:
            raise RunnerContractError("G7 method ratios must be nonempty")
        if not isinstance(scales, Mapping) or set(scales) != set(ratios):
            raise RunnerContractError("G7 method ratio/scale names differ")
        if per_method and set(ratios) != set(per_method):
            raise RunnerContractError("G7 method names change within an update")
        scale_snapshot: dict[str, float] = {}
        for name in sorted(ratios):
            if not isinstance(name, str) or not name:
                raise RunnerContractError("G7 stream name is invalid")
            values = _flatten_floats(ratios[name], f"G7 ratio {name}")
            if any(value < 0.0 for value in values):
                raise RunnerContractError("G7 ratios must be nonnegative")
            per_method.setdefault(name, []).extend(values)
            scale = _scalar_float(scales[name], f"G7 scale {name}")
            if not 0.0 <= scale <= 1.0:
                raise RunnerContractError("G7 method scales must lie in [0,1]")
            scale_snapshot[name] = scale
        scale_snapshots.append(scale_snapshot)
        totals = _flatten_floats(
            telemetry["total_method_over_base"], "G7 total ratio"
        )
        if any(value < 0.0 for value in totals):
            raise RunnerContractError("G7 total ratios must be nonnegative")
        total_values.extend(totals)
        r_prev = _flatten_floats(telemetry["r_prev"], "G7 r_prev")
        if any(value < 0.0 for value in r_prev):
            raise RunnerContractError("G7 r_prev must be nonnegative")
        r_prev_values.extend(r_prev)
        prev_scale = _scalar_float(
            telemetry["abs_tanh_s_prev"], "G7 abs_tanh_s_prev"
        )
        if not 0.0 <= prev_scale <= 1.0:
            raise RunnerContractError("G7 abs_tanh_s_prev must lie in [0,1]")
        prev_scales.append(prev_scale)

    first_scales = scale_snapshots[0]
    if any(snapshot != first_scales for snapshot in scale_snapshots[1:]):
        raise RunnerContractError("G7 method scales changed inside accumulation")
    if any(value != prev_scales[0] for value in prev_scales[1:]):
        raise RunnerContractError("G7 s_prev changed inside accumulation")
    if not r_prev_values:
        raise RunnerContractError("G7 r_prev telemetry is mandatory")

    return RunnerG7Update(
        u_pre=u_pre,
        per_method_over_base={
            name: tuple(values) for name, values in sorted(per_method.items())
        },
        total_method_over_base=tuple(total_values),
        abs_tanh_method_scales=first_scales,
        r_prev=tuple(r_prev_values),
        abs_tanh_s_prev=prev_scales[0],
        head_observations=len(observations),
    )


def _validate_prediction_shape(prediction: Any, label: str) -> AP2Prediction:
    if not isinstance(prediction, AP2Prediction):
        raise RunnerContractError(f"{label} must be an AP2Prediction")
    if prediction.delta_fy.shape != (1, AP2_HORIZON, 2):
        raise RunnerContractError(f"{label}.delta_fy shape mismatch")
    if prediction.raw_actions.shape != (1, AP2_HORIZON, 3):
        raise RunnerContractError(f"{label}.raw_actions shape mismatch")
    if prediction.bounded_future_actions.shape != (1, AP2_HORIZON - 1, 3):
        raise RunnerContractError(f"{label}.bounded_future_actions shape mismatch")
    if torch.count_nonzero(prediction.raw_actions[..., 1]).item() != 0:
        raise RunnerContractError(f"{label} predicted nonzero strafe")
    return prediction


def _prediction_k0_fy(prediction: AP2Prediction) -> torch.Tensor:
    return prediction.raw_actions[0, 0, (0, 2)]


def _prediction_reconstruction_error(
    prediction: AP2Prediction,
    prev_fy: torch.Tensor,
) -> float:
    expected = prev_fy.unsqueeze(-2) + torch.cumsum(
        prediction.delta_fy, dim=-2
    )
    actual = prediction.raw_actions[..., (0, 2)]
    error = torch.max(torch.abs(expected - actual))
    return _tensor_float(error)


def _range_counts(prediction: AP2Prediction) -> tuple[int, int]:
    controlled = prediction.raw_actions[..., (0, 2)]
    violations = torch.any(torch.abs(controlled) > ACTION_MAX_ABS, dim=-1)
    return int(violations.sum().item()), int(violations.numel())


def _quartile_errors(
    errors: Sequence[float], reset_positions: Sequence[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    starts = list(reset_positions)
    ends = starts[1:] + [len(errors)]
    first: list[float] = []
    last: list[float] = []
    for start, end in zip(starts, ends):
        length = end - start
        width = max(1, math.ceil(length / 4))
        first.extend(errors[start : start + width])
        last.extend(errors[end - width : end])
    return tuple(first), tuple(last)


@dataclass
class _MutableArmState:
    arm: ArmName
    callbacks: ArmCallbacks
    controller: ActionFilterController
    controller_state: ActionFilterState | None = None
    self_prev: torch.Tensor | None = None
    counts: dict[str, int] = field(default_factory=dict)
    row_losses: list[float] = field(default_factory=list)
    update_row_losses: list[float] = field(default_factory=list)
    update_row_positions: list[int] = field(default_factory=list)
    update_original_indices: list[int] = field(default_factory=list)
    update_g7_observations: list[Any] = field(default_factory=list)
    g6_updates: list[G6Update] = field(default_factory=list)
    g7_updates: list[RunnerG7Update] = field(default_factory=list)
    transitions: list[K0ControllerReceipt] = field(default_factory=list)
    branch2_sources: list[PrevSource] = field(default_factory=list)
    reconstruction_errors: list[float] = field(default_factory=list)
    self_errors: list[float] = field(default_factory=list)
    range_violation_count: int = 0
    range_observation_count: int = 0

    def __post_init__(self) -> None:
        self.counts = {
            "rows": 0,
            "feature_forwards": 0,
            "aux_forwards": 0,
            "head_forwards": 0,
            "track_loss_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "controller_steps": 0,
            "static_resets": 0,
            "nonfinite_resets": 0,
            "branch1_logged_rows": 0,
            "branch2_logged_rows": 0,
            "branch2_self_rows": 0,
            "g6_updates": 0,
            "g7_updates": 0,
            "g9_transitions": 0,
            "expert_future_leak_count": 0,
            "self_state_expert_overwrite_count": 0,
        }

    def frozen_counts(self) -> ArmCounts:
        return ArmCounts(**self.counts)


def _reset_arm_state(
    state: _MutableArmState,
    row: RunnerRow,
    reference: torch.Tensor | None = None,
) -> None:
    state.controller_state = state.controller.reset(row.logged_prev_action)
    if reference is None:
        state.self_prev = torch.tensor(
            [[row.logged_prev_action[0], row.logged_prev_action[2]]],
            dtype=torch.float32,
        ).detach()
    else:
        state.self_prev = reference.new_tensor(
            [[row.logged_prev_action[0], row.logged_prev_action[2]]]
        ).detach()


AUDIT_COUNTER_NAMES = (
    "expert_future_leak_count",
    "self_state_expert_overwrite_count",
)


def _apply_adapter_audit_counters(
    state: _MutableArmState, *, required: bool = False
) -> None:
    """Overwrite the placeholder leak counters with adapter-audited values.

    The structural receipt still requires both counters to be exactly zero, so
    any nonzero adapter observation fails the count receipt closed.  Under a
    production plan (``required=True``) a missing callback is itself a
    contract violation: placeholder zeroes must never pass silently.
    """

    if state.callbacks.audit_counters is None:
        if required:
            raise RunnerContractError(
                f"{state.arm} callbacks provide no adapter audit_counters; "
                "the production plan forbids fail-open placeholder zeroes"
            )
        return
    observed = state.callbacks.audit_counters()
    if not isinstance(observed, Mapping):
        raise RunnerContractError(
            f"{state.arm} audit_counters must return a mapping"
        )
    for name in AUDIT_COUNTER_NAMES:
        if name not in observed:
            raise RunnerContractError(
                f"{state.arm} audit_counters is missing {name!r}"
            )
        value = observed[name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise RunnerContractError(
                f"{state.arm} audit counter {name} must be an integer"
            )
        if int(value) < 0:
            raise RunnerContractError(
                f"{state.arm} audit counter {name} must be nonnegative"
            )
        state.counts[name] = int(value)


def _frozen_counts_pass(
    counts: Mapping[ArmName, ArmCounts], expected_static_resets: int
) -> bool:
    structural = {
        "rows": SMOKE_ROWS,
        "feature_forwards": SMOKE_ROWS,
        "aux_forwards": SMOKE_ROWS,
        "head_forwards": SMOKE_ROWS * 2,
        "track_loss_calls": SMOKE_ROWS * 2,
        "backward_calls": SMOKE_ROWS,
        "optimizer_steps": SMOKE_UPDATES,
        "controller_steps": SMOKE_ROWS,
        "static_resets": expected_static_resets,
        "nonfinite_resets": 0,
        "branch1_logged_rows": SMOKE_ROWS,
        "g7_updates": SMOKE_UPDATES,
        "g9_transitions": SMOKE_ROWS,
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }
    for arm in ARM_ORDER:
        for name, expected in structural.items():
            if getattr(counts[arm], name) != expected:
                return False
    ctrl = counts[S_CTRL]
    self_counts = counts[S_SELF]
    return (
        ctrl.branch2_logged_rows == SMOKE_ROWS
        and ctrl.branch2_self_rows == 0
        and ctrl.g6_updates == SMOKE_UPDATES
        and self_counts.branch2_logged_rows == SMOKE_WARMUP_UPDATES * GRAD_ACCUM
        and self_counts.branch2_self_rows
        == SMOKE_ROWS - SMOKE_WARMUP_UPDATES * GRAD_ACCUM
        and self_counts.g6_updates == 0
    )


def run_paired_smoke(
    rows: Sequence[RunnerRow],
    *,
    callbacks: Mapping[ArmName, ArmCallbacks],
    hooks: RunnerTelemetryHooks,
    strafe_reset_original_indices: Set[int],
    expected_static_reset_original_indices: Set[int],
    controller_config: ActionFilterConfig = DEFAULT_CONFIG,
    require_audit_counters: bool = False,
) -> PairedRunResult:
    """Run the exact 256-row, 128-update callback-level paired recurrence.

    ``require_audit_counters=True`` is the production-plan marker: every
    arm's callbacks must then carry a real adapter ``audit_counters``
    callable, checked before any callback runs, so the count receipt can
    never pass on silent placeholder zeroes.
    """

    frozen_rows = _validate_rows(rows)
    if set(callbacks) != set(ARM_ORDER):
        raise RunnerContractError(f"callbacks must contain exactly {ARM_ORDER!r}")
    if not isinstance(hooks, RunnerTelemetryHooks):
        raise RunnerContractError("hooks must be RunnerTelemetryHooks")
    if require_audit_counters:
        for arm in ARM_ORDER:
            if callbacks[arm].audit_counters is None:
                raise RunnerContractError(
                    f"{arm} callbacks provide no adapter audit_counters; "
                    "the production plan forbids fail-open placeholder zeroes"
                )
    strafe_resets = _validate_index_set(
        strafe_reset_original_indices, "strafe_reset_original_indices"
    )
    expected_resets = _validate_index_set(
        expected_static_reset_original_indices,
        "expected_static_reset_original_indices",
    )
    reset_plan = _build_reset_plan(frozen_rows, strafe_resets)
    observed_reset_indices = frozenset(
        row.original_row_index
        for row, reasons in zip(frozen_rows, reset_plan)
        if reasons
    )
    if observed_reset_indices != expected_resets:
        raise RunnerContractError(
            "static reset plan does not match the frozen expected receipt"
        )

    init_hashes = {
        arm: checkpoint_init_sha256(callbacks[arm].checkpoint_state)
        for arm in ARM_ORDER
    }
    if len(set(init_hashes.values())) != 1:
        raise RunnerContractError("paired arms do not share checkpoint init SHA")
    init_sha = init_hashes[S_CTRL]

    states: dict[ArmName, _MutableArmState] = {
        arm: _MutableArmState(
            arm=arm,
            callbacks=callbacks[arm],
            controller=ActionFilterController(controller_config),
        )
        for arm in ARM_ORDER
    }

    for row_position, (row, reset_reasons) in enumerate(
        zip(frozen_rows, reset_plan)
    ):
        u_pre = row_position // GRAD_ACCUM
        for arm in ARM_ORDER:
            state = states[arm]
            reset = bool(reset_reasons)
            if reset:
                _reset_arm_state(state, row)
                state.counts["static_resets"] += 1
            event = RowEvent(
                arm=arm,
                row_position=row_position,
                original_row_index=row.original_row_index,
                u_pre=u_pre,
                reset=reset,
                reset_reasons=reset_reasons,
            )
            feature_result = state.callbacks.feature_forward(
                row.observation, event
            )
            state.counts["feature_forwards"] += 1
            if not isinstance(feature_result, FeatureForwardResult):
                raise RunnerContractError(
                    "feature_forward must return FeatureForwardResult"
                )
            reference = _finite_tensor(
                feature_result.reference_tensor, "feature reference_tensor"
            )
            if state.self_prev is None or state.controller_state is None:
                raise RunnerContractError("runner state was not initialized by reset")
            state.self_prev = state.self_prev.to(
                device=reference.device, dtype=reference.dtype
            ).detach()
            logged_prev = reference.new_tensor(
                [[row.logged_prev_action[0], row.logged_prev_action[2]]]
            )
            target_actions = row.target_actions.to(
                device=reference.device, dtype=reference.dtype
            ).unsqueeze(0)

            aux_result = state.callbacks.aux_forward(
                feature_result.value, row.aux_targets, event
            )
            state.counts["aux_forwards"] += 1
            if not isinstance(aux_result, AuxForwardResult):
                raise RunnerContractError(
                    "aux_forward must return AuxForwardResult"
                )
            aux_loss = _scalar_tensor(aux_result.loss, "L_aux")

            branch1_event = HeadEvent(
                arm=arm,
                row_position=row_position,
                original_row_index=row.original_row_index,
                u_pre=u_pre,
                branch="branch1",
                prev_source="logged",
            )
            branch1 = state.callbacks.head_forward(
                feature_result.value, logged_prev, branch1_event
            )
            state.counts["head_forwards"] += 1
            state.counts["branch1_logged_rows"] += 1
            if not isinstance(branch1, HeadForwardResult):
                raise RunnerContractError(
                    "head_forward must return HeadForwardResult"
                )
            prediction1 = _validate_prediction_shape(
                branch1.prediction, "branch1 prediction"
            )
            _finite_tensor(prediction1.raw_actions, "branch1 raw_actions")
            _finite_tensor(prediction1.delta_fy, "branch1 delta_fy")

            if arm == S_CTRL or u_pre < SMOKE_WARMUP_UPDATES:
                branch2_prev = logged_prev
                branch2_source: PrevSource = "logged"
                state.counts["branch2_logged_rows"] += 1
            else:
                branch2_prev = state.self_prev
                branch2_source = "self"
                state.counts["branch2_self_rows"] += 1
            state.branch2_sources.append(branch2_source)
            branch2_event = HeadEvent(
                arm=arm,
                row_position=row_position,
                original_row_index=row.original_row_index,
                u_pre=u_pre,
                branch="branch2",
                prev_source=branch2_source,
            )
            branch2 = state.callbacks.head_forward(
                feature_result.value, branch2_prev, branch2_event
            )
            state.counts["head_forwards"] += 1
            if not isinstance(branch2, HeadForwardResult):
                raise RunnerContractError(
                    "head_forward must return HeadForwardResult"
                )
            prediction2 = _validate_prediction_shape(
                branch2.prediction, "branch2 prediction"
            )
            k0_fy = _prediction_k0_fy(prediction2)
            if bool((~torch.isfinite(k0_fy)).detach().any().cpu().item()):
                _reset_arm_state(state, row, reference)
                state.counts["nonfinite_resets"] += 1
                receipt = NonfiniteResetReceipt(
                    arm=arm,
                    row_position=row_position,
                    original_row_index=row.original_row_index,
                    u_pre=u_pre,
                    logged_prev_fy=(
                        row.logged_prev_action[0],
                        row.logged_prev_action[2],
                    ),
                    controller_state_after_reset=state.controller_state,
                    self_prev_after_reset=state.self_prev.detach().clone(),
                    nonfinite_reset_count=state.counts["nonfinite_resets"],
                )
                if hooks.on_g9_transition is not None:
                    hooks.on_g9_transition(
                        K0ControllerReceipt(
                            arm=arm,
                            row_position=row_position,
                            original_row_index=row.original_row_index,
                            u_pre=u_pre,
                            reset=reset,
                            reset_reasons=reset_reasons,
                            branch2_prev_source=branch2_source,
                            raw_k0_fy=None,
                            post_safety_clamp=None,
                            rate_limited=None,
                            filtered_sent=None,
                            self_prev_after_fy=receipt.logged_prev_fy,
                            reconstruction_error=None,
                            range_violation_count=0,
                            range_observation_count=0,
                            synchronized_nonfinite_reset=True,
                        )
                    )
                raise RunnerNonfiniteActionError(receipt)
            _finite_tensor(prediction2.raw_actions, "branch2 raw_actions")
            _finite_tensor(prediction2.delta_fy, "branch2 delta_fy")
            _finite_tensor(
                prediction2.bounded_future_actions,
                "branch2 bounded_future_actions",
            )

            track1 = _scalar_tensor(
                state.callbacks.track_loss(
                    prediction1, target_actions, branch1_event
                ),
                "L1",
            )
            state.counts["track_loss_calls"] += 1
            track2 = _scalar_tensor(
                state.callbacks.track_loss(
                    prediction2, target_actions, branch2_event
                ),
                "L2",
            )
            state.counts["track_loss_calls"] += 1
            row_loss = aux_loss + 0.5 * track1 + 0.5 * track2
            _scalar_tensor(row_loss, "paired row loss")
            scaled_loss = row_loss / GRAD_ACCUM
            state.callbacks.backward(
                BackwardEvent(
                    arm=arm,
                    row_position=row_position,
                    original_row_index=row.original_row_index,
                    u_pre=u_pre,
                    unscaled_loss=row_loss,
                    scaled_loss=scaled_loss,
                )
            )
            state.counts["backward_calls"] += 1

            raw_k0 = (float(k0_fy[0].item()), float(k0_fy[1].item()))
            next_controller_state, transition = state.controller.step(
                state.controller_state, raw_k0
            )
            state.controller_state = next_controller_state
            state.counts["controller_steps"] += 1
            state.self_prev = reference.new_tensor(
                [transition.next_prev_fy]
            ).detach()
            reconstruction_error = _prediction_reconstruction_error(
                prediction2, branch2_prev
            )
            range_violations, range_observations = _range_counts(prediction2)
            state.reconstruction_errors.append(reconstruction_error)
            state.self_errors.append(_tensor_float(track2))
            state.range_violation_count += range_violations
            state.range_observation_count += range_observations
            k0_receipt = K0ControllerReceipt(
                arm=arm,
                row_position=row_position,
                original_row_index=row.original_row_index,
                u_pre=u_pre,
                reset=reset,
                reset_reasons=reset_reasons,
                branch2_prev_source=branch2_source,
                raw_k0_fy=raw_k0,
                post_safety_clamp=transition.bounded,
                rate_limited=transition.rate_limited,
                filtered_sent=transition.filtered,
                self_prev_after_fy=transition.next_prev_fy,
                reconstruction_error=reconstruction_error,
                range_violation_count=range_violations,
                range_observation_count=range_observations,
                synchronized_nonfinite_reset=False,
            )
            state.transitions.append(k0_receipt)
            state.counts["g9_transitions"] += 1
            if hooks.on_g9_transition is not None:
                hooks.on_g9_transition(k0_receipt)

            state.update_g7_observations.extend(
                (branch1.g7_telemetry, branch2.g7_telemetry)
            )
            loss_value = _tensor_float(row_loss)
            state.row_losses.append(loss_value)
            state.update_row_losses.append(loss_value)
            state.update_row_positions.append(row_position)
            state.update_original_indices.append(row.original_row_index)
            state.counts["rows"] += 1

        if (row_position + 1) % GRAD_ACCUM == 0:
            for arm in ARM_ORDER:
                state = states[arm]
                if (
                    len(state.update_row_losses) != GRAD_ACCUM
                    or len(state.update_g7_observations) != GRAD_ACCUM * 2
                ):
                    raise RunnerContractError("gradient accumulation count drift")
                update_event = OptimizerUpdateEvent(
                    arm=arm,
                    u_pre=u_pre,
                    row_positions=tuple(state.update_row_positions),
                    original_row_indices=tuple(state.update_original_indices),
                    row_loss_values=tuple(state.update_row_losses),
                    mean_loss=math.fsum(state.update_row_losses) / GRAD_ACCUM,
                )
                g7_update = _aggregate_g7(
                    u_pre, state.update_g7_observations
                )
                state.g7_updates.append(g7_update)
                state.counts["g7_updates"] += 1
                if hooks.on_g7_update is not None:
                    hooks.on_g7_update(arm, g7_update)
                if arm == S_CTRL:
                    g6_update = hooks.g6_update(update_event)
                    if not isinstance(g6_update, G6Update):
                        raise RunnerContractError(
                            "G6 hook must return a G6Update"
                        )
                    if g6_update.u_pre != u_pre:
                        raise RunnerContractError("G6 hook returned wrong u_pre")
                    state.g6_updates.append(g6_update)
                    state.counts["g6_updates"] += 1
                state.callbacks.optimizer_step(update_event)
                state.counts["optimizer_steps"] += 1
                state.update_row_losses.clear()
                state.update_row_positions.clear()
                state.update_original_indices.clear()
                state.update_g7_observations.clear()

    reset_positions = tuple(
        index for index, reasons in enumerate(reset_plan) if reasons
    )
    arm_results: dict[ArmName, ArmRunResult] = {}
    frozen_counts: dict[ArmName, ArmCounts] = {}
    for arm in ARM_ORDER:
        state = states[arm]
        first_errors, last_errors = _quartile_errors(
            state.self_errors, reset_positions
        )
        _apply_adapter_audit_counters(state, required=require_audit_counters)
        counts = state.frozen_counts()
        frozen_counts[arm] = counts
        arm_results[arm] = ArmRunResult(
            arm=arm,
            counts=counts,
            g6_updates=tuple(state.g6_updates),
            g7_updates=tuple(state.g7_updates),
            g9=G9Telemetry(
                expected_static_resets=len(expected_resets),
                observed_static_resets=counts.static_resets,
                nonfinite_reset_count=counts.nonfinite_resets,
                range_violation_count=state.range_violation_count,
                range_observation_count=state.range_observation_count,
                reconstruction_errors=tuple(state.reconstruction_errors),
                first_quartile_self_errors=first_errors,
                last_quartile_self_errors=last_errors,
                transitions=tuple(state.transitions),
            ),
            row_losses=tuple(state.row_losses),
            branch2_sources=tuple(state.branch2_sources),
        )

    passed = _frozen_counts_pass(frozen_counts, len(expected_resets))
    count_receipt = CountReceipt(
        checkpoint_init_sha256=init_sha,
        arms=frozen_counts,
        expected_static_resets=len(expected_resets),
        passed=passed,
    )
    if not passed:
        raise RunnerContractError("paired runner count receipt failed")
    return PairedRunResult(
        checkpoint_init_sha256=init_sha,
        count_receipt=count_receipt,
        arms=arm_results,
        static_reset_original_indices=tuple(sorted(observed_reset_indices)),
    )


__all__ = [
    "ARM_ORDER",
    "AUDIT_COUNTER_NAMES",
    "GRAD_ACCUM",
    "SMOKE_ROWS",
    "SMOKE_UPDATES",
    "SMOKE_WARMUP_UPDATES",
    "S_CTRL",
    "S_SELF",
    "ArmCallbacks",
    "ArmCounts",
    "ArmRunResult",
    "AuxForwardResult",
    "BackwardEvent",
    "CountReceipt",
    "FeatureForwardResult",
    "G9Telemetry",
    "HeadEvent",
    "HeadForwardResult",
    "K0ControllerReceipt",
    "NonfiniteResetReceipt",
    "OptimizerUpdateEvent",
    "PairedRunResult",
    "RowEvent",
    "RunnerContractError",
    "RunnerG7Update",
    "RunnerNonfiniteActionError",
    "RunnerRow",
    "RunnerTelemetryHooks",
    "checkpoint_init_sha256",
    "run_paired_smoke",
]
