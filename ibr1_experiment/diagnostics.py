"""Non-behavioral diagnostics frozen for the IBR1 authoritative smoke.

Observers may use ``torch.autograd.grad`` to inspect live losses against the
dedicated probe, but they never accumulate parameter ``.grad`` or add a
training ``backward`` pass.  They do not rewrite clipping, call
``optimizer.step`` themselves, change ``zero_grad``, or consume RNG.  Missing,
duplicate, out-of-order, mixed-dtype, or nonfinite evidence fails closed before
scientific gate adjudication.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import torch
from torch import nn

from f2_experiment.assembly_data import F2AssemblyContractError
from f2_experiment.assembly_model import F2ArmModules, G6Instrument
from f2_experiment.evaluation import G6Update
from f2_experiment.runner import (
    GRAD_ACCUM,
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    HeadEvent,
    HeadForwardResult,
    OptimizerUpdateEvent,
    RunnerRow,
)

from .assembly_model import (
    ENGINE_TO_FAMILY_ARM,
    FAMILY_TO_ENGINE_ARM,
    IBR1_AUX_COMPONENTS,
    IBR1_CTRL,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    IBR1_SELF,
)
from .model import IBR1Prediction


DIAGNOSTIC_EPS = 1e-12
AXIS_NAMES = ("forward", "yaw")
EVAL_MODES = ("logged", "self")
EVAL_SNAPSHOTS = (
    "update0_IBR1-SELF",
    "update128_IBR1-CTRL",
    "update128_IBR1-SELF",
)
EXACT_G6_UPDATE_FIELD = "exact_g6_update"
EXACT_G6_UPDATE_KEYS = (
    "u_pre",
    "aux_reachable",
    "track_reachable",
    "cosine_total_track",
    "signed_projection",
    "aux_track_ratio",
    "per_aux_ratios",
)


class IBR1DiagnosticsContractError(F2AssemblyContractError):
    """Engineering fail-closed raised before any IBR1 gate verdict."""


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IBR1DiagnosticsContractError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IBR1DiagnosticsContractError(f"{label} is nonfinite")
    return numeric


def exact_g6_update_mapping(update: G6Update, label: str) -> dict[str, Any]:
    if not isinstance(update, G6Update):
        raise IBR1DiagnosticsContractError(f"{label} must be a G6Update")
    if isinstance(update.u_pre, bool) or not isinstance(update.u_pre, int):
        raise IBR1DiagnosticsContractError(f"{label}.u_pre must be an integer")
    if update.u_pre < 0:
        raise IBR1DiagnosticsContractError(f"{label}.u_pre must be nonnegative")
    if not isinstance(update.aux_reachable, bool):
        raise IBR1DiagnosticsContractError(
            f"{label}.aux_reachable must be boolean"
        )
    if not isinstance(update.track_reachable, bool):
        raise IBR1DiagnosticsContractError(
            f"{label}.track_reachable must be boolean"
        )

    optional_scalars: dict[str, float | None] = {}
    for name in (
        "cosine_total_track",
        "signed_projection",
        "aux_track_ratio",
    ):
        value = getattr(update, name)
        if value is None:
            optional_scalars[name] = None
            continue
        if not isinstance(value, float):
            raise IBR1DiagnosticsContractError(
                f"{label}.{name} must be a binary64 float or null"
            )
        optional_scalars[name] = _finite_float(value, f"{label}.{name}")
    cosine = optional_scalars["cosine_total_track"]
    if cosine is not None and not -1.0 <= cosine <= 1.0:
        raise IBR1DiagnosticsContractError(
            f"{label}.cosine_total_track must be in [-1, 1]"
        )
    ratio = optional_scalars["aux_track_ratio"]
    if ratio is not None and ratio < 0.0:
        raise IBR1DiagnosticsContractError(
            f"{label}.aux_track_ratio must be nonnegative"
        )

    per_aux_ratios: dict[str, float] | None = None
    if update.per_aux_ratios is not None:
        if not isinstance(update.per_aux_ratios, Mapping) or set(
            update.per_aux_ratios
        ) != set(IBR1_AUX_COMPONENTS):
            raise IBR1DiagnosticsContractError(
                f"{label}.per_aux_ratios auxiliary names drifted"
            )
        per_aux_ratios = {}
        for name in IBR1_AUX_COMPONENTS:
            value = update.per_aux_ratios[name]
            if not isinstance(value, float):
                raise IBR1DiagnosticsContractError(
                    f"{label}.per_aux_ratios.{name} must be a binary64 float"
                )
            numeric = _finite_float(value, f"{label}.per_aux_ratios.{name}")
            if numeric < 0.0:
                raise IBR1DiagnosticsContractError(
                    f"{label}.per_aux_ratios.{name} must be nonnegative"
                )
            per_aux_ratios[name] = numeric

    if update.u_pre < 8:
        if any(value is not None for value in optional_scalars.values()) or (
            per_aux_ratios is not None
        ):
            raise IBR1DiagnosticsContractError(
                f"{label} emits forbidden pre-window geometry"
            )
    elif (
        optional_scalars["cosine_total_track"] is None
        or optional_scalars["signed_projection"] is None
        or optional_scalars["aux_track_ratio"] is None
        or per_aux_ratios is not None
    ):
        raise IBR1DiagnosticsContractError(
            f"{label} must contain complete B* gradient-window geometry"
        )

    return {
        "u_pre": update.u_pre,
        "aux_reachable": update.aux_reachable,
        "track_reachable": update.track_reachable,
        **optional_scalars,
        "per_aux_ratios": per_aux_ratios,
    }


def validate_exact_g6_update_mapping(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(EXACT_G6_UPDATE_KEYS):
        raise IBR1DiagnosticsContractError(f"{label} schema drifted")
    return exact_g6_update_mapping(
        G6Update(
            u_pre=value["u_pre"],
            aux_reachable=value["aux_reachable"],
            track_reachable=value["track_reachable"],
            cosine_total_track=value["cosine_total_track"],
            signed_projection=value["signed_projection"],
            aux_track_ratio=value["aux_track_ratio"],
            per_aux_ratios=value["per_aux_ratios"],
        ),
        label,
    )


def _finite_vector(value: torch.Tensor, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise IBR1DiagnosticsContractError(f"{label} must be a vector tensor")
    vector = value.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(vector).all().item()):
        raise IBR1DiagnosticsContractError(f"{label} is nonfinite")
    return vector


def _vector_norm(vector: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(vector).item())


def _dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.dot(left, right).item())


def _cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm <= DIAGNOSTIC_EPS or right_norm <= DIAGNOSTIC_EPS:
        return 0.0
    value = _dot(left, right) / (left_norm * right_norm)
    return max(-1.0, min(1.0, value))


def _projection(vector: torch.Tensor, track: torch.Tensor, track_norm: float) -> float:
    if track_norm <= DIAGNOSTIC_EPS:
        return 0.0
    return _dot(vector, track) / track_norm


def _grad_norm(parameters: Sequence[nn.Parameter], label: str) -> float:
    squares: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        detached = gradient.detach().to(device="cpu", dtype=torch.float64)
        if not bool(torch.isfinite(detached).all().item()):
            raise IBR1DiagnosticsContractError(f"{label} gradient is nonfinite")
        squares.append(torch.sum(detached * detached))
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).item())


def _parameter_snapshot(parameters: Sequence[nn.Parameter]) -> tuple[torch.Tensor, ...]:
    values: list[torch.Tensor] = []
    for parameter in parameters:
        detached = parameter.detach().to(device="cpu", dtype=torch.float64).clone()
        if not bool(torch.isfinite(detached).all().item()):
            raise IBR1DiagnosticsContractError("parameter snapshot is nonfinite")
        values.append(detached)
    return tuple(values)


def _parameter_update_norm(
    parameters: Sequence[nn.Parameter],
    before: Sequence[torch.Tensor],
    label: str,
) -> float:
    if len(parameters) != len(before):
        raise IBR1DiagnosticsContractError(f"{label} snapshot cardinality drift")
    squares: list[torch.Tensor] = []
    for parameter, previous in zip(parameters, before):
        current = parameter.detach().to(device="cpu", dtype=torch.float64)
        difference = current - previous
        if not bool(torch.isfinite(difference).all().item()):
            raise IBR1DiagnosticsContractError(f"{label} update is nonfinite")
        squares.append(torch.sum(difference * difference))
    return float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0


@dataclass
class _ActiveOptimizerUpdate:
    event: OptimizerUpdateEvent
    engine_arm: str
    optimizer: torch.optim.AdamW
    full_parameters: tuple[nn.Parameter, ...]
    base_parameters: tuple[nn.Parameter, ...]
    head_parameters: tuple[nn.Parameter, ...]
    base_before: tuple[torch.Tensor, ...]
    head_before: tuple[torch.Tensor, ...]
    record: dict[str, Any]
    pre_hook_calls: int = 0
    post_hook_calls: int = 0


class GradientDiagnosticsCollector:
    """Collect absolute G6 contribution and untouched AdamW geometry."""

    def __init__(
        self,
        *,
        aux_coefficients: Mapping[str, float] = IBR1_FROZEN_AUX_COEFFICIENTS,
        expected_gradient_updates: int = 128,
        expected_optimizer_updates_per_arm: int = 128,
    ) -> None:
        if set(aux_coefficients) != set(IBR1_AUX_COMPONENTS):
            raise IBR1DiagnosticsContractError("diagnostic aux coefficient keys drift")
        self.aux_coefficients = {
            name: _finite_float(aux_coefficients[name], f"lambda {name}")
            for name in IBR1_AUX_COMPONENTS
        }
        if any(value <= 0.0 for value in self.aux_coefficients.values()):
            raise IBR1DiagnosticsContractError(
                "raw per-aux norms require strictly positive frozen lambdas"
            )
        self.expected_gradient_updates = int(expected_gradient_updates)
        self.expected_optimizer_updates_per_arm = int(
            expected_optimizer_updates_per_arm
        )
        if self.expected_gradient_updates <= 0 or self.expected_optimizer_updates_per_arm <= 0:
            raise IBR1DiagnosticsContractError("expected update counts must be positive")
        self.gradient_records: list[dict[str, Any]] = []
        self.optimizer_records: list[dict[str, Any]] = []
        self._active_by_optimizer: dict[int, _ActiveOptimizerUpdate] = {}

    def observe_contributions(
        self,
        event: OptimizerUpdateEvent,
        *,
        g_track: torch.Tensor,
        g_aux: torch.Tensor,
        g_aux_joint: torch.Tensor,
        per_aux: Mapping[str, torch.Tensor],
    ) -> None:
        if not isinstance(event, OptimizerUpdateEvent) or event.arm != S_CTRL:
            raise IBR1DiagnosticsContractError(
                "absolute G6 contribution diagnostics are S-CTRL only"
            )
        if event.u_pre != len(self.gradient_records):
            raise IBR1DiagnosticsContractError("gradient diagnostic clock discontinuity")
        track = _finite_vector(g_track, "g_track")
        aux = _finite_vector(g_aux, "g_aux")
        joint_aux = _finite_vector(g_aux_joint, "g_aux_joint")
        if track.shape != aux.shape or joint_aux.shape != aux.shape:
            raise IBR1DiagnosticsContractError(
                "g_track/g_aux/joint-aux vector shape drift"
            )
        if set(per_aux) != set(IBR1_AUX_COMPONENTS):
            raise IBR1DiagnosticsContractError("per-aux contribution key drift")
        per_aux_vectors = {
            name: _finite_vector(per_aux[name], f"per_aux[{name}]")
            for name in IBR1_AUX_COMPONENTS
        }
        if any(vector.shape != track.shape for vector in per_aux_vectors.values()):
            raise IBR1DiagnosticsContractError("per-aux contribution shape drift")
        aggregate_discrepancy = torch.abs(joint_aux - aux)
        # The production graph crosses a frozen BF16 LLM.  Three independent
        # VJPs are useful descriptive geometry, but their FP64 sum is not a
        # valid reconstruction of the one-pass aggregate VJP.  Integrity is
        # instead proved by a second one-pass joint VJP over the exact ordered
        # per-aux scalar tensors.  It must be exactly equal under
        # ``torch.equal`` on the same device and dtype as the live aggregate
        # VJP; no empirical rounding allowance is authoritative.
        if not torch.equal(joint_aux, aux):
            raise IBR1DiagnosticsContractError(
                "joint weighted per-aux VJP does not reconstruct aggregate g_aux"
            )

        track_norm = _vector_norm(track)
        aux_norm = _vector_norm(aux)
        total = track + aux
        total_norm = _vector_norm(total)
        dot = _dot(aux, track)
        denominator = max(track_norm, DIAGNOSTIC_EPS)
        per_weighted_norm: dict[str, float] = {}
        per_raw_norm: dict[str, float] = {}
        per_cosine: dict[str, float] = {}
        per_projection: dict[str, float] = {}
        per_near_zero: dict[str, bool] = {}
        for name, vector in per_aux_vectors.items():
            norm = _vector_norm(vector)
            per_weighted_norm[name] = norm
            per_raw_norm[name] = norm / self.aux_coefficients[name]
            per_cosine[name] = _cosine(
                vector, track, left_norm=norm, right_norm=track_norm
            )
            per_projection[name] = _projection(vector, track, track_norm)
            per_near_zero[name] = norm <= DIAGNOSTIC_EPS

        record = {
            "u_pre": event.u_pre,
            "engine_arm": S_CTRL,
            "arm": IBR1_CTRL,
            "grad_accum": event.grad_accum,
            "track_grad_norm": track_norm,
            "weighted_aux_grad_norm": aux_norm,
            "total_grad_norm": total_norm,
            "weighted_aux_track_dot": dot,
            "weighted_aux_track_cosine": _cosine(
                aux, track, left_norm=aux_norm, right_norm=track_norm
            ),
            "weighted_aux_signed_projection": _projection(aux, track, track_norm),
            "per_aux_weighted_grad_norm": per_weighted_norm,
            "per_aux_raw_grad_norm_derived_from_frozen_lambda": per_raw_norm,
            "per_aux_cosine_to_track": per_cosine,
            "per_aux_signed_projection_to_track": per_projection,
            "track_norm_below_eps": track_norm <= DIAGNOSTIC_EPS,
            "weighted_aux_norm_below_eps": aux_norm <= DIAGNOSTIC_EPS,
            "total_norm_below_eps": total_norm <= DIAGNOSTIC_EPS,
            "per_aux_norm_below_eps": per_near_zero,
            "actual_ratio_denominator": denominator,
            "per_aux_aggregate_discrepancy_norm": _vector_norm(
                aggregate_discrepancy
            ),
            "per_aux_aggregate_rounding_bound_norm": 0.0,
        }
        self.gradient_records.append(record)

    def attach_exact_g6_update(
        self,
        event: OptimizerUpdateEvent,
        update: G6Update,
    ) -> None:
        if not isinstance(event, OptimizerUpdateEvent) or event.arm != S_CTRL:
            raise IBR1DiagnosticsContractError(
                "exact G6 update attachment is S-CTRL only"
            )
        if len(self.gradient_records) != event.u_pre + 1:
            raise IBR1DiagnosticsContractError(
                "exact G6 update attachment clock is not the latest gradient record"
            )
        record = self.gradient_records[event.u_pre]
        if record.get("u_pre") != event.u_pre:
            raise IBR1DiagnosticsContractError(
                "exact G6 update attachment record clock drifted"
            )
        if EXACT_G6_UPDATE_FIELD in record:
            raise IBR1DiagnosticsContractError(
                "exact G6 update was attached more than once"
            )
        mapping = exact_g6_update_mapping(
            update,
            f"exact G6 update {event.u_pre}",
        )
        if mapping["u_pre"] != event.u_pre:
            raise IBR1DiagnosticsContractError(
                "exact G6 update attachment clock differs from the live event"
            )
        record[EXACT_G6_UPDATE_FIELD] = mapping

    def begin_optimizer_update(
        self,
        event: OptimizerUpdateEvent,
        *,
        optimizer: torch.optim.AdamW,
        modules: F2ArmModules,
    ) -> None:
        if not isinstance(event, OptimizerUpdateEvent):
            raise IBR1DiagnosticsContractError("optimizer event type mismatch")
        if event.arm not in (S_CTRL, S_SELF):
            raise IBR1DiagnosticsContractError("unknown engine arm")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise IBR1DiagnosticsContractError("optimizer must be AdamW")
        if not isinstance(modules, F2ArmModules) or not isinstance(
            modules.model.action_head, nn.Module
        ):
            raise IBR1DiagnosticsContractError("optimizer modules are malformed")
        optimizer_id = id(optimizer)
        if optimizer_id in self._active_by_optimizer:
            raise IBR1DiagnosticsContractError("nested optimizer diagnostic update")
        observed_for_arm = sum(
            record["engine_arm"] == event.arm for record in self.optimizer_records
        )
        if event.u_pre != observed_for_arm:
            raise IBR1DiagnosticsContractError("optimizer diagnostic clock discontinuity")

        full_parameters = tuple(modules.trainable_parameters())
        base_parameters = tuple(
            parameter
            for parameter in modules.base.proj.parameters()
            if parameter.requires_grad
        )
        head_parameters = tuple(
            parameter
            for parameter in modules.model.action_head.parameters()
            if parameter.requires_grad
        )
        if not full_parameters or not base_parameters or not head_parameters:
            raise IBR1DiagnosticsContractError("optimizer diagnostic parameter set is empty")
        record = {
            "u_pre": event.u_pre,
            "engine_arm": event.arm,
            "arm": ENGINE_TO_FAMILY_ARM[event.arm],
            "pre_clip_full_grad_norm": _grad_norm(
                full_parameters, "pre-clip full"
            ),
            "base_proj_pre_clip_grad_norm": _grad_norm(
                base_parameters, "pre-clip base.proj"
            ),
            "action_head_pre_clip_grad_norm": _grad_norm(
                head_parameters, "pre-clip action head"
            ),
        }
        self._active_by_optimizer[optimizer_id] = _ActiveOptimizerUpdate(
            event=event,
            engine_arm=event.arm,
            optimizer=optimizer,
            full_parameters=full_parameters,
            base_parameters=base_parameters,
            head_parameters=head_parameters,
            base_before=_parameter_snapshot(base_parameters),
            head_before=_parameter_snapshot(head_parameters),
            record=record,
        )

    def optimizer_pre_step_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        active = self._active_by_optimizer.get(id(optimizer))
        if active is None:
            raise IBR1DiagnosticsContractError(
                "AdamW pre-hook ran outside a registered update"
            )
        active.pre_hook_calls += 1
        if active.pre_hook_calls != 1:
            raise IBR1DiagnosticsContractError("AdamW pre-hook ran more than once")
        active.record.update(
            {
                "post_clip_full_grad_norm": _grad_norm(
                    active.full_parameters, "post-clip full"
                ),
                "base_proj_post_clip_grad_norm": _grad_norm(
                    active.base_parameters, "post-clip base.proj"
                ),
                "action_head_post_clip_grad_norm": _grad_norm(
                    active.head_parameters, "post-clip action head"
                ),
            }
        )

    def optimizer_post_step_hook(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        active = self._active_by_optimizer.get(id(optimizer))
        if active is None:
            raise IBR1DiagnosticsContractError(
                "AdamW post-hook ran outside a registered update"
            )
        if active.pre_hook_calls != 1:
            raise IBR1DiagnosticsContractError("AdamW post-hook preceded pre-hook")
        active.post_hook_calls += 1
        if active.post_hook_calls != 1:
            raise IBR1DiagnosticsContractError("AdamW post-hook ran more than once")
        active.record.update(
            {
                "base_proj_parameter_update_norm": _parameter_update_norm(
                    active.base_parameters,
                    active.base_before,
                    "base.proj parameter",
                ),
                "action_head_parameter_update_norm": _parameter_update_norm(
                    active.head_parameters,
                    active.head_before,
                    "action-head parameter",
                ),
            }
        )

    def finish_optimizer_update(self, optimizer: torch.optim.AdamW) -> None:
        active = self._active_by_optimizer.pop(id(optimizer), None)
        if active is None:
            raise IBR1DiagnosticsContractError("optimizer update was not begun")
        if active.pre_hook_calls != 1 or active.post_hook_calls != 1:
            raise IBR1DiagnosticsContractError(
                "optimizer step hooks did not observe exactly one real step"
            )
        required = (
            "post_clip_full_grad_norm",
            "base_proj_post_clip_grad_norm",
            "action_head_post_clip_grad_norm",
            "base_proj_parameter_update_norm",
            "action_head_parameter_update_norm",
        )
        if any(name not in active.record for name in required):
            raise IBR1DiagnosticsContractError("optimizer diagnostic fields are missing")
        self.optimizer_records.append(active.record)

    def abort_optimizer_update(self, optimizer: torch.optim.AdamW) -> None:
        self._active_by_optimizer.pop(id(optimizer), None)

    def finalize(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if self._active_by_optimizer:
            raise IBR1DiagnosticsContractError("optimizer diagnostics remain active")
        if len(self.gradient_records) != self.expected_gradient_updates:
            raise IBR1DiagnosticsContractError(
                "gradient geometry cardinality mismatch"
            )
        for index, record in enumerate(self.gradient_records):
            if record.get("u_pre") != index:
                raise IBR1DiagnosticsContractError(
                    "gradient geometry clock mismatch"
                )
            exact_update = validate_exact_g6_update_mapping(
                record.get(EXACT_G6_UPDATE_FIELD),
                f"gradient record {index} exact G6 update",
            )
            if exact_update["u_pre"] != index:
                raise IBR1DiagnosticsContractError(
                    "gradient record exact G6 update clock mismatch"
                )
        expected_optimizer = 2 * self.expected_optimizer_updates_per_arm
        if len(self.optimizer_records) != expected_optimizer:
            raise IBR1DiagnosticsContractError(
                "optimizer geometry cardinality mismatch"
            )
        for engine_arm in (S_CTRL, S_SELF):
            arm_records = [
                record
                for record in self.optimizer_records
                if record["engine_arm"] == engine_arm
            ]
            if [record["u_pre"] for record in arm_records] != list(
                range(self.expected_optimizer_updates_per_arm)
            ):
                raise IBR1DiagnosticsContractError(
                    f"optimizer geometry clock mismatch for {engine_arm}"
                )
        return (
            {
                "schema_version": 1,
                "analysis_class": "ibr1_gradient_geometry",
                "family_id": "IBR1",
                "deciding_arm": IBR1_CTRL,
                "records": list(self.gradient_records),
                "internal_test": "sealed",
                "internal_test_opened": False,
            },
            {
                "schema_version": 1,
                "analysis_class": "ibr1_optimizer_geometry",
                "family_id": "IBR1",
                "records": list(self.optimizer_records),
                "records_per_arm": {
                    IBR1_CTRL: self.expected_optimizer_updates_per_arm,
                    IBR1_SELF: self.expected_optimizer_updates_per_arm,
                },
                "internal_test": "sealed",
                "internal_test_opened": False,
            },
        )


class IBR1G6Instrument(G6Instrument):
    """F2 G6 gate producer plus pre-clear absolute contribution evidence."""

    def __init__(
        self,
        probe: Sequence[nn.Parameter],
        collector: GradientDiagnosticsCollector,
        *,
        block_mode: str = "bstar",
        rows_per_update: int = GRAD_ACCUM,
    ) -> None:
        if not isinstance(collector, GradientDiagnosticsCollector):
            raise IBR1DiagnosticsContractError(
                "collector must be GradientDiagnosticsCollector"
            )
        super().__init__(
            probe, block_mode=block_mode, rows_per_update=rows_per_update
        )
        probe_devices = {parameter.device for parameter in self.probe}
        probe_dtypes = {parameter.dtype for parameter in self.probe}
        if len(probe_devices) != 1 or probe_dtypes != {torch.float32}:
            raise IBR1DiagnosticsContractError(
                "joint G6 probe must use one device and float32 parameters"
            )
        self._probe_device = next(iter(probe_devices))
        self.collector = collector
        self._sum_joint_aux: torch.Tensor | None = None
        self._row_direct_aux: torch.Tensor | None = None

    def _grad_vector(self, loss: Any, label: str) -> torch.Tensor:
        if not isinstance(loss, torch.Tensor):
            raise IBR1DiagnosticsContractError(f"{label} must be a torch.Tensor")
        if loss.numel() != 1:
            raise IBR1DiagnosticsContractError(f"{label} must be a scalar loss")
        requires_probe_connection = label == "G6 aux_loss" or label.startswith(
            "G6 per-aux "
        )
        if not loss.requires_grad:
            if requires_probe_connection:
                raise IBR1DiagnosticsContractError(
                    f"{label} must remain connected to the G6 probe"
                )
            return self._zero_vector()
        gradients = torch.autograd.grad(
            loss,
            self.probe,
            retain_graph=True,
            allow_unused=True,
        )
        if requires_probe_connection and all(
            gradient is None for gradient in gradients
        ):
            raise IBR1DiagnosticsContractError(
                f"{label} must reach at least one G6 probe parameter"
            )
        pieces: list[torch.Tensor] = []
        for parameter, gradient in zip(self.probe, gradients):
            if gradient is None:
                pieces.append(
                    torch.zeros(parameter.numel(), dtype=torch.float64)
                )
            else:
                pieces.append(
                    gradient.detach().reshape(-1).to("cpu").to(torch.float64)
                )
        vector = torch.cat(pieces)
        if not bool(torch.isfinite(vector).all().item()):
            raise IBR1DiagnosticsContractError(f"{label} probe gradient is nonfinite")
        if label == "G6 aux_loss":
            if self._row_direct_aux is not None:
                raise IBR1DiagnosticsContractError(
                    "G6 direct aux gradient was captured more than once per row"
                )
            self._row_direct_aux = vector
        return vector

    def _joint_grad_vector(
        self, losses: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        active_losses = tuple(loss for loss in losses if loss.requires_grad)
        if active_losses:
            gradients = torch.autograd.grad(
                active_losses,
                self.probe,
                grad_outputs=tuple(torch.ones_like(loss) for loss in active_losses),
                retain_graph=True,
                allow_unused=True,
            )
        else:
            gradients = (None,) * len(self.probe)
        pieces: list[torch.Tensor] = []
        for parameter, gradient in zip(self.probe, gradients):
            if gradient is None:
                pieces.append(
                    torch.zeros(parameter.numel(), dtype=torch.float64)
                )
            else:
                pieces.append(
                    gradient.detach().reshape(-1).to("cpu").to(torch.float64)
                )
        vector = torch.cat(pieces)
        if not bool(torch.isfinite(vector).all().item()):
            raise IBR1DiagnosticsContractError(
                "G6 joint weighted per-aux probe gradient is nonfinite"
            )
        return vector

    def observe_row(
        self,
        *,
        aux_loss: torch.Tensor,
        track1: torch.Tensor,
        track2: torch.Tensor,
        per_aux_losses: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self._row_direct_aux is not None:
            raise IBR1DiagnosticsContractError(
                "G6 direct aux gradient residue detected before row"
            )
        if not isinstance(per_aux_losses, Mapping) or set(per_aux_losses) != set(
            IBR1_AUX_COMPONENTS
        ):
            raise IBR1DiagnosticsContractError(
                "joint G6 diagnostics require the exact per-aux loss set"
            )
        if not isinstance(aux_loss, torch.Tensor) or aux_loss.numel() != 1:
            raise IBR1DiagnosticsContractError("joint G6 aux loss must be scalar")
        if aux_loss.device != self._probe_device or aux_loss.dtype != torch.float32:
            raise IBR1DiagnosticsContractError(
                "joint G6 aggregate aux loss must match the float32 probe device"
            )
        ordered_losses: list[torch.Tensor] = []
        for name in IBR1_AUX_COMPONENTS:
            loss = per_aux_losses[name]
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise IBR1DiagnosticsContractError(
                    f"joint G6 per-aux loss {name} must be scalar"
                )
            if not loss.requires_grad:
                raise IBR1DiagnosticsContractError(
                    f"joint G6 per-aux loss {name} must remain graph-connected"
                )
            if loss.device != aux_loss.device or loss.dtype != aux_loss.dtype:
                raise IBR1DiagnosticsContractError(
                    "joint G6 per-aux loss device/dtype drift"
                )
            ordered_losses.append(loss)
        joint_loss = ordered_losses[0]
        for loss in ordered_losses[1:]:
            joint_loss = joint_loss + loss
        if not torch.equal(aux_loss.detach(), joint_loss.detach()):
            raise IBR1DiagnosticsContractError(
                "ordered per-aux scalars do not reconstruct aggregate aux loss"
            )

        # The name-to-loss meaning is frozen by the authority-bound
        # ArmExecutor/adapter source.  This live check proves that the exact
        # three named tensors are complete as an aggregate; it deliberately
        # does not treat separately rounded VJPs as semantic identifiers.
        super().observe_row(
            aux_loss=aux_loss,
            track1=track1,
            track2=track2,
            per_aux_losses=per_aux_losses,
        )
        if self._row_direct_aux is None:
            raise IBR1DiagnosticsContractError(
                "G6 direct aux gradient was not captured"
            )
        joint_vector = self._joint_grad_vector(ordered_losses)
        if not torch.equal(self._row_direct_aux, joint_vector):
            raise IBR1DiagnosticsContractError(
                "joint weighted per-aux VJP does not reconstruct row aggregate g_aux"
            )
        if self._sum_joint_aux is None:
            self._sum_joint_aux = self._zero_vector()
        self._sum_joint_aux = self._sum_joint_aux + joint_vector
        self._row_direct_aux = None

    def _clear(self) -> None:
        super()._clear()
        self._sum_joint_aux = None
        self._row_direct_aux = None

    def emit_update(self, event: OptimizerUpdateEvent):
        if event.grad_accum != self.rows_per_update:
            raise IBR1DiagnosticsContractError(
                "G6 diagnostic grad_accum differs from the live accumulator"
            )
        if self._sum_aux is None or self._sum_track is None:
            raise IBR1DiagnosticsContractError("G6 diagnostic accumulators are empty")
        if not self._per_aux_sums:
            raise IBR1DiagnosticsContractError("G6 per-aux diagnostics are empty")
        if self._sum_joint_aux is None:
            raise IBR1DiagnosticsContractError("G6 joint aux diagnostics are empty")
        divisor = float(event.grad_accum)
        g_aux = self._sum_aux.detach().clone() / divisor
        g_aux_joint = self._sum_joint_aux.detach().clone() / divisor
        g_track = self._sum_track.detach().clone() / divisor
        per_aux = {
            name: vector.detach().clone() / divisor
            for name, vector in self._per_aux_sums.items()
        }
        self.collector.observe_contributions(
            event,
            g_track=g_track,
            g_aux=g_aux,
            g_aux_joint=g_aux_joint,
            per_aux=per_aux,
        )
        update = super().emit_update(event)
        self.collector.attach_exact_g6_update(event, update)
        return update


class OptimizerDiagnosticsHandle:
    """Wrap one real callback and observe its existing clip/step path."""

    def __init__(
        self,
        callbacks: ArmCallbacks,
        *,
        optimizer: torch.optim.AdamW,
        modules: F2ArmModules,
        collector: GradientDiagnosticsCollector,
        engine_arm: str,
    ) -> None:
        if engine_arm not in (S_CTRL, S_SELF):
            raise IBR1DiagnosticsContractError("unknown optimizer engine arm")
        if not isinstance(callbacks, ArmCallbacks):
            raise IBR1DiagnosticsContractError("callbacks must be ArmCallbacks")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise IBR1DiagnosticsContractError("optimizer must be AdamW")
        self.original = callbacks.optimizer_step
        self.optimizer = optimizer
        self.modules = modules
        self.collector = collector
        self.engine_arm = engine_arm
        self.pre_handle = optimizer.register_step_pre_hook(
            collector.optimizer_pre_step_hook
        )
        self.post_handle = optimizer.register_step_post_hook(
            collector.optimizer_post_step_hook
        )
        self.callbacks = replace(callbacks, optimizer_step=self.optimizer_step)

    def optimizer_step(self, event: OptimizerUpdateEvent) -> None:
        if event.arm != self.engine_arm:
            raise IBR1DiagnosticsContractError(
                "optimizer callback received the wrong engine arm"
            )
        self.collector.begin_optimizer_update(
            event, optimizer=self.optimizer, modules=self.modules
        )
        try:
            self.original(event)
            self.collector.finish_optimizer_update(self.optimizer)
        except BaseException:
            self.collector.abort_optimizer_update(self.optimizer)
            raise

    def close(self) -> None:
        self.pre_handle.remove()
        self.post_handle.remove()


def _require_fp32_prediction(prediction: Any, label: str) -> IBR1Prediction:
    if not isinstance(prediction, IBR1Prediction):
        raise IBR1DiagnosticsContractError(f"{label} is not an IBR1Prediction")
    tensors = (
        prediction.raw_fy,
        prediction.delta_fy,
        prediction.latent_delta_fy,
        prediction.cumulative_latent_fy,
        prediction.additive_prebound_fy,
        prediction.normalizer_fy,
        prediction.prebound_overshoot_fy,
        prediction.boundary_margin_fy,
    )
    reference = tensors[0]
    if reference.dtype != torch.float32:
        raise IBR1DiagnosticsContractError(
            "authoritative IBR1 geometry must use torch.float32"
        )
    if any(tensor.device != reference.device or tensor.dtype != reference.dtype for tensor in tensors):
        raise IBR1DiagnosticsContractError("IBR1 geometry dtype/device drift")
    if any(tensor.shape != (1, 8, 2) for tensor in tensors):
        raise IBR1DiagnosticsContractError("IBR1 geometry tensor shape drift")
    if any(not bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise IBR1DiagnosticsContractError("IBR1 geometry contains nonfinite values")
    mask = prediction.prebound_violation_mask
    if (
        not isinstance(mask, torch.Tensor)
        or mask.shape != (1, 8, 2)
        or mask.dtype != torch.bool
        or mask.device != reference.device
    ):
        raise IBR1DiagnosticsContractError(
            "IBR1 prebound violation mask shape/dtype/device drift"
        )
    expected_mask = torch.abs(prediction.additive_prebound_fy) > 1.0
    if not torch.equal(mask, expected_mask):
        raise IBR1DiagnosticsContractError(
            "IBR1 prebound violation mask differs from additive geometry"
        )
    return prediction


def _matrix(value: torch.Tensor) -> list[list[float]]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    return [[float(cell) for cell in row] for row in tensor[0].tolist()]


def _mask_matrix(value: torch.Tensor) -> list[list[bool]]:
    tensor = value.detach().to(device="cpu", dtype=torch.bool)
    return [[bool(cell) for cell in row] for row in tensor[0].tolist()]


def _vector2(value: torch.Tensor, label: str) -> list[float]:
    if not isinstance(value, torch.Tensor) or value.shape != (1, 2):
        raise IBR1DiagnosticsContractError(f"{label} must have shape (1,2)")
    if value.dtype != torch.float32 or not bool(torch.isfinite(value).all().item()):
        raise IBR1DiagnosticsContractError(f"{label} must be finite float32")
    return [float(cell) for cell in value.detach().to("cpu")[0].tolist()]


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise IBR1DiagnosticsContractError("nearest-rank universe is empty")
    ordered = sorted(_finite_float(value, "quantile value") for value in values)
    rank = max(1, min(len(ordered), math.ceil(quantile * len(ordered))))
    return ordered[rank - 1]


def _quantile_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise IBR1DiagnosticsContractError("quantile breakdown universe is empty")
    finite = [_finite_float(value, "quantile value") for value in values]
    return {
        "support": len(finite),
        "max": max(finite),
        "p50": _nearest_rank(finite, 0.5),
        "p90": _nearest_rank(finite, 0.9),
        "p99": _nearest_rank(finite, 0.99),
    }


def _time_bin(u_pre: int) -> str:
    if 0 <= u_pre <= 7:
        return "0-7"
    if 8 <= u_pre <= 15:
        return "8-15"
    lower = 16 + ((u_pre - 16) // 16) * 16
    return f"{lower}-{lower + 15}"


class GeometryCollector:
    """Collect strict SMK-TRAIN branch2 and EVAL-FIX IBR1 geometry."""

    def __init__(
        self,
        *,
        expected_training_rows_per_arm: int = 256,
        expected_eval_rows_per_snapshot_mode: int = 512,
    ) -> None:
        self.expected_training_rows_per_arm = int(expected_training_rows_per_arm)
        self.expected_eval_rows_per_snapshot_mode = int(
            expected_eval_rows_per_snapshot_mode
        )
        if self.expected_training_rows_per_arm <= 0 or self.expected_eval_rows_per_snapshot_mode <= 0:
            raise IBR1DiagnosticsContractError("geometry expected counts must be positive")
        self.training_records: list[dict[str, Any]] = []
        self.eval_records: list[dict[str, Any]] = []

    def observe_training(
        self,
        result: HeadForwardResult,
        prev_fy: torch.Tensor,
        event: HeadEvent,
    ) -> None:
        if event.branch != "branch2":
            raise IBR1DiagnosticsContractError(
                "training geometry may observe branch2 only"
            )
        if event.arm not in (S_CTRL, S_SELF):
            raise IBR1DiagnosticsContractError("unknown training engine arm")
        if event.u_pre != event.row_position // GRAD_ACCUM:
            raise IBR1DiagnosticsContractError(
                "training geometry u_pre differs from the frozen row clock"
            )
        prediction = _require_fp32_prediction(
            result.prediction, "training prediction"
        )
        prev = _vector2(prev_fy, "training prev_fy")
        observed = sum(
            record["engine_arm"] == event.arm for record in self.training_records
        )
        if event.row_position != observed:
            raise IBR1DiagnosticsContractError(
                "training geometry row-position discontinuity"
            )
        reconstructed = prev_fy.unsqueeze(-2) + torch.cumsum(
            prediction.delta_fy, dim=-2
        )
        geometry_reconstruction = (
            prediction.additive_prebound_fy / prediction.normalizer_fy
        )
        geometry_error = float(
            torch.max(torch.abs(geometry_reconstruction - prediction.raw_fy))
            .detach()
            .to("cpu")
            .item()
        )
        telescoping_error = float(
            torch.max(torch.abs(reconstructed - prediction.raw_fy))
            .detach()
            .to("cpu")
            .item()
        )
        record = {
            "arm": ENGINE_TO_FAMILY_ARM[event.arm],
            "engine_arm": event.arm,
            "row_position": event.row_position,
            "original_row_index": event.original_row_index,
            "u_pre": event.u_pre,
            "row_within_update": event.row_position % GRAD_ACCUM,
            "branch": event.branch,
            "prev_source": event.prev_source,
            "prev_fy": prev,
            "latent_delta_fy": _matrix(prediction.latent_delta_fy),
            "cumulative_latent_fy": _matrix(prediction.cumulative_latent_fy),
            "additive_prebound_fy": _matrix(prediction.additive_prebound_fy),
            "normalizer_fy": _matrix(prediction.normalizer_fy),
            "raw_fy": _matrix(prediction.raw_fy),
            "realized_delta_fy": _matrix(prediction.delta_fy),
            "prebound_violation_mask": _mask_matrix(
                prediction.prebound_violation_mask
            ),
            "prebound_overshoot_fy": _matrix(
                prediction.prebound_overshoot_fy
            ),
            "boundary_margin_fy": _matrix(prediction.boundary_margin_fy),
            "geometry_reconstruction_error": geometry_error,
            "telescoping_reconstruction_error": telescoping_error,
        }
        self.training_records.append(record)

    def observe_eval(
        self,
        prediction_value: Any,
        prev_fy: torch.Tensor,
        row: RunnerRow,
        *,
        family_arm: str,
        snapshot: str,
        mode: str,
        position: int,
    ) -> None:
        prediction = _require_fp32_prediction(prediction_value, "EVAL prediction")
        if family_arm not in (IBR1_CTRL, IBR1_SELF):
            raise IBR1DiagnosticsContractError("unknown IBR1 EVAL arm")
        if snapshot not in EVAL_SNAPSHOTS or mode not in EVAL_MODES:
            raise IBR1DiagnosticsContractError("unknown IBR1 EVAL snapshot/mode")
        expected_arm = IBR1_SELF if "IBR1-SELF" in snapshot else IBR1_CTRL
        if family_arm != expected_arm:
            raise IBR1DiagnosticsContractError("EVAL snapshot/arm identity mismatch")
        if not isinstance(row, RunnerRow):
            raise IBR1DiagnosticsContractError("EVAL geometry requires RunnerRow")
        prior_cells = sum(
            record["snapshot"] == snapshot and record["mode"] == mode
            for record in self.eval_records
        )
        if prior_cells % 16 != 0 or position != prior_cells // 16:
            raise IBR1DiagnosticsContractError(
                "EVAL geometry position/mode clock discontinuity"
            )
        prev = _vector2(prev_fy, "EVAL prev_fy")
        target = row.target_actions[:, (0, 2)].detach().to(
            device="cpu", dtype=torch.float64
        )
        if target.shape != (8, 2) or not bool(torch.isfinite(target).all().item()):
            raise IBR1DiagnosticsContractError("EVAL target geometry is invalid")
        raw = prediction.raw_fy.detach().to(device="cpu", dtype=torch.float64)[0]
        latent = prediction.latent_delta_fy.detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        cumulative = prediction.cumulative_latent_fy.detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        prebound = prediction.additive_prebound_fy.detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        overshoot = prediction.prebound_overshoot_fy.detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        margin = prediction.boundary_margin_fy.detach().to(
            device="cpu", dtype=torch.float64
        )[0]
        for horizon in range(8):
            for axis, axis_name in enumerate(AXIS_NAMES):
                self.eval_records.append(
                    {
                        "arm": family_arm,
                        "engine_arm": FAMILY_TO_ENGINE_ARM[family_arm],
                        "snapshot": snapshot,
                        "mode": mode,
                        "row_position": position,
                        "original_row_index": row.original_row_index,
                        "horizon": horizon,
                        "axis": axis_name,
                        "prev_fy": prev[axis],
                        "latent_delta_fy": float(latent[horizon, axis].item()),
                        "cumulative_latent_fy": float(
                            cumulative[horizon, axis].item()
                        ),
                        "additive_prebound_fy": float(
                            prebound[horizon, axis].item()
                        ),
                        "raw_fy": float(raw[horizon, axis].item()),
                        "target_fy": float(target[horizon, axis].item()),
                        "absolute_error": float(
                            abs(raw[horizon, axis] - target[horizon, axis]).item()
                        ),
                        "prebound_violation": bool(
                            abs(prebound[horizon, axis].item()) > 1.0
                        ),
                        "overshoot": float(overshoot[horizon, axis].item()),
                        "boundary_margin": float(margin[horizon, axis].item()),
                    }
                )

    def _validate_training(self) -> dict[str, Any]:
        expected_total = 2 * self.expected_training_rows_per_arm
        if len(self.training_records) != expected_total:
            raise IBR1DiagnosticsContractError("training geometry cardinality mismatch")
        keys: set[tuple[Any, ...]] = set()
        arm_summaries: dict[str, Any] = {}
        for engine_arm in (S_CTRL, S_SELF):
            family_arm = ENGINE_TO_FAMILY_ARM[engine_arm]
            records = [
                record
                for record in self.training_records
                if record["engine_arm"] == engine_arm
            ]
            if [record["row_position"] for record in records] != list(
                range(self.expected_training_rows_per_arm)
            ):
                raise IBR1DiagnosticsContractError(
                    f"training geometry rows are incomplete for {family_arm}"
                )
            horizon_violations = 0
            overshoot_all: list[float] = []
            overshoot_violating: list[float] = []
            overshoot_by_axis = {axis: [] for axis in AXIS_NAMES}
            overshoot_by_horizon = {str(index): [] for index in range(8)}
            overshoot_by_update: dict[str, list[float]] = {}
            overshoot_by_time_bin: dict[str, list[float]] = {}
            by_axis = {axis: 0 for axis in AXIS_NAMES}
            by_horizon = {str(index): 0 for index in range(8)}
            by_update: dict[str, int] = {}
            by_time_bin: dict[str, int] = {}
            by_row_within_update = {"0": 0, "1": 0}
            positive = {axis: 0 for axis in AXIS_NAMES}
            negative = {axis: 0 for axis in AXIS_NAMES}
            joint_boundary_counts: list[dict[str, Any]] = []
            original_indices: set[int] = set()
            geometry_error_max = 0.0
            telescoping_error_max = 0.0
            k0 = 0
            future = 0
            for record in records:
                key = (
                    record["engine_arm"],
                    record["row_position"],
                    record["original_row_index"],
                )
                if key in keys:
                    raise IBR1DiagnosticsContractError("duplicate training geometry row")
                keys.add(key)
                if record["original_row_index"] in original_indices:
                    raise IBR1DiagnosticsContractError(
                        "duplicate original row in training geometry"
                    )
                original_indices.add(record["original_row_index"])
                if record["branch"] != "branch2":
                    raise IBR1DiagnosticsContractError("non-branch2 training geometry")
                if (
                    record["u_pre"] != record["row_position"] // GRAD_ACCUM
                    or record["row_within_update"]
                    != record["row_position"] % GRAD_ACCUM
                ):
                    raise IBR1DiagnosticsContractError(
                        "training geometry frozen clock fields drifted"
                    )
                geometry_error_max = max(
                    geometry_error_max,
                    _finite_float(
                        record["geometry_reconstruction_error"],
                        "geometry reconstruction error",
                    ),
                )
                telescoping_error_max = max(
                    telescoping_error_max,
                    _finite_float(
                        record["telescoping_reconstruction_error"],
                        "telescoping reconstruction error",
                    ),
                )
                mask = record["prebound_violation_mask"]
                prebound = record["additive_prebound_fy"]
                overshoot = record["prebound_overshoot_fy"]
                update_key = str(record["u_pre"])
                bin_key = _time_bin(int(record["u_pre"]))
                row_within_key = str(record["row_within_update"])
                row_any_axis_violations = 0
                for horizon in range(8):
                    any_axis = any(mask[horizon])
                    if any_axis:
                        horizon_violations += 1
                        row_any_axis_violations += 1
                        by_horizon[str(horizon)] += 1
                        by_row_within_update[row_within_key] += 1
                        if horizon == 0:
                            k0 += 1
                        else:
                            future += 1
                    for axis, axis_name in enumerate(AXIS_NAMES):
                        value = _finite_float(
                            overshoot[horizon][axis], "training overshoot"
                        )
                        overshoot_all.append(value)
                        overshoot_by_axis[axis_name].append(value)
                        overshoot_by_horizon[str(horizon)].append(value)
                        overshoot_by_update.setdefault(update_key, []).append(value)
                        overshoot_by_time_bin.setdefault(bin_key, []).append(value)
                        if value > 0.0:
                            overshoot_violating.append(value)
                        joint = {
                            "u_pre": record["u_pre"],
                            "row_within_update": record["row_within_update"],
                            "horizon": horizon,
                            "axis": axis_name,
                            "observations": 1,
                            "positive_boundary_count": 0,
                            "negative_boundary_count": 0,
                            "violation_count": int(mask[horizon][axis]),
                        }
                        if mask[horizon][axis]:
                            by_axis[axis_name] += 1
                            if prebound[horizon][axis] > 1.0:
                                positive[axis_name] += 1
                                joint["positive_boundary_count"] = 1
                            elif prebound[horizon][axis] < -1.0:
                                negative[axis_name] += 1
                                joint["negative_boundary_count"] = 1
                        joint_boundary_counts.append(joint)
                by_update[update_key] = (
                    by_update.get(update_key, 0) + row_any_axis_violations
                )
                by_time_bin[bin_key] = (
                    by_time_bin.get(bin_key, 0) + row_any_axis_violations
                )
            denominator = self.expected_training_rows_per_arm * 8
            rate = horizon_violations / denominator
            if len(overshoot_all) != self.expected_training_rows_per_arm * 8 * 2:
                raise IBR1DiagnosticsContractError("overshoot universe mismatch")
            all_quantiles = _quantile_summary(overshoot_all)
            violating_quantiles: dict[str, Any] = {
                "support": len(overshoot_violating),
                "max": None,
                "p50": None,
                "p90": None,
                "p99": None,
            }
            if overshoot_violating:
                violating_quantiles.update(
                    {
                        "max": max(overshoot_violating),
                        "p50": _nearest_rank(overshoot_violating, 0.5),
                        "p90": _nearest_rank(overshoot_violating, 0.9),
                        "p99": _nearest_rank(overshoot_violating, 0.99),
                    }
                )
            arm_summaries[family_arm] = {
                "rows": len(records),
                "I2_any_axis_violation_count": horizon_violations,
                "I2_any_axis_denominator": denominator,
                "I2_any_axis_violation_rate": rate,
                "I2_pass": rate < 0.05,
                "axis_violation_counts": by_axis,
                "horizon_violation_counts": by_horizon,
                "update_violation_counts": by_update,
                "row_within_update_violation_counts": by_row_within_update,
                "time_bin_violation_counts": by_time_bin,
                "positive_boundary_counts": positive,
                "negative_boundary_counts": negative,
                "arm_by_axis_by_horizon_by_update_counts": joint_boundary_counts,
                "k0_violation_count": k0,
                "k1_to_k7_violation_count": future,
                "overshoot_all_axis_cells": all_quantiles,
                "overshoot_quantiles_by_axis": {
                    key: _quantile_summary(values)
                    for key, values in overshoot_by_axis.items()
                },
                "overshoot_quantiles_by_horizon": {
                    key: _quantile_summary(values)
                    for key, values in overshoot_by_horizon.items()
                },
                "overshoot_quantiles_by_u_pre": {
                    key: _quantile_summary(values)
                    for key, values in sorted(
                        overshoot_by_update.items(), key=lambda item: int(item[0])
                    )
                },
                "overshoot_quantiles_by_time_bin": {
                    key: _quantile_summary(values)
                    for key, values in overshoot_by_time_bin.items()
                },
                "overshoot_violating_only_descriptive": violating_quantiles,
                "geometry_reconstruction_error_max": geometry_error_max,
                "telescoping_reconstruction_error_max": telescoping_error_max,
            }
        return {
            "schema_version": 1,
            "analysis_class": "ibr1_training_geometry_summary",
            "family_id": "IBR1",
            "deciding_branch": "branch2",
            "arms": arm_summaries,
            "I2_pass": all(summary["I2_pass"] for summary in arm_summaries.values()),
            "threshold": {"operator": "<", "value": 0.05},
            "internal_test": "sealed",
            "internal_test_opened": False,
        }

    def _validate_eval(self) -> dict[str, Any]:
        per_snapshot = self.expected_eval_rows_per_snapshot_mode * 2 * 8 * 2
        expected_total = len(EVAL_SNAPSHOTS) * per_snapshot
        if len(self.eval_records) != expected_total:
            raise IBR1DiagnosticsContractError("EVAL geometry cardinality mismatch")
        keyed: dict[tuple[Any, ...], dict[str, Any]] = {}
        counts: dict[str, int] = {}
        counts_by_snapshot_mode: dict[str, int] = {}
        positions_by_snapshot_mode: dict[str, dict[int, set[int]]] = {}
        for record in self.eval_records:
            key = (
                record["snapshot"],
                record["mode"],
                record["row_position"],
                record["original_row_index"],
                record["horizon"],
                record["axis"],
            )
            if key in keyed:
                raise IBR1DiagnosticsContractError("duplicate EVAL geometry cell")
            keyed[key] = record
            counts[record["snapshot"]] = counts.get(record["snapshot"], 0) + 1
            snapshot_mode = f"{record['snapshot']}::{record['mode']}"
            counts_by_snapshot_mode[snapshot_mode] = (
                counts_by_snapshot_mode.get(snapshot_mode, 0) + 1
            )
            positions = positions_by_snapshot_mode.setdefault(snapshot_mode, {})
            positions.setdefault(record["row_position"], set()).add(
                record["original_row_index"]
            )
        if any(counts.get(snapshot) != per_snapshot for snapshot in EVAL_SNAPSHOTS):
            raise IBR1DiagnosticsContractError("EVAL snapshot coverage mismatch")
        expected_cells_per_mode = self.expected_eval_rows_per_snapshot_mode * 16
        for snapshot in EVAL_SNAPSHOTS:
            for mode in EVAL_MODES:
                snapshot_mode = f"{snapshot}::{mode}"
                if counts_by_snapshot_mode.get(snapshot_mode) != expected_cells_per_mode:
                    raise IBR1DiagnosticsContractError(
                        "EVAL snapshot/mode cardinality mismatch"
                    )
                positions = positions_by_snapshot_mode.get(snapshot_mode, {})
                if set(positions) != set(
                    range(self.expected_eval_rows_per_snapshot_mode)
                ):
                    raise IBR1DiagnosticsContractError(
                        "EVAL snapshot/mode position coverage mismatch"
                    )
                if any(len(originals) != 1 for originals in positions.values()):
                    raise IBR1DiagnosticsContractError(
                        "EVAL position maps to multiple original rows"
                    )

        update0 = {
            key[1:]: record
            for key, record in keyed.items()
            if key[0] == "update0_IBR1-SELF"
        }
        update128 = {
            key[1:]: record
            for key, record in keyed.items()
            if key[0] == "update128_IBR1-SELF"
        }
        if set(update0) != set(update128):
            raise IBR1DiagnosticsContractError(
                "SELF update0/update128 EVAL pair coverage mismatch"
            )
        pairs = [
            {
                "mode": key[0],
                "row_position": key[1],
                "original_row_index": key[2],
                "horizon": key[3],
                "axis": key[4],
                "raw_fy_delta": update128[key]["raw_fy"] - update0[key]["raw_fy"],
                "absolute_error_delta": update128[key]["absolute_error"]
                - update0[key]["absolute_error"],
                "overshoot_delta": update128[key]["overshoot"]
                - update0[key]["overshoot"],
            }
            for key in sorted(update0)
        ]
        return {
            "schema_version": 1,
            "analysis_class": "ibr1_eval_geometry_summary",
            "family_id": "IBR1",
            "role": "mandatory descriptive telemetry; not an additional mechanism gate",
            "records": len(self.eval_records),
            "records_by_snapshot": counts,
            "records_by_snapshot_mode": counts_by_snapshot_mode,
            "self_update0_to_update128_pair_count": len(pairs),
            "self_update0_to_update128_pairs": pairs,
            "adds_scientific_threshold": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }

    def finalize(
        self,
    ) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], dict[str, Any]]:
        training_summary = self._validate_training()
        eval_summary = self._validate_eval()
        return (
            tuple(self.training_records),
            tuple(self.eval_records),
            {
                "schema_version": 1,
                "analysis_class": "ibr1_diagnostics_summary",
                "family_id": "IBR1",
                "training_geometry": training_summary,
                "eval_geometry": eval_summary,
                "engineering_fail_closed": False,
                "internal_test": "sealed",
                "internal_test_opened": False,
            },
        )


def wrap_training_head_forward(
    callbacks: ArmCallbacks, collector: GeometryCollector
) -> ArmCallbacks:
    """Observe only branch2 and return the original result object unchanged."""

    original = callbacks.head_forward

    def head_forward(
        features: Any, prev_fy: torch.Tensor, event: HeadEvent
    ) -> HeadForwardResult:
        result = original(features, prev_fy, event)
        if event.branch == "branch2":
            collector.observe_training(result, prev_fy, event)
        return result

    return replace(callbacks, head_forward=head_forward)


def wrap_eval_predictor(
    predictor: Callable[..., Any],
    collector: GeometryCollector,
    *,
    family_arm: str,
    snapshot: str,
) -> Callable[..., Any]:
    """Observe one frozen EVAL predictor without changing its return value."""

    def wrapped(
        row: RunnerRow,
        prev_fy: torch.Tensor,
        *,
        mode: str,
        reset: bool,
        position: int,
    ) -> Any:
        prediction = predictor(
            row,
            prev_fy,
            mode=mode,
            reset=reset,
            position=position,
        )
        collector.observe_eval(
            prediction,
            prev_fy,
            row,
            family_arm=family_arm,
            snapshot=snapshot,
            mode=mode,
            position=position,
        )
        return prediction

    return wrapped


__all__ = [
    "AXIS_NAMES",
    "DIAGNOSTIC_EPS",
    "EVAL_MODES",
    "EVAL_SNAPSHOTS",
    "GeometryCollector",
    "GradientDiagnosticsCollector",
    "IBR1DiagnosticsContractError",
    "IBR1G6Instrument",
    "OptimizerDiagnosticsHandle",
    "wrap_eval_predictor",
    "wrap_training_head_forward",
]
