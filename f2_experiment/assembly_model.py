"""Model-side production assembly for the F2 paired-smoke lifecycle.

This module owns the executable model surface of the approved F2 assembly
plan (blockers 7, 8, and 11) under the 2026-07-18 Fable PRIMARY merged
adjudication:

* the SA-B0 / SA-B1 / SA-Hstar package factories with a shared AP2 head and
  the byte-identical shared controller contract (ruling d: the smoke package
  is SA-Hstar; SA-B0/SA-B1 must be constructible and unit-tested now but run
  only at formal; the SA-B1 ``tim_q`` q slot keeps its dimension with a
  zero-fill placeholder);
* bit-identical paired-arm construction via a single deep copy with a
  ``checkpoint_init_sha256`` equality proof;
* the frozen AdamW optimizer contract (ruling f) together with a
  parameter-membership/overlap/missing-coverage/LR/weight-decay receipt;
* the real G6 graph/gradient instrument on the frozen ``base.proj`` probe
  surface (ruling c: bstar is the deciding block mode); and
* the :class:`~f2_experiment.runner.ArmCallbacks` assembly that the frozen
  paired runner drives (feature/aux/head forward, track loss, backward,
  optimizer step, and the adapter audit-counter hook).

It never loads data, receipts, checkpoints, or the sealed internal test; the
data-side and lifecycle-side assemblies live in their own modules.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import copy
from dataclasses import dataclass, fields as dataclass_fields
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Literal

import torch
from torch import nn

from third_party.OpenTrackVLA.harness.base_repro.polar_cot import polar_cot_loss
from third_party.OpenTrackVLA.harness.base_repro.tim import roi_pool_candidate

from .assembly import (
    CalRowAudit,
    SmokeArmAssembly,
    SmokeAssemblyPlan,
    _resolve_frozen_token_ledger,
)
from .assembly_data import (
    F2AssemblyContractError,
    build_runner_rows,
    ensure_observation_packet,
    frozen_cache_roots,
    ordered_support_rows,
    resolve_frozen_base_hf_dir,
    smoke_reset_sets,
)
from .evaluation import G6_FALLBACK_RATIO_MAX, G6_GRADIENT_CLOCK, G6Update
from .model import (
    F2AP2Model,
    F2ModelContractError,
    ap2_track_loss,
    assert_prev_free_tensors,
    assert_step0_controlled_axis_persistence,
)
from .opentrack_adapter import (
    OpenTrackVLAF2ObservationAdapter,
    _detach_tree,
    _state_tensors,
)
from .runner import (
    GRAD_ACCUM,
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    AuxForwardResult,
    BackwardEvent,
    FeatureForwardResult,
    HeadEvent,
    HeadForwardResult,
    OptimizerUpdateEvent,
    RowEvent,
    RunnerRow,
    checkpoint_init_sha256,
)
from .reproducibility import (
    configure_cuda_reproducibility,
    cuda_reproducibility_receipt,
)
from .support import (
    ARCHITECTURE_LOCK,
    FROZEN_TRAIN_RELATIVE,
    build_frozen_support,
    parse_train_jsonl,
)


PackageName = Literal["SA-B0", "SA-B1", "SA-Hstar"]
PACKAGE_NAMES: tuple[PackageName, ...] = ("SA-B0", "SA-B1", "SA-Hstar")
SMOKE_PACKAGE: PackageName = "SA-Hstar"

PACKAGE_AUX_COMPONENTS: Mapping[PackageName, tuple[str, ...]] = {
    "SA-B0": (),
    "SA-B1": ("L_cot",),
    "SA-Hstar": ("L_cot", "L_future", "L_verify"),
}

TRAINABLE_BASE_MODULES = ("proj",)
G6_PROBE_SURFACE = "base.proj"
G6_BLOCK_MODES = ("bstar", "per_aux")
G6_ACCUMULATOR_EPS = 1e-12

OBSERVATION_REQUIRED_KEYS = (
    "coarse_tokens",
    "coarse_tidx",
    "fine_tokens",
    "fine_tidx",
    "instruction",
)
OBSERVATION_OPTIONAL_KEYS = ("yaw_hist", "yaw_curr")
OBSERVATION_FORBIDDEN_KEYS = (
    "step_actions",
    "actions",
    "waypoints",
    "prev_action",
    "target_actions",
    "delta_vel",
    "delta_pos",
    "motors",
    "polar_theta_idx",
    "polar_dist_idx",
    "polar_invalid",
) + tuple(
    f"fut_{kind}_{horizon}"
    for kind in ("valid", "vis", "theta_idx", "dist_idx")
    for horizon in (4, 8, 16)
)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise F2AssemblyContractError(f"{label} must be a positive integer")
    return value


def _finite_positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2AssemblyContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise F2AssemblyContractError(f"{label} must be finite and positive")
    return result


def _finite_nonnegative_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2AssemblyContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise F2AssemblyContractError(f"{label} must be finite and nonnegative")
    return result


def _validate_package_name(package: Any) -> PackageName:
    if package not in PACKAGE_NAMES:
        raise F2AssemblyContractError(
            f"package must be one of {list(PACKAGE_NAMES)!r}, got {package!r}"
        )
    return package


def _validate_aux_coefficients(
    package: PackageName, aux_coefficients: Any
) -> dict[str, float]:
    if not isinstance(aux_coefficients, Mapping):
        raise F2AssemblyContractError("aux_coefficients must be a mapping")
    expected = set(PACKAGE_AUX_COMPONENTS[package])
    supplied = set(aux_coefficients)
    if supplied != expected:
        raise F2AssemblyContractError(
            f"{package} aux_coefficients keys must be exactly "
            f"{sorted(expected)!r}, got {sorted(supplied)!r}"
        )
    frozen: dict[str, float] = {}
    for name in sorted(expected):
        value = _finite_positive_float(
            aux_coefficients[name], f"aux_coefficients[{name!r}]"
        )
        if value > 1.0:
            raise F2AssemblyContractError(
                f"aux_coefficients[{name!r}] must lie in (0,1] per the frozen "
                "CAL calibration rule"
            )
        frozen[name] = value
    return frozen


# ---------------------------------------------------------------------------
# Package adapters (SA-B1 and SA-B0 variants of the frozen full adapter)
# ---------------------------------------------------------------------------


class ConfidenceTIMObservationAdapter(OpenTrackVLAF2ObservationAdapter):
    """SA-B1 adapter: Polar soft token plus the 4-token confidence TIM.

    The Future, Cognitive Event Bank, Orchestrator, and q self-correctness
    modules are deleted outright so that no parameter, buffer, or forward
    path for them exists.  The official base encoding, TIM delayed-write
    machinery, reset semantics, and audit counters are inherited byte-for-byte
    from the frozen full adapter.  Per the adjudication (ruling d) the
    ``tim_q`` feature keeps its ``D+2`` dimension: the confidence slot is
    live and the q slot is a zero-fill placeholder.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        tim_tokens: int = 4,
    ) -> None:
        super().__init__(base_model, tim_tokens=tim_tokens)
        for name in ("future", "self_correctness", "events", "orchestrator"):
            delattr(self, name)

    @property
    def method_dims(self) -> dict[str, int]:
        return {"polar": self.D, "tim_q": self.D + 2}

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        batch = _positive_int(batch_size, "batch_size")
        tim_state = self.tim.init_state(batch, device)
        tim_state["last_gate"] = torch.zeros(batch, device=device)
        return {
            "tim": tim_state,
            "pending_candidate": torch.zeros(
                batch,
                self.tim.n_tokens,
                self.D,
                device=device,
            ),
            "pending_confidence": torch.zeros(batch, device=device),
            "pending_q_write": torch.zeros(batch, device=device),
            "pending_invalid": torch.ones(
                batch, dtype=torch.bool, device=device
            ),
            "has_pending": torch.zeros(
                batch, dtype=torch.bool, device=device
            ),
        }

    def encode_step(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Sequence[str],
        previous_state: Mapping[str, Any],
        *,
        reset_mask: bool | torch.Tensor | None = None,
        distractor_rate: torch.Tensor | None = None,
        yaw_hist: torch.Tensor | None = None,
        yaw_curr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del distractor_rate  # SA-B1 has no event/orchestrator consumer.
        if coarse_tokens.ndim != 3 or fine_tokens.ndim != 3:
            raise F2AssemblyContractError(
                "coarse_tokens and fine_tokens must have shape (B,N,D)"
            )
        batch_size = coarse_tokens.shape[0]
        if fine_tokens.shape[0] != batch_size or len(instructions) != batch_size:
            raise F2AssemblyContractError(
                "observation inputs must share the batch axis"
            )
        device = self.base.act_token.device
        state = self._prepare_state(
            previous_state,
            batch_size=batch_size,
            device=device,
            reset_mask=reset_mask,
        )
        tim_state = self._apply_pending_tim(state)
        h_act, fine_projected = self._encode_official_base(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            tim_state,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
        )

        cot_output = self.cot(h_act)
        cot_decoded = self.cot.decode(cot_output)
        confidence = cot_decoded["confidence"]
        invalid_prediction = cot_decoded["invalid_pred"]
        candidate = roi_pool_candidate(
            fine_projected,
            cot_decoded["theta_idx"],
            n_theta=self.cot.n_theta,
            n_tokens=self.tim.n_tokens,
        )
        tim_mean = tim_state["mem"].mean(dim=1).float()
        q_placeholder = torch.zeros_like(confidence)

        method_features = {
            "polar": self.polar_token(cot_output),
            "tim_q": torch.cat(
                [
                    tim_mean,
                    confidence.unsqueeze(-1),
                    q_placeholder.unsqueeze(-1),
                ],
                dim=-1,
            ),
        }
        method_alphas = {
            "polar": torch.ones_like(confidence),
            "tim_q": torch.ones_like(confidence),
        }

        new_state = _detach_tree(
            {
                "tim": tim_state,
                "pending_candidate": candidate,
                "pending_confidence": confidence,
                "pending_q_write": q_placeholder,
                "pending_invalid": invalid_prediction,
                "has_pending": torch.ones(
                    batch_size, dtype=torch.bool, device=device
                ),
            }
        )
        if any(
            tensor.requires_grad or tensor.grad_fn is not None
            for tensor in _state_tensors(new_state)
        ):
            raise F2AssemblyContractError(
                "perception state must be detached at every step"
            )
        return {
            "base_features": h_act,
            "h_act": h_act,
            "method_features": method_features,
            "method_alphas": method_alphas,
            "cot": cot_output,
            "cot_decoded": cot_decoded,
            "new_state": new_state,
            "audit_counters": self.audit_counters(),
        }

    def compute_aux_losses(
        self,
        output: Mapping[str, Any],
        targets: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """SA-B1 keeps only the Polar-CoT auxiliary loss."""

        if not isinstance(output, Mapping) or not isinstance(targets, Mapping):
            raise F2AssemblyContractError("output and targets must be mappings")
        device = output["h_act"].device
        theta_target = self._target(targets, "theta_idx", "polar_theta_idx").to(
            device=device, dtype=torch.long
        )
        distance_target = self._target(targets, "dist_idx", "polar_dist_idx").to(
            device=device, dtype=torch.long
        )
        invalid_target = self._target(targets, "invalid", "polar_invalid").to(
            device=device, dtype=torch.float32
        )
        cot_loss = polar_cot_loss(
            output["cot"],
            theta_target,
            distance_target,
            invalid_target,
        )
        return {"loss": cot_loss, "L_aux": cot_loss, "L_cot": cot_loss}


class NullTIMObservationAdapter(OpenTrackVLAF2ObservationAdapter):
    """SA-B0 adapter: official base encoding with matched null TIM slots.

    All method modules (Polar-CoT, TIM, polar token, Future, events,
    orchestrator, q head) are deleted; the LLM sequence still receives the
    same number of TIM slot tokens as SA-Hstar, but they are identically zero
    so that the token count and FLOPs stay matched while no method
    information can flow.  ``method_dims`` is empty and there is no auxiliary
    loss.  SA-B0 is stateless: its perception state is the empty mapping.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        tim_tokens: int = 4,
    ) -> None:
        super().__init__(base_model, tim_tokens=tim_tokens)
        self.n_tim_slots = int(self.tim.n_tokens)
        for name in (
            "cot",
            "tim",
            "polar_token",
            "future",
            "self_correctness",
            "events",
            "orchestrator",
        ):
            delattr(self, name)

    @property
    def method_dims(self) -> dict[str, int]:
        return {}

    def init_state(
        self,
        batch_size: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        _positive_int(batch_size, "batch_size")
        del device
        return {}

    def encode_step(
        self,
        coarse_tokens: torch.Tensor,
        coarse_tidx: torch.Tensor,
        fine_tokens: torch.Tensor,
        fine_tidx: torch.Tensor,
        instructions: Sequence[str],
        previous_state: Mapping[str, Any],
        *,
        reset_mask: bool | torch.Tensor | None = None,
        distractor_rate: torch.Tensor | None = None,
        yaw_hist: torch.Tensor | None = None,
        yaw_curr: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        del distractor_rate
        if coarse_tokens.ndim != 3 or fine_tokens.ndim != 3:
            raise F2AssemblyContractError(
                "coarse_tokens and fine_tokens must have shape (B,N,D)"
            )
        batch_size = coarse_tokens.shape[0]
        if fine_tokens.shape[0] != batch_size or len(instructions) != batch_size:
            raise F2AssemblyContractError(
                "observation inputs must share the batch axis"
            )
        device = self.base.act_token.device
        # Validates the (empty) state and reset mask shapes fail-closed.
        self._prepare_state(
            previous_state,
            batch_size=batch_size,
            device=device,
            reset_mask=reset_mask,
        )
        null_tim_slots = torch.zeros(
            batch_size, self.n_tim_slots, self.D, device=device
        )
        h_act, _fine_projected = self._encode_official_base(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            {"mem": null_tim_slots},
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
        )
        return {
            "base_features": h_act,
            "h_act": h_act,
            "method_features": {},
            "method_alphas": {},
            "null_tim_slots": null_tim_slots,
            "new_state": {},
            "audit_counters": self.audit_counters(),
        }

    def compute_aux_losses(
        self,
        output: Mapping[str, Any],
        targets: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """SA-B0 has no auxiliary loss; return a graph-connected zero."""

        del targets
        if not isinstance(output, Mapping):
            raise F2AssemblyContractError("output must be a mapping")
        zero = output["h_act"].sum() * 0.0
        return {"loss": zero, "L_aux": zero}


# ---------------------------------------------------------------------------
# Arm modules and package factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class F2ArmModules:
    """One arm's complete module set with a hashable full state."""

    package: PackageName
    base: nn.Module
    adapter: nn.Module
    model: F2AP2Model
    aux_coefficients: Mapping[str, float]

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        seen: set[int] = set()
        for module in (self.adapter, self.model):
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    yield parameter

    def named_full_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Prefixed parameter names; ``base`` is reached through ``adapter``."""

        for prefix, module in (("adapter", self.adapter), ("model", self.model)):
            for name, parameter in module.named_parameters():
                yield f"{prefix}.{name}", parameter

    def full_state_dict(self) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        for prefix, module in (("adapter", self.adapter), ("model", self.model)):
            for name, tensor in module.state_dict().items():
                state[f"{prefix}.{name}"] = tensor
        if not state:
            raise F2AssemblyContractError("arm full state dict is empty")
        return state


def _validate_base_interface(base: nn.Module) -> None:
    if not isinstance(base, nn.Module):
        raise F2AssemblyContractError("base must be an nn.Module")
    proj = getattr(base, "proj", None)
    if not isinstance(proj, nn.Module):
        raise F2AssemblyContractError(
            "base must expose the official 'proj' projector module"
        )
    if not any(True for _ in proj.parameters()):
        raise F2AssemblyContractError("base.proj has no parameters to probe")


def _freeze_base_except_proj(base: nn.Module) -> None:
    """Mirror main-v1 ``trainable_base_modules=['proj']``; F2 has no planner."""

    proj_ids = {id(parameter) for parameter in base.proj.parameters()}
    for parameter in base.parameters():
        parameter.requires_grad_(id(parameter) in proj_ids)


def build_package(
    package: PackageName,
    base: nn.Module,
    *,
    device: torch.device | str,
    aux_coefficients: Mapping[str, float],
) -> F2ArmModules:
    """Build one arm's modules for the requested assembly package."""

    name = _validate_package_name(package)
    _validate_base_interface(base)
    frozen_coefficients = _validate_aux_coefficients(name, aux_coefficients)
    target_device = torch.device(device)
    base = base.to(target_device)
    if name == "SA-Hstar":
        adapter: nn.Module = OpenTrackVLAF2ObservationAdapter(base)
    elif name == "SA-B1":
        adapter = ConfidenceTIMObservationAdapter(base)
    else:
        adapter = NullTIMObservationAdapter(base)
    adapter = adapter.to(target_device)
    model = F2AP2Model(
        d_model=int(base.D), method_dims=adapter.method_dims
    ).to(target_device)
    _freeze_base_except_proj(base)
    return F2ArmModules(
        package=name,
        base=base,
        adapter=adapter,
        model=model,
        aux_coefficients=frozen_coefficients,
    )


# ---------------------------------------------------------------------------
# Frozen optimizer contract and parameter receipt (adjudication ruling f)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimizerContract:
    """AdamW hyperparameters frozen by the merged adjudication (ruling f)."""

    base_lr: float = 2e-5
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8

    def __post_init__(self) -> None:
        _finite_positive_float(self.base_lr, "base_lr")
        _finite_positive_float(self.head_lr, "head_lr")
        _finite_nonnegative_float(self.weight_decay, "weight_decay")
        _finite_positive_float(self.grad_clip, "grad_clip")
        if (
            not isinstance(self.betas, tuple)
            or len(self.betas) != 2
            or any(
                not 0.0 <= _finite_nonnegative_float(beta, "betas") < 1.0
                for beta in self.betas
            )
        ):
            raise F2AssemblyContractError("betas must be a pair in [0,1)")
        _finite_positive_float(self.eps, "eps")

    def to_dict(self) -> dict[str, Any]:
        return {
            "optimizer": "AdamW",
            "base_lr": self.base_lr,
            "head_lr": self.head_lr,
            "weight_decay": self.weight_decay,
            "grad_clip_norm": self.grad_clip,
            "betas": list(self.betas),
            "eps": self.eps,
        }


def build_arm_optimizer(
    arm: F2ArmModules,
    contract: OptimizerContract,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """Build the frozen four-group AdamW and its parameter receipt.

    Groups: ``base.proj`` at ``base_lr``; adapter modules plus the F2AP2Model
    ordinary head at ``head_lr``; method LayerScales and ``s_prev`` at
    ``head_lr`` with zero weight decay.  Any overlap, any trainable parameter
    missing from the groups, or any frozen parameter inside a group fails
    closed before an optimizer is returned.
    """

    if not isinstance(arm, F2ArmModules):
        raise F2AssemblyContractError("arm must be an F2ArmModules")
    if not isinstance(contract, OptimizerContract):
        raise F2AssemblyContractError("contract must be an OptimizerContract")

    base_param_ids = {id(parameter) for parameter in arm.base.parameters()}
    proj_parameters = [
        parameter
        for parameter in arm.base.proj.parameters()
        if parameter.requires_grad
    ]
    if not proj_parameters:
        raise F2AssemblyContractError("base.proj has no trainable parameters")
    adapter_parameters = [
        parameter
        for parameter in arm.adapter.parameters()
        if parameter.requires_grad and id(parameter) not in base_param_ids
    ]
    model_groups = arm.model.optimizer_parameter_groups(
        head_lr=contract.head_lr,
        head_weight_decay=contract.weight_decay,
    )
    groups_by_name = {group["name"]: group for group in model_groups}
    ordinary_head_parameters = list(groups_by_name["ordinary_head"]["params"])

    groups: list[dict[str, Any]] = [
        {
            "name": "base_proj",
            "params": proj_parameters,
            "lr": contract.base_lr,
            "weight_decay": contract.weight_decay,
        },
        {
            "name": "adapter_and_ordinary_head",
            "params": adapter_parameters + ordinary_head_parameters,
            "lr": contract.head_lr,
            "weight_decay": contract.weight_decay,
        },
    ]
    if "method_layerscales" in groups_by_name:
        groups.append(
            {
                "name": "method_layerscales",
                "params": list(groups_by_name["method_layerscales"]["params"]),
                "lr": contract.head_lr,
                "weight_decay": 0.0,
            }
        )
    groups.append(
        {
            "name": "prev_layerscale",
            "params": list(groups_by_name["prev_layerscale"]["params"]),
            "lr": contract.head_lr,
            "weight_decay": 0.0,
        }
    )

    name_by_id = {
        id(parameter): name for name, parameter in arm.named_full_parameters()
    }
    grouped_ids: list[int] = []
    for group in groups:
        for parameter in group["params"]:
            grouped_ids.append(id(parameter))
    if len(grouped_ids) != len(set(grouped_ids)):
        raise F2AssemblyContractError(
            "optimizer parameter groups overlap"
        )
    trainable_by_id = {
        id(parameter): parameter for parameter in arm.trainable_parameters()
    }
    nontrainable_in_groups = sorted(
        name_by_id.get(identifier, f"<unnamed:{identifier}>")
        for identifier in grouped_ids
        if identifier not in trainable_by_id
    )
    if nontrainable_in_groups:
        raise F2AssemblyContractError(
            "optimizer groups contain frozen parameters: "
            f"{nontrainable_in_groups!r}"
        )
    missing_from_groups = sorted(
        name_by_id.get(identifier, f"<unnamed:{identifier}>")
        for identifier in trainable_by_id
        if identifier not in set(grouped_ids)
    )
    if missing_from_groups:
        raise F2AssemblyContractError(
            "trainable parameters missing from optimizer groups: "
            f"{missing_from_groups!r}"
        )

    receipt_groups: dict[str, Any] = {}
    for group in groups:
        names = sorted(
            name_by_id.get(id(parameter), f"<unnamed:{id(parameter)}>")
            for parameter in group["params"]
        )
        receipt_groups[group["name"]] = {
            "parameter_names": names,
            "parameter_count": len(names),
            "numel": int(
                sum(parameter.numel() for parameter in group["params"])
            ),
            "lr": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
        }

    frozen_count = sum(
        1
        for _name, parameter in arm.named_full_parameters()
        if not parameter.requires_grad
    )
    receipt = {
        "schema_version": 1,
        "analysis_class": "f2_arm_optimizer_parameter_receipt",
        "architecture_lock": ARCHITECTURE_LOCK,
        "package": arm.package,
        "contract": contract.to_dict(),
        "groups": receipt_groups,
        "group_order": [group["name"] for group in groups],
        "trainable_parameter_count": len(trainable_by_id),
        "frozen_parameter_count": frozen_count,
        "trainable_base_modules": list(TRAINABLE_BASE_MODULES),
        "overlap_count": 0,
        "missing_from_groups": [],
        "nontrainable_in_groups": [],
        "coverage_exact": True,
    }
    optimizer = torch.optim.AdamW(
        groups, betas=contract.betas, eps=contract.eps
    )
    return optimizer, receipt


# ---------------------------------------------------------------------------
# Bit-identical paired arms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmAssembly:
    modules: F2ArmModules
    optimizer: torch.optim.AdamW
    parameter_receipt: Mapping[str, Any]


@dataclass(frozen=True)
class PairedArms:
    """Both smoke arms with the shared bit-identical init SHA proof."""

    package: PackageName
    seed: int
    device: str
    checkpoint_init_sha256: str
    arms: Mapping[str, ArmAssembly]


def build_paired_arms(
    base: nn.Module,
    *,
    package: PackageName = SMOKE_PACKAGE,
    seed: int = 0,
    device: torch.device | str,
    contract: OptimizerContract,
    aux_coefficients: Mapping[str, float],
) -> PairedArms:
    """Build S-CTRL, deep-copy S-SELF, and prove bit-identical init SHAs."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise F2AssemblyContractError("seed must be a nonnegative integer")
    if not isinstance(contract, OptimizerContract):
        raise F2AssemblyContractError("contract must be an OptimizerContract")
    target_device = torch.device(device)
    torch.manual_seed(seed)
    ctrl_modules = build_package(
        package,
        base,
        device=target_device,
        aux_coefficients=aux_coefficients,
    )
    self_modules = copy.deepcopy(ctrl_modules)
    if self_modules.adapter.base is not self_modules.base:
        raise F2AssemblyContractError(
            "deep copy broke the adapter/base aliasing inside one arm"
        )
    ctrl_sha = checkpoint_init_sha256(ctrl_modules.full_state_dict())
    self_sha = checkpoint_init_sha256(self_modules.full_state_dict())
    if ctrl_sha != self_sha:
        raise F2AssemblyContractError(
            "paired arms are not bit-identical at initialization: "
            f"{ctrl_sha} != {self_sha}"
        )
    arms: dict[str, ArmAssembly] = {}
    for arm_name, modules in (("S-CTRL", ctrl_modules), ("S-SELF", self_modules)):
        optimizer, receipt = build_arm_optimizer(modules, contract)
        arms[arm_name] = ArmAssembly(
            modules=modules,
            optimizer=optimizer,
            parameter_receipt=receipt,
        )
    return PairedArms(
        package=ctrl_modules.package,
        seed=seed,
        device=str(target_device),
        checkpoint_init_sha256=ctrl_sha,
        arms=arms,
    )


# ---------------------------------------------------------------------------
# G6 gradient instrumentation (S-CTRL only; probe surface = base.proj)
# ---------------------------------------------------------------------------


class G6Instrument:
    """Real gradient-geometry probe on the frozen ``base.proj`` surface.

    ``observe_row`` must be called inside the backward callback before
    ``event.scaled_loss.backward()``; ``emit_update`` is the runner's
    ``g6_update`` hook.  Accumulation is float64 on CPU.  A loss that does not
    reach the probe (``allow_unused`` gradient of ``None`` or a loss without a
    graph) contributes a zero vector: that is reachability evidence, not an
    error.
    """

    def __init__(
        self,
        probe: Sequence[nn.Parameter],
        *,
        block_mode: str = "bstar",
        rows_per_update: int = GRAD_ACCUM,
    ) -> None:
        if block_mode not in G6_BLOCK_MODES:
            raise F2AssemblyContractError(
                f"block_mode must be one of {list(G6_BLOCK_MODES)!r}"
            )
        probe_parameters = tuple(probe)
        if not probe_parameters or any(
            not isinstance(parameter, nn.Parameter)
            for parameter in probe_parameters
        ):
            raise F2AssemblyContractError(
                "probe must be a nonempty sequence of nn.Parameter"
            )
        self.probe = probe_parameters
        self.block_mode = block_mode
        self.rows_per_update = _positive_int(rows_per_update, "rows_per_update")
        self._rows_observed = 0
        self._expected_u_pre = 0
        self._sum_aux: torch.Tensor | None = None
        self._sum_track: torch.Tensor | None = None
        self._per_aux_sums: dict[str, torch.Tensor] | None = None
        self._per_aux_evidence: list[dict[str, Any]] = []

    def receipt_contract(self) -> dict[str, Any]:
        return {
            "probe_surface": G6_PROBE_SURFACE,
            "probe_parameter_count": len(self.probe),
            "probe_numel": int(
                sum(parameter.numel() for parameter in self.probe)
            ),
            "block_mode": self.block_mode,
            "deciding_block_mode": "bstar",
            "fallback_per_aux_ratio_max": G6_FALLBACK_RATIO_MAX,
            "accumulator": "float64 CPU vector sums",
            "arm": S_CTRL,
        }

    def fallback_evidence(self) -> dict[str, Any]:
        """Per-aux ratio series for the adjudicated 0.75 fallback report.

        PRIMARY ruling c requires the per-aux fallback to be evaluated and
        reported in the same receipt while bstar stays the deciding gate;
        this accessor surfaces the evidence without ever mixing the two
        ratio families inside one :class:`G6Update`.
        """

        return {
            "deciding_block_mode": "bstar",
            "block_mode": self.block_mode,
            "fallback_per_aux_ratio_max": G6_FALLBACK_RATIO_MAX,
            "per_aux_ratio_series": tuple(
                {
                    "u_pre": entry["u_pre"],
                    "ratios": dict(entry["ratios"]),
                }
                for entry in self._per_aux_evidence
            ),
        }

    def _zero_vector(self) -> torch.Tensor:
        return torch.zeros(
            sum(parameter.numel() for parameter in self.probe),
            dtype=torch.float64,
        )

    def _grad_vector(self, loss: Any, label: str) -> torch.Tensor:
        if not isinstance(loss, torch.Tensor):
            raise F2AssemblyContractError(f"{label} must be a torch.Tensor")
        if loss.numel() != 1:
            raise F2AssemblyContractError(f"{label} must be a scalar loss")
        if not loss.requires_grad:
            return self._zero_vector()
        gradients = torch.autograd.grad(
            loss,
            self.probe,
            retain_graph=True,
            allow_unused=True,
        )
        pieces: list[torch.Tensor] = []
        for parameter, gradient in zip(self.probe, gradients):
            if gradient is None:
                pieces.append(
                    torch.zeros(
                        parameter.numel(), dtype=torch.float64
                    )
                )
            else:
                pieces.append(
                    gradient.detach().reshape(-1).to("cpu").to(torch.float64)
                )
        vector = torch.cat(pieces)
        if not bool(torch.isfinite(vector).all().item()):
            raise F2AssemblyContractError(f"{label} probe gradient is nonfinite")
        return vector

    def observe_row(
        self,
        *,
        aux_loss: torch.Tensor,
        track1: torch.Tensor,
        track2: torch.Tensor,
        per_aux_losses: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        if self._rows_observed >= self.rows_per_update:
            raise F2AssemblyContractError(
                "G6 observe_row called more times than rows per update"
            )
        if per_aux_losses is not None:
            if not isinstance(per_aux_losses, Mapping) or not per_aux_losses:
                raise F2AssemblyContractError(
                    "per_aux_losses must be a nonempty mapping when provided"
                )
        elif self.block_mode == "per_aux":
            raise F2AssemblyContractError(
                "per_aux block mode requires a nonempty per_aux_losses"
            )
        track_combined = 0.5 * track1 + 0.5 * track2
        g_aux = self._grad_vector(aux_loss, "G6 aux_loss")
        g_track = self._grad_vector(track_combined, "G6 track loss")
        if self._sum_aux is None:
            self._sum_aux = self._zero_vector()
            self._sum_track = self._zero_vector()
        self._sum_aux = self._sum_aux + g_aux
        self._sum_track = self._sum_track + g_track
        if per_aux_losses is not None:
            if self._per_aux_sums is None:
                self._per_aux_sums = {}
            for name in sorted(per_aux_losses):
                if not isinstance(name, str) or not name:
                    raise F2AssemblyContractError(
                        "per_aux loss names must be nonempty strings"
                    )
                vector = self._grad_vector(
                    per_aux_losses[name], f"G6 per-aux {name}"
                )
                if name in self._per_aux_sums:
                    self._per_aux_sums[name] = self._per_aux_sums[name] + vector
                else:
                    self._per_aux_sums[name] = vector
        self._rows_observed += 1

    def _clear(self) -> None:
        self._rows_observed = 0
        self._sum_aux = None
        self._sum_track = None
        self._per_aux_sums = None

    def emit_update(self, event: OptimizerUpdateEvent) -> G6Update:
        if not isinstance(event, OptimizerUpdateEvent):
            raise F2AssemblyContractError(
                "G6 emit_update requires an OptimizerUpdateEvent"
            )
        if event.arm != S_CTRL:
            raise F2AssemblyContractError(
                f"G6 instrumentation is S-CTRL only, got {event.arm!r}"
            )
        if event.u_pre != self._expected_u_pre:
            raise F2AssemblyContractError(
                f"G6 clock discontinuity: expected u_pre="
                f"{self._expected_u_pre}, got {event.u_pre}"
            )
        if self._rows_observed != self.rows_per_update:
            raise F2AssemblyContractError(
                f"G6 update {event.u_pre} aggregated {self._rows_observed} "
                f"rows; expected {self.rows_per_update}"
            )
        if self._sum_aux is None or self._sum_track is None:
            raise F2AssemblyContractError(
                "G6 accumulators are empty at emit time"
            )
        sum_aux = self._sum_aux
        sum_track = self._sum_track
        sum_total = sum_aux + sum_track
        aux_norm = float(torch.linalg.vector_norm(sum_aux).item())
        track_norm = float(torch.linalg.vector_norm(sum_track).item())
        total_norm = float(torch.linalg.vector_norm(sum_total).item())
        aux_reachable = aux_norm > 0.0
        track_reachable = track_norm > 0.0

        evidence_ratios: dict[str, float] | None = None
        if self._per_aux_sums:
            evidence_ratios = {
                name: float(torch.linalg.vector_norm(vector).item())
                / max(track_norm, G6_ACCUMULATOR_EPS)
                for name, vector in sorted(self._per_aux_sums.items())
            }
            self._per_aux_evidence.append(
                {"u_pre": event.u_pre, "ratios": evidence_ratios}
            )

        cosine: float | None = None
        projection: float | None = None
        aux_track_ratio: float | None = None
        per_aux_ratios: dict[str, float] | None = None
        if event.u_pre in G6_GRADIENT_CLOCK:
            dot = float(torch.dot(sum_total, sum_track).item())
            cosine = dot / (
                max(total_norm, G6_ACCUMULATOR_EPS)
                * max(track_norm, G6_ACCUMULATOR_EPS)
            )
            cosine = max(-1.0, min(1.0, cosine))
            projection = dot / max(track_norm, G6_ACCUMULATOR_EPS)
            if self.block_mode == "bstar":
                aux_track_ratio = aux_norm / max(
                    track_norm, G6_ACCUMULATOR_EPS
                )
            else:
                per_aux_ratios = evidence_ratios
                if not per_aux_ratios:
                    raise F2AssemblyContractError(
                        "per_aux block mode has no aux gradient blocks"
                    )
        self._clear()
        self._expected_u_pre += 1
        return G6Update(
            u_pre=event.u_pre,
            aux_reachable=aux_reachable,
            track_reachable=track_reachable,
            cosine_total_track=cosine,
            signed_projection=projection,
            aux_track_ratio=aux_track_ratio,
            per_aux_ratios=per_aux_ratios,
        )


# ---------------------------------------------------------------------------
# Runner-facing arm executor and callbacks
# ---------------------------------------------------------------------------


def _extract_observation(observation: Any) -> dict[str, Any]:
    """Structurally gate the observation and batch it for the adapter.

    Blocker 10 (GPT-5.6 sol review P1-2): the only accepted observation type
    is :class:`~f2_experiment.assembly_data.ObservationPacket`.  Arbitrary
    mappings and packet-like duck-typed objects are rejected outright, so the
    validated packet constructor is the sole path into the encoder and expert
    actions or future labels cannot be smuggled in structurally.
    """

    packet = ensure_observation_packet(observation)
    return {
        "coarse_tokens": packet.coarse_tokens.unsqueeze(0),
        "coarse_tidx": packet.coarse_tidx.unsqueeze(0),
        "fine_tokens": packet.fine_tokens.unsqueeze(0),
        "fine_tidx": packet.fine_tidx.unsqueeze(0),
        "instruction": packet.instruction,
        "yaw_hist": (
            None if packet.yaw_hist is None else packet.yaw_hist.unsqueeze(0)
        ),
        "yaw_curr": (
            None if packet.yaw_curr is None else packet.yaw_curr.reshape(1, 1)
        ),
    }


def _tensor_storage_ptr(tensor: torch.Tensor) -> int:
    """Storage-level identity: catches views and aliases, not just objects."""

    try:
        return int(tensor.untyped_storage().data_ptr())
    except (AttributeError, RuntimeError):
        return int(tensor.data_ptr())


def _observation_storage_ptrs(extracted: Mapping[str, Any]) -> frozenset[int]:
    return frozenset(
        _tensor_storage_ptr(value)
        for value in extracted.values()
        if isinstance(value, torch.Tensor)
    )


def _aliased_target_names(
    storage_ptrs: frozenset[int],
    *,
    aux_targets: Any = None,
    target_actions: torch.Tensor | None = None,
) -> list[str]:
    """Names of expert-side tensors sharing storage with the observation."""

    offenders: list[str] = []
    if (
        isinstance(target_actions, torch.Tensor)
        and _tensor_storage_ptr(target_actions) in storage_ptrs
    ):
        offenders.append("target_actions")
    if aux_targets is not None:
        mapping = (
            aux_targets.as_targets()
            if hasattr(aux_targets, "as_targets")
            else aux_targets
        )
        if isinstance(mapping, Mapping):
            for name, value in mapping.items():
                if (
                    isinstance(value, torch.Tensor)
                    and _tensor_storage_ptr(value) in storage_ptrs
                ):
                    offenders.append(str(name))
    return sorted(offenders)


def _assert_row_targets_not_aliased(
    row: RunnerRow,
    storage_ptrs: frozenset[int],
    adapter: nn.Module,
    label: str,
) -> None:
    offenders = _aliased_target_names(
        storage_ptrs,
        aux_targets=row.aux_targets,
        target_actions=row.target_actions,
    )
    if offenders:
        adapter.record_expert_future_leak(len(offenders))
        raise F2AssemblyContractError(
            f"{label}: expert target tensors share storage with the "
            f"observation: {offenders!r}"
        )


def _aux_target_mapping(aux_targets: Any) -> Mapping[str, torch.Tensor]:
    """Normalize label packets (``as_targets`` protocol) or plain mappings."""

    if hasattr(aux_targets, "as_targets"):
        aux_targets = aux_targets.as_targets()
    if not isinstance(aux_targets, Mapping) or not aux_targets:
        raise F2AssemblyContractError(
            "aux_targets must be a nonempty label mapping or an "
            "AuxTargetPacket-style object"
        )
    return aux_targets


class _RowScratch:
    """Per-row loss capture with strict one-read-one-clear semantics."""

    __slots__ = (
        "aux_loss",
        "track1",
        "track2",
        "per_aux_components",
        "observation_storage_ptrs",
    )

    def __init__(self) -> None:
        self.aux_loss: torch.Tensor | None = None
        self.track1: torch.Tensor | None = None
        self.track2: torch.Tensor | None = None
        self.per_aux_components: dict[str, torch.Tensor] = {}
        self.observation_storage_ptrs: frozenset[int] = frozenset()

    def assert_clear(self, label: str) -> None:
        if (
            self.aux_loss is not None
            or self.track1 is not None
            or self.track2 is not None
        ):
            raise F2AssemblyContractError(
                f"row scratch residue detected at {label}; the previous row "
                "was not consumed exactly once"
            )

    def clear(self) -> None:
        self.aux_loss = None
        self.track1 = None
        self.track2 = None
        self.per_aux_components = {}
        self.observation_storage_ptrs = frozenset()


class ArmExecutor:
    """Bind one arm's modules and optimizer to the frozen runner callbacks."""

    def __init__(
        self,
        arm: F2ArmModules,
        optimizer: torch.optim.AdamW,
        contract: OptimizerContract,
        *,
        g6: G6Instrument | None = None,
    ) -> None:
        if not isinstance(arm, F2ArmModules):
            raise F2AssemblyContractError("arm must be an F2ArmModules")
        if not isinstance(optimizer, torch.optim.AdamW):
            raise F2AssemblyContractError("optimizer must be a torch AdamW")
        if not isinstance(contract, OptimizerContract):
            raise F2AssemblyContractError("contract must be an OptimizerContract")
        if g6 is not None and not isinstance(g6, G6Instrument):
            raise F2AssemblyContractError("g6 must be a G6Instrument or None")
        # GPT-5.6 sol review P1-6: audit counters must never fail open.  A
        # production arm whose adapter cannot report the forbidden-dataflow
        # counters is rejected at construction, before any callback exists.
        if not callable(getattr(arm.adapter, "audit_counters", None)) or not callable(
            getattr(arm.adapter, "assert_audit_counters_clean", None)
        ):
            raise F2AssemblyContractError(
                "arm adapter must expose audit_counters and "
                "assert_audit_counters_clean; refusing fail-open callbacks"
            )
        self.arm = arm
        self.optimizer = optimizer
        self.contract = contract
        self.g6 = g6
        self._scratch = _RowScratch()
        self._perception_state: Mapping[str, Any] | None = None
        self._device = arm.base.act_token.device if hasattr(
            arm.base, "act_token"
        ) else next(arm.model.parameters()).device

    def callbacks(self) -> ArmCallbacks:
        return ArmCallbacks(
            checkpoint_state=self.arm.full_state_dict(),
            feature_forward=self.feature_forward,
            aux_forward=self.aux_forward,
            head_forward=self.head_forward,
            track_loss=self.track_loss,
            backward=self.backward,
            optimizer_step=self.optimizer_step,
            audit_counters=self.arm.adapter.audit_counters,
        )

    def feature_forward(
        self, observation: Any, event: RowEvent
    ) -> FeatureForwardResult:
        self._scratch.assert_clear(f"feature_forward row {event.row_position}")
        extracted = _extract_observation(observation)
        if self._perception_state is None:
            self._perception_state = self.arm.adapter.init_state(1, self._device)
            if not event.reset:
                raise F2AssemblyContractError(
                    "first runner row must carry a reset"
                )
        output = self.arm.adapter.encode_step(
            extracted["coarse_tokens"],
            extracted["coarse_tidx"],
            extracted["fine_tokens"],
            extracted["fine_tidx"],
            [extracted["instruction"]],
            self._perception_state,
            reset_mask=bool(event.reset),
            yaw_hist=extracted.get("yaw_hist"),
            yaw_curr=extracted.get("yaw_curr"),
        )
        self._perception_state = output["new_state"]
        self._scratch.observation_storage_ptrs = _observation_storage_ptrs(
            extracted
        )
        return FeatureForwardResult(
            value=output, reference_tensor=output["h_act"]
        )

    def aux_forward(
        self, features: Any, aux_targets: Any, event: RowEvent
    ) -> AuxForwardResult:
        if not isinstance(features, Mapping) or "h_act" not in features:
            raise F2AssemblyContractError(
                "aux_forward features must be the adapter encode output"
            )
        if self._scratch.aux_loss is not None:
            raise F2AssemblyContractError(
                f"aux_forward called twice for row {event.row_position}"
            )
        components = PACKAGE_AUX_COMPONENTS[self.arm.package]
        weighted = features["h_act"].sum() * 0.0
        weighted_components: dict[str, torch.Tensor] = {}
        if components:
            aux_targets = _aux_target_mapping(aux_targets)
            leaked = _aliased_target_names(
                self._scratch.observation_storage_ptrs,
                aux_targets=aux_targets,
            )
            if leaked:
                self.arm.adapter.record_expert_future_leak(len(leaked))
                raise F2AssemblyContractError(
                    "expert label tensors are aliased into the observation: "
                    f"{leaked!r}"
                )
            losses = self.arm.adapter.compute_aux_losses(features, aux_targets)
            for name in components:
                if name not in losses:
                    raise F2AssemblyContractError(
                        f"adapter aux losses are missing {name!r}"
                    )
                term = self.arm.aux_coefficients[name] * losses[name]
                weighted_components[name] = term
                weighted = weighted + term
        self._scratch.per_aux_components = weighted_components
        self._scratch.aux_loss = weighted
        return AuxForwardResult(loss=weighted)

    def head_forward(
        self, features: Any, prev_fy: torch.Tensor, event: HeadEvent
    ) -> HeadForwardResult:
        if not isinstance(features, Mapping) or "base_features" not in features:
            raise F2AssemblyContractError(
                "head_forward features must be the adapter encode output"
            )
        output = self.arm.model(
            features["base_features"],
            prev_fy,
            method_features=features["method_features"],
            method_alphas=features["method_alphas"],
        )
        return HeadForwardResult(
            prediction=output.prediction,
            g7_telemetry=output.fused_context.telemetry.detached(),
        )

    def track_loss(
        self,
        prediction: Any,
        target_actions: torch.Tensor,
        event: HeadEvent,
    ) -> torch.Tensor:
        if (
            isinstance(target_actions, torch.Tensor)
            and _tensor_storage_ptr(target_actions)
            in self._scratch.observation_storage_ptrs
        ):
            self.arm.adapter.record_expert_future_leak()
            raise F2AssemblyContractError(
                "expert target_actions share storage with the observation "
                f"at row {event.row_position}"
            )
        loss = ap2_track_loss(prediction, target_actions).total
        if event.branch == "branch1":
            if self._scratch.track1 is not None:
                raise F2AssemblyContractError(
                    f"branch1 track loss recorded twice for row "
                    f"{event.row_position}"
                )
            self._scratch.track1 = loss
        else:
            if self._scratch.track2 is not None:
                raise F2AssemblyContractError(
                    f"branch2 track loss recorded twice for row "
                    f"{event.row_position}"
                )
            self._scratch.track2 = loss
        return loss

    def backward(self, event: BackwardEvent) -> None:
        scratch = self._scratch
        if (
            scratch.aux_loss is None
            or scratch.track1 is None
            or scratch.track2 is None
        ):
            raise F2AssemblyContractError(
                f"backward at row {event.row_position} is missing captured "
                "losses; each row must record aux, branch1, and branch2 "
                "exactly once"
            )
        if self.g6 is not None:
            self.g6.observe_row(
                aux_loss=scratch.aux_loss,
                track1=scratch.track1,
                track2=scratch.track2,
                per_aux_losses=scratch.per_aux_components or None,
            )
        event.scaled_loss.backward()
        scratch.clear()

    def optimizer_step(self, event: OptimizerUpdateEvent) -> None:
        del event
        torch.nn.utils.clip_grad_norm_(
            list(self.arm.trainable_parameters()), self.contract.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.arm.adapter.assert_audit_counters_clean()


def make_backward_callback(
    executor: ArmExecutor,
) -> Callable[[BackwardEvent], None]:
    """Expose the executor's G6-then-backward callback (plan section 3.7)."""

    if not isinstance(executor, ArmExecutor):
        raise F2AssemblyContractError("executor must be an ArmExecutor")
    return executor.backward


def make_optimizer_step_callback(
    executor: ArmExecutor,
) -> Callable[[OptimizerUpdateEvent], None]:
    """Expose the executor's clip/step/zero-grad/audit callback."""

    if not isinstance(executor, ArmExecutor):
        raise F2AssemblyContractError("executor must be an ArmExecutor")
    return executor.optimizer_step


def build_arm_callbacks(
    arm: F2ArmModules,
    optimizer: torch.optim.AdamW,
    contract: OptimizerContract,
    *,
    g6: G6Instrument | None = None,
) -> tuple[ArmCallbacks, ArmExecutor]:
    """Assemble one arm's runner callbacks; returns the executor for audits."""

    executor = ArmExecutor(arm, optimizer, contract, g6=g6)
    return executor.callbacks(), executor


# ---------------------------------------------------------------------------
# Real base checkpoint loading (train_pfem.py recipe, fail-closed)
# ---------------------------------------------------------------------------

_VENDORED_OPENTRACKVLA_ROOT = (
    Path(__file__).resolve().parents[1] / "third_party" / "OpenTrackVLA"
)

BASE_CONFIG_REQUIRED_KEYS = (
    "freeze_llm",
    "n_waypoints",
    "max_time",
    "beta_nav",
    "use_angle_tvi",
    "use_tanh_actions",
    "vision_feat_dim",
)

# Adjudication ruling b freeze protocol: the per-aux lambda values freeze in
# the CAL audit receipt with PRIMARY authority and are then written here as
# literals bound by source receipt v4.  While this is None the production
# smoke plan fails closed; the CAL audit itself runs with the unweighted
# placeholder below because per-component gradient norms are lambda-free.
#
# 2026-07-20 Windows CUDA seeded CAL v3: two seed-0 cuda:0 CAL runs under
# windows_cuda_deterministic_v1 produced byte-identical receipts (SHA-256
# e2b87cc3...fee778).  The prior MPS literals remain preserved in their
# historical receipt but are superseded for this Windows assembly by the
# verbatim proposal below, frozen before any smoke update/EVAL result.
FROZEN_AUX_COEFFICIENTS: Mapping[str, float] | None = {
    "L_cot": 0.0195,
    "L_future": 0.34,
    "L_verify": 0.5,
}
CAL_PLACEHOLDER_AUX_COEFFICIENTS: Mapping[str, float] = {
    "L_cot": 1.0,
    "L_future": 1.0,
    "L_verify": 1.0,
}


def default_device() -> torch.device:
    """cuda > mps > cpu, mirroring the main-v1 train_pfem device policy."""

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _vendored_module(name: str) -> Any:
    """Import a vendored top-level OpenTrackVLA module (train_pfem pattern).

    ``third_party/OpenTrackVLA/model.py`` imports its siblings as top-level
    modules, so the vendored root must be on ``sys.path`` exactly as
    ``scripts/train_pfem.py`` arranges it.
    """

    root = str(_VENDORED_OPENTRACKVLA_ROOT)
    if not _VENDORED_OPENTRACKVLA_ROOT.is_dir():
        raise F2AssemblyContractError(
            f"vendored OpenTrackVLA tree is missing: {root}"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def load_base_checkpoint(
    base_hf_dir: str | Path | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Load the frozen official OpenTrackVLA base from the HF checkpoint.

    Mirrors the sanctioned ``train_pfem.py`` recipe: offline env, local Qwen
    resolution, ``ModelConfig`` from the checkpoint ``config.json``, and the
    ``model.``-prefixed safetensors state.  Any missing or unexpected tensor
    after prefix stripping fails closed (the live checkpoint loads 0/0).
    The returned base is put in ``eval()`` mode (frozen-LLM discipline; no
    F2 module uses dropout or batch norm).
    """

    hf_dir = resolve_frozen_base_hf_dir(base_hf_dir)
    config_path = hf_dir / "config.json"
    state_path = hf_dir / "model.safetensors"
    for label, path in (("config.json", config_path), ("model.safetensors", state_path)):
        if not path.is_file():
            raise F2AssemblyContractError(
                f"base HF checkpoint {label} is missing: {path}"
            )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    local_weights = _vendored_module("local_weights")
    qwen_path = local_weights.resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=None,
        env_var="QWEN_MODEL_PATH",
        candidates=local_weights.default_qwen_candidates(),
    )
    os.environ["QWEN_MODEL_PATH"] = str(qwen_path)
    vendored = _vendored_module("model")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise F2AssemblyContractError(
            f"base HF config.json is unreadable: {config_path}"
        ) from exc
    missing_config = [
        key for key in BASE_CONFIG_REQUIRED_KEYS if key not in config
    ]
    if missing_config:
        raise F2AssemblyContractError(
            f"base HF config.json is missing keys: {missing_config!r}"
        )
    model_config = vendored.ModelConfig(
        llm_name=str(qwen_path),
        freeze_llm=bool(config["freeze_llm"]),
        n_waypoints=int(config["n_waypoints"]),
        max_time=int(config["max_time"]),
        beta_nav=float(config["beta_nav"]),
        use_angle_tvi=bool(config["use_angle_tvi"]),
        use_tanh_actions=bool(config["use_tanh_actions"]),
        alpha_xy=config.get("alpha_xy"),
    )
    base = vendored.OpenTrackVLA(
        model_config, vision_feat_dim=int(config["vision_feat_dim"])
    )
    from safetensors.torch import load_file as load_safetensors

    state = load_safetensors(str(state_path))
    prefix = "model."
    stripped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not key.startswith(prefix):
            raise F2AssemblyContractError(
                f"base checkpoint key {key!r} lacks the {prefix!r} prefix"
            )
        stripped[key[len(prefix):]] = value
    missing, unexpected = base.load_state_dict(stripped, strict=False)
    if missing or unexpected:
        raise F2AssemblyContractError(
            "base checkpoint state does not exactly match the vendored "
            f"architecture: missing={sorted(missing)[:5]!r}, "
            f"unexpected={sorted(unexpected)[:5]!r}"
        )
    base.eval()
    report = {
        "base_hf_dir": str(hf_dir),
        "qwen_path": str(qwen_path),
        "d_model": int(base.D),
        "vision_feat_dim": int(config["vision_feat_dim"]),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "module_mode": "eval",
    }
    return base, report


# ---------------------------------------------------------------------------
# EVAL-FIX row predictor (frozen inference; runner-branch2-aligned recurrence)
# ---------------------------------------------------------------------------


class EvalRowPredictor:
    """Frozen-inference EVAL-FIX predictor over one arm's live modules.

    ``assembly.run_eval_fix`` calls it under ``torch.no_grad`` as
    ``predictor(row, prev_tensor, mode=..., reset=..., position=...)``.
    ``position == 0`` starts a fresh pass (fresh perception state);
    afterwards the same mode must proceed in strict position order.  The
    perception state lives in this object only, so evaluation passes never
    disturb the training executor's recurrent state.
    """

    def __init__(self, arm: F2ArmModules) -> None:
        if not isinstance(arm, F2ArmModules):
            raise F2AssemblyContractError("arm must be an F2ArmModules")
        self.arm = arm
        self._device = (
            arm.base.act_token.device
            if hasattr(arm.base, "act_token")
            else next(arm.model.parameters()).device
        )
        self._state: Mapping[str, Any] | None = None
        self._mode: str | None = None
        self._position = -1
        arm.base.eval()
        arm.adapter.eval()
        arm.model.eval()

    def __call__(
        self,
        row: RunnerRow,
        prev_fy: torch.Tensor,
        *,
        mode: str,
        reset: bool,
        position: int,
    ) -> Any:
        if not isinstance(row, RunnerRow):
            raise F2AssemblyContractError("EVAL predictor requires a RunnerRow")
        if mode not in ("logged", "self"):
            raise F2AssemblyContractError(f"unknown EVAL mode {mode!r}")
        if position == 0:
            if not reset:
                raise F2AssemblyContractError(
                    "EVAL pass must start with a reset row at position 0"
                )
            self._state = self.arm.adapter.init_state(1, self._device)
            self._mode = mode
            self._position = 0
        else:
            if mode != self._mode or position != self._position + 1:
                raise F2AssemblyContractError(
                    "EVAL predictor calls broke the per-mode pass order: "
                    f"got mode={mode!r} position={position}, expected "
                    f"mode={self._mode!r} position={self._position + 1}"
                )
            self._position = position
        extracted = _extract_observation(row.observation)
        _assert_row_targets_not_aliased(
            row,
            _observation_storage_ptrs(extracted),
            self.arm.adapter,
            f"EVAL[{mode}][{position}]",
        )
        output = self.arm.adapter.encode_step(
            extracted["coarse_tokens"],
            extracted["coarse_tidx"],
            extracted["fine_tokens"],
            extracted["fine_tidx"],
            [extracted["instruction"]],
            self._state,
            reset_mask=bool(reset),
            yaw_hist=extracted.get("yaw_hist"),
            yaw_curr=extracted.get("yaw_curr"),
        )
        self._state = output["new_state"]
        reference = output["h_act"]
        prev = prev_fy.to(device=reference.device, dtype=reference.dtype)
        model_output = self.arm.model(
            output["base_features"],
            prev,
            method_features=output["method_features"],
            method_alphas=output["method_alphas"],
        )
        return model_output.prediction


def build_eval_row_predictor(arm: F2ArmModules) -> EvalRowPredictor:
    """Build the EVAL-FIX predictor for one live arm."""

    return EvalRowPredictor(arm)


# ---------------------------------------------------------------------------
# CAL zero-update row auditor (integration seam: assembly.run_cal_audit)
# ---------------------------------------------------------------------------


class CalRowAuditor:
    """Stateful per-row CAL auditor over one SA-Hstar arm; zero updates.

    Per row it reports HS6 step-0 parity, the prev-free observation-graph
    assertion, and the ``base.proj`` probe-surface gradient norms of the
    three frozen aux blocks and the track loss (``allow_unused``; an
    unreachable loss reports norm 0.0 rather than raising).  No optimizer
    is ever constructed and no parameter is ever mutated.
    """

    def __init__(self, arm: F2ArmModules, *, seed: int | None = None) -> None:
        if not isinstance(arm, F2ArmModules):
            raise F2AssemblyContractError("arm must be an F2ArmModules")
        components = set(PACKAGE_AUX_COMPONENTS[arm.package])
        if components != {"L_cot", "L_future", "L_verify"}:
            raise F2AssemblyContractError(
                "CAL audit requires the full SA-Hstar aux block set; "
                f"{arm.package} provides {sorted(components)!r}"
            )
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
        ):
            raise F2AssemblyContractError(
                "CAL seed must be a nonnegative integer or None"
            )
        self.arm = arm
        self.seed = seed
        self.probe = tuple(
            parameter for parameter in arm.base.proj.parameters()
        )
        if not self.probe:
            raise F2AssemblyContractError("CAL probe surface is empty")
        self._device = (
            arm.base.act_token.device
            if hasattr(arm.base, "act_token")
            else next(arm.model.parameters()).device
        )
        self.initial_state_sha256 = checkpoint_init_sha256(
            arm.full_state_dict()
        )
        self._state: Mapping[str, Any] | None = None
        self._position = -1

    def context_receipt(self) -> dict[str, Any]:
        """Reproducibility binding for the CAL audit receipt (review P1-1)."""

        receipt = {
            "seed": self.seed,
            "device": str(self._device),
            "package": self.arm.package,
            "probe_surface": G6_PROBE_SURFACE,
            "initialization": (
                "torch.manual_seed(seed) followed by build_package, the "
                "byte-identical smoke arm initialization path"
            ),
            "checkpoint_init_sha256": self.initial_state_sha256,
        }
        if self._device.type == "cuda":
            receipt["cuda_reproducibility"] = cuda_reproducibility_receipt()
        return receipt

    def _probe_grad_norm(self, loss: torch.Tensor, label: str) -> float:
        if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
            raise F2AssemblyContractError(f"{label} must be a scalar loss")
        if not loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            loss,
            self.probe,
            retain_graph=True,
            allow_unused=True,
        )
        total = 0.0
        for gradient in gradients:
            if gradient is None:
                continue
            value = float(
                torch.linalg.vector_norm(
                    gradient.detach().to("cpu").to(torch.float64)
                ).item()
            )
            total += value * value
        result = math.sqrt(total)
        if not math.isfinite(result):
            raise F2AssemblyContractError(f"{label} probe gradient is nonfinite")
        return result

    def __call__(
        self,
        row: RunnerRow,
        reasons: Sequence[str],
        position: int,
    ) -> CalRowAudit:
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            raise F2AssemblyContractError("CAL position must be a nonnegative int")
        if not isinstance(row, RunnerRow):
            raise F2AssemblyContractError("CAL auditor requires a RunnerRow")
        reset = bool(tuple(reasons))
        if position == 0:
            if not reset:
                raise F2AssemblyContractError(
                    "CAL position 0 must carry a reset reason"
                )
            self._state = self.arm.adapter.init_state(1, self._device)
            self._position = 0
        else:
            if position != self._position + 1 or self._state is None:
                raise F2AssemblyContractError(
                    f"CAL audit position discontinuity: got {position}, "
                    f"expected {self._position + 1}"
                )
            self._position = position

        # Review P1-2: the prev leaf exists BEFORE the observation encoding
        # and is the only P_prev input downstream, so the graph audit below
        # genuinely proves the encoding graph is independent of it.
        prev_leaf = torch.tensor(
            [[row.logged_prev_action[0], row.logged_prev_action[2]]],
            dtype=torch.float32,
            device=self._device,
            requires_grad=True,
        )
        extracted = _extract_observation(row.observation)
        _assert_row_targets_not_aliased(
            row,
            _observation_storage_ptrs(extracted),
            self.arm.adapter,
            f"CAL position {position}",
        )
        output = self.arm.adapter.encode_step(
            extracted["coarse_tokens"],
            extracted["coarse_tidx"],
            extracted["fine_tokens"],
            extracted["fine_tidx"],
            [extracted["instruction"]],
            self._state,
            reset_mask=reset,
            yaw_hist=extracted.get("yaw_hist"),
            yaw_curr=extracted.get("yaw_curr"),
        )
        self._state = output["new_state"]

        reference = output["h_act"]
        prev_leaf = prev_leaf.to(dtype=reference.dtype)
        if not prev_leaf.requires_grad:
            raise F2AssemblyContractError("CAL prev leaf lost its grad flag")

        if not output["h_act"].requires_grad:
            raise F2AssemblyContractError(
                "CAL h_act carries no graph; the probe path is dead"
            )
        # A tensor with no autograd graph at all provably cannot depend on
        # the prev leaf (e.g. tim_q on reset rows: detached TIM memory plus
        # the no-grad Polar confidence decode), so the graph assertion only
        # audits graph-carrying tensors.
        audited = {"base_h_act": output["h_act"]}
        for name, tensor in output["method_features"].items():
            if tensor.requires_grad:
                audited[f"method_{name}"] = tensor
        prev_free = True
        try:
            assert_prev_free_tensors(audited, prev_leaf)
        except F2ModelContractError:
            prev_free = False

        model_output = self.arm.model(
            output["base_features"],
            prev_leaf,
            method_features=output["method_features"],
            method_alphas=output["method_alphas"],
        )
        step0_parity = True
        try:
            assert_step0_controlled_axis_persistence(
                model_output.prediction, prev_leaf
            )
        except F2ModelContractError:
            step0_parity = False

        targets = _aux_target_mapping(row.aux_targets)
        losses = self.arm.adapter.compute_aux_losses(output, targets)
        aux_grad_norms: dict[str, float] = {}
        for name in ("L_cot", "L_future", "L_verify"):
            if name not in losses:
                raise F2AssemblyContractError(
                    f"CAL adapter aux losses are missing {name!r}"
                )
            aux_grad_norms[name] = self._probe_grad_norm(
                losses[name], f"CAL {name}"
            )
        target_actions = row.target_actions.to(
            device=reference.device, dtype=reference.dtype
        ).unsqueeze(0)
        track_loss = ap2_track_loss(
            model_output.prediction, target_actions
        ).total
        track_grad_norm = self._probe_grad_norm(track_loss, "CAL track loss")
        audit_kwargs: dict[str, Any] = {
            "step0_parity": step0_parity,
            "prev_free": prev_free,
            "aux_grad_norms": aux_grad_norms,
            "track_grad_norm": track_grad_norm,
        }
        # Forward-compatible seed/device binding: fill them as soon as the
        # lifecycle-side CalRowAudit dataclass grows the fields.
        audit_field_names = {
            field.name for field in dataclass_fields(CalRowAudit)
        }
        if "seed" in audit_field_names:
            audit_kwargs["seed"] = self.seed
        if "device" in audit_field_names:
            audit_kwargs["device"] = str(self._device)
        return CalRowAudit(**audit_kwargs)


CAL_CONTEXT_SEED = 0

_ACTIVE_CAL_AUDITOR: CalRowAuditor | None = None


def build_cal_audit_context(
    base: nn.Module,
    *,
    device: torch.device | str,
    seed: int = CAL_CONTEXT_SEED,
) -> CalRowAuditor:
    """Build the seeded CAL audit context (GPT-5.6 sol review P1-1).

    ``torch.manual_seed(seed)`` followed by ``build_package`` is exactly the
    initialization path ``build_paired_arms`` uses for the smoke arms, so
    with the same seed the CAL gradient statistics are computed on the
    byte-identical frozen smoke initial weights and reproduce run to run.
    The seed is recorded on the auditor for receipt binding.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise F2AssemblyContractError("CAL seed must be a nonnegative integer")
    torch.manual_seed(seed)
    arm = build_package(
        SMOKE_PACKAGE,
        base,
        device=device,
        aux_coefficients=CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    )
    return CalRowAuditor(arm, seed=seed)


def active_cal_context_receipt() -> dict[str, Any]:
    """Reproducibility context of the live CAL audit (seed/device/SHA)."""

    if _ACTIVE_CAL_AUDITOR is None:
        raise F2AssemblyContractError(
            "no active CAL audit context; audit_cal_row position 0 has not run"
        )
    return _ACTIVE_CAL_AUDITOR.context_receipt()


def audit_cal_row(
    row: RunnerRow,
    reasons: Sequence[str],
    position: int,
) -> CalRowAudit:
    """Integration seam for ``assembly.run_cal_audit`` (stateful across rows).

    Position 0 (re)builds the default CAL context: the frozen base HF
    checkpoint and the SA-Hstar package on the default device, seeded with
    ``CAL_CONTEXT_SEED`` so the audited weights are byte-identical to the
    seed-0 smoke initialization (review P1-1); coefficients stay at the
    unweighted placeholder because per-component gradient norms are
    lambda-free.  A call before any position-0 initialization fails closed.
    """

    global _ACTIVE_CAL_AUDITOR
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        raise F2AssemblyContractError("CAL position must be a nonnegative int")
    if position == 0:
        target_device = default_device()
        if target_device.type == "cuda":
            configure_cuda_reproducibility()
        base, _report = load_base_checkpoint()
        _ACTIVE_CAL_AUDITOR = build_cal_audit_context(
            base,
            device=target_device,
            seed=CAL_CONTEXT_SEED,
        )
    if _ACTIVE_CAL_AUDITOR is None:
        raise F2AssemblyContractError(
            "CAL audit must start at position 0 to initialize its context"
        )
    return _ACTIVE_CAL_AUDITOR(row, reasons, position)


# ---------------------------------------------------------------------------
# Production smoke plan (integration seam: assembly.run_production_smoke)
# ---------------------------------------------------------------------------


def _checkpoint_payload_factory(
    arm_assembly: ArmAssembly,
    *,
    seed: int,
    device: str,
) -> Callable[[], dict[str, Any]]:
    def _payload() -> dict[str, Any]:
        return {
            "model": arm_assembly.modules.full_state_dict(),
            "optimizer": arm_assembly.optimizer.state_dict(),
            "seed": int(seed),
            "device": str(device),
        }

    return _payload


def _cross_check_optimizer_contract(
    contract: OptimizerContract, declared: Any
) -> None:
    if declared is None:
        return
    if not isinstance(declared, Mapping):
        raise F2AssemblyContractError(
            "receipt optimizer_contract must be a mapping"
        )
    expected = {
        "base_lr": contract.base_lr,
        "head_lr": contract.head_lr,
        "weight_decay": contract.weight_decay,
        "grad_clip_norm": contract.grad_clip,
        "betas": list(contract.betas),
        "eps": contract.eps,
    }
    for key, value in expected.items():
        if key in declared and declared[key] != value:
            raise F2AssemblyContractError(
                f"receipt optimizer_contract.{key}={declared[key]!r} differs "
                f"from the frozen OptimizerContract value {value!r}"
            )


def _receipt_base_hf_dir(receipt_document: Mapping[str, Any]) -> str | None:
    asset_binding = receipt_document.get("asset_binding")
    if isinstance(asset_binding, Mapping):
        base_hf = asset_binding.get("base_hf")
        if isinstance(base_hf, Mapping):
            path = base_hf.get("path")
            if isinstance(path, str) and path:
                return path
    return None


def build_production_smoke_plan(
    project_root: str | Path,
    receipt_document: Mapping[str, Any],
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    base: nn.Module | None = None,
    aux_coefficients: Mapping[str, float] | None = None,
) -> SmokeAssemblyPlan:
    """Assemble the real smoke plan (plan section 4 wiring).

    Loads the frozen supports through the fail-closed ``assembly_data``
    loaders, loads the frozen base HF checkpoint (unless a base is
    injected), builds the bit-identical SA-Hstar paired arms, and wires the
    runner callbacks, the EVAL-FIX predictors, the checkpoint payload
    accessors, and the shared G6 instrument.  Fails closed while the
    ruling-b lambda literals are not frozen.
    """

    root = Path(project_root).expanduser().resolve()
    if not isinstance(receipt_document, Mapping):
        raise F2AssemblyContractError("receipt_document must be a mapping")
    package = receipt_document.get("smoke_package", SMOKE_PACKAGE)
    if package != SMOKE_PACKAGE:
        raise F2AssemblyContractError(
            f"smoke package must be {SMOKE_PACKAGE!r} (adjudication ruling "
            f"d); receipt declares {package!r}"
        )
    block_mode = str(receipt_document.get("block_mode", "bstar"))
    if block_mode != "bstar":
        raise F2AssemblyContractError(
            "the production smoke instrument is wired for the deciding "
            f"bstar block mode (ruling c); receipt declares {block_mode!r}"
        )
    coefficients = (
        aux_coefficients
        if aux_coefficients is not None
        else FROZEN_AUX_COEFFICIENTS
    )
    if coefficients is None:
        raise F2AssemblyContractError(
            "aux lambda coefficients are not frozen yet (adjudication "
            "ruling b): run the CAL audit, obtain the PRIMARY freeze, and "
            "write FROZEN_AUX_COEFFICIENTS as source literals before any "
            "production smoke"
        )
    contract = OptimizerContract()
    _cross_check_optimizer_contract(
        contract, receipt_document.get("optimizer_contract")
    )
    target_device = (
        torch.device(device) if device is not None else default_device()
    )
    if target_device.type == "cuda" and target_device.index is None:
        target_device = torch.device("cuda:0")
    cuda_reproducibility = None
    if target_device.type == "cuda":
        cuda_reproducibility = configure_cuda_reproducibility()

    train_path = (root / FROZEN_TRAIN_RELATIVE).resolve()
    if not train_path.is_file():
        raise F2AssemblyContractError(
            f"frozen train JSONL is missing: {train_path}"
        )
    support_receipt = build_frozen_support(train_path)
    raw_rows = parse_train_jsonl(train_path.read_bytes())
    base_root, cache_root = frozen_cache_roots(root)
    token_ledger = _resolve_frozen_token_ledger(root, receipt_document)
    smoke_rows = build_runner_rows(
        rows=raw_rows,
        receipt=support_receipt,
        support_name="SMK-TRAIN",
        base_root=base_root,
        cache_root=cache_root,
        token_ledger=token_ledger,
    )
    eval_rows = build_runner_rows(
        rows=raw_rows,
        receipt=support_receipt,
        support_name="EVAL-FIX",
        base_root=base_root,
        cache_root=cache_root,
        token_ledger=token_ledger,
    )
    eval_raw_rows = tuple(
        row
        for _index, row in ordered_support_rows(
            raw_rows, support_receipt, "EVAL-FIX"
        )
    )
    smoke_strafe, smoke_expected = smoke_reset_sets(
        support_receipt, "SMK-TRAIN"
    )
    eval_strafe, _eval_expected = smoke_reset_sets(support_receipt, "EVAL-FIX")

    if base is None:
        base, _load_report = load_base_checkpoint(
            _receipt_base_hf_dir(receipt_document)
        )
    paired = build_paired_arms(
        base,
        package=package,
        seed=seed,
        device=target_device,
        contract=contract,
        aux_coefficients=coefficients,
    )
    ctrl = paired.arms[S_CTRL]
    g6 = G6Instrument(
        tuple(ctrl.modules.base.proj.parameters()), block_mode=block_mode
    )
    arms: dict[str, SmokeArmAssembly] = {}
    for arm_name in (S_CTRL, S_SELF):
        arm_assembly = paired.arms[arm_name]
        callbacks, _executor = build_arm_callbacks(
            arm_assembly.modules,
            arm_assembly.optimizer,
            contract,
            g6=g6 if arm_name == S_CTRL else None,
        )
        arms[arm_name] = SmokeArmAssembly(
            callbacks=callbacks,
            eval_predictor=EvalRowPredictor(arm_assembly.modules),
            checkpoint_payload=_checkpoint_payload_factory(
                arm_assembly,
                seed=seed,
                device=paired.device,
            ),
        )
    plan_kwargs: dict[str, Any] = {
        "smoke_rows": tuple(smoke_rows),
        "eval_rows": tuple(eval_rows),
        "eval_raw_rows": eval_raw_rows,
        "strafe_reset_original_indices": frozenset(smoke_strafe | eval_strafe),
        "expected_static_reset_original_indices": frozenset(smoke_expected),
        "arms": arms,
        "g6_update": g6.emit_update,
    }
    # Reproducibility metadata (review P3): recorded as soon as the
    # lifecycle-side SmokeAssemblyPlan dataclass grows the fields.
    plan_field_names = {
        field.name for field in dataclass_fields(SmokeAssemblyPlan)
    }
    if "seed" in plan_field_names:
        plan_kwargs["seed"] = seed
    if "device" in plan_field_names:
        plan_kwargs["device"] = paired.device
    if "checkpoint_init_sha256" in plan_field_names:
        plan_kwargs["checkpoint_init_sha256"] = paired.checkpoint_init_sha256
    if "cuda_reproducibility" in plan_field_names:
        plan_kwargs["cuda_reproducibility"] = cuda_reproducibility
    if "g6_fallback_evidence" in plan_field_names:
        plan_kwargs["g6_fallback_evidence"] = g6.fallback_evidence
    return SmokeAssemblyPlan(**plan_kwargs)


# ---------------------------------------------------------------------------
# Checkpoint-based EVAL predictor (integration seam: forensics EVAL command)
# ---------------------------------------------------------------------------


def build_eval_row_predictor_from_checkpoint(
    project_root: str | Path,
    receipt_document: Mapping[str, Any],
    arm: str,
    payload: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    base: nn.Module | None = None,
) -> EvalRowPredictor:
    """Rebuild a frozen inference arm from a verified checkpoint payload.

    ``payload`` is the output of ``assembly.load_arm_checkpoint_verified``.
    The tensor state must split exactly into ``adapter.*``/``model.*``
    prefixes, load strictly, and the reconstructed modules must hash back
    to the payload's ``checkpoint_init_sha256`` bit-for-bit.
    """

    del project_root  # identity comes from the receipt and payload SHAs
    if not isinstance(receipt_document, Mapping):
        raise F2AssemblyContractError("receipt_document must be a mapping")
    if not isinstance(payload, Mapping) or "model" not in payload:
        raise F2AssemblyContractError(
            "checkpoint payload must be a mapping with a 'model' state"
        )
    if payload.get("arm") is not None and payload["arm"] != arm:
        raise F2AssemblyContractError(
            f"checkpoint payload belongs to {payload['arm']!r}, not {arm!r}"
        )
    package = receipt_document.get("smoke_package", SMOKE_PACKAGE)
    if package != SMOKE_PACKAGE:
        raise F2AssemblyContractError(
            f"smoke package must be {SMOKE_PACKAGE!r}; receipt declares "
            f"{package!r}"
        )
    target_device = (
        torch.device(device) if device is not None else default_device()
    )
    if base is None:
        base, _load_report = load_base_checkpoint(
            _receipt_base_hf_dir(receipt_document)
        )
    # Aux coefficients never enter evaluation; the placeholder only
    # satisfies the package constructor contract.
    modules = build_package(
        package,
        base,
        device=target_device,
        aux_coefficients=CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    )
    state = payload["model"]
    if not isinstance(state, Mapping) or not state:
        raise F2AssemblyContractError("checkpoint model state is empty")
    adapter_state: dict[str, torch.Tensor] = {}
    model_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("adapter."):
            adapter_state[key[len("adapter."):]] = value
        elif key.startswith("model."):
            model_state[key[len("model."):]] = value
        else:
            raise F2AssemblyContractError(
                f"checkpoint tensor {key!r} has an unknown prefix"
            )
    try:
        modules.adapter.load_state_dict(adapter_state, strict=True)
        modules.model.load_state_dict(model_state, strict=True)
    except RuntimeError as exc:
        raise F2AssemblyContractError(
            "checkpoint tensor set does not exactly match the rebuilt "
            f"{package} modules: {exc}"
        ) from exc
    rebuilt_sha = checkpoint_init_sha256(modules.full_state_dict())
    expected_sha = payload.get("checkpoint_init_sha256")
    if expected_sha is not None and rebuilt_sha != expected_sha:
        raise F2AssemblyContractError(
            "rebuilt checkpoint state SHA does not match the payload: "
            f"{rebuilt_sha} != {expected_sha}"
        )
    return EvalRowPredictor(modules)


__all__ = [
    "ArmAssembly",
    "ArmExecutor",
    "BASE_CONFIG_REQUIRED_KEYS",
    "CAL_CONTEXT_SEED",
    "CAL_PLACEHOLDER_AUX_COEFFICIENTS",
    "CalRowAuditor",
    "ConfidenceTIMObservationAdapter",
    "EvalRowPredictor",
    "F2ArmModules",
    "F2AssemblyContractError",
    "FROZEN_AUX_COEFFICIENTS",
    "G6Instrument",
    "G6_BLOCK_MODES",
    "G6_PROBE_SURFACE",
    "NullTIMObservationAdapter",
    "OBSERVATION_FORBIDDEN_KEYS",
    "OBSERVATION_OPTIONAL_KEYS",
    "OBSERVATION_REQUIRED_KEYS",
    "OptimizerContract",
    "PACKAGE_AUX_COMPONENTS",
    "PACKAGE_NAMES",
    "PackageName",
    "PairedArms",
    "SMOKE_PACKAGE",
    "TRAINABLE_BASE_MODULES",
    "active_cal_context_receipt",
    "audit_cal_row",
    "build_arm_callbacks",
    "build_arm_optimizer",
    "build_cal_audit_context",
    "build_eval_row_predictor",
    "build_eval_row_predictor_from_checkpoint",
    "build_package",
    "build_paired_arms",
    "build_production_smoke_plan",
    "default_device",
    "load_base_checkpoint",
    "make_backward_callback",
    "make_optimizer_step_callback",
]
