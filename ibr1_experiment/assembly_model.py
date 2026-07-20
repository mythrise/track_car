"""Model assembly for the preregistered IBR1 successor family.

The module intentionally reuses the sealed F2 observation adapter, optimizer
contract, runner arm containers, and checkpoint tensor hash.  It replaces only
the AP2 action module with :class:`IBR1AP2Model` and records the public
``IBR1-CTRL``/``IBR1-SELF`` identity separately from the runner's frozen
``S-CTRL``/``S-SELF`` keys.

No data, CAL row, or internal-test sample is loaded here.  The one filesystem
helper reads only the immutable F2 negative-result authority documents needed
to prove that seed-0 IBR1 starts from the same tensor bytes as the sealed F2
comparator.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from f2_experiment.assembly_model import (
    ArmAssembly,
    F2ArmModules,
    OptimizerContract,
    build_arm_optimizer,
)
from f2_experiment.assembly_data import F2AssemblyContractError
from f2_experiment.opentrack_adapter import OpenTrackVLAF2ObservationAdapter
from f2_experiment.runner import S_CTRL, S_SELF, checkpoint_init_sha256
from f2_experiment.support import canonical_json_sha256

from .model import IBR1AP2Model, IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


IBR1_PACKAGE = "SA-Hstar"
IBR1_CTRL = "IBR1-CTRL"
IBR1_SELF = "IBR1-SELF"
ENGINE_TO_FAMILY_ARM: Mapping[str, str] = {
    S_CTRL: IBR1_CTRL,
    S_SELF: IBR1_SELF,
}
FAMILY_TO_ENGINE_ARM: Mapping[str, str] = {
    family: engine for engine, family in ENGINE_TO_FAMILY_ARM.items()
}

IBR1_AUX_COMPONENTS = ("L_cot", "L_future", "L_verify")
IBR1_FROZEN_AUX_COEFFICIENTS: Mapping[str, float] = {
    "L_cot": 0.0195,
    "L_future": 0.34,
    "L_verify": 0.5,
}
IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS: Mapping[str, float] = {
    "L_cot": 1.0,
    "L_future": 1.0,
    "L_verify": 1.0,
}

F2_NEGATIVE_ADOPTION_RELATIVE = Path(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_negative_result_adoption_v1.json"
)
IBR1_PRIMARY_RELATIVE = Path(
    "experiments/windows_cuda_ibr1/preregistration/"
    "ibr1_primary_preregistration_v1.json"
)


class IBR1AssemblyContractError(F2AssemblyContractError):
    """Raised when IBR1 assembly identity or disjointness fails closed."""


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise IBR1AssemblyContractError(f"required authority file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IBR1AssemblyContractError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise IBR1AssemblyContractError(f"{label} must be a JSON object")
    return value


def _resolve_bound_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise IBR1AssemblyContractError(f"{label} has no bound path")
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _valid_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IBR1AssemblyContractError(f"{label} is not a lowercase SHA-256")
    return value


@dataclass(frozen=True)
class F2SealedInitEvidence:
    """Immutable evidence that supplies the inherited seed-0 tensor hash."""

    primary_path: str
    primary_sha256: str
    negative_adoption_path: str
    negative_adoption_sha256: str
    negative_seal_path: str
    negative_seal_sha256: str
    smoke_summary_path: str
    smoke_summary_sha256: str
    seed: int
    checkpoint_init_sha256: str

    def __post_init__(self) -> None:
        if self.seed != 0:
            raise IBR1AssemblyContractError("sealed F2 comparator seed must be 0")
        for field_name in (
            "primary_sha256",
            "negative_adoption_sha256",
            "negative_seal_sha256",
            "smoke_summary_sha256",
            "checkpoint_init_sha256",
        ):
            _valid_sha256(getattr(self, field_name), field_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": {
                "path": self.primary_path,
                "sha256": self.primary_sha256,
            },
            "negative_adoption": {
                "path": self.negative_adoption_path,
                "sha256": self.negative_adoption_sha256,
            },
            "negative_seal": {
                "path": self.negative_seal_path,
                "sha256": self.negative_seal_sha256,
            },
            "smoke_summary": {
                "path": self.smoke_summary_path,
                "sha256": self.smoke_summary_sha256,
            },
            "seed": self.seed,
            "checkpoint_init_sha256": self.checkpoint_init_sha256,
        }


def read_sealed_f2_init_evidence(
    project_root: str | Path,
) -> F2SealedInitEvidence:
    """Resolve the inherited init hash through the frozen adoption chain.

    Only JSON authority metadata is read.  The sealed internal-test subtree is
    neither resolved nor accessed.
    """

    root = Path(project_root).expanduser().resolve()
    primary_path = (root / IBR1_PRIMARY_RELATIVE).resolve()
    primary = _load_json(primary_path, "IBR1 PRIMARY index")
    primary_payload = dict(primary)
    primary_payload_sha = primary_payload.pop("receipt_payload_sha256", None)
    if (
        primary.get("analysis_class") != "ibr1_primary_preregistration_index"
        or primary.get("family_id") != IBR1_FAMILY_ID
        or primary.get("architecture_lock") != IBR1_ARCHITECTURE_LOCK
        or primary.get("internal_test") != "sealed"
        or primary.get("internal_test_opened") is not False
        or primary_payload_sha != canonical_json_sha256(primary_payload)
    ):
        raise IBR1AssemblyContractError("IBR1 PRIMARY identity/self-hash is invalid")
    adoption_path = (root / F2_NEGATIVE_ADOPTION_RELATIVE).resolve()
    primary_adoption = primary.get("component_bindings", {}).get(
        "negative_result_adoption"
    )
    if not isinstance(primary_adoption, Mapping):
        raise IBR1AssemblyContractError(
            "IBR1 PRIMARY has no negative-result adoption binding"
        )
    if _resolve_bound_path(
        root,
        primary_adoption.get("path"),
        "PRIMARY negative-result adoption",
    ) != adoption_path:
        raise IBR1AssemblyContractError(
            "IBR1 PRIMARY binds a different negative-result adoption path"
        )
    adoption_sha = _sha256_file(adoption_path)
    if adoption_sha != primary_adoption.get("sha256"):
        raise IBR1AssemblyContractError(
            "IBR1 negative-result adoption SHA differs from PRIMARY"
        )
    adoption = _load_json(adoption_path, "IBR1 negative-result adoption")
    adoption_payload = dict(adoption)
    adoption_payload_sha = adoption_payload.pop("receipt_payload_sha256", None)
    if (
        adoption.get("analysis_class") != "ibr1_negative_result_adoption"
        or adoption.get("family_id") != IBR1_FAMILY_ID
        or adoption.get("internal_test") != "sealed"
        or adoption.get("internal_test_opened") is not False
        or adoption_payload_sha != canonical_json_sha256(adoption_payload)
        or adoption_payload_sha != primary_adoption.get("payload_sha256")
    ):
        raise IBR1AssemblyContractError(
            "IBR1 negative-result adoption identity/seal is invalid"
        )
    adopted_parent = adoption.get("adopted_parent")
    if not isinstance(adopted_parent, Mapping):
        raise IBR1AssemblyContractError("negative-result adoption has no parent")
    if (
        adopted_parent.get("decision") != "FAIL_STOP"
        or adopted_parent.get("valid_input") is not True
        or adopted_parent.get("engineering_failure") is not False
        or adopted_parent.get("scientific_negative_result") is not True
    ):
        raise IBR1AssemblyContractError("adopted F2 result is not a valid seal")

    seal_path = _resolve_bound_path(
        root, adopted_parent.get("path"), "adopted F2 negative seal"
    )
    seal_sha = _sha256_file(seal_path)
    if seal_sha != adopted_parent.get("sha256"):
        raise IBR1AssemblyContractError(
            "adopted F2 negative seal SHA differs from its frozen binding"
        )
    seal = _load_json(seal_path, "F2 negative-result seal")
    run = seal.get("run")
    combined = seal.get("gate_outcomes", {}).get("combined")
    if (
        seal.get("analysis_class")
        != "f2_authoritative_smoke_negative_result_seal"
        or seal.get("internal_test") != "sealed"
        or seal.get("internal_test_opened") is not False
        or not isinstance(run, Mapping)
        or run.get("valid_input") is not True
        or run.get("engineering_failure") is not False
        or run.get("scientific_negative_result") is not True
        or not isinstance(combined, Mapping)
        or combined.get("status") != "FAIL"
        or combined.get("decision") != "STOP"
        or combined.get("formal_training_authorized") is not False
    ):
        raise IBR1AssemblyContractError("F2 negative-result seal content is invalid")

    summary_binding = seal.get("evidence", {}).get("smoke_summary")
    if not isinstance(summary_binding, Mapping):
        raise IBR1AssemblyContractError("F2 seal has no smoke-summary binding")
    summary_path = _resolve_bound_path(
        root, summary_binding.get("path"), "F2 smoke summary"
    )
    summary_sha = _sha256_file(summary_path)
    if summary_sha != summary_binding.get("sha256"):
        raise IBR1AssemblyContractError(
            "F2 smoke summary SHA differs from the negative seal binding"
        )
    summary = _load_json(summary_path, "F2 smoke summary")
    init_sha = _valid_sha256(
        summary.get("checkpoint_init_sha256"), "F2 checkpoint init SHA"
    )
    update0 = summary.get("checkpoints", {}).get("update0")
    if (
        summary.get("analysis_class") != "f2_production_smoke_summary"
        or summary.get("seed") != 0
        or summary.get("internal_test") != "sealed"
        or summary.get("internal_test_opened") is not False
        or summary.get("passed") is not False
        or summary.get("formal_training_authorized") is not False
        or not isinstance(update0, Mapping)
    ):
        raise IBR1AssemblyContractError("F2 smoke summary identity is invalid")
    for engine_arm in (S_CTRL, S_SELF):
        arm_evidence = update0.get(engine_arm)
        if (
            not isinstance(arm_evidence, Mapping)
            or arm_evidence.get("state_sha256") != init_sha
        ):
            raise IBR1AssemblyContractError(
                f"F2 update-0 {engine_arm} state does not match the init SHA"
            )

    return F2SealedInitEvidence(
        primary_path=IBR1_PRIMARY_RELATIVE.as_posix(),
        primary_sha256=_sha256_file(primary_path),
        negative_adoption_path=F2_NEGATIVE_ADOPTION_RELATIVE.as_posix(),
        negative_adoption_sha256=adoption_sha,
        negative_seal_path=str(adopted_parent["path"]),
        negative_seal_sha256=seal_sha,
        smoke_summary_path=str(summary_binding["path"]),
        smoke_summary_sha256=summary_sha,
        seed=0,
        checkpoint_init_sha256=init_sha,
    )


def _validate_aux_coefficients(
    coefficients: Mapping[str, float],
) -> dict[str, float]:
    if not isinstance(coefficients, Mapping):
        raise IBR1AssemblyContractError("aux_coefficients must be a mapping")
    if set(coefficients) != set(IBR1_AUX_COMPONENTS):
        raise IBR1AssemblyContractError(
            "IBR1 aux_coefficients must contain exactly "
            f"{list(IBR1_AUX_COMPONENTS)!r}"
        )
    frozen: dict[str, float] = {}
    for name in IBR1_AUX_COMPONENTS:
        value = coefficients[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise IBR1AssemblyContractError(f"aux coefficient {name} is invalid")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0.0:
            raise IBR1AssemblyContractError(
                f"aux coefficient {name} must be finite and nonnegative"
            )
        frozen[name] = numeric
    return frozen


def _validate_base(base: nn.Module) -> None:
    if not isinstance(base, nn.Module):
        raise IBR1AssemblyContractError("base must be an nn.Module")
    proj = getattr(base, "proj", None)
    if not isinstance(proj, nn.Module) or not any(True for _ in proj.parameters()):
        raise IBR1AssemblyContractError(
            "base must expose a nonempty official proj module"
        )
    try:
        dimension = int(base.D)
    except (AttributeError, TypeError, ValueError) as exc:
        raise IBR1AssemblyContractError("base must expose integer D") from exc
    if dimension <= 0:
        raise IBR1AssemblyContractError("base.D must be positive")


def _freeze_base_except_proj(base: nn.Module) -> None:
    proj_ids = {id(parameter) for parameter in base.proj.parameters()}
    for parameter in base.parameters():
        parameter.requires_grad_(id(parameter) in proj_ids)


def build_ibr1_package(
    base: nn.Module,
    *,
    device: torch.device | str,
    aux_coefficients: Mapping[str, float] = IBR1_FROZEN_AUX_COEFFICIENTS,
) -> F2ArmModules:
    """Build one independent SA-Hstar arm with the IBR1 action model."""

    _validate_base(base)
    coefficients = _validate_aux_coefficients(aux_coefficients)
    target_device = torch.device(device)
    base = base.to(target_device)
    adapter = OpenTrackVLAF2ObservationAdapter(base).to(target_device)
    model = IBR1AP2Model(
        d_model=int(base.D), method_dims=adapter.method_dims
    ).to(target_device)
    _freeze_base_except_proj(base)
    if adapter.base is not base:
        raise IBR1AssemblyContractError("adapter/base aliasing is broken")
    return F2ArmModules(
        package=IBR1_PACKAGE,
        base=base,
        adapter=adapter,
        model=model,
        aux_coefficients=coefficients,
    )


def _ibr1_parameter_receipt(
    receipt: Mapping[str, Any], *, engine_arm: str
) -> dict[str, Any]:
    family_arm = ENGINE_TO_FAMILY_ARM[engine_arm]
    return {
        **dict(receipt),
        "analysis_class": "ibr1_arm_optimizer_parameter_receipt",
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "model_class": "ibr1_experiment.model.IBR1AP2Model",
        "package": IBR1_PACKAGE,
        "engine_arm": engine_arm,
        "family_arm": family_arm,
    }


@dataclass(frozen=True)
class IBR1PairedArms:
    """Runner-ready paired arms plus explicit IBR1/F2 identity evidence."""

    family_id: str
    architecture_lock: str
    package: str
    seed: int
    device: str
    checkpoint_init_sha256: str
    parent_f2_evidence: F2SealedInitEvidence
    arm_mapping: Mapping[str, str]
    arms: Mapping[str, ArmAssembly]

    def public_arms(self) -> dict[str, ArmAssembly]:
        return {
            self.arm_mapping[engine_arm]: assembly
            for engine_arm, assembly in self.arms.items()
        }

    def identity_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": "ibr1_paired_arm_initialization_receipt",
            "family_id": self.family_id,
            "architecture_lock": self.architecture_lock,
            "package": self.package,
            "seed": self.seed,
            "device": self.device,
            "checkpoint_init_sha256": self.checkpoint_init_sha256,
            "parent_f2_init_evidence": self.parent_f2_evidence.to_dict(),
            "arm_mapping_to_engine": {
                family: engine for engine, family in self.arm_mapping.items()
            },
            "engine_arms": list(self.arms),
            "family_arms": [self.arm_mapping[arm] for arm in self.arms],
            "parameters_shared_between_arms": False,
            "optimizers_shared_between_arms": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }


def _parameter_ids(assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for _name, parameter in assembly.modules.named_full_parameters()
    }


def _optimizer_parameter_ids(assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for group in assembly.optimizer.param_groups
        for parameter in group["params"]
    }


def build_ibr1_paired_arms(
    base: nn.Module,
    *,
    seed: int,
    device: torch.device | str,
    contract: OptimizerContract,
    parent_f2_evidence: F2SealedInitEvidence,
    aux_coefficients: Mapping[str, float] = IBR1_FROZEN_AUX_COEFFICIENTS,
) -> IBR1PairedArms:
    """Build CTRL once, deep-copy SELF, and prove identity/disjointness."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise IBR1AssemblyContractError("seed must be a nonnegative integer")
    if not isinstance(contract, OptimizerContract):
        raise IBR1AssemblyContractError("contract must be an OptimizerContract")
    if not isinstance(parent_f2_evidence, F2SealedInitEvidence):
        raise IBR1AssemblyContractError(
            "parent_f2_evidence must be verified F2SealedInitEvidence"
        )
    if seed != parent_f2_evidence.seed:
        raise IBR1AssemblyContractError(
            "IBR1 seed differs from the sealed F2 comparator seed"
        )
    validated_coefficients = _validate_aux_coefficients(aux_coefficients)
    if validated_coefficients != dict(IBR1_FROZEN_AUX_COEFFICIENTS):
        raise IBR1AssemblyContractError(
            "paired smoke aux coefficients differ from the preregistered "
            "IBR1 frozen values"
        )

    target_device = torch.device(device)
    torch.manual_seed(seed)
    ctrl_modules = build_ibr1_package(
        base,
        device=target_device,
        aux_coefficients=validated_coefficients,
    )
    self_modules = copy.deepcopy(ctrl_modules)
    if self_modules.adapter.base is not self_modules.base:
        raise IBR1AssemblyContractError(
            "deep copy broke adapter/base aliasing inside SELF"
        )
    ctrl_sha = checkpoint_init_sha256(ctrl_modules.full_state_dict())
    self_sha = checkpoint_init_sha256(self_modules.full_state_dict())
    if ctrl_sha != self_sha:
        raise IBR1AssemblyContractError(
            "IBR1 paired arms are not bit-identical at initialization"
        )
    if ctrl_sha != parent_f2_evidence.checkpoint_init_sha256:
        raise IBR1AssemblyContractError(
            "IBR1 initialization differs from the sealed seed-0 F2 comparator"
        )

    arms: dict[str, ArmAssembly] = {}
    for engine_arm, modules in (
        (S_CTRL, ctrl_modules),
        (S_SELF, self_modules),
    ):
        optimizer, f2_receipt = build_arm_optimizer(modules, contract)
        arms[engine_arm] = ArmAssembly(
            modules=modules,
            optimizer=optimizer,
            parameter_receipt=_ibr1_parameter_receipt(
                f2_receipt, engine_arm=engine_arm
            ),
        )

    ctrl = arms[S_CTRL]
    self_arm = arms[S_SELF]
    if ctrl.optimizer is self_arm.optimizer:
        raise IBR1AssemblyContractError("paired arms share an optimizer object")
    if _parameter_ids(ctrl) & _parameter_ids(self_arm):
        raise IBR1AssemblyContractError("paired arms share parameter objects")
    if _optimizer_parameter_ids(ctrl) & _optimizer_parameter_ids(self_arm):
        raise IBR1AssemblyContractError(
            "paired optimizers reference shared parameter objects"
        )
    for assembly in arms.values():
        if _optimizer_parameter_ids(assembly) != {
            id(parameter) for parameter in assembly.modules.trainable_parameters()
        }:
            raise IBR1AssemblyContractError(
                "optimizer coverage differs from the arm trainable parameters"
            )

    return IBR1PairedArms(
        family_id=IBR1_FAMILY_ID,
        architecture_lock=IBR1_ARCHITECTURE_LOCK,
        package=IBR1_PACKAGE,
        seed=seed,
        device=str(target_device),
        checkpoint_init_sha256=ctrl_sha,
        parent_f2_evidence=parent_f2_evidence,
        arm_mapping=dict(ENGINE_TO_FAMILY_ARM),
        arms=arms,
    )


__all__ = [
    "ENGINE_TO_FAMILY_ARM",
    "F2SealedInitEvidence",
    "FAMILY_TO_ENGINE_ARM",
    "IBR1AssemblyContractError",
    "IBR1PairedArms",
    "IBR1_AUX_COMPONENTS",
    "IBR1_CTRL",
    "IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS",
    "IBR1_FROZEN_AUX_COEFFICIENTS",
    "IBR1_PACKAGE",
    "IBR1_SELF",
    "build_ibr1_package",
    "build_ibr1_paired_arms",
    "read_sealed_f2_init_evidence",
]
