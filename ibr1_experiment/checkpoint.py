"""Family-strict immutable checkpoints for the IBR1 smoke lifecycle.

The archived F2 checkpoint format is deliberately not accepted here.  IBR1
checkpoints are update-boundary snapshots for evaluation and evidence only;
they omit optimizer and RNG state and therefore do not provide a resume path.
Both the tensor payload and its exclusive-write JSON sidecar bind the IBR1
family identity, source files, public/engine arm mapping, and the final IBR1
assembly receipt.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch

from f2_experiment.assembly_model import (
    ArmAssembly,
    OptimizerContract,
    build_arm_optimizer,
)
from f2_experiment.opentrack_adapter import OpenTrackVLAF2ObservationAdapter
from f2_experiment.runner import checkpoint_init_sha256

from .assembly_model import (
    ENGINE_TO_FAMILY_ARM,
    F2SealedInitEvidence,
    IBR1AssemblyContractError,
    IBR1PairedArms,
    IBR1_PACKAGE,
    read_sealed_f2_init_evidence,
)
from .authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    IBR1AuthorityError,
    verify_assembly_receipt,
)
from .model import IBR1AP2Model, IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_PAYLOAD_CLASS = "ibr1_arm_checkpoint_payload"
CHECKPOINT_SIDECAR_CLASS = "ibr1_arm_checkpoint_sidecar"
IBR1_MODEL_CLASS = "ibr1_experiment.model.IBR1AP2Model"
IBR1_ADAPTER_CLASS = (
    "f2_experiment.opentrack_adapter.OpenTrackVLAF2ObservationAdapter"
)
ALLOWED_CHECKPOINT_UPDATES = (0, 128)
MODEL_SOURCE_RELATIVE = "ibr1_experiment/model.py"
CHECKPOINT_SOURCE_RELATIVE = "ibr1_experiment/checkpoint.py"
SOURCE_RELATIVES = (MODEL_SOURCE_RELATIVE, CHECKPOINT_SOURCE_RELATIVE)

SNAPSHOT_POLICY: Mapping[str, Any] = {
    "purpose": "immutable_update_boundary_snapshot",
    "allowed_u_pre": [0, 128],
    "mid_run_resume": "forbidden",
    "optimizer_state_included": False,
    "rng_state_included": False,
}

_IDENTITY_KEYS = {
    "schema_version",
    "family_id",
    "architecture_lock",
    "model_class",
    "adapter_class",
    "model_source_sha256",
    "source_sha256",
    "family_arm",
    "engine_arm",
    "u_pre",
    "checkpoint_tensor_sha256",
    "final_assembly_receipt",
    "state_schema",
    "snapshot_policy",
    "internal_test",
    "internal_test_opened",
}
_PAYLOAD_KEYS = _IDENTITY_KEYS | {
    "analysis_class",
    "adapter_state",
    "model_state",
}
_SIDECAR_KEYS = _IDENTITY_KEYS | {
    "analysis_class",
    "checkpoint_file",
    "checkpoint_file_sha256",
}


class IBR1CheckpointContractError(IBR1AssemblyContractError):
    """Raised when an IBR1 checkpoint cannot be trusted or loaded."""


@dataclass(frozen=True)
class IBR1EvaluationSnapshot:
    """Verified CPU-only state for constructing a separate eval model.

    This object intentionally carries no optimizer or RNG state and is never
    hydrated into the live paired-smoke arms by the checkpoint loader.
    """

    checkpoint_path: str
    checkpoint_file_sha256: str
    sidecar_path: str
    sidecar_sha256: str
    checkpoint_tensor_sha256: str
    family_arm: str
    engine_arm: str
    u_pre: int
    final_assembly_receipt: Mapping[str, str]
    _adapter_state: Mapping[str, torch.Tensor] = field(repr=False)
    _model_state: Mapping[str, torch.Tensor] = field(repr=False)
    evaluation_only: bool = True
    resume_supported: bool = False
    optimizer_state_included: bool = False
    rng_state_included: bool = False

    def materialize_adapter_state(self) -> dict[str, torch.Tensor]:
        """Return a fresh mutable copy for a separately built eval adapter."""

        return _clone_state(self._adapter_state, "adapter")

    def materialize_model_state(self) -> dict[str, torch.Tensor]:
        """Return a fresh mutable copy for a separately built eval model."""

        return _clone_state(self._model_state, "model")

    @property
    def adapter_state(self) -> Mapping[str, torch.Tensor]:
        """Read-only mapping whose tensors are fresh copies on every access."""

        return MappingProxyType(self.materialize_adapter_state())

    @property
    def model_state(self) -> Mapping[str, torch.Tensor]:
        """Read-only mapping whose tensors are fresh copies on every access."""

        return MappingProxyType(self.materialize_model_state())


@dataclass(frozen=True)
class _CreatedFile:
    path: Path
    device: int
    inode: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1CheckpointContractError(message)


def _valid_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _sha256_file(path: Path, label: str) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise IBR1CheckpointContractError(
            f"cannot read {label}: {path}"
        ) from exc


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IBR1CheckpointContractError(
            f"{label} is unreadable: {path}"
        ) from exc
    _require(isinstance(document, dict), f"{label} must be a JSON object")
    return document


def _created_file(handle: Any, path: Path) -> _CreatedFile:
    stat = os.fstat(handle.fileno())
    return _CreatedFile(path=path, device=stat.st_dev, inode=stat.st_ino)


def _cleanup_created_file(created: _CreatedFile, label: str) -> None:
    try:
        if not created.path.exists():
            return
        current = created.path.stat()
        _require(
            (current.st_dev, current.st_ino)
            == (created.device, created.inode),
            f"refusing to clean up a replaced {label}: {created.path}",
        )
        created.path.unlink()
    except OSError as exc:
        raise IBR1CheckpointContractError(
            f"cannot clean up the partially created {label}: {created.path}"
        ) from exc


def _write_sidecar_bytes(handle: Any, encoded: bytes) -> None:
    """Narrow write seam used by the partial-write rollback test."""

    handle.write(encoded)


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    created: _CreatedFile | None = None
    try:
        handle = path.open("xb")
        created = _created_file(handle, path)
        with handle:
            _write_sidecar_bytes(handle, encoded)
    except FileExistsError as exc:
        raise IBR1CheckpointContractError(
            f"refusing to overwrite an existing checkpoint sidecar: {path}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - rollback every partial write
        if created is not None:
            _cleanup_created_file(created, "checkpoint sidecar")
        if isinstance(exc, IBR1CheckpointContractError):
            raise
        raise IBR1CheckpointContractError(
            f"cannot write IBR1 checkpoint sidecar: {path}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected, key=repr)
    _require(
        not missing and not unexpected,
        f"{label} keys are not exact; missing={missing!r}, "
        f"unexpected={unexpected!r}",
    )


def _qualified_class_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _validate_u_pre(value: Any, label: str = "u_pre") -> int:
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value in ALLOWED_CHECKPOINT_UPDATES,
        f"{label} must be one of {ALLOWED_CHECKPOINT_UPDATES!r}; "
        "mid-run checkpoints/resume are forbidden",
    )
    return value


def _validate_arm_pair(family_arm: Any, engine_arm: Any) -> tuple[str, str]:
    _require(
        isinstance(engine_arm, str) and engine_arm in ENGINE_TO_FAMILY_ARM,
        f"unknown IBR1 engine arm {engine_arm!r}",
    )
    expected_family = ENGINE_TO_FAMILY_ARM[engine_arm]
    _require(
        family_arm == expected_family,
        f"IBR1 family arm {family_arm!r} does not map to engine arm "
        f"{engine_arm!r}; expected {expected_family!r}",
    )
    return expected_family, engine_arm


def _validate_arm_assembly(value: Any) -> ArmAssembly:
    _require(isinstance(value, ArmAssembly), "arm_assembly must be an ArmAssembly")
    modules = value.modules
    _require(
        type(modules.model) is IBR1AP2Model,
        "checkpoint model must be the exact IBR1AP2Model class",
    )
    _require(
        type(modules.adapter) is OpenTrackVLAF2ObservationAdapter,
        "checkpoint adapter must be the exact frozen OpenTrackVLA adapter",
    )
    _require(
        _qualified_class_name(modules.model) == IBR1_MODEL_CLASS,
        "checkpoint model class identity drifted",
    )
    _require(
        _qualified_class_name(modules.adapter) == IBR1_ADAPTER_CLASS,
        "checkpoint adapter class identity drifted",
    )
    _require(
        modules.adapter.base is modules.base,
        "checkpoint adapter/base aliasing is broken",
    )
    return value


def _validate_parameter_receipt_arm(
    arm_assembly: ArmAssembly,
    *,
    family_arm: str,
    engine_arm: str,
) -> None:
    receipt = arm_assembly.parameter_receipt
    _require(
        isinstance(receipt, Mapping),
        "arm assembly parameter receipt must be a mapping",
    )
    _require(
        receipt.get("analysis_class")
        == "ibr1_arm_optimizer_parameter_receipt"
        and receipt.get("family_id") == IBR1_FAMILY_ID
        and receipt.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK,
        "arm assembly parameter receipt is not IBR1 family authority",
    )
    _require(
        receipt.get("model_class") == IBR1_MODEL_CLASS,
        "arm assembly parameter receipt has a different model class",
    )
    _require(
        receipt.get("family_arm") == family_arm
        and receipt.get("engine_arm") == engine_arm,
        "arm assembly parameter receipt belongs to a different IBR1 arm",
    )


def _parameter_ids(arm_assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for _name, parameter in arm_assembly.modules.named_full_parameters()
    }


def _optimizer_parameter_ids(arm_assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for group in arm_assembly.optimizer.param_groups
        for parameter in group["params"]
    }


def _expected_ibr1_parameter_receipt(
    arm_assembly: ArmAssembly,
    *,
    family_arm: str,
    engine_arm: str,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    expected_optimizer, f2_receipt = build_arm_optimizer(
        arm_assembly.modules,
        OptimizerContract(),
    )
    expected_receipt = {
        **dict(f2_receipt),
        "analysis_class": "ibr1_arm_optimizer_parameter_receipt",
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "model_class": IBR1_MODEL_CLASS,
        "package": IBR1_PACKAGE,
        "engine_arm": engine_arm,
        "family_arm": family_arm,
    }
    return expected_optimizer, expected_receipt


def _validate_live_optimizer_contract(
    arm_assembly: ArmAssembly,
    *,
    family_arm: str,
    engine_arm: str,
) -> None:
    expected_optimizer, expected_receipt = _expected_ibr1_parameter_receipt(
        arm_assembly,
        family_arm=family_arm,
        engine_arm=engine_arm,
    )
    _require(
        dict(arm_assembly.parameter_receipt) == expected_receipt,
        f"{family_arm} stored parameter receipt differs from the mechanically "
        "rebuilt frozen receipt",
    )
    actual = arm_assembly.optimizer
    _require(
        type(actual) is torch.optim.AdamW,
        f"{family_arm} optimizer must be the exact AdamW class",
    )
    _require(
        len(actual.param_groups) == len(expected_optimizer.param_groups),
        f"{family_arm} optimizer group count drifted",
    )
    contract = OptimizerContract()
    for position, (observed, expected) in enumerate(
        zip(actual.param_groups, expected_optimizer.param_groups)
    ):
        _require(
            observed.get("name") == expected.get("name"),
            f"{family_arm} optimizer group order/name drifted at {position}",
        )
        _require(
            [id(parameter) for parameter in observed["params"]]
            == [id(parameter) for parameter in expected["params"]],
            f"{family_arm} optimizer parameter IDs/order drifted at {position}",
        )
        _require(
            observed.get("lr") == expected.get("lr")
            and observed.get("weight_decay") == expected.get("weight_decay"),
            f"{family_arm} optimizer lr/weight_decay drifted at {position}",
        )
        _require(
            tuple(observed.get("betas", ())) == contract.betas
            and observed.get("eps") == contract.eps,
            f"{family_arm} optimizer betas/eps drifted at {position}",
        )
    _require(
        tuple(actual.defaults.get("betas", ())) == contract.betas
        and actual.defaults.get("eps") == contract.eps,
        f"{family_arm} optimizer defaults drifted",
    )


def _validate_paired_device(paired_arms: IBR1PairedArms) -> None:
    try:
        declared = torch.device(paired_arms.device)
    except (TypeError, RuntimeError) as exc:
        raise IBR1CheckpointContractError(
            "paired-arm declared device is invalid"
        ) from exc
    _require(
        paired_arms.device == str(declared),
        "paired-arm device is not canonically declared",
    )
    for engine_arm, arm_assembly in paired_arms.arms.items():
        for section, module in (
            ("adapter", arm_assembly.modules.adapter),
            ("model", arm_assembly.modules.model),
        ):
            for name, tensor in module.state_dict().items():
                _require(
                    tensor.device == declared,
                    f"{engine_arm} {section} state {name!r} differs from "
                    "the paired-arm declared device",
                )


def _validate_paired_arm_identity(
    paired_arms: Any,
    arm_assembly: Any,
    engine_arm: Any,
    *,
    project_root: Path,
) -> tuple[
    IBR1PairedArms,
    ArmAssembly,
    str,
    str,
    F2SealedInitEvidence,
]:
    _require(
        isinstance(paired_arms, IBR1PairedArms),
        "paired_arms must be an IBR1PairedArms authority object",
    )
    _require(
        isinstance(engine_arm, str) and engine_arm in ENGINE_TO_FAMILY_ARM,
        f"unknown IBR1 engine arm {engine_arm!r}",
    )
    family_arm, validated_engine = _validate_arm_pair(
        ENGINE_TO_FAMILY_ARM[engine_arm], engine_arm
    )
    _require(
        paired_arms.family_id == IBR1_FAMILY_ID
        and paired_arms.architecture_lock == IBR1_ARCHITECTURE_LOCK
        and paired_arms.package == IBR1_PACKAGE,
        "paired-arm family/package identity drifted",
    )
    _require(
        dict(paired_arms.arm_mapping) == dict(ENGINE_TO_FAMILY_ARM),
        "paired-arm public/engine mapping drifted",
    )
    _require(
        set(paired_arms.arms) == set(ENGINE_TO_FAMILY_ARM),
        "paired-arm engine coverage is not exact",
    )
    try:
        live_parent = read_sealed_f2_init_evidence(project_root)
    except IBR1AssemblyContractError as exc:
        raise IBR1CheckpointContractError(
            "paired-arm parent F2 evidence failed live seal-chain verification"
        ) from exc
    _require(
        isinstance(paired_arms.parent_f2_evidence, F2SealedInitEvidence)
        and paired_arms.parent_f2_evidence == live_parent,
        "paired-arm parent F2 evidence differs from the live sealed chain",
    )
    _require(
        paired_arms.seed == live_parent.seed
        and paired_arms.checkpoint_init_sha256
        == live_parent.checkpoint_init_sha256,
        "paired-arm seed/init binding differs from live F2 evidence",
    )
    _valid_sha256(
        paired_arms.checkpoint_init_sha256,
        "paired-arm checkpoint init SHA",
    )
    selected = paired_arms.arms[validated_engine]
    _require(
        arm_assembly is selected,
        "arm_assembly is not the exact paired_arms object for the engine arm",
    )

    validated_arms: dict[str, ArmAssembly] = {}
    for current_engine, current_family in ENGINE_TO_FAMILY_ARM.items():
        current = _validate_arm_assembly(paired_arms.arms[current_engine])
        _validate_parameter_receipt_arm(
            current,
            family_arm=current_family,
            engine_arm=current_engine,
        )
        _validate_live_optimizer_contract(
            current,
            family_arm=current_family,
            engine_arm=current_engine,
        )
        optimizer_ids = _optimizer_parameter_ids(current)
        trainable_ids = {
            id(parameter)
            for parameter in current.modules.trainable_parameters()
        }
        _require(
            optimizer_ids == trainable_ids,
            f"{current_family} optimizer coverage drifted",
        )
        validated_arms[current_engine] = current

    engine_order = tuple(ENGINE_TO_FAMILY_ARM)
    left = validated_arms[engine_order[0]]
    right = validated_arms[engine_order[1]]
    _require(left.optimizer is not right.optimizer, "paired arms share optimizer")
    _require(
        not (_parameter_ids(left) & _parameter_ids(right)),
        "paired arms share parameter objects",
    )
    _require(
        not (_optimizer_parameter_ids(left) & _optimizer_parameter_ids(right)),
        "paired optimizers share parameter objects",
    )
    _validate_paired_device(paired_arms)
    return paired_arms, selected, family_arm, validated_engine, live_parent


def _validate_tensor(name: str, tensor: Any, section: str) -> torch.Tensor:
    _require(isinstance(name, str) and bool(name), f"{section} has an invalid key")
    _require(
        isinstance(tensor, torch.Tensor),
        f"{section} state {name!r} is not a tensor",
    )
    _require(
        tensor.layout == torch.strided
        and not tensor.is_quantized
        and tensor.device.type != "meta",
        f"{section} state {name!r} uses unsupported storage",
    )
    if tensor.is_floating_point() or tensor.is_complex():
        _require(
            not bool((~torch.isfinite(tensor)).detach().any().cpu().item()),
            f"{section} state {name!r} is nonfinite",
        )
    return tensor


def _sorted_state_names(
    state: Mapping[str, torch.Tensor], section: str
) -> list[str]:
    names = list(state)
    _require(
        all(isinstance(name, str) and bool(name) for name in names),
        f"{section} state keys must be nonempty strings",
    )
    return sorted(names)


def _clone_state(
    state: Mapping[str, torch.Tensor], section: str
) -> dict[str, torch.Tensor]:
    _require(isinstance(state, Mapping) and bool(state), f"{section} state is empty")
    result: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        validated = _validate_tensor(name, tensor, section)
        result[name] = validated.detach().cpu().contiguous().clone()
    return result


def _state_schema(
    adapter_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for section, state in (
        ("adapter", adapter_state),
        ("model", model_state),
    ):
        _require(
            isinstance(state, Mapping) and bool(state),
            f"{section} state must be a nonempty mapping",
        )
        section_schema: dict[str, dict[str, Any]] = {}
        for name in _sorted_state_names(state, section):
            tensor = _validate_tensor(name, state[name], section)
            section_schema[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        result[section] = section_schema
    return result


def checkpoint_tensor_sha256(
    *,
    adapter_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> str:
    """Hash checkpoint tensors in the inherited F2 full-state domain.

    IBR1 persists adapter/model state in two payload sections, while the
    sealed comparator and paired runner hash one flat state whose names are
    prefixed with ``adapter.`` and ``model.``.  Normalize the sections back to
    that frozen naming contract before hashing so update-0 sidecars, the
    sealed initialization, lifecycle checks, and count/gate receipts all bind
    the same tensor identity.
    """

    full_state: dict[str, torch.Tensor] = {}
    for section, state in (
        ("adapter", adapter_state),
        ("model", model_state),
    ):
        _require(
            isinstance(state, Mapping) and bool(state),
            f"{section} state must be a nonempty mapping",
        )
        for name in _sorted_state_names(state, section):
            tensor = _validate_tensor(name, state[name], section)
            qualified_name = f"{section}.{name}"
            _require(
                qualified_name not in full_state,
                f"checkpoint full-state key collision: {qualified_name!r}",
            )
            full_state[qualified_name] = tensor
    return checkpoint_init_sha256(full_state)


def _source_binding(project_root: Path) -> dict[str, str]:
    result = {
        relative: _sha256_file(project_root / relative, "IBR1 source")
        for relative in SOURCE_RELATIVES
    }
    return dict(sorted(result.items()))


def _bound_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _verify_final_assembly_receipt(
    *,
    project_root: Path,
    receipt_path: str | Path,
    expected_sha256: str,
) -> dict[str, str]:
    expected_sha = _valid_sha256(
        expected_sha256, "final assembly receipt SHA"
    )
    resolved = Path(receipt_path).expanduser().resolve()
    actual_sha = _sha256_file(resolved, "final IBR1 assembly receipt")
    _require(
        actual_sha == expected_sha,
        "final IBR1 assembly receipt bytes differ from the expected SHA",
    )
    try:
        document = verify_assembly_receipt(
            project_root,
            resolved,
            required_phase=ASSEMBLY_PHASE_FINAL,
        )
    except IBR1AuthorityError as exc:
        raise IBR1CheckpointContractError(
            "final IBR1 assembly receipt failed live authority verification"
        ) from exc
    _require(
        isinstance(document, Mapping),
        "final assembly verifier did not return a receipt mapping",
    )
    _require(
        document.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS,
        "checkpoint authority is not an IBR1 assembly receipt",
    )
    _require(
        document.get("family_id") == IBR1_FAMILY_ID,
        "final assembly receipt belongs to a different family",
    )
    _require(
        document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK,
        "final assembly receipt has a different architecture lock",
    )
    _require(
        document.get("phase") == ASSEMBLY_PHASE_FINAL,
        "checkpoint must bind the final, not bootstrap, IBR1 assembly receipt",
    )
    _require(
        isinstance(document.get("lambda_freeze_binding"), Mapping)
        and bool(document["lambda_freeze_binding"]),
        "final assembly receipt has no lambda adoption freeze binding",
    )
    _require(
        document.get("internal_test") == "sealed"
        and document.get("internal_test_opened") is False,
        "final assembly receipt does not preserve the internal-test seal",
    )
    return {
        "path": _bound_path(project_root, resolved),
        "sha256": actual_sha,
        "receipt_payload_sha256": _valid_sha256(
            document.get("receipt_payload_sha256"),
            "final assembly receipt payload SHA",
        ),
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }


def _target_state_schema(arm_assembly: ArmAssembly) -> dict[str, Any]:
    modules = arm_assembly.modules
    return _state_schema(
        modules.adapter.state_dict(),
        modules.model.state_dict(),
    )


def _identity_metadata(
    *,
    source_sha256: Mapping[str, str],
    family_arm: str,
    engine_arm: str,
    u_pre: int,
    tensor_sha256: str,
    final_receipt_binding: Mapping[str, str],
    state_schema: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "model_class": IBR1_MODEL_CLASS,
        "adapter_class": IBR1_ADAPTER_CLASS,
        "model_source_sha256": source_sha256[MODEL_SOURCE_RELATIVE],
        "source_sha256": dict(source_sha256),
        "family_arm": family_arm,
        "engine_arm": engine_arm,
        "u_pre": u_pre,
        "checkpoint_tensor_sha256": tensor_sha256,
        "final_assembly_receipt": dict(final_receipt_binding),
        "state_schema": dict(state_schema),
        "snapshot_policy": dict(SNAPSHOT_POLICY),
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


def save_ibr1_arm_checkpoint(
    path: str | Path,
    *,
    paired_arms: IBR1PairedArms,
    arm_assembly: ArmAssembly,
    engine_arm: str,
    u_pre: int,
    final_assembly_receipt_path: str | Path,
    final_assembly_receipt_sha256: str,
    project_root: str | Path,
) -> dict[str, Any]:
    """Exclusively save one immutable IBR1 update-boundary snapshot."""

    root = Path(project_root).expanduser().resolve()
    paired, assembly, validated_family, validated_engine, live_parent = (
        _validate_paired_arm_identity(
            paired_arms,
            arm_assembly,
            engine_arm,
            project_root=root,
        )
    )
    validated_u_pre = _validate_u_pre(u_pre)
    if validated_u_pre == 0:
        for current_engine, current in paired.arms.items():
            observed_init_sha = checkpoint_init_sha256(
                current.modules.full_state_dict()
            )
            _require(
                observed_init_sha == live_parent.checkpoint_init_sha256,
                f"update-0 {current_engine} state differs from the live "
                "sealed F2 init SHA",
            )
    source_sha = _source_binding(root)
    receipt_binding = _verify_final_assembly_receipt(
        project_root=root,
        receipt_path=final_assembly_receipt_path,
        expected_sha256=final_assembly_receipt_sha256,
    )

    destination = Path(path).expanduser().resolve()
    sidecar_path = destination.with_name(destination.name + ".receipt.json")
    _require(
        not destination.exists(),
        f"refusing to overwrite an existing IBR1 checkpoint: {destination}",
    )
    _require(
        not sidecar_path.exists(),
        f"refusing to overwrite an existing checkpoint sidecar: {sidecar_path}",
    )

    adapter_state = _clone_state(
        assembly.modules.adapter.state_dict(), "adapter"
    )
    model_state = _clone_state(assembly.modules.model.state_dict(), "model")
    schema = _state_schema(adapter_state, model_state)
    tensor_sha = checkpoint_tensor_sha256(
        adapter_state=adapter_state,
        model_state=model_state,
    )
    identity = _identity_metadata(
        source_sha256=source_sha,
        family_arm=validated_family,
        engine_arm=validated_engine,
        u_pre=validated_u_pre,
        tensor_sha256=tensor_sha,
        final_receipt_binding=receipt_binding,
        state_schema=schema,
    )
    payload = {
        **identity,
        "analysis_class": CHECKPOINT_PAYLOAD_CLASS,
        "adapter_state": adapter_state,
        "model_state": model_state,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_created: _CreatedFile | None = None
    try:
        handle = destination.open("xb")
        checkpoint_created = _created_file(handle, destination)
        with handle:
            torch.save(payload, handle)
    except FileExistsError as exc:
        raise IBR1CheckpointContractError(
            f"refusing to overwrite an existing IBR1 checkpoint: {destination}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 - rollback partial torch.save
        if checkpoint_created is not None:
            _cleanup_created_file(checkpoint_created, "IBR1 checkpoint")
        if isinstance(exc, IBR1CheckpointContractError):
            raise
        raise IBR1CheckpointContractError(
            f"cannot write IBR1 checkpoint: {destination}"
        ) from exc

    try:
        checkpoint_file_sha = _sha256_file(destination, "IBR1 checkpoint")
        sidecar = {
            **identity,
            "analysis_class": CHECKPOINT_SIDECAR_CLASS,
            "checkpoint_file": destination.name,
            "checkpoint_file_sha256": checkpoint_file_sha,
        }
        sidecar_sha = _exclusive_write_json(sidecar_path, sidecar)
    except Exception:
        _require(
            checkpoint_created is not None,
            "checkpoint rollback token is unexpectedly absent",
        )
        _cleanup_created_file(checkpoint_created, "IBR1 checkpoint")
        raise
    return {
        "path": str(destination),
        "file_sha256": checkpoint_file_sha,
        "tensor_sha256": tensor_sha,
        "sidecar": str(sidecar_path),
        "sidecar_sha256": sidecar_sha,
        "family_arm": validated_family,
        "engine_arm": validated_engine,
        "u_pre": validated_u_pre,
        "resume_supported": False,
    }


def _validate_static_identity(
    document: Mapping[str, Any],
    *,
    expected_analysis_class: str,
    expected_family_arm: str,
    expected_engine_arm: str,
    expected_u_pre: int,
    expected_source_sha256: Mapping[str, str],
    expected_receipt_binding: Mapping[str, str],
    expected_state_schema: Mapping[str, Any],
    label: str,
) -> None:
    _require(
        document.get("analysis_class") == expected_analysis_class,
        f"{label} is not an IBR1 checkpoint artifact",
    )
    _require(
        document.get("schema_version") == CHECKPOINT_SCHEMA_VERSION,
        f"{label} schema version is unsupported",
    )
    _require(
        document.get("family_id") == IBR1_FAMILY_ID,
        f"{label} belongs to a different family",
    )
    _require(
        document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK,
        f"{label} has a different architecture lock",
    )
    _require(
        document.get("model_class") == IBR1_MODEL_CLASS,
        f"{label} has a non-IBR1 model class",
    )
    _require(
        document.get("adapter_class") == IBR1_ADAPTER_CLASS,
        f"{label} has a different adapter class",
    )
    _require(
        document.get("source_sha256") == dict(expected_source_sha256),
        f"{label} source SHA binding differs from the current IBR1 source",
    )
    _require(
        document.get("model_source_sha256")
        == expected_source_sha256[MODEL_SOURCE_RELATIVE],
        f"{label} IBR1 model source SHA is invalid",
    )
    _require(
        document.get("family_arm") == expected_family_arm
        and document.get("engine_arm") == expected_engine_arm,
        f"{label} belongs to a different IBR1 arm",
    )
    _require(
        document.get("u_pre") == expected_u_pre,
        f"{label} u_pre differs from the required snapshot",
    )
    _valid_sha256(
        document.get("checkpoint_tensor_sha256"),
        f"{label} checkpoint tensor SHA",
    )
    _require(
        document.get("final_assembly_receipt")
        == dict(expected_receipt_binding),
        f"{label} is bound to a different final assembly receipt",
    )
    _require(
        document.get("state_schema") == dict(expected_state_schema),
        f"{label} adapter/model key, shape, or dtype schema differs from "
        "the target IBR1 assembly",
    )
    _require(
        document.get("snapshot_policy") == dict(SNAPSHOT_POLICY),
        f"{label} snapshot/no-resume policy is invalid",
    )
    _require(
        document.get("internal_test") == "sealed"
        and document.get("internal_test_opened") is False,
        f"{label} does not preserve the internal-test seal",
    )


def load_ibr1_arm_checkpoint_verified(
    path: str | Path,
    *,
    paired_arms: IBR1PairedArms,
    expected_arm_assembly: ArmAssembly,
    expected_engine_arm: str,
    expected_u_pre: int,
    expected_checkpoint_file_sha256: str,
    expected_sidecar_sha256: str,
    expected_final_assembly_receipt_path: str | Path,
    expected_final_assembly_receipt_sha256: str,
    project_root: str | Path,
) -> IBR1EvaluationSnapshot:
    """Verify an IBR1 snapshot without mutating either live smoke arm.

    The caller must supply checkpoint-file and sidecar SHAs from an external
    lifecycle manifest (normally the values returned by the exclusive save).
    Every sidecar, source, final-authority, paired-arm, and target-schema check
    runs before ``torch.load``.  The returned state is CPU-only and explicitly
    evaluation-only; optimizer/RNG state is absent and the live model and
    optimizer are untouched.
    """

    root = Path(project_root).expanduser().resolve()
    _paired, assembly, family_arm, engine_arm, live_parent = (
        _validate_paired_arm_identity(
            paired_arms,
            expected_arm_assembly,
            expected_engine_arm,
            project_root=root,
        )
    )
    u_pre = _validate_u_pre(expected_u_pre, "expected_u_pre")
    expected_file_sha = _valid_sha256(
        expected_checkpoint_file_sha256,
        "externally anchored checkpoint file SHA",
    )
    expected_sidecar_sha = _valid_sha256(
        expected_sidecar_sha256,
        "externally anchored checkpoint sidecar SHA",
    )

    destination = Path(path).expanduser().resolve()
    _require(
        destination.is_file(), f"IBR1 checkpoint is missing: {destination}"
    )
    sidecar_path = destination.with_name(destination.name + ".receipt.json")
    observed_sidecar_sha = _sha256_file(
        sidecar_path, "IBR1 checkpoint sidecar"
    )
    _require(
        observed_sidecar_sha == expected_sidecar_sha,
        "IBR1 checkpoint sidecar differs from the external lifecycle anchor",
    )
    checkpoint_file_sha = _sha256_file(destination, "IBR1 checkpoint")
    _require(
        checkpoint_file_sha == expected_file_sha,
        "IBR1 checkpoint file differs from the external lifecycle anchor",
    )

    sidecar = _load_json(sidecar_path, "IBR1 checkpoint sidecar")
    _require_exact_keys(sidecar, _SIDECAR_KEYS, "IBR1 checkpoint sidecar")
    source_sha = _source_binding(root)
    receipt_binding = _verify_final_assembly_receipt(
        project_root=root,
        receipt_path=expected_final_assembly_receipt_path,
        expected_sha256=expected_final_assembly_receipt_sha256,
    )
    target_schema = _target_state_schema(assembly)
    _validate_static_identity(
        sidecar,
        expected_analysis_class=CHECKPOINT_SIDECAR_CLASS,
        expected_family_arm=family_arm,
        expected_engine_arm=engine_arm,
        expected_u_pre=u_pre,
        expected_source_sha256=source_sha,
        expected_receipt_binding=receipt_binding,
        expected_state_schema=target_schema,
        label="IBR1 checkpoint sidecar",
    )
    _require(
        sidecar.get("checkpoint_file") == destination.name,
        "IBR1 checkpoint sidecar names a different checkpoint file",
    )
    _require(
        sidecar.get("checkpoint_file_sha256")
        == checkpoint_file_sha
        == expected_file_sha,
        "IBR1 checkpoint bytes do not match the sidecar/external anchor",
    )

    try:
        payload = torch.load(
            destination,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:  # noqa: BLE001 - all load failures fail closed
        raise IBR1CheckpointContractError(
            f"IBR1 checkpoint cannot be deserialized: {destination}"
        ) from exc
    _require(isinstance(payload, dict), "IBR1 checkpoint payload must be a dict")
    _require_exact_keys(payload, _PAYLOAD_KEYS, "IBR1 checkpoint payload")
    _validate_static_identity(
        payload,
        expected_analysis_class=CHECKPOINT_PAYLOAD_CLASS,
        expected_family_arm=family_arm,
        expected_engine_arm=engine_arm,
        expected_u_pre=u_pre,
        expected_source_sha256=source_sha,
        expected_receipt_binding=receipt_binding,
        expected_state_schema=target_schema,
        label="IBR1 checkpoint payload",
    )

    adapter_state = payload["adapter_state"]
    model_state = payload["model_state"]
    payload_schema = _state_schema(adapter_state, model_state)
    _require(
        payload_schema == target_schema
        and payload_schema == sidecar["state_schema"]
        and payload_schema == payload["state_schema"],
        "IBR1 checkpoint adapter/model key, shape, or dtype mismatch",
    )
    tensor_sha = checkpoint_tensor_sha256(
        adapter_state=adapter_state,
        model_state=model_state,
    )
    _require(
        tensor_sha == sidecar["checkpoint_tensor_sha256"]
        and tensor_sha == payload["checkpoint_tensor_sha256"],
        "IBR1 checkpoint tensor SHA mismatch",
    )
    if u_pre == 0:
        inherited_state = {
            **{
                f"adapter.{name}": tensor
                for name, tensor in adapter_state.items()
            },
            **{
                f"model.{name}": tensor for name, tensor in model_state.items()
            },
        }
        _require(
            checkpoint_init_sha256(inherited_state)
            == live_parent.checkpoint_init_sha256,
            "update-0 checkpoint payload differs from the live sealed F2 init SHA",
        )

    verified_adapter = _clone_state(adapter_state, "adapter")
    verified_model = _clone_state(model_state, "model")
    _require(
        all(tensor.device.type == "cpu" for tensor in verified_adapter.values())
        and all(tensor.device.type == "cpu" for tensor in verified_model.values()),
        "verified evaluation snapshot must be CPU-only",
    )
    return IBR1EvaluationSnapshot(
        checkpoint_path=str(destination),
        checkpoint_file_sha256=checkpoint_file_sha,
        sidecar_path=str(sidecar_path),
        sidecar_sha256=observed_sidecar_sha,
        checkpoint_tensor_sha256=tensor_sha,
        family_arm=family_arm,
        engine_arm=engine_arm,
        u_pre=u_pre,
        final_assembly_receipt=MappingProxyType(dict(receipt_binding)),
        _adapter_state=MappingProxyType(verified_adapter),
        _model_state=MappingProxyType(verified_model),
    )


__all__ = [
    "ALLOWED_CHECKPOINT_UPDATES",
    "CHECKPOINT_PAYLOAD_CLASS",
    "CHECKPOINT_SCHEMA_VERSION",
    "CHECKPOINT_SIDECAR_CLASS",
    "IBR1CheckpointContractError",
    "IBR1EvaluationSnapshot",
    "checkpoint_tensor_sha256",
    "load_ibr1_arm_checkpoint_verified",
    "save_ibr1_arm_checkpoint",
]
