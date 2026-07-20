"""Pure production assembly for the preregistered IBR1 paired smoke.

This module stops at object wiring.  It verifies and loads the frozen support
plane, constructs the two live IBR1 arms, and binds the existing runner
callbacks to the preregistered diagnostics.  It never calls a callback,
performs a forward/backward pass, steps an optimizer, evaluates a row, or
writes a checkpoint.

The production entry point is intentionally narrower than the component
factory used by unit tests: seed 0, ``cuda:0``, the final IBR1 assembly
authority, deterministic CUDA, and the sealed F2 initialization evidence are
mandatory and cannot be overridden by callers.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass, field, replace
import hashlib
from pathlib import Path
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary

import torch
import torch.optim.optimizer as torch_optimizer

from f2_experiment.assembly_data import (
    TokenHashLedger,
    build_runner_rows,
    build_train_token_ledger,
    frozen_cache_roots,
    ordered_support_rows,
    smoke_reset_sets,
)
from f2_experiment.assembly_model import (
    ArmAssembly,
    ArmExecutor,
    EvalRowPredictor,
    F2ArmModules,
    OptimizerContract,
    build_arm_callbacks,
    build_arm_optimizer,
    build_eval_row_predictor,
    load_base_checkpoint,
)
from f2_experiment.opentrack_adapter import OpenTrackVLAF2ObservationAdapter
from f2_experiment.reproducibility import (
    configure_cuda_reproducibility,
    validate_cuda_reproducibility_receipt,
)
from f2_experiment.runner import (
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    OptimizerUpdateEvent,
    RunnerRow,
    checkpoint_init_sha256,
)
from f2_experiment.support import (
    FROZEN_TRAIN_SHA256,
    FROZEN_TRAIN_RELATIVE,
    SUPPORT_EXPECTATIONS,
    build_frozen_support,
    canonical_json_sha256,
    parse_train_jsonl,
)

from .assembly_model import (
    ENGINE_TO_FAMILY_ARM,
    F2SealedInitEvidence,
    IBR1AssemblyContractError,
    IBR1PairedArms,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    IBR1_PACKAGE,
    IBR1_CTRL,
    IBR1_SELF,
    build_ibr1_paired_arms,
    read_sealed_f2_init_evidence,
)
from .authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    verify_assembly_receipt,
)
from .checkpoint import ALLOWED_CHECKPOINT_UPDATES
from .diagnostics import (
    EVAL_SNAPSHOTS,
    GeometryCollector,
    GradientDiagnosticsCollector,
    IBR1G6Instrument,
    OptimizerDiagnosticsHandle,
    wrap_eval_predictor,
    wrap_training_head_forward,
)
from .model import IBR1AP2Model, IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


IBR1_SMOKE_SEED = 0
IBR1_SMOKE_DEVICE = "cuda:0"
IBR1_SMOKE_SUPPORT_ORDER = ("SMK-TRAIN", "EVAL-FIX")
IBR1_SMOKE_TRAIN_ROWS = 256
IBR1_SMOKE_EVAL_ROWS = 512

_SMOKE_DATA_LOAD_PROVENANCE_KEY = object()
_PRODUCTION_SMOKE_PROVENANCE_KEY = object()

_EVAL_SNAPSHOTS_BY_FAMILY_ARM: Mapping[str, frozenset[str]] = {
    IBR1_CTRL: frozenset({"update128_IBR1-CTRL"}),
    IBR1_SELF: frozenset(
        {"update0_IBR1-SELF", "update128_IBR1-SELF"}
    ),
}


class IBR1SmokeContractError(IBR1AssemblyContractError):
    """Raised before execution when the production smoke wiring drifts."""


@dataclass(frozen=True)
class _SmokeDataLoadProvenance:
    key: object
    project_root: str
    receipt_payload_sha256: str
    train_sha256: str
    support_order_sha256: tuple[tuple[str, str], ...]
    reset_set_sha256: tuple[tuple[str, str], ...]
    token_ledger_sha256: str
    token_ledger_file_count: int
    data_identity_sha256: str


@dataclass(frozen=True)
class _ProductionSmokeProvenance:
    key: object
    project_root: str
    final_receipt_path: str
    final_receipt_file_sha256: str
    final_receipt_payload_sha256: str
    final_receipt_analysis_class: str
    smoke_data_object_id: int
    smoke_data_identity_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1SmokeContractError(message)


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
        raise IBR1SmokeContractError(f"cannot read {label}: {path}") from exc


def _frozen_mapping(value: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return MappingProxyType(copy.deepcopy(dict(value)))


def _root_relative(root: Path, path: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as exc:
        raise IBR1SmokeContractError(
            f"{label} must remain inside the project root: {path}"
        ) from exc


def _validate_final_receipt(document: Mapping[str, Any]) -> None:
    _require(
        document.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS
        and document.get("family_id") == IBR1_FAMILY_ID
        and document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK
        and document.get("phase") == ASSEMBLY_PHASE_FINAL,
        "smoke authority must be the final IBR1 assembly receipt",
    )
    _require(
        isinstance(document.get("lambda_freeze_binding"), Mapping)
        and bool(document["lambda_freeze_binding"]),
        "final IBR1 assembly receipt has no lambda-adoption freeze",
    )
    _require(
        document.get("formal_training_authorized") is False,
        "IBR1 smoke receipt must keep formal training unauthorized",
    )
    _require(
        document.get("internal_test") == "sealed"
        and document.get("internal_test_opened") is False,
        "IBR1 smoke receipt does not preserve the internal-test seal",
    )
    _valid_sha256(
        document.get("receipt_payload_sha256"),
        "final assembly receipt payload SHA",
    )


def _final_receipt_binding(
    root: Path,
    path: Path,
    document: Mapping[str, Any],
) -> Mapping[str, str]:
    return MappingProxyType(
        {
            "path": _root_relative(root, path, "final assembly receipt"),
            "sha256": _sha256_file(path, "final assembly receipt"),
            "receipt_payload_sha256": _valid_sha256(
                document.get("receipt_payload_sha256"),
                "final assembly receipt payload SHA",
            ),
            "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        }
    )


def _support_observation(
    document: Mapping[str, Any],
) -> Mapping[str, Any]:
    binding = document.get("support_binding")
    _require(
        isinstance(binding, Mapping)
        and isinstance(binding.get("observation"), Mapping),
        "final assembly receipt carries no frozen support observation",
    )
    return binding["observation"]


def _asset_observation(document: Mapping[str, Any]) -> Mapping[str, Any]:
    binding = document.get("asset_binding")
    _require(
        isinstance(binding, Mapping)
        and isinstance(binding.get("observation"), Mapping),
        "final assembly receipt carries no frozen asset observation",
    )
    return binding["observation"]


def _token_ledger_binding(
    document: Mapping[str, Any], ledger: TokenHashLedger
) -> Mapping[str, Any]:
    assets = _asset_observation(document)
    frozen_sha = _valid_sha256(
        assets.get("token_ledger_sha256"), "receipt token ledger SHA"
    )
    frozen_count = assets.get("token_ledger_file_count")
    _require(
        isinstance(frozen_count, int)
        and not isinstance(frozen_count, bool)
        and frozen_count > 0,
        "receipt token ledger file count must be a positive integer",
    )
    _require(
        ledger.ledger_sha256 == frozen_sha,
        "TOKEN_LEDGER_MISMATCH: rebuilt train token ledger differs from "
        "the final IBR1 assembly receipt",
    )
    _require(
        ledger.token_files == frozen_count,
        "token ledger file count differs from the final IBR1 receipt",
    )
    return MappingProxyType(
        {
            "anchor": "final_assembly_receipt.asset_binding.observation",
            "sha256": frozen_sha,
            "file_count": frozen_count,
        }
    )


def _receipt_support_order(
    document: Mapping[str, Any], support_name: str
) -> tuple[int, ...]:
    supports = _support_observation(document).get("supports")
    _require(isinstance(supports, Mapping), "support observation is malformed")
    support = supports.get(support_name)
    _require(
        isinstance(support, Mapping),
        f"receipt has no {support_name} support binding",
    )
    order = support.get("ordered_original_indices")
    _require(
        isinstance(order, Sequence) and not isinstance(order, (str, bytes)),
        f"receipt {support_name} order is malformed",
    )
    normalized = tuple(order)
    _require(
        len(normalized) == SUPPORT_EXPECTATIONS[support_name].rows
        and all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in normalized
        )
        and len(set(normalized)) == len(normalized),
        f"receipt {support_name} order cardinality/identity drifted",
    )
    _require(
        canonical_json_sha256(list(normalized))
        == SUPPORT_EXPECTATIONS[support_name].sha256,
        f"receipt {support_name} order SHA drifted",
    )
    return normalized


@dataclass(frozen=True)
class IBR1SmokeData:
    """Frozen SMK-TRAIN/EVAL-FIX inputs and their receipt bindings."""

    smoke_rows: tuple[RunnerRow, ...]
    eval_rows: tuple[RunnerRow, ...]
    eval_raw_rows: tuple[Mapping[str, Any], ...]
    smoke_strafe_reset_original_indices: frozenset[int]
    eval_strafe_reset_original_indices: frozenset[int]
    smoke_expected_static_reset_original_indices: frozenset[int]
    eval_expected_static_reset_original_indices: frozenset[int]
    token_ledger: TokenHashLedger
    token_ledger_binding: Mapping[str, Any]
    support_order: Mapping[str, tuple[int, ...]]
    _load_provenance: _SmokeDataLoadProvenance | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require(
            len(self.smoke_rows) == IBR1_SMOKE_TRAIN_ROWS,
            "IBR1 SMK-TRAIN must contain exactly 256 RunnerRows",
        )
        _require(
            len(self.eval_rows) == IBR1_SMOKE_EVAL_ROWS,
            "IBR1 EVAL-FIX must contain exactly 512 RunnerRows",
        )
        _require(
            len(self.eval_raw_rows) == IBR1_SMOKE_EVAL_ROWS
            and all(isinstance(row, Mapping) for row in self.eval_raw_rows),
            "IBR1 EVAL-FIX must retain exactly 512 raw row mappings",
        )
        _require(
            isinstance(self.token_ledger, TokenHashLedger),
            "IBR1 smoke data requires a TokenHashLedger",
        )
        _require(
            tuple(self.support_order) == IBR1_SMOKE_SUPPORT_ORDER,
            "IBR1 support order keys must be SMK-TRAIN then EVAL-FIX",
        )
        expected_orders = {
            "SMK-TRAIN": tuple(
                row.original_row_index for row in self.smoke_rows
            ),
            "EVAL-FIX": tuple(row.original_row_index for row in self.eval_rows),
        }
        for name in IBR1_SMOKE_SUPPORT_ORDER:
            observed = tuple(self.support_order[name])
            _require(
                observed == expected_orders[name]
                and len(observed) == SUPPORT_EXPECTATIONS[name].rows
                and len(set(observed)) == len(observed),
                f"{name} RunnerRow order differs from its frozen support order",
            )
        reset_contracts = (
            (
                "SMK-TRAIN",
                self.smoke_strafe_reset_original_indices,
                self.smoke_expected_static_reset_original_indices,
            ),
            (
                "EVAL-FIX",
                self.eval_strafe_reset_original_indices,
                self.eval_expected_static_reset_original_indices,
            ),
        )
        for name, strafe, expected_static in reset_contracts:
            universe = frozenset(self.support_order[name])
            _require(
                isinstance(strafe, frozenset)
                and isinstance(expected_static, frozenset)
                and strafe <= universe
                and expected_static <= universe,
                f"{name} reset sets escape the frozen support",
            )
            _require(
                len(expected_static)
                == SUPPORT_EXPECTATIONS[name].static_resets,
                f"{name} static reset count drifted",
            )
        ledger_binding = self.token_ledger_binding
        _require(
            isinstance(ledger_binding, Mapping)
            and ledger_binding.get("sha256") == self.token_ledger.ledger_sha256
            and ledger_binding.get("file_count") == self.token_ledger.token_files,
            "IBR1 token ledger object differs from its recorded binding",
        )

    @property
    def strafe_reset_original_indices(self) -> frozenset[int]:
        return frozenset(
            self.smoke_strafe_reset_original_indices
            | self.eval_strafe_reset_original_indices
        )


def load_ibr1_smoke_data(
    project_root: str | Path,
    final_assembly_receipt: Mapping[str, Any],
) -> IBR1SmokeData:
    """Load only the frozen train-derived smoke/eval supports."""

    root = Path(project_root).expanduser().resolve()
    _validate_final_receipt(final_assembly_receipt)
    train_path = (root / FROZEN_TRAIN_RELATIVE).resolve()
    _require(train_path.is_file(), f"frozen train JSONL is missing: {train_path}")
    support_receipt = build_frozen_support(train_path)
    try:
        raw_rows = parse_train_jsonl(train_path.read_bytes())
    except OSError as exc:
        raise IBR1SmokeContractError(
            f"cannot read frozen train JSONL: {train_path}"
        ) from exc
    token_ledger = build_train_token_ledger(root)
    ledger_binding = _token_ledger_binding(
        final_assembly_receipt, token_ledger
    )
    base_root, cache_root = frozen_cache_roots(root)

    support_orders: dict[str, tuple[int, ...]] = {}
    for support_name in IBR1_SMOKE_SUPPORT_ORDER:
        live_order = tuple(
            index
            for index, _row in ordered_support_rows(
                raw_rows, support_receipt, support_name
            )
        )
        _require(
            live_order
            == _receipt_support_order(final_assembly_receipt, support_name),
            f"live {support_name} order differs from final receipt authority",
        )
        support_orders[support_name] = live_order

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
        MappingProxyType(copy.deepcopy(dict(row)))
        for _index, row in ordered_support_rows(
            raw_rows, support_receipt, "EVAL-FIX"
        )
    )
    smoke_strafe, smoke_expected = smoke_reset_sets(
        support_receipt, "SMK-TRAIN"
    )
    eval_strafe, eval_expected = smoke_reset_sets(
        support_receipt, "EVAL-FIX"
    )
    data = IBR1SmokeData(
        smoke_rows=tuple(smoke_rows),
        eval_rows=tuple(eval_rows),
        eval_raw_rows=eval_raw_rows,
        smoke_strafe_reset_original_indices=frozenset(smoke_strafe),
        eval_strafe_reset_original_indices=frozenset(eval_strafe),
        smoke_expected_static_reset_original_indices=frozenset(smoke_expected),
        eval_expected_static_reset_original_indices=frozenset(eval_expected),
        token_ledger=token_ledger,
        token_ledger_binding=ledger_binding,
        support_order=MappingProxyType(support_orders),
    )
    return _attach_smoke_data_load_provenance(
        root,
        final_assembly_receipt,
        data,
        train_sha256=support_receipt.train_sha256,
    )


@dataclass(frozen=True)
class IBR1EvalPredictorBinding:
    """One fresh predictor and its exact live-arm/snapshot identity."""

    raw_predictor: EvalRowPredictor
    predictor: Callable[..., Any]
    arm_assembly: ArmAssembly
    engine_arm: str
    family_arm: str
    snapshot: str

    def __post_init__(self) -> None:
        _require(
            self.raw_predictor.arm is self.arm_assembly.modules,
            "EVAL predictor is not bound to the exact live arm modules",
        )
        _require(
            ENGINE_TO_FAMILY_ARM.get(self.engine_arm) == self.family_arm,
            "EVAL predictor engine/public arm mapping drifted",
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.predictor(*args, **kwargs)


@dataclass(frozen=True)
class IBR1EvalPredictorFactory:
    """Create a fresh, geometry-wrapped predictor without running it."""

    arm_assembly: ArmAssembly
    engine_arm: str
    family_arm: str
    collector: GeometryCollector

    def __post_init__(self) -> None:
        _require(
            ENGINE_TO_FAMILY_ARM.get(self.engine_arm) == self.family_arm,
            "EVAL predictor factory arm mapping drifted",
        )

    def __call__(self, snapshot: str) -> IBR1EvalPredictorBinding:
        _require(
            snapshot in EVAL_SNAPSHOTS,
            f"unknown IBR1 EVAL snapshot {snapshot!r}",
        )
        _require(
            snapshot in _EVAL_SNAPSHOTS_BY_FAMILY_ARM[self.family_arm],
            f"snapshot {snapshot!r} is not valid for {self.family_arm}",
        )
        raw = build_eval_row_predictor(self.arm_assembly.modules)
        _require(
            raw.arm is self.arm_assembly.modules,
            "fresh EVAL predictor factory returned a different arm binding",
        )
        wrapped = wrap_eval_predictor(
            raw,
            self.collector,
            family_arm=self.family_arm,
            snapshot=snapshot,
        )
        return IBR1EvalPredictorBinding(
            raw_predictor=raw,
            predictor=wrapped,
            arm_assembly=self.arm_assembly,
            engine_arm=self.engine_arm,
            family_arm=self.family_arm,
            snapshot=snapshot,
        )


@dataclass(frozen=True)
class IBR1CheckpointTarget:
    """Exact live identity handed to the immutable checkpoint writer later."""

    project_root: str
    paired_arms: IBR1PairedArms
    arm_assembly: ArmAssembly
    engine_arm: str
    family_arm: str
    u_pre: int
    final_assembly_receipt_path: str
    final_assembly_receipt_sha256: str

    def __post_init__(self) -> None:
        _require(
            self.u_pre in ALLOWED_CHECKPOINT_UPDATES,
            "IBR1 checkpoint identity is not an allowed update boundary",
        )
        _require(
            self.paired_arms.arms.get(self.engine_arm) is self.arm_assembly
            and ENGINE_TO_FAMILY_ARM.get(self.engine_arm) == self.family_arm,
            "checkpoint identity is not bound to the exact paired arm",
        )
        _valid_sha256(
            self.final_assembly_receipt_sha256,
            "checkpoint final assembly receipt SHA",
        )

    def writer_kwargs(self) -> dict[str, Any]:
        """Return arguments for ``save_ibr1_arm_checkpoint``; do not write."""

        return {
            "paired_arms": self.paired_arms,
            "arm_assembly": self.arm_assembly,
            "engine_arm": self.engine_arm,
            "u_pre": self.u_pre,
            "final_assembly_receipt_path": self.final_assembly_receipt_path,
            "final_assembly_receipt_sha256": (
                self.final_assembly_receipt_sha256
            ),
            "project_root": self.project_root,
        }

    def identity_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "analysis_class": "ibr1_live_checkpoint_target_identity",
            "family_id": IBR1_FAMILY_ID,
            "engine_arm": self.engine_arm,
            "family_arm": self.family_arm,
            "u_pre": self.u_pre,
            "checkpoint_init_sha256": self.paired_arms.checkpoint_init_sha256,
            "final_assembly_receipt": {
                "path": self.final_assembly_receipt_path,
                "sha256": self.final_assembly_receipt_sha256,
            },
            "checkpoint_written": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }


@dataclass(frozen=True)
class IBR1CheckpointIdentityAccessor:
    """Produce update-boundary checkpoint identities without snapshotting."""

    project_root: str
    paired_arms: IBR1PairedArms
    arm_assembly: ArmAssembly
    engine_arm: str
    family_arm: str
    final_assembly_receipt_path: str
    final_assembly_receipt_sha256: str

    def __call__(self, u_pre: int) -> IBR1CheckpointTarget:
        _require(
            isinstance(u_pre, int)
            and not isinstance(u_pre, bool)
            and u_pre in ALLOWED_CHECKPOINT_UPDATES,
            f"checkpoint u_pre must be one of {ALLOWED_CHECKPOINT_UPDATES!r}",
        )
        return IBR1CheckpointTarget(
            project_root=self.project_root,
            paired_arms=self.paired_arms,
            arm_assembly=self.arm_assembly,
            engine_arm=self.engine_arm,
            family_arm=self.family_arm,
            u_pre=u_pre,
            final_assembly_receipt_path=self.final_assembly_receipt_path,
            final_assembly_receipt_sha256=self.final_assembly_receipt_sha256,
        )


@dataclass(frozen=True)
class IBR1SmokeArm:
    """One exact live arm with wrapped callbacks and deferred accessors."""

    engine_arm: str
    family_arm: str
    assembly: ArmAssembly
    modules: F2ArmModules
    optimizer: torch.optim.AdamW
    callbacks: ArmCallbacks
    executor: ArmExecutor
    optimizer_diagnostics: OptimizerDiagnosticsHandle
    eval_predictor_factory: IBR1EvalPredictorFactory
    checkpoint_identity: IBR1CheckpointIdentityAccessor

    def __post_init__(self) -> None:
        _require(
            ENGINE_TO_FAMILY_ARM.get(self.engine_arm) == self.family_arm,
            "smoke arm public/engine mapping drifted",
        )
        _require(
            self.assembly.modules is self.modules
            and self.assembly.optimizer is self.optimizer,
            "smoke arm live module/optimizer identity drifted",
        )
        _require(
            self.executor.arm is self.modules
            and self.executor.optimizer is self.optimizer,
            "runner executor is not bound to the exact live arm",
        )
        _require(
            self.optimizer_diagnostics.modules is self.modules
            and self.optimizer_diagnostics.optimizer is self.optimizer
            and self.optimizer_diagnostics.engine_arm == self.engine_arm,
            "optimizer diagnostics are not bound to the exact live arm",
        )
        _require(
            self.eval_predictor_factory.arm_assembly is self.assembly
            and self.checkpoint_identity.arm_assembly is self.assembly,
            "deferred predictor/checkpoint accessor arm identity drifted",
        )


def _index_set_identity(values: frozenset[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "sorted_original_indices": ordered,
        "canonical_sha256": canonical_json_sha256(ordered),
    }


def _smoke_data_identity_receipt(data: IBR1SmokeData) -> dict[str, Any]:
    return {
        "support_order": {
            name: {
                "count": len(data.support_order[name]),
                "ordered_original_indices": list(data.support_order[name]),
                "canonical_sha256": canonical_json_sha256(
                    list(data.support_order[name])
                ),
            }
            for name in IBR1_SMOKE_SUPPORT_ORDER
        },
        "reset_sets": {
            "SMK-TRAIN.strafe": _index_set_identity(
                data.smoke_strafe_reset_original_indices
            ),
            "SMK-TRAIN.expected_static": _index_set_identity(
                data.smoke_expected_static_reset_original_indices
            ),
            "EVAL-FIX.strafe": _index_set_identity(
                data.eval_strafe_reset_original_indices
            ),
            "EVAL-FIX.expected_static": _index_set_identity(
                data.eval_expected_static_reset_original_indices
            ),
        },
        "token_ledger": {
            "sha256": data.token_ledger.ledger_sha256,
            "file_count": data.token_ledger.token_files,
            "binding": dict(data.token_ledger_binding),
        },
    }


def _smoke_data_support_order_shas(
    data: IBR1SmokeData,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            name,
            canonical_json_sha256(list(data.support_order[name])),
        )
        for name in IBR1_SMOKE_SUPPORT_ORDER
    )


def _smoke_data_reset_set_shas(
    data: IBR1SmokeData,
) -> tuple[tuple[str, str], ...]:
    receipt = _smoke_data_identity_receipt(data)["reset_sets"]
    return tuple(
        (name, receipt[name]["canonical_sha256"])
        for name in (
            "SMK-TRAIN.strafe",
            "SMK-TRAIN.expected_static",
            "EVAL-FIX.strafe",
            "EVAL-FIX.expected_static",
        )
    )


def _attach_smoke_data_load_provenance(
    root: Path,
    document: Mapping[str, Any],
    data: IBR1SmokeData,
    *,
    train_sha256: str,
) -> IBR1SmokeData:
    """Mark data produced by the frozen loader's private verified path."""

    _require(
        train_sha256 == FROZEN_TRAIN_SHA256,
        "IBR1 smoke loader train SHA differs from the frozen train split",
    )
    payload_sha = _valid_sha256(
        document.get("receipt_payload_sha256"),
        "smoke-data final receipt payload SHA",
    )
    identity_sha = canonical_json_sha256(_smoke_data_identity_receipt(data))
    provenance = _SmokeDataLoadProvenance(
        key=_SMOKE_DATA_LOAD_PROVENANCE_KEY,
        project_root=str(root),
        receipt_payload_sha256=payload_sha,
        train_sha256=train_sha256,
        support_order_sha256=_smoke_data_support_order_shas(data),
        reset_set_sha256=_smoke_data_reset_set_shas(data),
        token_ledger_sha256=data.token_ledger.ledger_sha256,
        token_ledger_file_count=data.token_ledger.token_files,
        data_identity_sha256=identity_sha,
    )
    return replace(data, _load_provenance=provenance)


def _validate_loaded_smoke_data_provenance(
    root: Path,
    document: Mapping[str, Any],
    data: IBR1SmokeData,
) -> str:
    _require(
        isinstance(data, IBR1SmokeData),
        "production smoke data is not verified IBR1SmokeData",
    )
    provenance = data._load_provenance
    _require(
        type(provenance) is _SmokeDataLoadProvenance
        and provenance.key is _SMOKE_DATA_LOAD_PROVENANCE_KEY,
        "production smoke data lacks private frozen-loader provenance",
    )
    assert provenance is not None
    payload_sha = _valid_sha256(
        document.get("receipt_payload_sha256"),
        "production smoke-data receipt payload SHA",
    )
    _require(
        provenance.project_root == str(root)
        and provenance.receipt_payload_sha256 == payload_sha
        and provenance.train_sha256 == FROZEN_TRAIN_SHA256,
        "production smoke-data root/train/receipt provenance drifted",
    )
    support_shas = _smoke_data_support_order_shas(data)
    _require(
        support_shas == provenance.support_order_sha256,
        "production smoke support order drifted after frozen loading",
    )
    for name, order_sha in support_shas:
        _require(
            order_sha == SUPPORT_EXPECTATIONS[name].sha256
            and tuple(data.support_order[name])
            == _receipt_support_order(document, name),
            f"production {name} order lost frozen receipt provenance",
        )
    _require(
        _smoke_data_reset_set_shas(data) == provenance.reset_set_sha256,
        "production smoke reset sets drifted after frozen loading",
    )
    assets = _asset_observation(document)
    _require(
        data.token_ledger_binding.get("anchor")
        == "final_assembly_receipt.asset_binding.observation"
        and data.token_ledger.ledger_sha256
        == provenance.token_ledger_sha256
        == data.token_ledger_binding.get("sha256")
        == assets.get("token_ledger_sha256")
        and data.token_ledger.token_files
        == provenance.token_ledger_file_count
        == data.token_ledger_binding.get("file_count")
        == assets.get("token_ledger_file_count"),
        "production smoke token ledger lost final-receipt provenance",
    )
    identity_sha = canonical_json_sha256(_smoke_data_identity_receipt(data))
    _require(
        identity_sha == provenance.data_identity_sha256,
        "production smoke data identity drifted after frozen loading",
    )
    return identity_sha


def _issue_production_smoke_provenance(
    root: Path,
    receipt_path: Path,
    document: Mapping[str, Any],
    receipt_binding: Mapping[str, str],
    data: IBR1SmokeData,
) -> _ProductionSmokeProvenance:
    data_identity_sha = _validate_loaded_smoke_data_provenance(
        root, document, data
    )
    return _ProductionSmokeProvenance(
        key=_PRODUCTION_SMOKE_PROVENANCE_KEY,
        project_root=str(root),
        final_receipt_path=str(receipt_path),
        final_receipt_file_sha256=_valid_sha256(
            receipt_binding.get("sha256"),
            "production final receipt file SHA",
        ),
        final_receipt_payload_sha256=_valid_sha256(
            document.get("receipt_payload_sha256"),
            "production final receipt payload SHA",
        ),
        final_receipt_analysis_class=ASSEMBLY_RECEIPT_CLASS,
        smoke_data_object_id=id(data),
        smoke_data_identity_sha256=data_identity_sha,
    )


def _validate_production_smoke_provenance(
    root: Path,
    receipt_path: Path,
    document: Mapping[str, Any],
    receipt_binding: Mapping[str, str],
    data: IBR1SmokeData,
    provenance: _ProductionSmokeProvenance,
) -> None:
    _require(
        type(provenance) is _ProductionSmokeProvenance
        and provenance.key is _PRODUCTION_SMOKE_PROVENANCE_KEY,
        "production smoke plan lacks private live provenance",
    )
    _require(
        provenance.project_root == str(root)
        and provenance.final_receipt_path == str(receipt_path)
        and provenance.smoke_data_object_id == id(data),
        "production smoke live root/receipt/data identity drifted",
    )
    live_document = verify_assembly_receipt(
        root,
        receipt_path,
        required_phase=ASSEMBLY_PHASE_FINAL,
    )
    _validate_final_receipt(live_document)
    live_binding = _final_receipt_binding(root, receipt_path, live_document)
    _require(
        dict(live_document) == dict(document)
        and dict(live_binding) == dict(receipt_binding)
        and live_binding["sha256"]
        == provenance.final_receipt_file_sha256
        and live_binding["receipt_payload_sha256"]
        == provenance.final_receipt_payload_sha256
        and live_binding["analysis_class"]
        == provenance.final_receipt_analysis_class
        == ASSEMBLY_RECEIPT_CLASS,
        "production final receipt path/file/payload/class drifted",
    )
    data_identity_sha = _validate_loaded_smoke_data_provenance(
        root, live_document, data
    )
    _require(
        data_identity_sha == provenance.smoke_data_identity_sha256,
        "production frozen smoke-data provenance drifted",
    )


@dataclass(slots=True, eq=False, weakref_slot=True)
class IBR1SmokePlan:
    """Execution-ready objects whose construction performs no experiment."""

    project_root: str
    final_assembly_receipt_path: str
    final_assembly_receipt: Mapping[str, Any]
    final_assembly_receipt_binding: Mapping[str, str]
    paired_arms: IBR1PairedArms
    data: IBR1SmokeData
    geometry_collector: GeometryCollector
    gradient_collector: GradientDiagnosticsCollector
    g6: IBR1G6Instrument
    _arms: Mapping[str, IBR1SmokeArm]
    optimizer_contract: OptimizerContract
    seed: int
    device: str
    checkpoint_init_sha256: str
    cuda_reproducibility: Mapping[str, Any] | None
    base_load_report: Mapping[str, Any]
    formal_training_authorized: bool = field(default=False, init=False)
    internal_test: str = field(default="sealed", init=False)
    internal_test_opened: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_final_receipt(self.final_assembly_receipt)
        _require(
            self.seed == self.paired_arms.seed
            and self.device == self.paired_arms.device
            and self.checkpoint_init_sha256
            == self.paired_arms.checkpoint_init_sha256,
            "smoke plan seed/device/init identity differs from paired arms",
        )
        _require(
            tuple(self._arms) == (S_CTRL, S_SELF),
            "smoke plan engine arm order must be S-CTRL then S-SELF",
        )
        for engine_arm in (S_CTRL, S_SELF):
            _require(
                self._arms[engine_arm].assembly
                is self.paired_arms.arms[engine_arm],
                f"smoke plan {engine_arm} is not the exact paired assembly",
            )

    def _validate_identity_entry(self) -> None:
        """Component plans have no authority capability to validate."""

    def _validate_execution_entry(self) -> None:
        """Component plans are test-only and never authority-eligible."""

    @property
    def production_context(self) -> bool:
        return False

    @property
    def authority_eligible(self) -> bool:
        return False

    @property
    def arms(self) -> Mapping[str, IBR1SmokeArm]:
        self._validate_execution_entry()
        return self._arms

    @property
    def smoke_rows(self) -> tuple[RunnerRow, ...]:
        self._validate_execution_entry()
        return self.data.smoke_rows

    @property
    def eval_rows(self) -> tuple[RunnerRow, ...]:
        self._validate_execution_entry()
        return self.data.eval_rows

    @property
    def eval_raw_rows(self) -> tuple[Mapping[str, Any], ...]:
        self._validate_execution_entry()
        return self.data.eval_raw_rows

    @property
    def g6_update(self) -> Callable[[OptimizerUpdateEvent], Any]:
        self._validate_execution_entry()
        return self.g6.emit_update

    def public_arms(self) -> dict[str, IBR1SmokeArm]:
        return {
            ENGINE_TO_FAMILY_ARM[engine_arm]: arm
            for engine_arm, arm in self.arms.items()
        }

    def identity_receipt(self) -> dict[str, Any]:
        self._validate_identity_entry()
        return {
            "schema_version": 1,
            "analysis_class": "ibr1_production_smoke_assembly_plan",
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "seed": self.seed,
            "device": self.device,
            "checkpoint_init_sha256": self.checkpoint_init_sha256,
            "cuda_reproducibility": (
                dict(self.cuda_reproducibility)
                if self.cuda_reproducibility is not None
                else None
            ),
            "final_assembly_receipt": dict(
                self.final_assembly_receipt_binding
            ),
            "support_order": {
                name: list(self.data.support_order[name])
                for name in IBR1_SMOKE_SUPPORT_ORDER
            },
            "support_rows": {
                "SMK-TRAIN": len(self.smoke_rows),
                "EVAL-FIX": len(self.eval_rows),
                "EVAL-FIX_raw": len(self.eval_raw_rows),
            },
            "reset_sets": {
                "SMK-TRAIN": {
                    "strafe": _index_set_identity(
                        self.data.smoke_strafe_reset_original_indices
                    ),
                    "expected_static": _index_set_identity(
                        self.data.smoke_expected_static_reset_original_indices
                    ),
                },
                "EVAL-FIX": {
                    "strafe": _index_set_identity(
                        self.data.eval_strafe_reset_original_indices
                    ),
                    "expected_static": _index_set_identity(
                        self.data.eval_expected_static_reset_original_indices
                    ),
                },
            },
            "token_ledger_binding": dict(self.data.token_ledger_binding),
            "engine_to_family_arm": dict(ENGINE_TO_FAMILY_ARM),
            "production_context": self.production_context,
            "authority_eligible": self.authority_eligible,
            "callbacks_executed_during_assembly": False,
            "checkpoint_written_during_assembly": False,
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }

    def close(self) -> None:
        """Remove optimizer hooks.  Safe to call more than once."""

        registration = _PRODUCTION_PLAN_REGISTRATIONS.get(self)
        if registration is not None:
            cleanup_error: BaseException | None = None
            for arm_registration in reversed(registration.arm_bindings):
                for handle in (
                    arm_registration.pre_hook_handle,
                    arm_registration.post_hook_handle,
                ):
                    try:
                        handle.remove()
                    except BaseException as exc:
                        if cleanup_error is None:
                            cleanup_error = exc
            object.__setattr__(self, "_closed", True)
            if cleanup_error is not None:
                raise cleanup_error
            return
        if self._closed:
            return
        for engine_arm in reversed((S_CTRL, S_SELF)):
            self._arms[engine_arm].optimizer_diagnostics.close()
        self._closed = True

    def __copy__(self) -> IBR1SmokePlan:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> IBR1SmokePlan:
        return self

    def __enter__(self) -> IBR1SmokePlan:
        self._validate_execution_entry()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()


_PRODUCTION_PLAN_CLASS_MARKER = object()


class _IBR1ProductionSmokePlan(IBR1SmokePlan):
    """Private capability-bearing plan; never constructed by component APIs."""

    __slots__ = ("_production_class_marker",)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_production_class_marker":
            raise AttributeError("production plan capability is immutable")
        super().__setattr__(name, value)

    def _live_provenance(self) -> _ProductionSmokeProvenance:
        _require(
            getattr(self, "_production_class_marker", None)
            is _PRODUCTION_PLAN_CLASS_MARKER,
            "production smoke plan has no immutable class capability",
        )
        provenance = _PRODUCTION_PLAN_PROVENANCE.get(self)
        _require(
            type(provenance) is _ProductionSmokeProvenance
            and provenance.key is _PRODUCTION_SMOKE_PROVENANCE_KEY,
            "production smoke plan has no registered live provenance",
        )
        registration = _PRODUCTION_PLAN_REGISTRATIONS.get(self)
        _require(
            type(registration) is _ProductionPlanRegistration
            and registration.provenance is provenance,
            "production smoke plan has no registered execution binding",
        )
        assert provenance is not None
        return provenance

    def _validate_identity_entry(self) -> None:
        _validate_registered_production_plan(self, self._live_provenance())

    def _validate_execution_entry(self) -> None:
        _validate_registered_production_plan(self, self._live_provenance())

    @property
    def production_context(self) -> bool:
        self._validate_identity_entry()
        return True

    @property
    def authority_eligible(self) -> bool:
        self._validate_identity_entry()
        return True


@dataclass(frozen=True)
class _ProductionArmRegistration:
    engine_arm: str
    family_arm: str
    arm: IBR1SmokeArm
    assembly: ArmAssembly
    modules: F2ArmModules
    optimizer: torch.optim.AdamW
    callbacks: ArmCallbacks
    callback_fields: tuple[Any, ...]
    executor: ArmExecutor
    optimizer_diagnostics: OptimizerDiagnosticsHandle
    eval_predictor_factory: IBR1EvalPredictorFactory
    checkpoint_identity: IBR1CheckpointIdentityAccessor
    pre_hook_handle: Any
    post_hook_handle: Any


@dataclass(frozen=True)
class _ProductionPlanRegistration:
    provenance: _ProductionSmokeProvenance
    project_root: str
    final_assembly_receipt_path: str
    final_assembly_receipt: Mapping[str, Any]
    final_assembly_receipt_binding: Mapping[str, str]
    paired_arms: IBR1PairedArms
    paired_arm_mapping: Mapping[str, str]
    paired_arm_assemblies: Mapping[str, ArmAssembly]
    data: IBR1SmokeData
    geometry_collector: GeometryCollector
    gradient_collector: GradientDiagnosticsCollector
    g6: IBR1G6Instrument
    arms: Mapping[str, IBR1SmokeArm]
    optimizer_contract: OptimizerContract
    seed: int
    device: str
    checkpoint_init_sha256: str
    cuda_reproducibility: Mapping[str, Any] | None
    base_load_report: Mapping[str, Any]
    formal_training_authorized: bool
    internal_test: str
    internal_test_opened: bool
    arm_bindings: tuple[_ProductionArmRegistration, ...]


_PRODUCTION_PLAN_PROVENANCE: WeakKeyDictionary[
    _IBR1ProductionSmokePlan, _ProductionSmokeProvenance
] = WeakKeyDictionary()
_PRODUCTION_PLAN_REGISTRATIONS: WeakKeyDictionary[
    _IBR1ProductionSmokePlan, _ProductionPlanRegistration
] = WeakKeyDictionary()


_REGISTERED_CALLBACK_FIELDS = (
    "checkpoint_state",
    "feature_forward",
    "aux_forward",
    "head_forward",
    "track_loss",
    "backward",
    "optimizer_step",
    "audit_counters",
)


def _capture_production_plan_registration(
    plan: _IBR1ProductionSmokePlan,
    provenance: _ProductionSmokeProvenance,
) -> _ProductionPlanRegistration:
    arm_bindings: list[_ProductionArmRegistration] = []
    for engine_arm in (S_CTRL, S_SELF):
        arm = plan._arms[engine_arm]
        diagnostics = arm.optimizer_diagnostics
        arm_bindings.append(
            _ProductionArmRegistration(
                engine_arm=engine_arm,
                family_arm=arm.family_arm,
                arm=arm,
                assembly=arm.assembly,
                modules=arm.modules,
                optimizer=arm.optimizer,
                callbacks=arm.callbacks,
                callback_fields=tuple(
                    getattr(arm.callbacks, name)
                    for name in _REGISTERED_CALLBACK_FIELDS
                ),
                executor=arm.executor,
                optimizer_diagnostics=diagnostics,
                eval_predictor_factory=arm.eval_predictor_factory,
                checkpoint_identity=arm.checkpoint_identity,
                pre_hook_handle=diagnostics.pre_handle,
                post_hook_handle=diagnostics.post_handle,
            )
        )
    return _ProductionPlanRegistration(
        provenance=provenance,
        project_root=plan.project_root,
        final_assembly_receipt_path=plan.final_assembly_receipt_path,
        final_assembly_receipt=plan.final_assembly_receipt,
        final_assembly_receipt_binding=plan.final_assembly_receipt_binding,
        paired_arms=plan.paired_arms,
        paired_arm_mapping=plan.paired_arms.arm_mapping,
        paired_arm_assemblies=plan.paired_arms.arms,
        data=plan.data,
        geometry_collector=plan.geometry_collector,
        gradient_collector=plan.gradient_collector,
        g6=plan.g6,
        arms=plan._arms,
        optimizer_contract=plan.optimizer_contract,
        seed=plan.seed,
        device=plan.device,
        checkpoint_init_sha256=plan.checkpoint_init_sha256,
        cuda_reproducibility=plan.cuda_reproducibility,
        base_load_report=plan.base_load_report,
        formal_training_authorized=plan.formal_training_authorized,
        internal_test=plan.internal_test,
        internal_test_opened=plan.internal_test_opened,
        arm_bindings=tuple(arm_bindings),
    )


def _require_registered_scalar(
    plan: _IBR1ProductionSmokePlan,
    registration: _ProductionPlanRegistration,
    name: str,
) -> None:
    observed = object.__getattribute__(plan, name)
    expected = getattr(registration, name)
    _require(
        type(observed) is type(expected) and observed == expected,
        f"registered production plan field {name!r} drifted",
    )


def _require_registered_object(
    plan: _IBR1ProductionSmokePlan,
    registration: _ProductionPlanRegistration,
    name: str,
    registration_name: str | None = None,
) -> None:
    expected_name = registration_name or name
    _require(
        object.__getattribute__(plan, name)
        is getattr(registration, expected_name),
        f"registered production plan field {name!r} drifted",
    )


def _validate_registered_execution_bindings(
    plan: _IBR1ProductionSmokePlan,
    registration: _ProductionPlanRegistration,
) -> None:
    _require(
        type(plan) is _IBR1ProductionSmokePlan
        and registration.provenance
        is _PRODUCTION_PLAN_PROVENANCE.get(plan),
        "production smoke plan registration identity drifted",
    )
    for name in (
        "project_root",
        "final_assembly_receipt_path",
        "seed",
        "device",
        "checkpoint_init_sha256",
        "formal_training_authorized",
        "internal_test",
        "internal_test_opened",
    ):
        _require_registered_scalar(plan, registration, name)
    for plan_name, registration_name in (
        ("final_assembly_receipt", None),
        ("final_assembly_receipt_binding", None),
        ("paired_arms", None),
        ("data", None),
        ("geometry_collector", None),
        ("gradient_collector", None),
        ("g6", None),
        ("_arms", "arms"),
        ("optimizer_contract", None),
        ("cuda_reproducibility", None),
        ("base_load_report", None),
    ):
        _require_registered_object(
            plan,
            registration,
            plan_name,
            registration_name,
        )

    paired = registration.paired_arms
    _require(
        paired.arm_mapping is registration.paired_arm_mapping
        and paired.arms is registration.paired_arm_assemblies
        and tuple(paired.arms) == (S_CTRL, S_SELF),
        "registered production paired-arm binding drifted",
    )
    _require(
        registration.g6.collector is registration.gradient_collector,
        "registered production G6 collector binding drifted",
    )
    for arm_registration in registration.arm_bindings:
        engine_arm = arm_registration.engine_arm
        arm = registration.arms.get(engine_arm)
        _require(
            arm is arm_registration.arm
            and paired.arms.get(engine_arm) is arm_registration.assembly,
            f"registered production {engine_arm} arm binding drifted",
        )
        _require(
            arm.engine_arm == engine_arm
            and arm.family_arm == arm_registration.family_arm
            and arm.assembly is arm_registration.assembly
            and arm.modules is arm_registration.modules
            and arm.optimizer is arm_registration.optimizer
            and arm.callbacks is arm_registration.callbacks
            and arm.executor is arm_registration.executor
            and arm.optimizer_diagnostics
            is arm_registration.optimizer_diagnostics
            and arm.eval_predictor_factory
            is arm_registration.eval_predictor_factory
            and arm.checkpoint_identity
            is arm_registration.checkpoint_identity,
            f"registered production {engine_arm} execution wiring drifted",
        )
        _require(
            all(
                getattr(arm.callbacks, name) is expected
                for name, expected in zip(
                    _REGISTERED_CALLBACK_FIELDS,
                    arm_registration.callback_fields,
                )
            ),
            f"registered production {engine_arm} callbacks drifted",
        )
        executor = arm_registration.executor
        _require(
            executor.arm is arm_registration.modules
            and executor.optimizer is arm_registration.optimizer
            and executor.contract is registration.optimizer_contract
            and executor.g6
            is (registration.g6 if engine_arm == S_CTRL else None),
            f"registered production {engine_arm} executor binding drifted",
        )
        diagnostics = arm_registration.optimizer_diagnostics
        _require(
            diagnostics.optimizer is arm_registration.optimizer
            and diagnostics.modules is arm_registration.modules
            and diagnostics.collector is registration.gradient_collector
            and diagnostics.engine_arm == engine_arm
            and diagnostics.callbacks is arm_registration.callbacks
            and diagnostics.pre_handle
            is arm_registration.pre_hook_handle
            and diagnostics.post_handle
            is arm_registration.post_hook_handle,
            f"registered production {engine_arm} diagnostics binding drifted",
        )
        eval_factory = arm_registration.eval_predictor_factory
        _require(
            eval_factory.arm_assembly is arm_registration.assembly
            and eval_factory.engine_arm == engine_arm
            and eval_factory.family_arm == arm_registration.family_arm
            and eval_factory.collector is registration.geometry_collector,
            f"registered production {engine_arm} eval binding drifted",
        )
        checkpoint = arm_registration.checkpoint_identity
        _require(
            checkpoint.project_root == registration.project_root
            and checkpoint.paired_arms is paired
            and checkpoint.arm_assembly is arm_registration.assembly
            and checkpoint.engine_arm == engine_arm
            and checkpoint.family_arm == arm_registration.family_arm
            and checkpoint.final_assembly_receipt_path
            == registration.final_assembly_receipt_path
            and checkpoint.final_assembly_receipt_sha256
            == registration.final_assembly_receipt_binding.get("sha256"),
            f"registered production {engine_arm} checkpoint binding drifted",
        )


def _validate_registered_production_plan(
    plan: _IBR1ProductionSmokePlan,
    provenance: _ProductionSmokeProvenance,
) -> None:
    registration = _PRODUCTION_PLAN_REGISTRATIONS.get(plan)
    _require(
        type(registration) is _ProductionPlanRegistration
        and registration.provenance is provenance,
        "production smoke plan has no registered execution binding",
    )
    assert registration is not None
    _validate_registered_execution_bindings(plan, registration)
    _validate_production_smoke_provenance(
        Path(plan.project_root).resolve(),
        Path(plan.final_assembly_receipt_path).resolve(),
        plan.final_assembly_receipt,
        plan.final_assembly_receipt_binding,
        plan.data,
        provenance,
    )
    _require(
        plan.seed == IBR1_SMOKE_SEED and plan.device == IBR1_SMOKE_DEVICE,
        "production IBR1 smoke is fixed to seed 0 on cuda:0",
    )
    _require(
        plan.cuda_reproducibility is not None,
        "production IBR1 smoke has no CUDA reproducibility receipt",
    )
    _require(
        torch.cuda.is_available(),
        "production IBR1 smoke requires live CUDA availability",
    )
    validate_cuda_reproducibility_receipt(plan.cuda_reproducibility)


def _register_production_plan(
    plan: _IBR1ProductionSmokePlan,
    provenance: _ProductionSmokeProvenance,
) -> None:
    _require(
        type(plan) is _IBR1ProductionSmokePlan
        and plan not in _PRODUCTION_PLAN_PROVENANCE
        and plan not in _PRODUCTION_PLAN_REGISTRATIONS,
        "production plan capability registration is not fresh",
    )
    _require(
        type(provenance) is _ProductionSmokeProvenance
        and provenance.key is _PRODUCTION_SMOKE_PROVENANCE_KEY,
        "production plan capability provenance is invalid",
    )
    object.__setattr__(
        plan, "_production_class_marker", _PRODUCTION_PLAN_CLASS_MARKER
    )
    registration = _capture_production_plan_registration(plan, provenance)
    _PRODUCTION_PLAN_PROVENANCE[plan] = provenance
    _PRODUCTION_PLAN_REGISTRATIONS[plan] = registration
    try:
        _validate_registered_production_plan(plan, provenance)
    except BaseException:
        _PRODUCTION_PLAN_PROVENANCE.pop(plan, None)
        _PRODUCTION_PLAN_REGISTRATIONS.pop(plan, None)
        raise


def _module_state_marker(assembly: ArmAssembly) -> tuple[Any, ...]:
    marker: list[Any] = []
    for name, tensor in sorted(assembly.modules.full_state_dict().items()):
        marker.append(
            (
                name,
                tuple(tensor.shape),
                str(tensor.dtype),
                str(tensor.device),
                tensor.untyped_storage().data_ptr(),
                tensor._version,
            )
        )
    marker.extend(
        (name, parameter.requires_grad)
        for name, parameter in assembly.modules.named_full_parameters()
    )
    return tuple(marker)


def _optimizer_state_marker(assembly: ArmAssembly) -> dict[str, Any]:
    state = assembly.optimizer.state_dict()
    _require(
        not state["state"],
        "fresh IBR1 smoke optimizer unexpectedly carries state",
    )
    return copy.deepcopy(state)


def _parameter_ids(assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for _name, parameter in assembly.modules.named_full_parameters()
    }


def _arm_named_module_objects(
    assembly: ArmAssembly,
) -> dict[int, tuple[torch.nn.Module, tuple[str, ...]]]:
    observed: dict[int, tuple[torch.nn.Module, list[str]]] = {}
    for root_name, root_module in (
        ("base", assembly.modules.base),
        ("adapter", assembly.modules.adapter),
        ("model", assembly.modules.model),
    ):
        _require(
            isinstance(root_module, torch.nn.Module),
            f"arm {root_name} root is not a torch module",
        )
        for relative_name, module in root_module.named_modules(
            remove_duplicate=False
        ):
            qualified = (
                root_name
                if not relative_name
                else f"{root_name}.{relative_name}"
            )
            record = observed.setdefault(id(module), (module, []))
            record[1].append(qualified)
    return {
        object_id: (module, tuple(sorted(set(names))))
        for object_id, (module, names) in observed.items()
    }


@dataclass(frozen=True)
class _TensorStorageInterval:
    names: tuple[str, ...]
    device: str
    start: int
    stop: int


def _tensor_storage_interval(
    tensor: torch.Tensor,
    names: tuple[str, ...],
) -> _TensorStorageInterval | None:
    if tensor.numel() == 0:
        return None
    _require(
        tensor.layout == torch.strided,
        f"registered tensor {names!r} has unsupported layout {tensor.layout}",
    )
    minimum_element = tensor.storage_offset()
    maximum_element = tensor.storage_offset()
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (size - 1) * stride
        minimum_element += min(0, delta)
        maximum_element += max(0, delta)
    element_size = tensor.element_size()
    storage = tensor.untyped_storage()
    storage_start = storage.data_ptr()
    storage_stop = storage_start + storage.nbytes()
    start = storage_start + minimum_element * element_size
    stop = storage_start + (maximum_element + 1) * element_size
    _require(
        storage_start <= start < stop <= storage_stop,
        f"registered tensor {names!r} escapes its underlying storage",
    )
    return _TensorStorageInterval(
        names=names,
        device=str(tensor.device),
        start=start,
        stop=stop,
    )


def _arm_tensor_storage_intervals(
    assembly: ArmAssembly,
) -> tuple[_TensorStorageInterval, ...]:
    tensors: dict[int, tuple[torch.Tensor, list[str]]] = {}
    for _module, module_names in _arm_named_module_objects(assembly).values():
        module_name = module_names[0]
        for kind, values in (
            ("parameter", _module.named_parameters(recurse=False)),
            ("buffer", _module.named_buffers(recurse=False)),
        ):
            for name, tensor in values:
                record = tensors.setdefault(id(tensor), (tensor, []))
                record[1].append(f"{module_name}.{name} ({kind})")
    intervals: list[_TensorStorageInterval] = []
    for tensor, names in tensors.values():
        interval = _tensor_storage_interval(
            tensor,
            tuple(sorted(set(names))),
        )
        if interval is not None:
            intervals.append(interval)
    return tuple(intervals)


def _require_cross_arm_isolation(
    ctrl: ArmAssembly,
    self_arm: ArmAssembly,
) -> None:
    ctrl_modules = _arm_named_module_objects(ctrl)
    self_modules = _arm_named_module_objects(self_arm)
    shared_modules = set(ctrl_modules) & set(self_modules)
    _require(
        not shared_modules,
        "paired arms share named_modules objects: "
        + ", ".join(
            f"{ctrl_modules[object_id][1]!r} / "
            f"{self_modules[object_id][1]!r}"
            for object_id in sorted(shared_modules)
        ),
    )

    ctrl_intervals = _arm_tensor_storage_intervals(ctrl)
    self_intervals = _arm_tensor_storage_intervals(self_arm)
    for left in ctrl_intervals:
        for right in self_intervals:
            if left.device != right.device:
                continue
            _require(
                left.stop <= right.start or right.stop <= left.start,
                "paired arms overlap parameter/buffer storage on "
                f"{left.device}: {left.names!r} [{left.start},{left.stop}) / "
                f"{right.names!r} [{right.start},{right.stop})",
            )


def _optimizer_parameter_ids(assembly: ArmAssembly) -> set[int]:
    return {
        id(parameter)
        for group in assembly.optimizer.param_groups
        for parameter in group["params"]
    }


def _require_no_global_optimizer_hooks() -> None:
    """Reject process-wide optimizer hooks before wiring a smoke plan.

    Torch 2.6 keeps these registries in ``torch.optim.optimizer`` as ordered
    mappings.  Treat a missing or changed registry as unsafe rather than
    silently assuming that no process-wide callbacks are installed.
    """

    for attribute, label in (
        ("_global_optimizer_pre_hooks", "pre-hooks"),
        ("_global_optimizer_post_hooks", "post-hooks"),
    ):
        registry = getattr(torch_optimizer, attribute, None)
        _require(
            isinstance(registry, Mapping),
            f"cannot verify global optimizer {label} registry",
        )
        _require(
            not registry,
            f"global optimizer {label} are installed; refusing smoke wiring",
        )


def _require_fresh_optimizer_hooks(
    optimizer: torch.optim.AdamW, family_arm: str
) -> None:
    for attribute, label in (
        ("_optimizer_step_pre_hooks", "step pre-hooks"),
        ("_optimizer_step_post_hooks", "step post-hooks"),
    ):
        hooks = getattr(optimizer, attribute, None)
        _require(
            isinstance(hooks, Mapping) and not hooks,
            f"{family_arm} optimizer already has {label}; refusing double wiring",
        )


def _expected_parameter_receipt(
    assembly: ArmAssembly,
    *,
    engine_arm: str,
    family_arm: str,
    contract: OptimizerContract,
) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    expected_optimizer, f2_receipt = build_arm_optimizer(
        assembly.modules, contract
    )
    return expected_optimizer, {
        **dict(f2_receipt),
        "analysis_class": "ibr1_arm_optimizer_parameter_receipt",
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "model_class": "ibr1_experiment.model.IBR1AP2Model",
        "package": IBR1_PACKAGE,
        "engine_arm": engine_arm,
        "family_arm": family_arm,
    }


def _optimizer_group_metadata(group: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in group.items() if key != "params"}


def _validate_live_optimizer_contract(
    assembly: ArmAssembly,
    *,
    engine_arm: str,
    family_arm: str,
    contract: OptimizerContract,
) -> None:
    actual = assembly.optimizer
    _require(
        type(actual) is torch.optim.AdamW,
        f"{family_arm} optimizer must be the exact torch AdamW class",
    )
    expected, expected_receipt = _expected_parameter_receipt(
        assembly,
        engine_arm=engine_arm,
        family_arm=family_arm,
        contract=contract,
    )
    _require(
        isinstance(assembly.parameter_receipt, Mapping)
        and dict(assembly.parameter_receipt) == expected_receipt,
        f"{family_arm} parameter receipt differs from a mechanical rebuild",
    )
    _require(
        dict(actual.defaults) == dict(expected.defaults),
        f"{family_arm} AdamW defaults drifted from the frozen contract",
    )
    _require(
        len(actual.param_groups) == len(expected.param_groups),
        f"{family_arm} AdamW parameter-group count drifted",
    )
    for position, (observed_group, expected_group) in enumerate(
        zip(actual.param_groups, expected.param_groups)
    ):
        _require(
            _optimizer_group_metadata(observed_group)
            == _optimizer_group_metadata(expected_group),
            f"{family_arm} AdamW group metadata/order drifted at {position}",
        )
        _require(
            [id(parameter) for parameter in observed_group["params"]]
            == [id(parameter) for parameter in expected_group["params"]],
            f"{family_arm} AdamW parameter membership/order drifted at {position}",
        )

    grouped_ids = [
        id(parameter)
        for group in actual.param_groups
        for parameter in group["params"]
    ]
    trainable_ids = [
        id(parameter) for parameter in assembly.modules.trainable_parameters()
    ]
    _require(
        len(grouped_ids) == len(set(grouped_ids)),
        f"{family_arm} AdamW parameter groups overlap",
    )
    _require(
        set(grouped_ids) == set(trainable_ids),
        f"{family_arm} AdamW trainable-parameter coverage drifted",
    )
    _require(
        not actual.state_dict()["state"],
        f"{family_arm} optimizer is not a fresh update-0 AdamW",
    )
    _require_fresh_optimizer_hooks(actual, family_arm)


def _validate_live_device(
    paired_arms: IBR1PairedArms, *, production_context: bool
) -> torch.device:
    try:
        declared = torch.device(paired_arms.device)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise IBR1SmokeContractError(
            "paired-arm declared device is invalid"
        ) from exc
    _require(
        paired_arms.device == str(declared),
        "paired-arm device declaration is not canonical",
    )
    if production_context:
        _require(
            declared == torch.device(IBR1_SMOKE_DEVICE),
            "production paired arms must declare cuda:0",
        )
        _require(
            torch.cuda.is_available(),
            "production paired arms require live CUDA availability",
        )
    for engine_arm, assembly in paired_arms.arms.items():
        for name, tensor in assembly.modules.full_state_dict().items():
            _require(
                tensor.device == declared,
                f"{engine_arm} live tensor {name!r} differs from declared device",
            )
        for group_position, group in enumerate(assembly.optimizer.param_groups):
            for parameter_position, parameter in enumerate(group["params"]):
                _require(
                    parameter.device == declared,
                    f"{engine_arm} optimizer parameter "
                    f"{group_position}:{parameter_position} differs from "
                    "declared device",
                )
    return declared


def _validate_exact_paired_arms(
    paired_arms: IBR1PairedArms,
    *,
    project_root: Path,
    contract: OptimizerContract,
    production_context: bool,
) -> None:
    _require(
        paired_arms.family_id == IBR1_FAMILY_ID
        and paired_arms.architecture_lock == IBR1_ARCHITECTURE_LOCK
        and paired_arms.package == IBR1_PACKAGE,
        "paired-arm IBR1 family/package identity drifted",
    )
    _require(
        isinstance(paired_arms.seed, int)
        and not isinstance(paired_arms.seed, bool)
        and paired_arms.seed >= 0,
        "paired-arm seed must be a nonnegative integer",
    )
    _require(
        isinstance(paired_arms.parent_f2_evidence, F2SealedInitEvidence),
        "paired arms have no sealed F2 initialization evidence",
    )
    _require(
        paired_arms.seed == paired_arms.parent_f2_evidence.seed
        and paired_arms.checkpoint_init_sha256
        == paired_arms.parent_f2_evidence.checkpoint_init_sha256,
        "paired-arm seed/init differs from its sealed F2 evidence",
    )
    _valid_sha256(
        paired_arms.checkpoint_init_sha256,
        "paired-arm checkpoint initialization SHA",
    )
    _require(
        dict(paired_arms.arm_mapping) == dict(ENGINE_TO_FAMILY_ARM),
        "paired-arm public/engine mapping drifted",
    )
    _require(
        tuple(paired_arms.arms) == (S_CTRL, S_SELF),
        "paired-arm engine coverage/order must be S-CTRL then S-SELF",
    )
    if production_context:
        live_evidence = read_sealed_f2_init_evidence(project_root)
        _require(
            paired_arms.parent_f2_evidence == live_evidence,
            "production paired arms differ from live sealed F2 evidence",
        )

    validated: dict[str, ArmAssembly] = {}
    for engine_arm in (S_CTRL, S_SELF):
        family_arm = ENGINE_TO_FAMILY_ARM[engine_arm]
        assembly = paired_arms.arms[engine_arm]
        _require(
            type(assembly) is ArmAssembly,
            f"{family_arm} must be the exact ArmAssembly class",
        )
        modules = assembly.modules
        _require(
            type(modules) is F2ArmModules
            and modules.package == IBR1_PACKAGE
            and dict(modules.aux_coefficients)
            == dict(IBR1_FROZEN_AUX_COEFFICIENTS),
            f"{family_arm} module/package/lambda identity drifted",
        )
        _require(
            type(modules.adapter) is OpenTrackVLAF2ObservationAdapter
            and type(modules.model) is IBR1AP2Model
            and modules.adapter.base is modules.base,
            f"{family_arm} exact adapter/model/base identity drifted",
        )
        _validate_live_optimizer_contract(
            assembly,
            engine_arm=engine_arm,
            family_arm=family_arm,
            contract=contract,
        )
        live_sha = checkpoint_init_sha256(modules.full_state_dict())
        _require(
            live_sha == paired_arms.checkpoint_init_sha256,
            f"{family_arm} live tensors differ from paired init SHA",
        )
        validated[engine_arm] = assembly

    ctrl = validated[S_CTRL]
    self_arm = validated[S_SELF]
    for label, left, right in (
        ("ArmAssembly", ctrl, self_arm),
        ("modules", ctrl.modules, self_arm.modules),
        ("base", ctrl.modules.base, self_arm.modules.base),
        ("adapter", ctrl.modules.adapter, self_arm.modules.adapter),
        ("model", ctrl.modules.model, self_arm.modules.model),
        ("optimizer", ctrl.optimizer, self_arm.optimizer),
    ):
        _require(left is not right, f"paired arms share the {label} object")
    _require(
        not (_parameter_ids(ctrl) & _parameter_ids(self_arm)),
        "paired arms share parameter objects",
    )
    _require(
        not (
            _optimizer_parameter_ids(ctrl)
            & _optimizer_parameter_ids(self_arm)
        ),
        "paired optimizers share parameter membership",
    )
    _require_cross_arm_isolation(ctrl, self_arm)
    _validate_live_device(
        paired_arms, production_context=production_context
    )


def _bound_method_owner(value: Any) -> Any:
    return getattr(value, "__self__", None)


def _validate_callback_binding(
    callbacks: ArmCallbacks,
    executor: ArmExecutor,
    handle: OptimizerDiagnosticsHandle,
) -> None:
    for name in (
        "feature_forward",
        "aux_forward",
        "track_loss",
        "backward",
    ):
        _require(
            _bound_method_owner(getattr(callbacks, name)) is executor,
            f"callback {name} is not bound to the exact ArmExecutor",
        )
    _require(
        _bound_method_owner(handle.original) is executor,
        "optimizer diagnostics did not wrap the exact executor step callback",
    )
    _require(
        _bound_method_owner(callbacks.optimizer_step) is handle,
        "final optimizer callback is not bound to its diagnostics handle",
    )


def _validate_checkpoint_state_binding(
    callbacks: ArmCallbacks, modules: F2ArmModules
) -> None:
    live_state = modules.full_state_dict()
    callback_state = callbacks.checkpoint_state
    _require(
        set(callback_state) == set(live_state),
        "checkpoint callback state keys differ from the exact live arm",
    )
    for name, live_tensor in live_state.items():
        callback_tensor = callback_state[name]
        _require(
            isinstance(callback_tensor, torch.Tensor)
            and callback_tensor.untyped_storage().data_ptr()
            == live_tensor.untyped_storage().data_ptr(),
            f"checkpoint callback state {name!r} is not live-arm bound",
        )


def build_ibr1_smoke_plan_from_components(
    *,
    project_root: str | Path,
    final_assembly_receipt_path: str | Path,
    final_assembly_receipt: Mapping[str, Any],
    final_assembly_receipt_binding: Mapping[str, str],
    paired_arms: IBR1PairedArms,
    data: IBR1SmokeData,
    cuda_reproducibility: Mapping[str, Any] | None,
    base_load_report: Mapping[str, Any] | None = None,
) -> IBR1SmokePlan:
    """Wire a non-authoritative test/component plan without executing it."""

    return _build_ibr1_smoke_plan_from_components(
        project_root=project_root,
        final_assembly_receipt_path=final_assembly_receipt_path,
        final_assembly_receipt=final_assembly_receipt,
        final_assembly_receipt_binding=final_assembly_receipt_binding,
        paired_arms=paired_arms,
        data=data,
        cuda_reproducibility=cuda_reproducibility,
        base_load_report=base_load_report,
        production_provenance=None,
    )


def _build_ibr1_smoke_plan_from_components(
    *,
    project_root: str | Path,
    final_assembly_receipt_path: str | Path,
    final_assembly_receipt: Mapping[str, Any],
    final_assembly_receipt_binding: Mapping[str, str],
    paired_arms: IBR1PairedArms,
    data: IBR1SmokeData,
    cuda_reproducibility: Mapping[str, Any] | None,
    base_load_report: Mapping[str, Any] | None,
    production_provenance: _ProductionSmokeProvenance | None,
) -> IBR1SmokePlan:
    """Private shared builder; only live production provenance grants authority."""

    root = Path(project_root).expanduser().resolve()
    receipt_path = Path(final_assembly_receipt_path).expanduser().resolve()
    _require_no_global_optimizer_hooks()
    _validate_final_receipt(final_assembly_receipt)
    _require(
        isinstance(paired_arms, IBR1PairedArms),
        "paired_arms must be an IBR1PairedArms",
    )
    _require(
        isinstance(data, IBR1SmokeData),
        "data must be verified IBR1SmokeData",
    )
    binding = dict(final_assembly_receipt_binding)
    _require(
        binding.get("path") == _root_relative(root, receipt_path, "final receipt"),
        "final receipt binding path differs from the assembly argument",
    )
    _require(
        binding.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS
        and binding.get("receipt_payload_sha256")
        == final_assembly_receipt.get("receipt_payload_sha256"),
        "final receipt document/binding identity differs",
    )
    receipt_sha = _valid_sha256(
        binding.get("sha256"), "final receipt binding SHA"
    )
    _require(
        paired_arms.arm_mapping == dict(ENGINE_TO_FAMILY_ARM),
        "paired IBR1 arm mapping drifted",
    )
    contract = OptimizerContract()
    normalized_cuda_receipt = (
        MappingProxyType(
            validate_cuda_reproducibility_receipt(cuda_reproducibility)
        )
        if cuda_reproducibility is not None
        else None
    )
    production_context = production_provenance is not None
    if production_context:
        _require(
            normalized_cuda_receipt is not None,
            "production IBR1 smoke has no CUDA reproducibility receipt",
        )
        assert production_provenance is not None
        _validate_production_smoke_provenance(
            root,
            receipt_path,
            final_assembly_receipt,
            binding,
            data,
            production_provenance,
        )
    _validate_exact_paired_arms(
        paired_arms,
        project_root=root,
        contract=contract,
        production_context=production_context,
    )
    module_markers = {
        arm: _module_state_marker(paired_arms.arms[arm])
        for arm in (S_CTRL, S_SELF)
    }
    optimizer_markers = {
        arm: _optimizer_state_marker(paired_arms.arms[arm])
        for arm in (S_CTRL, S_SELF)
    }

    geometry = GeometryCollector()
    gradient = GradientDiagnosticsCollector()
    ctrl = paired_arms.arms[S_CTRL]
    g6 = IBR1G6Instrument(
        tuple(ctrl.modules.base.proj.parameters()), gradient
    )
    built_arms: dict[str, IBR1SmokeArm] = {}
    open_handles: list[OptimizerDiagnosticsHandle] = []
    try:
        for engine_arm in (S_CTRL, S_SELF):
            family_arm = ENGINE_TO_FAMILY_ARM[engine_arm]
            assembly = paired_arms.arms[engine_arm]
            callbacks, executor = build_arm_callbacks(
                assembly.modules,
                assembly.optimizer,
                contract,
                g6=g6 if engine_arm == S_CTRL else None,
            )
            callbacks = wrap_training_head_forward(callbacks, geometry)
            diagnostics = OptimizerDiagnosticsHandle(
                callbacks,
                optimizer=assembly.optimizer,
                modules=assembly.modules,
                collector=gradient,
                engine_arm=engine_arm,
            )
            open_handles.append(diagnostics)
            callbacks = diagnostics.callbacks
            _validate_callback_binding(callbacks, executor, diagnostics)
            _validate_checkpoint_state_binding(callbacks, assembly.modules)
            _require(
                executor.g6 is (g6 if engine_arm == S_CTRL else None),
                f"{engine_arm} G6 executor binding drifted",
            )
            eval_factory = IBR1EvalPredictorFactory(
                arm_assembly=assembly,
                engine_arm=engine_arm,
                family_arm=family_arm,
                collector=geometry,
            )
            checkpoint_identity = IBR1CheckpointIdentityAccessor(
                project_root=str(root),
                paired_arms=paired_arms,
                arm_assembly=assembly,
                engine_arm=engine_arm,
                family_arm=family_arm,
                final_assembly_receipt_path=str(receipt_path),
                final_assembly_receipt_sha256=receipt_sha,
            )
            built_arms[engine_arm] = IBR1SmokeArm(
                engine_arm=engine_arm,
                family_arm=family_arm,
                assembly=assembly,
                modules=assembly.modules,
                optimizer=assembly.optimizer,
                callbacks=callbacks,
                executor=executor,
                optimizer_diagnostics=diagnostics,
                eval_predictor_factory=eval_factory,
                checkpoint_identity=checkpoint_identity,
            )

        for engine_arm in (S_CTRL, S_SELF):
            assembly = paired_arms.arms[engine_arm]
            _require(
                _module_state_marker(assembly) == module_markers[engine_arm],
                f"{engine_arm} module tensors changed during diagnostics wiring",
            )
            _require(
                assembly.optimizer.state_dict()
                == optimizer_markers[engine_arm],
                f"{engine_arm} optimizer state changed during diagnostics wiring",
            )
        _require(
            built_arms[S_CTRL].callbacks is not built_arms[S_SELF].callbacks
            and built_arms[S_CTRL].executor is not built_arms[S_SELF].executor
            and built_arms[S_CTRL].optimizer_diagnostics
            is not built_arms[S_SELF].optimizer_diagnostics,
            "IBR1 smoke arms share callback/executor/diagnostics objects",
        )
        plan_class = (
            _IBR1ProductionSmokePlan
            if production_context
            else IBR1SmokePlan
        )
        plan = plan_class(
            project_root=str(root),
            final_assembly_receipt_path=str(receipt_path),
            final_assembly_receipt=_frozen_mapping(
                final_assembly_receipt, "final assembly receipt"
            ),
            final_assembly_receipt_binding=MappingProxyType(binding),
            paired_arms=paired_arms,
            data=data,
            geometry_collector=geometry,
            gradient_collector=gradient,
            g6=g6,
            _arms=MappingProxyType(built_arms),
            optimizer_contract=contract,
            seed=paired_arms.seed,
            device=paired_arms.device,
            checkpoint_init_sha256=paired_arms.checkpoint_init_sha256,
            cuda_reproducibility=normalized_cuda_receipt,
            base_load_report=_frozen_mapping(
                base_load_report or {}, "base load report"
            ),
        )
        _require_no_global_optimizer_hooks()
        if production_context:
            assert production_provenance is not None
            _require(
                type(plan) is _IBR1ProductionSmokePlan,
                "production builder returned the component plan class",
            )
            _register_production_plan(plan, production_provenance)
    except BaseException:
        for handle in reversed(open_handles):
            handle.close()
        raise
    return plan


def _receipt_base_hf_path(
    root: Path, document: Mapping[str, Any]
) -> Path:
    base_hf = _asset_observation(document).get("base_hf")
    _require(
        isinstance(base_hf, Mapping)
        and isinstance(base_hf.get("path"), str)
        and bool(base_hf["path"]),
        "final assembly receipt has no bound base HF path",
    )
    path = Path(base_hf["path"]).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def build_ibr1_production_smoke_plan(
    project_root: str | Path,
    final_assembly_receipt_path: str | Path,
) -> IBR1SmokePlan:
    """Build the fixed seed-0/cuda:0 IBR1 smoke plan and stop before run."""

    root = Path(project_root).expanduser().resolve()
    receipt_path = Path(final_assembly_receipt_path).expanduser().resolve()
    _require_no_global_optimizer_hooks()
    document = verify_assembly_receipt(
        root,
        receipt_path,
        required_phase=ASSEMBLY_PHASE_FINAL,
    )
    _validate_final_receipt(document)
    receipt_binding = _final_receipt_binding(root, receipt_path, document)

    # The base loader may allocate enough host/GPU memory to obscure an
    # invalid CPU fallback.  Reject the unsupported platform immediately
    # after authority verification and before loading supports or weights.
    _require(
        torch.cuda.is_available(),
        "production IBR1 smoke requires CUDA; CPU fallback is forbidden",
    )
    cuda_receipt = validate_cuda_reproducibility_receipt(
        configure_cuda_reproducibility()
    )
    data = load_ibr1_smoke_data(root, document)
    production_provenance = _issue_production_smoke_provenance(
        root,
        receipt_path,
        document,
        receipt_binding,
        data,
    )
    base, load_report = load_base_checkpoint(
        _receipt_base_hf_path(root, document)
    )
    parent_f2_evidence = read_sealed_f2_init_evidence(root)
    paired = build_ibr1_paired_arms(
        base,
        seed=IBR1_SMOKE_SEED,
        device=IBR1_SMOKE_DEVICE,
        contract=OptimizerContract(),
        parent_f2_evidence=parent_f2_evidence,
    )
    _require(
        paired.checkpoint_init_sha256
        == parent_f2_evidence.checkpoint_init_sha256,
        "production paired arms differ from sealed F2 init evidence",
    )
    return _build_ibr1_smoke_plan_from_components(
        project_root=root,
        final_assembly_receipt_path=receipt_path,
        final_assembly_receipt=document,
        final_assembly_receipt_binding=receipt_binding,
        paired_arms=paired,
        data=data,
        cuda_reproducibility=cuda_receipt,
        base_load_report=load_report,
        production_provenance=production_provenance,
    )


__all__ = [
    "IBR1CheckpointIdentityAccessor",
    "IBR1CheckpointTarget",
    "IBR1EvalPredictorBinding",
    "IBR1EvalPredictorFactory",
    "IBR1SmokeArm",
    "IBR1SmokeContractError",
    "IBR1SmokeData",
    "IBR1SmokePlan",
    "IBR1_SMOKE_DEVICE",
    "IBR1_SMOKE_EVAL_ROWS",
    "IBR1_SMOKE_SEED",
    "IBR1_SMOKE_SUPPORT_ORDER",
    "IBR1_SMOKE_TRAIN_ROWS",
    "build_ibr1_production_smoke_plan",
    "build_ibr1_smoke_plan_from_components",
    "load_ibr1_smoke_data",
]
