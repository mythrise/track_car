"""Single-use, fail-closed CAL entrypoint for the IBR1 experiment.

The frozen F2 CAL implementation remains the subordinate numeric kernel.  This
module wraps that kernel with IBR1-specific row evidence and is the only code
path allowed to emit the IBR1 numeric/core/envelope/execution-witness chain.
It deliberately exposes no standalone witness builder or signer.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import secrets
from statistics import median
import sys
import threading
import time
from types import FunctionType, ModuleType
from typing import Any, Literal

import f2_experiment.assembly as f2_assembly
import f2_experiment.assembly_model as f2_assembly_model
from f2_experiment.assembly import (
    CAL_AUDIT_RECEIPT_CLASS,
    LAMBDA_AUX_LOSSES,
    LAMBDA_SIGNIFICANT_DIGITS,
    LAMBDA_TARGET_FRACTION,
    LAMBDA_UPPER_BOUND,
    CalRowAudit,
)
from f2_experiment.model import AP2_HORIZON

from . import authority
from .model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


RAW_F2_CAL_FILENAME = "cal_audit_receipt_v1.json"
NUMERIC_EVIDENCE_FILENAME = "ibr1_cal_numeric_evidence.json"
CORE_RECEIPT_FILENAME = "ibr1_cal_core_receipt.json"
ENVELOPE_RECEIPT_FILENAME = "ibr1_cal_envelope.json"
EXECUTION_WITNESS_FILENAME = "ibr1_cal_execution_witness.json"

IBR1_CAL_ROWS = 512
IBR1_CAL_SUPPORT = "CAL"
IBR1_GEOMETRY_DTYPE = "torch.float32"
IBR1_RECONSTRUCTION_THRESHOLD = 1e-6
IBR1_CONTROLLED_AXES = 2
IBR1_CONTROLLED_SHAPE = (AP2_HORIZON, IBR1_CONTROLLED_AXES)
IBR1_CONTROLLED_CELLS_PER_ROW = AP2_HORIZON * IBR1_CONTROLLED_AXES
IBR1_CAL_CONTROLLED_CELLS = IBR1_CAL_ROWS * IBR1_CONTROLLED_CELLS_PER_ROW

_INTERPRETER_REGISTRY_NAME = (
    "_ibr1_calibration_process_registry_71b7d740e6764ef78c8436945e56f531"
)
_PRODUCTION_BINDING_REGISTRY_NAME = (
    "_ibr1_calibration_production_binding_77b5aab8e59a42f7a4973b4320cbf6e4"
)
_PRODUCTION_METHOD_BINDING_REGISTRY_NAME = (
    "_ibr1_calibration_production_method_binding_"
    "d30dcad57bd74f98a56e21e8f65a10cb"
)
TEST_ONLY_EVIDENCE_CLASS = "ibr1_cal_test_only_evidence"
TEST_ONLY_EVIDENCE_FILENAME = "ibr1_cal_test_only_evidence.json"
PARENT_CHALLENGE_ENV = "IBR1_CAL_PARENT_CHALLENGE"
PARENT_PID_ENV = "IBR1_CAL_PARENT_PID"


_TRUSTED_F2_ASSEMBLY_MODULE = f2_assembly
_TRUSTED_F2_ASSEMBLY_MODULE_FILE = Path(f2_assembly.__file__).resolve()
_TRUSTED_F2_RUNNER = f2_assembly.run_cal_audit
_TRUSTED_F2_RUNNER_CODE = _TRUSTED_F2_RUNNER.__code__
_TRUSTED_F2_MODEL_MODULE = f2_assembly_model
_TRUSTED_F2_MODEL_MODULE_FILE = Path(f2_assembly_model.__file__).resolve()
_TRUSTED_F2_AUDITOR_CLASS = f2_assembly_model.CalRowAuditor
_TRUSTED_CALIBRATION_MODEL_BINDING: tuple[
    ModuleType,
    Path,
    Callable[[Path], Any],
    Any,
    type[Any],
] | None = getattr(builtins, _PRODUCTION_BINDING_REGISTRY_NAME, None)

_DirectMethodBinding = tuple[str, FunctionType, Any, str, str]
_DirectMethodBindings = tuple[_DirectMethodBinding, ...]
_IBR1_AUDITOR_DIRECT_METHOD_NAMES = frozenset(
    {
        "__init__",
        "_assert_init_binding",
        "context_receipt",
        "_ibr1_geometry",
        "__call__",
    }
)
_F2_AUDITOR_DIRECT_METHOD_NAMES = frozenset(
    {"__init__", "context_receipt", "_probe_grad_norm", "__call__"}
)
_TRUSTED_AUDITOR_METHOD_BINDING: tuple[
    ModuleType,
    Path,
    type[Any],
    _DirectMethodBindings,
    ModuleType,
    Path,
    type[Any],
    _DirectMethodBindings,
] | None = getattr(builtins, _PRODUCTION_METHOD_BINDING_REGISTRY_NAME, None)
if (
    isinstance(_TRUSTED_AUDITOR_METHOD_BINDING, tuple)
    and len(_TRUSTED_AUDITOR_METHOD_BINDING) == 8
):
    # Preserve the first registered F2 class/module objects across lifecycle
    # reloads instead of silently trusting whatever mutable module attributes
    # happen to exist at reload time.
    _TRUSTED_F2_MODEL_MODULE = _TRUSTED_AUDITOR_METHOD_BINDING[4]
    _TRUSTED_F2_MODEL_MODULE_FILE = _TRUSTED_AUDITOR_METHOD_BINDING[5]
    _TRUSTED_F2_AUDITOR_CLASS = _TRUSTED_AUDITOR_METHOD_BINDING[6]


class IBR1CalibrationContractError(authority.IBR1AuthorityError):
    """Raised when the single-use IBR1 CAL lifecycle must stop closed."""


@dataclass(frozen=True)
class IBR1CalRowAudit:
    """One callback's subordinate F2 evidence and IBR1 geometry evidence.

    ``subordinate_audit`` is returned unchanged to the frozen F2 CAL kernel.
    The remaining fields are independently accumulated into the canonical
    IBR1 numeric evidence receipt.
    """

    subordinate_audit: CalRowAudit
    geometry_dtype: str
    zero_init_persistence: bool
    post_decode_abs_max: float
    controlled_tensor_shape: tuple[int, int]
    controlled_cells: int
    realized_delta_reconstruction_error: float
    prev_free_observation_graph: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1CalibrationContractError(message)


def _parent_orchestration_binding() -> dict[str, Any]:
    challenge = os.environ.get(PARENT_CHALLENGE_ENV)
    parent_pid_text = os.environ.get(PARENT_PID_ENV)
    if challenge is None and parent_pid_text is None:
        # Direct/unit invocation remains available for forensic receipts.  It
        # can never satisfy the live pair proof because the parent challenge is
        # deliberately absent.
        return {
            "analysis_class": "ibr1_cal_worker_parent_challenge",
            "parent_challenge": None,
            "parent_pid": None,
            "child_pid": os.getpid(),
        }
    _require(
        isinstance(challenge, str)
        and len(challenge) == 64
        and all(character in "0123456789abcdef" for character in challenge),
        "IBR1 CAL parent orchestration challenge is missing or malformed",
    )
    try:
        parent_pid = int(parent_pid_text or "")
    except ValueError as exc:
        raise IBR1CalibrationContractError(
            "IBR1 CAL parent orchestration PID is malformed"
        ) from exc
    _require(
        parent_pid > 0 and parent_pid != os.getpid(),
        "IBR1 CAL parent orchestration PID is invalid",
    )
    return {
        "analysis_class": "ibr1_cal_worker_parent_challenge",
        "parent_challenge": challenge,
        "parent_pid": parent_pid,
        "child_pid": os.getpid(),
    }


def _new_interpreter_registry(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "lock": threading.Lock(),
        "used": False,
        "process_start_token": (
            f"pid={pid};calibration_registry_ns={time.time_ns()};"
            f"nonce={secrets.token_hex(16)}"
        ),
        "module_import_token": secrets.token_hex(32),
    }


def _interpreter_registry() -> dict[str, Any]:
    """Return one PID-bound registry shared by every import alias/reload."""

    pid = os.getpid()
    namespace = vars(builtins)
    registry = namespace.get(_INTERPRETER_REGISTRY_NAME)
    if registry is None or (
        isinstance(registry, Mapping) and registry.get("pid") != pid
    ):
        candidate = _new_interpreter_registry(pid)
        namespace[_INTERPRETER_REGISTRY_NAME] = candidate
        registry = candidate
    _require(
        isinstance(registry, dict)
        and registry.get("pid") == pid
        and isinstance(registry.get("lock"), type(threading.Lock()))
        and isinstance(registry.get("used"), bool)
        and isinstance(registry.get("process_start_token"), str)
        and isinstance(registry.get("module_import_token"), str),
        "IBR1 interpreter CAL registry is malformed or PID-mismatched",
    )
    return registry


def _claim_process_session() -> dict[str, Any]:
    registry = _interpreter_registry()
    lock = registry["lock"]
    with lock:
        _require(
            registry["used"] is False,
            "IBR1 CAL is single-use per interpreter process",
        )
        registry["used"] = True
    return registry


def _root_relative(root: Path, path: Path, label: str) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise IBR1CalibrationContractError(
            f"{label} must stay inside the project root: {resolved}"
        ) from exc
    portable = relative.as_posix()
    _require(
        portable == PurePosixPath(portable).as_posix()
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} is not a clean project-relative path",
    )
    return portable


def _sha256_file(path: Path, label: str) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IBR1CalibrationContractError(f"{label} is unreadable") from exc
    _require(isinstance(document, dict), f"{label} must be a JSON object")
    expected = authority.canonical_json_bytes(document) + b"\n"
    _require(path.read_bytes() == expected, f"{label} is not canonical JSON plus LF")
    return document


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number) and number >= 0.0, f"{label} is invalid")
    return number


def _finite_number(value: Any, label: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{label} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} is nonfinite")
    return number


def _round_significant(value: float) -> float:
    _require(math.isfinite(value), "lambda proposal input is nonfinite")
    if value == 0.0:
        return 0.0
    exponent = math.floor(math.log10(abs(value)))
    return round(value, LAMBDA_SIGNIFICANT_DIGITS - 1 - exponent)


def _proposal_from_raw_medians(medians: Mapping[str, Any]) -> dict[str, float]:
    _require(
        set(medians) == set(LAMBDA_AUX_LOSSES),
        "raw CAL medians do not cover the frozen auxiliary losses",
    )
    normalized = {
        name: _finite_nonnegative(medians[name], f"raw median {name}")
        for name in LAMBDA_AUX_LOSSES
    }
    _require(
        all(value > 0.0 for value in normalized.values()),
        "raw CAL auxiliary median is zero",
    )
    minimum = min(normalized.values())
    return {
        name: _round_significant(
            min(
                LAMBDA_TARGET_FRACTION * minimum / normalized[name],
                LAMBDA_UPPER_BOUND,
            )
        )
        for name in LAMBDA_AUX_LOSSES
    }


def _validate_row_audit(value: Any, position: int) -> IBR1CalRowAudit:
    _require(
        isinstance(value, IBR1CalRowAudit),
        f"IBR1 row auditor returned the wrong type at position {position}",
    )
    subordinate = value.subordinate_audit
    _require(
        isinstance(subordinate, CalRowAudit),
        f"subordinate F2 row evidence has the wrong type at position {position}",
    )
    _require(
        subordinate.step0_parity is True,
        f"subordinate step0 persistence failed at position {position}",
    )
    _require(
        subordinate.prev_free is True and value.prev_free_observation_graph is True,
        f"prev-free observation graph failed at position {position}",
    )
    _require(
        value.geometry_dtype == IBR1_GEOMETRY_DTYPE,
        f"authoritative IBR1 CAL geometry is not FP32 at position {position}",
    )
    _require(
        value.zero_init_persistence is True,
        f"torch.equal zero-init persistence failed at position {position}",
    )
    abs_max = _finite_nonnegative(
        value.post_decode_abs_max, f"post-decode abs max[{position}]"
    )
    _require(abs_max <= 1.0, f"post-decode range failed at position {position}")
    _require(
        value.controlled_tensor_shape == IBR1_CONTROLLED_SHAPE
        and isinstance(value.controlled_cells, int)
        and not isinstance(value.controlled_cells, bool)
        and value.controlled_cells == IBR1_CONTROLLED_CELLS_PER_ROW,
        f"controlled tensor must be exactly 8 horizons x 2 axes at position {position}",
    )
    reconstruction = _finite_nonnegative(
        value.realized_delta_reconstruction_error,
        f"realized-delta reconstruction error[{position}]",
    )
    _require(
        reconstruction <= IBR1_RECONSTRUCTION_THRESHOLD,
        f"realized-delta reconstruction failed at position {position}",
    )
    _require(
        isinstance(subordinate.aux_grad_norms, Mapping)
        and set(subordinate.aux_grad_norms) == set(LAMBDA_AUX_LOSSES),
        f"auxiliary gradient evidence is malformed at position {position}",
    )
    for name in LAMBDA_AUX_LOSSES:
        _finite_nonnegative(
            subordinate.aux_grad_norms[name], f"{name} gradient norm[{position}]"
        )
    track_norm = _finite_nonnegative(
        subordinate.track_grad_norm, f"track gradient norm[{position}]"
    )
    _require(
        track_norm == 0.0,
        f"zero-update track gradient is nonzero at position {position}",
    )
    return value


def _callback_transcript_record(
    row: Any,
    reasons: Any,
    position: int,
    audit: IBR1CalRowAudit,
) -> dict[str, Any]:
    original_row_index = getattr(row, "original_row_index", None)
    sequence_id = getattr(row, "sequence_id", None)
    frame_idx = getattr(row, "frame_idx", None)
    mirrored = getattr(row, "mirrored", None)
    logged_prev_action = getattr(row, "logged_prev_action", None)
    _require(
        isinstance(original_row_index, int)
        and not isinstance(original_row_index, bool)
        and isinstance(sequence_id, str)
        and bool(sequence_id)
        and isinstance(frame_idx, int)
        and not isinstance(frame_idx, bool)
        and isinstance(mirrored, bool),
        f"CAL row identity is malformed at position {position}",
    )
    _require(
        isinstance(logged_prev_action, (list, tuple))
        and len(logged_prev_action) == 3,
        f"CAL logged previous action is malformed at position {position}",
    )
    _require(
        isinstance(reasons, (list, tuple))
        and all(isinstance(reason, str) and bool(reason) for reason in reasons),
        f"CAL reset reasons are malformed at position {position}",
    )
    subordinate = audit.subordinate_audit
    return {
        "position": position,
        "row_identity": {
            "original_row_index": original_row_index,
            "sequence_id": sequence_id,
            "frame_idx": frame_idx,
            "mirrored": mirrored,
            "logged_prev_action": [
                _finite_number(value, f"logged prev action[{position}]")
                for value in logged_prev_action
            ],
        },
        "reset_reasons": list(reasons),
        "subordinate_f2": {
            "step0_parity": subordinate.step0_parity,
            "prev_free": subordinate.prev_free,
            "track_grad_norm": float(subordinate.track_grad_norm),
            "aux_grad_norms": {
                name: float(subordinate.aux_grad_norms[name])
                for name in LAMBDA_AUX_LOSSES
            },
        },
        "ibr1": {
            "geometry_dtype": audit.geometry_dtype,
            "zero_init_persistence": audit.zero_init_persistence,
            "post_decode_abs_max": float(audit.post_decode_abs_max),
            "controlled_tensor_shape": list(audit.controlled_tensor_shape),
            "controlled_cells": audit.controlled_cells,
            "realized_delta_reconstruction_error": float(
                audit.realized_delta_reconstruction_error
            ),
            "prev_free_observation_graph": audit.prev_free_observation_graph,
        },
    }


def _seal_callback_transcript(records: list[dict[str, Any]]) -> dict[str, Any]:
    _require(
        len(records) == IBR1_CAL_ROWS,
        "callback transcript must contain exactly 512 records",
    )
    previous_sha = "0" * 64
    chained: list[dict[str, Any]] = []
    for position, record in enumerate(records):
        _require(
            record.get("position") == position,
            "callback transcript position clock drifted",
        )
        record_sha = authority.canonical_json_sha256(
            {"previous_sha256": previous_sha, "record": record}
        )
        chained.append(
            {
                **deepcopy(record),
                "previous_sha256": previous_sha,
                "record_sha256": record_sha,
            }
        )
        previous_sha = record_sha
    return {
        "schema_version": 1,
        "analysis_class": "ibr1_cal_callback_transcript",
        "rows": IBR1_CAL_ROWS,
        "chain_algorithm": (
            "sha256(canonical_json({previous_sha256,record_without_chain_fields}))"
        ),
        "initial_sha256": "0" * 64,
        "final_sha256": previous_sha,
        "records_sha256": authority.canonical_json_sha256(chained),
        "records": chained,
    }


def _bootstrap_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    _root_relative(root, path, "bootstrap receipt")
    payload_sha = document.get("receipt_payload_sha256")
    _require(
        isinstance(payload_sha, str) and len(payload_sha) == 64,
        "bootstrap receipt payload SHA is missing",
    )
    return {
        "filename": path.name,
        "sha256": _sha256_file(path, "bootstrap receipt"),
        "receipt_payload_sha256": payload_sha,
        "analysis_class": str(document.get("analysis_class")),
    }


def _self_hashed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    _require(
        "receipt_payload_sha256" not in result,
        "receipt already contains a payload self-hash",
    )
    result["receipt_payload_sha256"] = authority.canonical_json_sha256(result)
    return result


def _receipt_artifact_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    _root_relative(root, path, "CAL artifact")
    payload_sha = document.get("receipt_payload_sha256")
    _require(
        isinstance(payload_sha, str) and len(payload_sha) == 64,
        f"{path.name} has no receipt payload SHA",
    )
    return {
        "filename": path.name,
        "sha256": _sha256_file(path, "CAL artifact"),
        "receipt_payload_sha256": payload_sha,
        "analysis_class": str(document.get("analysis_class")),
    }


def _raw_artifact_binding(
    root: Path, path: Path, document: Mapping[str, Any]
) -> dict[str, str]:
    _root_relative(root, path, "raw F2 CAL artifact")
    return {
        "filename": path.name,
        "sha256": _sha256_file(path, "raw F2 CAL artifact"),
        "canonical_payload_sha256": authority.canonical_json_sha256(document),
        "analysis_class": str(document.get("analysis_class")),
    }


def _module_file(module: ModuleType, expected: Path, label: str) -> Path:
    module_name = getattr(module, "__name__", None)
    module_file = getattr(module, "__file__", None)
    spec = getattr(module, "__spec__", None)
    spec_origin = getattr(spec, "origin", None)
    _require(
        isinstance(module_name, str)
        and sys.modules.get(module_name) is module
        and isinstance(module_file, str)
        and isinstance(spec_origin, str),
        f"{label} is not bound to its imported module object",
    )
    current_file = Path(module_file).resolve()
    _require(
        current_file == expected and Path(spec_origin).resolve() == expected,
        f"{label} imported module file drifted",
    )
    return current_file


def _require_function_binding(
    actual: Any,
    *,
    expected: Any,
    expected_code: Any,
    module: ModuleType,
    module_file: Path,
    attribute: str,
    source_sha256: str,
    label: str,
) -> None:
    bound_file = _module_file(module, module_file, label)
    code = getattr(actual, "__code__", None)
    _require(
        actual is expected
        and vars(module).get(attribute) is actual
        and code is expected_code
        and getattr(actual, "__globals__", None) is vars(module)
        and isinstance(getattr(code, "co_filename", None), str)
        and Path(code.co_filename).resolve() == bound_file
        and _sha256_file(bound_file, label) == source_sha256,
        f"{label} identity drifted: real object/code/module binding changed",
    )


def _require_class_binding(
    actual: Any,
    *,
    expected: type[Any],
    module: ModuleType,
    module_file: Path,
    attribute: str,
    source_sha256: str,
    label: str,
) -> None:
    bound_file = _module_file(module, module_file, label)
    _require(
        actual is expected
        and vars(module).get(attribute) is actual
        and _sha256_file(bound_file, label) == source_sha256,
        f"{label} identity drifted: real class/module binding changed",
    )


def _capture_direct_method_bindings(
    auditor_class: type[Any],
    *,
    module: ModuleType,
    module_file: Path,
    expected_names: frozenset[str],
    label: str,
) -> _DirectMethodBindings:
    """Seal every method defined directly on one production auditor class."""

    bound_file = _module_file(module, module_file, label)
    direct_methods = {
        name: value
        for name, value in vars(auditor_class).items()
        if isinstance(value, FunctionType)
    }
    _require(
        frozenset(direct_methods) == expected_names,
        f"{label} direct method surface is malformed at registration",
    )
    bindings: list[_DirectMethodBinding] = []
    for name in sorted(expected_names):
        method = direct_methods[name]
        code = getattr(method, "__code__", None)
        module_name = getattr(method, "__module__", None)
        qualname = getattr(method, "__qualname__", None)
        _require(
            code is not None
            and module_name == module.__name__
            and qualname == f"{auditor_class.__qualname__}.{name}"
            and getattr(method, "__globals__", None) is vars(module)
            and isinstance(getattr(code, "co_filename", None), str)
            and Path(code.co_filename).resolve() == bound_file,
            f"{label}.{name} code/module/file binding is malformed at registration",
        )
        bindings.append((name, method, code, module_name, qualname))
    return tuple(bindings)


def _require_direct_method_bindings(
    actual_class: type[Any],
    *,
    expected_class: type[Any],
    bindings: _DirectMethodBindings,
    module: ModuleType,
    module_file: Path,
    expected_names: frozenset[str],
    source_sha256: str,
    label: str,
) -> None:
    """Fail closed if a registered direct method changed in memory or on disk."""

    bound_file = _module_file(module, module_file, label)
    _require(
        actual_class is expected_class
        and _sha256_file(bound_file, label) == source_sha256,
        f"{label} identity drifted: real class/module binding changed",
    )
    expected_bindings = {name: rest for name, *rest in bindings}
    current_direct_methods = {
        name: value
        for name, value in vars(actual_class).items()
        if isinstance(value, FunctionType)
    }
    _require(
        frozenset(expected_bindings) == expected_names
        and frozenset(current_direct_methods) == expected_names,
        f"{label} direct method surface drifted",
    )
    for name in sorted(expected_names):
        expected_method, expected_code, module_name, qualname = expected_bindings[name]
        actual_method = vars(actual_class).get(name)
        actual_code = getattr(actual_method, "__code__", None)
        _require(
            actual_method is expected_method
            and getattr(actual_class, name, None) is actual_method
            and actual_code is expected_code
            and getattr(actual_method, "__module__", None) == module_name
            and module_name == module.__name__
            and getattr(actual_method, "__qualname__", None) == qualname
            and qualname == f"{actual_class.__qualname__}.{name}"
            and getattr(actual_method, "__globals__", None) is vars(module)
            and isinstance(getattr(actual_code, "co_filename", None), str)
            and Path(actual_code.co_filename).resolve() == bound_file,
            f"{label}.{name} identity drifted: "
            "real method/code/module/file binding changed",
        )


def _require_no_direct_method_shadow(
    instance: Any,
    *,
    bindings: _DirectMethodBindings,
    label: str,
) -> None:
    namespace = getattr(instance, "__dict__", None)
    _require(isinstance(namespace, dict), f"{label} instance namespace is malformed")
    method_names = {binding[0] for binding in bindings}
    _require(
        method_names.isdisjoint(namespace),
        f"{label} instance shadows a registered direct method",
    )


def _bind_production_calibration_model_components(
    module: ModuleType,
    auditor_factory: Callable[[Path], Any],
    auditor_class: type[Any],
) -> None:
    """Capture model-side production objects once their circular import completes."""

    global _TRUSTED_AUDITOR_METHOD_BINDING, _TRUSTED_CALIBRATION_MODEL_BINDING
    _require(
        _TRUSTED_CALIBRATION_MODEL_BINDING is None
        and _TRUSTED_AUDITOR_METHOD_BINDING is None,
        "IBR1 CAL production model components were registered twice",
    )
    module_file_text = getattr(module, "__file__", None)
    _require(
        module is sys.modules.get("ibr1_experiment.calibration_model")
        and isinstance(module_file_text, str)
        and vars(module).get("build_ibr1_cal_row_auditor") is auditor_factory
        and vars(module).get("IBR1ModelCalRowAuditor") is auditor_class
        and callable(auditor_factory)
        and isinstance(auditor_class, type),
        "IBR1 CAL production model component registration is malformed",
    )
    code = getattr(auditor_factory, "__code__", None)
    module_file = Path(module_file_text).resolve()
    _require(
        code is not None
        and getattr(auditor_factory, "__globals__", None) is vars(module)
        and Path(code.co_filename).resolve() == module_file,
        "IBR1 CAL production factory code/module binding is malformed",
    )
    ibr1_method_bindings = _capture_direct_method_bindings(
        auditor_class,
        module=module,
        module_file=module_file,
        expected_names=_IBR1_AUDITOR_DIRECT_METHOD_NAMES,
        label="production IBR1 CAL auditor class",
    )
    f2_method_bindings = _capture_direct_method_bindings(
        _TRUSTED_F2_AUDITOR_CLASS,
        module=_TRUSTED_F2_MODEL_MODULE,
        module_file=_TRUSTED_F2_MODEL_MODULE_FILE,
        expected_names=_F2_AUDITOR_DIRECT_METHOD_NAMES,
        label="production F2 CAL auditor class",
    )
    _TRUSTED_CALIBRATION_MODEL_BINDING = (
        module,
        module_file,
        auditor_factory,
        code,
        auditor_class,
    )
    _TRUSTED_AUDITOR_METHOD_BINDING = (
        module,
        module_file,
        auditor_class,
        ibr1_method_bindings,
        _TRUSTED_F2_MODEL_MODULE,
        _TRUSTED_F2_MODEL_MODULE_FILE,
        _TRUSTED_F2_AUDITOR_CLASS,
        f2_method_bindings,
    )
    setattr(
        builtins,
        _PRODUCTION_BINDING_REGISTRY_NAME,
        _TRUSTED_CALIBRATION_MODEL_BINDING,
    )
    setattr(
        builtins,
        _PRODUCTION_METHOD_BINDING_REGISTRY_NAME,
        _TRUSTED_AUDITOR_METHOD_BINDING,
    )


def _bound_source_sha(
    source_binding: Mapping[str, Any],
    *,
    group: str,
    path: str,
    label: str,
) -> str:
    bindings = source_binding.get(group)
    value = bindings.get(path) if isinstance(bindings, Mapping) else None
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"live bootstrap source binding lacks {label}",
    )
    return value


def _resolve_production_components(
    source_binding: Mapping[str, Any],
) -> tuple[Callable[..., Mapping[str, Any]], Callable[[Path], Any], type[Any], dict[str, Any]]:
    """Resolve and identity-lock the only authority-producing components."""

    from . import calibration_model

    binding = _TRUSTED_CALIBRATION_MODEL_BINDING
    method_binding = _TRUSTED_AUDITOR_METHOD_BINDING
    _require(
        binding is not None and binding[0] is calibration_model,
        "IBR1 CAL production model components were not identity-bound at import",
    )
    _require(
        isinstance(method_binding, tuple)
        and len(method_binding) == 8
        and method_binding[0] is calibration_model,
        "IBR1 CAL production auditor methods were not identity-bound at import",
    )
    (
        calibration_model_module,
        calibration_model_file,
        trusted_auditor_factory,
        trusted_auditor_factory_code,
        trusted_auditor_class,
    ) = binding
    (
        method_calibration_model_module,
        method_calibration_model_file,
        method_auditor_class,
        trusted_ibr1_method_bindings,
        method_f2_model_module,
        method_f2_model_file,
        method_f2_auditor_class,
        trusted_f2_method_bindings,
    ) = method_binding
    _require(
        method_calibration_model_module is calibration_model_module
        and method_calibration_model_file == calibration_model_file
        and method_auditor_class is trusted_auditor_class
        and method_f2_model_module is _TRUSTED_F2_MODEL_MODULE
        and method_f2_model_file == _TRUSTED_F2_MODEL_MODULE_FILE
        and method_f2_auditor_class is _TRUSTED_F2_AUDITOR_CLASS,
        "IBR1 CAL registered class/method authority is internally inconsistent",
    )
    subordinate_runner = f2_assembly.run_cal_audit
    auditor_factory = calibration_model.build_ibr1_cal_row_auditor
    auditor_class = calibration_model.IBR1ModelCalRowAuditor
    f2_assembly_sha = _bound_source_sha(
        source_binding,
        group="inherited_f2_source_sha256",
        path="f2_experiment/assembly.py",
        label="F2 CAL lifecycle source",
    )
    f2_model_sha = _bound_source_sha(
        source_binding,
        group="inherited_f2_source_sha256",
        path="f2_experiment/assembly_model.py",
        label="F2 model-side CAL source",
    )
    calibration_model_sha = _bound_source_sha(
        source_binding,
        group="ibr1_source_sha256",
        path="ibr1_experiment/calibration_model.py",
        label="IBR1 calibration-model source",
    )
    _require_function_binding(
        subordinate_runner,
        expected=_TRUSTED_F2_RUNNER,
        expected_code=_TRUSTED_F2_RUNNER_CODE,
        module=_TRUSTED_F2_ASSEMBLY_MODULE,
        module_file=_TRUSTED_F2_ASSEMBLY_MODULE_FILE,
        attribute="run_cal_audit",
        source_sha256=f2_assembly_sha,
        label="production subordinate CAL kernel",
    )
    _require_function_binding(
        auditor_factory,
        expected=trusted_auditor_factory,
        expected_code=trusted_auditor_factory_code,
        module=calibration_model_module,
        module_file=calibration_model_file,
        attribute="build_ibr1_cal_row_auditor",
        source_sha256=calibration_model_sha,
        label="production IBR1 CAL auditor factory",
    )
    _require_class_binding(
        auditor_class,
        expected=trusted_auditor_class,
        module=calibration_model_module,
        module_file=calibration_model_file,
        attribute="IBR1ModelCalRowAuditor",
        source_sha256=calibration_model_sha,
        label="production IBR1 CAL auditor class",
    )
    _require_direct_method_bindings(
        auditor_class,
        expected_class=method_auditor_class,
        bindings=trusted_ibr1_method_bindings,
        module=method_calibration_model_module,
        module_file=method_calibration_model_file,
        expected_names=_IBR1_AUDITOR_DIRECT_METHOD_NAMES,
        source_sha256=calibration_model_sha,
        label="production IBR1 CAL auditor class",
    )
    _require_class_binding(
        f2_assembly_model.CalRowAuditor,
        expected=method_f2_auditor_class,
        module=method_f2_model_module,
        module_file=method_f2_model_file,
        attribute="CalRowAuditor",
        source_sha256=f2_model_sha,
        label="production F2 CAL auditor class",
    )
    _require_direct_method_bindings(
        f2_assembly_model.CalRowAuditor,
        expected_class=method_f2_auditor_class,
        bindings=trusted_f2_method_bindings,
        module=method_f2_model_module,
        module_file=method_f2_model_file,
        expected_names=_F2_AUDITOR_DIRECT_METHOD_NAMES,
        source_sha256=f2_model_sha,
        label="production F2 CAL auditor class",
    )
    bindings = {
        "subordinate_kernel": {
            "callable": "f2_experiment.assembly.run_cal_audit",
            "source_path": "f2_experiment/assembly.py",
            "source_sha256": f2_assembly_sha,
        },
        "f2_model_kernel": {
            "class": "f2_experiment.assembly_model.CalRowAuditor",
            "source_path": "f2_experiment/assembly_model.py",
            "source_sha256": f2_model_sha,
        },
        "row_auditor_factory": {
            "callable": (
                "ibr1_experiment.calibration_model.build_ibr1_cal_row_auditor"
            ),
            "source_path": "ibr1_experiment/calibration_model.py",
            "source_sha256": calibration_model_sha,
        },
        "row_auditor": {
            "class": "ibr1_experiment.calibration_model.IBR1ModelCalRowAuditor",
        },
        "ibr1_assembly": {
            "source_path": "ibr1_experiment/assembly_model.py",
            "source_sha256": _bound_source_sha(
                source_binding,
                group="ibr1_source_sha256",
                path="ibr1_experiment/assembly_model.py",
                label="IBR1 assembly-model source",
            ),
        },
    }
    return subordinate_runner, auditor_factory, auditor_class, bindings


def run_ibr1_cal_audit_once(
    project_root: str | Path,
    *,
    role: Literal["main", "reproduction"],
    bootstrap_receipt_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run exactly one controlled IBR1 CAL audit in this Python process.

    The production F2 kernel and IBR1 model-side auditor are resolved inside
    this function from their frozen qualified identities.  Numeric evidence is
    emitted only after 512 ordered callbacks and all IBR1 checks pass.  The
    execution witness is always the final artifact.
    """

    registry = _claim_process_session()
    started_wall_ns = time.time_ns()
    _require(role in ("main", "reproduction"), "IBR1 CAL role is invalid")
    orchestration_binding = _parent_orchestration_binding()

    root = Path(project_root).expanduser().resolve()
    bootstrap_path = Path(bootstrap_receipt_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    _root_relative(root, bootstrap_path, "bootstrap receipt")
    _root_relative(root, output, "CAL output directory")

    bootstrap_before = authority.verify_assembly_receipt(
        root,
        bootstrap_path,
        required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
    )
    _require(
        isinstance(bootstrap_before, Mapping),
        "bootstrap verifier returned no receipt",
    )
    source_before = deepcopy(bootstrap_before.get("source_binding"))
    _require(isinstance(source_before, Mapping), "bootstrap source binding is missing")
    ibr1_source_sha = source_before.get("ibr1_source_sha256")
    _require(
        isinstance(ibr1_source_sha, Mapping)
        and isinstance(ibr1_source_sha.get("ibr1_experiment/calibration.py"), str),
        "bootstrap does not bind the IBR1 calibration source",
    )
    (
        subordinate_runner,
        auditor_factory,
        auditor_class,
        production_bindings,
    ) = _resolve_production_components(source_before)
    bootstrap_binding = _bootstrap_binding(root, bootstrap_path, bootstrap_before)

    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise IBR1CalibrationContractError(
            "CAL output directory already exists; a fresh exclusive directory is required"
        ) from exc
    _require(not any(output.iterdir()), "fresh CAL output directory is not empty")

    ibr1_row_auditor = auditor_factory(root)
    _require(
        type(ibr1_row_auditor) is auditor_class,
        "production IBR1 CAL factory returned the wrong auditor class",
    )
    _require(
        type(ibr1_row_auditor.subordinate_auditor) is _TRUSTED_F2_AUDITOR_CLASS,
        "production IBR1 CAL factory returned a non-F2 subordinate auditor",
    )
    method_binding = _TRUSTED_AUDITOR_METHOD_BINDING
    _require(
        isinstance(method_binding, tuple) and len(method_binding) == 8,
        "production CAL auditor method binding disappeared after construction",
    )
    trusted_ibr1_method_bindings = method_binding[3]
    trusted_f2_method_bindings = method_binding[7]
    _require_no_direct_method_shadow(
        ibr1_row_auditor,
        bindings=trusted_ibr1_method_bindings,
        label="production IBR1 CAL auditor",
    )
    _require_no_direct_method_shadow(
        ibr1_row_auditor.subordinate_auditor,
        bindings=trusted_f2_method_bindings,
        label="production F2 CAL auditor",
    )
    trusted_ibr1_methods = {
        binding[0]: binding[1] for binding in trusted_ibr1_method_bindings
    }
    context_provider = trusted_ibr1_methods["context_receipt"].__get__(
        ibr1_row_auditor,
        auditor_class,
    )
    _require(
        callable(context_provider)
        and getattr(context_provider, "__self__", None) is ibr1_row_auditor
        and getattr(context_provider, "__func__", None)
        is trusted_ibr1_methods["context_receipt"],
        "production IBR1 row auditor context_receipt binding drifted",
    )

    callback_positions: list[int] = []
    row_audits: list[IBR1CalRowAudit] = []
    callback_records: list[dict[str, Any]] = []
    aux_norms: dict[str, list[float]] = {name: [] for name in LAMBDA_AUX_LOSSES}

    def verified_bootstrap_for_subordinate(
        observed_root: str | Path, observed_path: str | Path
    ) -> dict[str, Any]:
        _require(
            Path(observed_root).expanduser().resolve() == root
            and Path(observed_path).expanduser().resolve() == bootstrap_path,
            "subordinate CAL requested a different bootstrap authority",
        )
        return deepcopy(dict(bootstrap_before))

    def audited_row_callback(row: Any, reasons: Any, position: int) -> CalRowAudit:
        expected_position = len(callback_positions)
        _require(
            isinstance(position, int)
            and not isinstance(position, bool)
            and position == expected_position
            and position < IBR1_CAL_ROWS,
            "subordinate CAL callback clock is not ordered 0..511 exactly",
        )
        audit = _validate_row_audit(
            trusted_ibr1_methods["__call__"](
                ibr1_row_auditor,
                row,
                reasons,
                position,
            ),
            position,
        )
        callback_positions.append(position)
        row_audits.append(audit)
        callback_records.append(
            _callback_transcript_record(row, reasons, position, audit)
        )
        for name in LAMBDA_AUX_LOSSES:
            aux_norms[name].append(float(audit.subordinate_audit.aux_grad_norms[name]))
        return audit.subordinate_audit

    runner_result = subordinate_runner(
        root,
        receipt_path=bootstrap_path,
        output_dir=output,
        row_auditor=audited_row_callback,
        cal_context_provider=context_provider,
        verifier=verified_bootstrap_for_subordinate,
    )
    _require(isinstance(runner_result, Mapping), "subordinate CAL returned no result")
    _require(
        callback_positions == list(range(IBR1_CAL_ROWS))
        and len(row_audits) == IBR1_CAL_ROWS,
        "subordinate CAL did not execute exactly 512 ordered row callbacks",
    )
    callback_transcript = _seal_callback_transcript(callback_records)

    raw_path = output / RAW_F2_CAL_FILENAME
    returned_path = runner_result.get("path")
    _require(
        isinstance(returned_path, (str, Path))
        and Path(returned_path).expanduser().resolve() == raw_path,
        "subordinate CAL returned an unexpected raw receipt path",
    )
    _require(
        {path.name for path in output.iterdir()} == {RAW_F2_CAL_FILENAME},
        "subordinate CAL output contains prebuilt or unexpected artifacts",
    )
    raw = _load_canonical_json(raw_path, "raw F2 CAL receipt")
    _require(
        raw.get("analysis_class") == CAL_AUDIT_RECEIPT_CLASS
        and raw.get("support") == IBR1_CAL_SUPPORT
        and raw.get("rows") == IBR1_CAL_ROWS
        and raw.get("optimizer_updates") == 0
        and raw.get("internal_test") == "sealed"
        and raw.get("internal_test_opened") is False,
        "raw subordinate receipt is not the frozen zero-update F2 CAL kernel",
    )
    _require(
        raw.get("assembly_receipt_sha256") == bootstrap_binding["sha256"]
        and raw.get("assembly_receipt_payload_sha256")
        == bootstrap_binding["receipt_payload_sha256"],
        "raw subordinate receipt does not bind this IBR1 bootstrap",
    )

    raw_gradient = raw.get("gradient_calibration")
    _require(isinstance(raw_gradient, Mapping), "raw CAL gradient evidence is missing")
    raw_medians = raw_gradient.get("per_aux_grad_norm_median")
    _require(isinstance(raw_medians, Mapping), "raw CAL medians are missing")
    callback_medians = {
        name: float(median(aux_norms[name])) for name in LAMBDA_AUX_LOSSES
    }
    _require(
        dict(raw_medians) == callback_medians,
        "raw CAL medians differ from the 512 observed callback rows",
    )
    proposal = _proposal_from_raw_medians(raw_medians)
    raw_lambda = raw.get("lambda_calibration")
    _require(isinstance(raw_lambda, Mapping), "raw CAL lambda evidence is missing")
    _require(
        raw_lambda.get("proposed_lambda") == proposal,
        "raw CAL proposal differs from its medians",
    )
    _require(
        proposal == dict(authority.FROZEN_AUX_COEFFICIENTS),
        "IBR1 inherited lambda proposal drifted: STOP_NO_SMOKE_NO_LAMBDA_CHANGE",
    )

    bootstrap_after = authority.verify_assembly_receipt(
        root,
        bootstrap_path,
        required_phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
    )
    _require(
        bootstrap_after == bootstrap_before
        and bootstrap_after.get("source_binding") == source_before,
        "IBR1 source/bootstrap authority drifted during CAL",
    )

    post_decode_abs_max = max(audit.post_decode_abs_max for audit in row_audits)
    reconstruction_error_max = max(
        audit.realized_delta_reconstruction_error for audit in row_audits
    )
    numeric = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": authority.CAL_NUMERIC_EVIDENCE_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "support": IBR1_CAL_SUPPORT,
            "rows": IBR1_CAL_ROWS,
            "optimizer_updates": 0,
            "geometry_dtype": IBR1_GEOMETRY_DTYPE,
            "bootstrap_binding": bootstrap_binding,
            "source_binding": deepcopy(dict(source_before)),
            "cal_context": deepcopy(raw.get("cal_context")),
            "zero_init_persistence": {
                "checked_rows": IBR1_CAL_ROWS,
                "checked_cells": IBR1_CAL_CONTROLLED_CELLS,
                "per_row_shape": list(IBR1_CONTROLLED_SHAPE),
                "failures": 0,
                "contract": "torch.equal_same_dtype_device",
            },
            "post_decode_range": {
                "checked_rows": IBR1_CAL_ROWS,
                "checked_cells": IBR1_CAL_CONTROLLED_CELLS,
                "per_row_shape": list(IBR1_CONTROLLED_SHAPE),
                "violations": 0,
                "abs_max": float(post_decode_abs_max),
            },
            "realized_delta_reconstruction": {
                "checked_rows": IBR1_CAL_ROWS,
                "checked_cells": IBR1_CAL_CONTROLLED_CELLS,
                "per_row_shape": list(IBR1_CONTROLLED_SHAPE),
                "failures": 0,
                "error_max": float(reconstruction_error_max),
                "threshold": IBR1_RECONSTRUCTION_THRESHOLD,
            },
            "prev_free_observation_graph": {
                "checked_rows": IBR1_CAL_ROWS,
                "failures": 0,
            },
            "auxiliary_reachability": {
                "checked_rows": IBR1_CAL_ROWS,
                "failures": 0,
                "per_aux_grad_norm_median": callback_medians,
            },
            "callback_transcript": callback_transcript,
            "row_callback_count": IBR1_CAL_ROWS,
            "lambda_proposal": proposal,
            "proposal_role": "identity_no_drift_audit_not_coefficient_selection",
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    numeric_path = output / NUMERIC_EVIDENCE_FILENAME
    authority.exclusive_write_json(numeric_path, numeric)

    core = authority.build_cal_core_receipt(
        root,
        bootstrap_receipt_path=bootstrap_path,
        raw_f2_kernel_receipt_path=raw_path,
        numeric_evidence_receipt_path=numeric_path,
    )
    core_path = output / CORE_RECEIPT_FILENAME
    authority.exclusive_write_json(core_path, core)
    envelope = authority.build_cal_envelope(
        root,
        core_receipt_path=core_path,
        bootstrap_receipt_path=bootstrap_path,
    )
    envelope_path = output / ENVELOPE_RECEIPT_FILENAME
    authority.exclusive_write_json(envelope_path, envelope)

    _require(
        {path.name for path in output.iterdir()}
        == {
            RAW_F2_CAL_FILENAME,
            NUMERIC_EVIDENCE_FILENAME,
            CORE_RECEIPT_FILENAME,
            ENVELOPE_RECEIPT_FILENAME,
        },
        "CAL artifacts changed before execution-witness sealing",
    )
    ended_wall_ns = time.time_ns()
    while ended_wall_ns <= started_wall_ns:
        ended_wall_ns = time.time_ns()
    production_bindings = deepcopy(production_bindings)
    production_bindings["row_auditor"]["context_callable"] = (
        "ibr1_experiment.calibration_model.IBR1ModelCalRowAuditor.context_receipt"
    )
    production_bindings["actual_context"] = deepcopy(raw.get("cal_context"))
    witness = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": authority.CAL_EXECUTION_WITNESS_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "role": role,
            "process_identity": {
                "pid": registry["pid"],
                "process_start_token": registry["process_start_token"],
                "module_import_token": registry["module_import_token"],
            },
            "orchestration_binding": orchestration_binding,
            "audit_clock": {
                "started_ns": started_wall_ns,
                "ended_ns": ended_wall_ns,
                "callback_count": IBR1_CAL_ROWS,
                "first_position": 0,
                "last_position": IBR1_CAL_ROWS - 1,
                "ordered_positions_sha256": authority.canonical_json_sha256(
                    callback_positions
                ),
            },
            "bootstrap_binding": bootstrap_binding,
            "runner_binding": {
                "entrypoint": "run_ibr1_cal_audit_once",
                "source_path": "ibr1_experiment/calibration.py",
                "source_sha256": ibr1_source_sha[
                    "ibr1_experiment/calibration.py"
                ],
            },
            "production_bindings": production_bindings,
            "callback_transcript_binding": {
                "container_analysis_class": authority.CAL_NUMERIC_EVIDENCE_CLASS,
                "container_receipt_payload_sha256": numeric[
                    "receipt_payload_sha256"
                ],
                "analysis_class": callback_transcript["analysis_class"],
                "rows": callback_transcript["rows"],
                "records_sha256": callback_transcript["records_sha256"],
                "final_sha256": callback_transcript["final_sha256"],
            },
            "artifacts": {
                "raw_f2_kernel": _raw_artifact_binding(root, raw_path, raw),
                "numeric_evidence": _receipt_artifact_binding(
                    root, numeric_path, numeric
                ),
                "core": _receipt_artifact_binding(root, core_path, core),
                "envelope": _receipt_artifact_binding(
                    root, envelope_path, envelope
                ),
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    witness_path = output / EXECUTION_WITNESS_FILENAME
    authority.exclusive_write_json(witness_path, witness)

    return {
        "role": role,
        "orchestration_binding": orchestration_binding,
        "raw_f2_kernel": _raw_artifact_binding(root, raw_path, raw),
        "numeric_evidence": _receipt_artifact_binding(root, numeric_path, numeric),
        "core": _receipt_artifact_binding(root, core_path, core),
        "envelope": _receipt_artifact_binding(root, envelope_path, envelope),
        "execution_witness": _receipt_artifact_binding(
            root, witness_path, witness
        ),
        "formal_training_authorized": False,
    }


def _run_ibr1_cal_audit_test_only(
    project_root: str | Path,
    *,
    output_dir: str | Path,
    row_driver: Callable[[Callable[[Any, Any, int], CalRowAudit]], Any],
    row_auditor: Callable[[Any, Any, int], IBR1CalRowAudit],
) -> dict[str, Any]:
    """Exercise callback validation without creating any CAL authority layer.

    This private seam exists only for dummy/unit tests.  Its sole artifact is
    explicitly test-only and is rejected by every production authority
    verifier.  It cannot emit raw F2, numeric, core, envelope, or witness
    receipt classes.
    """

    _claim_process_session()
    _require(callable(row_driver), "test-only row driver is not callable")
    _require(callable(row_auditor), "test-only row auditor is not callable")
    root = Path(project_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    _root_relative(root, output, "test-only CAL output directory")
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise IBR1CalibrationContractError(
            "test-only CAL output directory already exists"
        ) from exc

    positions: list[int] = []
    records: list[dict[str, Any]] = []
    aux_norms: dict[str, list[float]] = {name: [] for name in LAMBDA_AUX_LOSSES}

    def callback(row: Any, reasons: Any, position: int) -> CalRowAudit:
        _require(
            isinstance(position, int)
            and not isinstance(position, bool)
            and position == len(positions)
            and position < IBR1_CAL_ROWS,
            "test-only callback clock is not ordered 0..511 exactly",
        )
        audit = _validate_row_audit(row_auditor(row, reasons, position), position)
        positions.append(position)
        records.append(_callback_transcript_record(row, reasons, position, audit))
        for name in LAMBDA_AUX_LOSSES:
            aux_norms[name].append(float(audit.subordinate_audit.aux_grad_norms[name]))
        return audit.subordinate_audit

    row_driver(callback)
    _require(
        positions == list(range(IBR1_CAL_ROWS)),
        "test-only row driver did not execute exactly 512 callbacks",
    )
    transcript = _seal_callback_transcript(records)
    medians = {name: float(median(aux_norms[name])) for name in LAMBDA_AUX_LOSSES}
    proposal = _proposal_from_raw_medians(medians)
    _require(
        proposal == dict(authority.FROZEN_AUX_COEFFICIENTS),
        "test-only lambda proposal drifted",
    )
    receipt = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": TEST_ONLY_EVIDENCE_CLASS,
            "test_only": True,
            "authority_eligible": False,
            "execution_witness_emitted": False,
            "rows": IBR1_CAL_ROWS,
            "geometry_dtype": IBR1_GEOMETRY_DTYPE,
            "controlled_cells": IBR1_CAL_CONTROLLED_CELLS,
            "per_aux_grad_norm_median": medians,
            "lambda_proposal": proposal,
            "callback_transcript": transcript,
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    receipt_path = output / TEST_ONLY_EVIDENCE_FILENAME
    authority.exclusive_write_json(receipt_path, receipt)
    _require(
        {path.name for path in output.iterdir()} == {TEST_ONLY_EVIDENCE_FILENAME},
        "test-only seam emitted a forbidden authority artifact",
    )
    return {
        "analysis_class": TEST_ONLY_EVIDENCE_CLASS,
        "path": str(receipt_path),
        "sha256": _sha256_file(receipt_path, "test-only CAL evidence"),
        "receipt_payload_sha256": receipt["receipt_payload_sha256"],
        "authority_eligible": False,
        "formal_training_authorized": False,
    }


__all__ = [
    "CORE_RECEIPT_FILENAME",
    "ENVELOPE_RECEIPT_FILENAME",
    "EXECUTION_WITNESS_FILENAME",
    "IBR1CalRowAudit",
    "IBR1CalibrationContractError",
    "NUMERIC_EVIDENCE_FILENAME",
    "RAW_F2_CAL_FILENAME",
    "TEST_ONLY_EVIDENCE_CLASS",
    "TEST_ONLY_EVIDENCE_FILENAME",
    "run_ibr1_cal_audit_once",
]
