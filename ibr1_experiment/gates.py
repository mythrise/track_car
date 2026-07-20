"""Fail-closed IBR1 mechanism gates and result seals.

This module is intentionally a consumer of already-produced telemetry.  It
does not run CAL, training, EVAL, CUDA code, or the sealed internal test.  The
six IBR1 gates are kept separate from the frozen F2 evaluator so that the
IBR1-specific cardinality, arm-pairing, and authority checks cannot silently
inherit a weaker contract.

The public functions accept ordinary JSON-like mappings in addition to the
dataclasses emitted by :mod:`f2_experiment.evaluation`.  Every receipt is
canonical, finite, self-hashed, and carries the permanent ``formal=false`` /
``internal_test=sealed`` policy.  Seal readers repeat those checks and bind
artifact bytes, making a post-hoc edit fail closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
import struct
from typing import Any

from f2_experiment.evaluation import (
    G6Update,
    G7Update,
    GateReceipt as F2GateReceipt,
    evaluate_g6,
    evaluate_g7,
    evaluate_g8,
    evaluate_g9,
)
from f2_experiment.assembly import EVAL_MODE_CONTRACT
from f2_experiment.controller import DEFAULT_CONFIG
from f2_experiment.model import ARCHITECTURE_LOCK as F2_ARCHITECTURE_LOCK
from f2_experiment.support import SUPPORT_EXPECTATIONS

from .assembly_model import (
    FAMILY_TO_ENGINE_ARM,
    IBR1_AUX_COMPONENTS,
    IBR1_CTRL,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    IBR1_SELF,
)
from .artifacts import (
    DIAGNOSTICS_MANIFEST_FILENAME,
    DIAGNOSTICS_SUMMARY_FILENAME,
    EVAL_GEOMETRY_FILENAME,
    EXPECTED_EVAL_RECORDS,
    EXPECTED_GRADIENT_RECORDS,
    EXPECTED_OPTIMIZER_RECORDS,
    EXPECTED_TRAINING_RECORDS,
    GRADIENT_GEOMETRY_FILENAME,
    OPTIMIZER_GEOMETRY_FILENAME,
    TRAINING_GEOMETRY_FILENAME,
)
from .authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    CAL_NUMERIC_EVIDENCE_CLASS,
    canonical_json_bytes,
    canonical_json_sha256,
)
from .checkpoint import CHECKPOINT_SIDECAR_CLASS
from .diagnostics import DIAGNOSTIC_EPS, GeometryCollector
from .eval_guard import (
    EVAL_GUARD_RECEIPT_CLASS,
    FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
    FROZEN_EVAL_ROWS,
    IBR1_EVAL_PHASES,
)
from .model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


I1_GATE_ID = "I1"
I2_GATE_ID = "I2"
I3_GATE_ID = "I3"
I4_GATE_ID = "I4"
I5_GATE_ID = "I5"
I6_GATE_ID = "I6"
IBR1_GATE_IDS = (I1_GATE_ID, I2_GATE_ID, I3_GATE_ID, I4_GATE_ID, I5_GATE_ID, I6_GATE_ID)

IBR1_GATE_RECEIPT_CLASS = "ibr1_preformal_mechanism_gate"
IBR1_COMBINED_GATE_RECEIPT_CLASS = "ibr1_preformal_mechanism_gates"
IBR1_CANDIDATE_LOCK_CLASS = "ibr1_candidate_lock_receipt"
IBR1_PASS_SEAL_CLASS = "ibr1_authoritative_smoke_pass_seal"
IBR1_NEGATIVE_SEAL_CLASS = "ibr1_authoritative_smoke_negative_result_seal"
IBR1_SEAL_SCHEMA_VERSION = 1
F2_COUNT_RECEIPT_CLASS = "f2_paired_runner_count_receipt"
F2_EVAL_PHASE_RECEIPT_CLASS = "f2_eval_fix_snapshot_receipt"

# These values are protocol constants, not fields supplied by a runner.  Keep
# private copies of the frozen F2 sources so a receipt cannot redefine the
# reset accounting or silently introduce a new controller/EVAL contract.
_FROZEN_SMOKE_STATIC_RESETS = SUPPORT_EXPECTATIONS["SMK-TRAIN"].static_resets
_FROZEN_EVAL_STATIC_RESETS = SUPPORT_EXPECTATIONS["EVAL-FIX"].static_resets
_FROZEN_CONTROLLER_CONFIG = deepcopy(DEFAULT_CONFIG.to_dict())
_FROZEN_EVAL_MODE_CONTRACT = deepcopy(EVAL_MODE_CONTRACT)

# The preregistered EVAL-FIX raw-row strata, expressed as inclusive position
# ranges in the frozen 512-row order.  These values are derived from the
# immutable train JSONL and the frozen EVAL-FIX support selection; result-seal
# construction deliberately uses this code-owned mapping instead of trusting
# the per-phase display summary.
_FROZEN_EVAL_STRATUM_POSITION_RANGES = {
    "change": (
        (11, 11), (17, 17), (29, 29), (68, 68), (70, 70), (78, 78),
        (80, 80), (88, 89), (101, 101), (103, 103), (127, 127),
        (135, 135), (137, 137), (145, 146), (154, 154), (156, 156),
        (160, 160), (186, 188), (190, 190), (199, 200), (207, 207),
        (209, 209), (217, 217), (223, 223), (227, 227), (231, 231),
        (236, 236), (238, 238), (246, 246), (248, 248), (250, 251),
        (261, 262), (295, 295), (300, 300), (308, 308), (310, 310),
        (318, 318), (347, 347), (349, 349), (391, 391), (393, 393),
        (398, 398), (401, 401), (409, 410), (419, 420), (433, 433),
        (435, 435), (460, 461), (469, 470), (472, 473), (477, 477),
        (487, 487), (490, 490), (498, 498), (501, 501), (509, 510),
    ),
    "turn": (
        (4, 9), (12, 17), (22, 26), (30, 63), (69, 72), (79, 81),
        (89, 91), (102, 103), (120, 120), (128, 129), (136, 138),
        (146, 148), (155, 157), (160, 160), (179, 179), (187, 187),
        (192, 192), (201, 201), (208, 215), (218, 220), (228, 228),
        (237, 240), (251, 253), (262, 262), (288, 292), (296, 302),
        (309, 312), (340, 341), (348, 349), (384, 385), (399, 402),
        (410, 411), (420, 420), (426, 427), (434, 435), (453, 453),
        (461, 462), (474, 475), (478, 482), (488, 493), (499, 502),
        (511, 511),
    ),
    "other": (
        (10, 11), (27, 29), (64, 68), (73, 78), (82, 88), (92, 101),
        (121, 127), (130, 135), (139, 145), (149, 154), (158, 159),
        (180, 186), (193, 200), (202, 207), (216, 217), (221, 227),
        (229, 236), (241, 250), (254, 261), (293, 295), (303, 308),
        (313, 319), (342, 347), (386, 398), (403, 409), (412, 419),
        (428, 433), (454, 460), (463, 473), (476, 477), (483, 487),
        (494, 498), (503, 510),
    ),
}
_FROZEN_EVAL_STRATUM_COUNTS = {
    "overall": FROZEN_EVAL_ROWS,
    "change": 69,
    "turn": 154,
    "other": 211,
}

I1_CAL_ROWS = 512
I1_CAL_CELLS = 8192
I1_CONTROLLED_SHAPE = (8, 2)
I1_RECONSTRUCTION_MAX = 1e-6
I2_ROWS_PER_ARM = 256
I2_RATE_MAX_EXCLUSIVE = 0.05
I3_GRADIENT_UPDATES = 128
I3_GRAD_ACCUM = 2
I3_AUX_REACHABLE_MIN = 127
I3_TRACK_REACHABLE_MIN = 120
I3_COSINE_MEDIAN_MIN = 0.6
I3_POSITIVE_PROJECTION_MIN = 108
I3_RATIO_MAX = 1.5
I6_RANGE_OBSERVATIONS = 2048
I6_RECONSTRUCTION_ROWS = 256
I6_RECONSTRUCTION_MAX = 1e-6

_LIFECYCLE_BINDING_KEYS = {
    "checkpoint_identity",
    "eval_order_guard_receipt",
    "final_assembly_receipt",
    "predictor_identity",
    "u_pre_identity",
}

_SIDECAR_KEYS = {
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
    "analysis_class",
    "checkpoint_file",
    "checkpoint_file_sha256",
}

_GATE_CHECK_NAMES: dict[str, frozenset[str]] = {
    "I1": frozenset(
        {
            "final_assembly_authority",
            "cal_rows",
            "cal_fp32",
            "zero_init",
            "post_decode_range",
            "reconstruction",
            "prev_free_observation",
            "update0_checkpoint_tensor_identity",
        }
    ),
    "I2": frozenset(
        {
            f"{arm}.{suffix}"
            for arm in (IBR1_CTRL, IBR1_SELF)
            for suffix in (
                "row_cardinality",
                "denominator",
                "violation_rate",
                "overshoot_quantiles",
            )
        }
    ),
    "I3": frozenset(
        {"inherited_G6", "absolute_gradient_records", "G6_absolute_cross_check"}
    ),
    "I4": frozenset({f"{arm}.inherited_G7" for arm in (IBR1_CTRL, IBR1_SELF)}),
    "I5": frozenset(
        {"inherited_G8", "all_registered_strata", "self_mode_overall_delta"}
    ),
    "I6": frozenset(
        {
            f"{arm}.{suffix}"
            for arm in (IBR1_CTRL, IBR1_SELF)
            for suffix in (
                "inherited_G9",
                "nonfinite_reset_count",
                "range_violation_count",
                "range_observation_count",
                "reconstruction_rows",
                "reconstruction_error_max",
            )
        }
    ),
}
_GATE_THRESHOLD_KEYS: dict[str, frozenset[str]] = {
    "I1": frozenset(
        {
            "cal_rows",
            "cal_cells",
            "controlled_shape",
            "post_decode_abs_max",
            "reconstruction_error_max",
        }
    ),
    "I2": frozenset(
        {"rows_per_arm", "denominator_per_arm", "violation_rate_max_exclusive"}
    ),
    "I3": frozenset(
        {
            "aux_reachable_updates_min",
            "track_reachable_updates_min",
            "cosine_total_track_median_min",
            "positive_signed_projection_min",
            "weighted_aux_over_track_median_max",
        }
    ),
    "I4": frozenset({"registry"}),
    "I5": frozenset({"self_mode_overall_delta_max", "registry"}),
    "I6": frozenset(
        {
            "nonfinite_reset_count",
            "range_violation_count",
            "range_observation_count",
            "reconstruction_rows",
            "reconstruction_error_max",
            "inherited_drift_and_reset_registry",
        }
    ),
}
_GATE_METRIC_KEYS: dict[str, frozenset[str]] = {
    "I1": frozenset(
        {
            "cal_rows",
            "cal_geometry_dtype",
            "cal_zero_init_failures",
            "cal_post_decode_violations",
            "cal_post_decode_abs_max",
            "cal_reconstruction_failures",
            "cal_reconstruction_error_max",
            "cal_prev_free_failures",
            "update0_checkpoint_tensor_sha256",
            "final_assembly_receipt_payload_sha256",
        }
    ),
    "I2": frozenset(
        {"raw_records", "summary_exact_match", "summary_sha256", "arms"}
    ),
    "I3": frozenset({"records", "inherited_G6", "gradient_geometry_sha256"}),
    "I4": frozenset({"arms"}),
    "I5": frozenset({"inherited_G8", "registered_strata"}),
    "I6": frozenset({"arms"}),
}


def _expected_gate_thresholds(gate_id: str) -> dict[str, Any]:
    return {
        "I1": {
            "cal_rows": I1_CAL_ROWS,
            "cal_cells": I1_CAL_CELLS,
            "controlled_shape": list(I1_CONTROLLED_SHAPE),
            "post_decode_abs_max": 1.0,
            "reconstruction_error_max": I1_RECONSTRUCTION_MAX,
        },
        "I2": {
            "rows_per_arm": I2_ROWS_PER_ARM,
            "denominator_per_arm": I2_ROWS_PER_ARM * 8,
            "violation_rate_max_exclusive": I2_RATE_MAX_EXCLUSIVE,
        },
        "I3": {
            "aux_reachable_updates_min": I3_AUX_REACHABLE_MIN,
            "track_reachable_updates_min": I3_TRACK_REACHABLE_MIN,
            "cosine_total_track_median_min": I3_COSINE_MEDIAN_MIN,
            "positive_signed_projection_min": I3_POSITIVE_PROJECTION_MIN,
            "weighted_aux_over_track_median_max": I3_RATIO_MAX,
        },
        "I4": {"registry": "frozen F2 G7"},
        "I5": {
            "self_mode_overall_delta_max": -1e-6,
            "registry": "frozen F2 G8",
        },
        "I6": {
            "nonfinite_reset_count": 0,
            "range_violation_count": 0,
            "range_observation_count": I6_RANGE_OBSERVATIONS,
            "reconstruction_rows": I6_RECONSTRUCTION_ROWS,
            "reconstruction_error_max": I6_RECONSTRUCTION_MAX,
            "inherited_drift_and_reset_registry": "frozen F2 G9",
        },
    }[gate_id]


def _expected_check_signature(gate_id: str, name: str) -> tuple[str, Any]:
    if gate_id == "I1":
        signatures = {
            "final_assembly_authority": (
                "==",
                {"phase": ASSEMBLY_PHASE_FINAL, "candidate_cap": 1},
            ),
            "cal_rows": ("==", I1_CAL_ROWS),
            "cal_fp32": ("==", "torch.float32"),
            "zero_init": (
                "frozen_zero_init_contract",
                {
                    "rows": I1_CAL_ROWS,
                    "cells": I1_CAL_CELLS,
                    "shape": list(I1_CONTROLLED_SHAPE),
                    "failures": 0,
                },
            ),
            "post_decode_range": (
                "rows/cells/shape/violations/abs_max",
                {
                    "rows": I1_CAL_ROWS,
                    "cells": I1_CAL_CELLS,
                    "shape": list(I1_CONTROLLED_SHAPE),
                    "violations": 0,
                    "abs_max": 1.0,
                },
            ),
            "reconstruction": (
                "rows/cells/shape/failures/error_max",
                {
                    "rows": I1_CAL_ROWS,
                    "cells": I1_CAL_CELLS,
                    "shape": list(I1_CONTROLLED_SHAPE),
                    "failures": 0,
                    "error_max": I1_RECONSTRUCTION_MAX,
                },
            ),
            "prev_free_observation": ("==", 0),
            "update0_checkpoint_tensor_identity": ("all", True),
        }
        return signatures[name]
    if gate_id == "I2":
        suffix = name.rsplit(".", 1)[1]
        return {
            "row_cardinality": ("==", I2_ROWS_PER_ARM),
            "denominator": ("==", I2_ROWS_PER_ARM * 8),
            "violation_rate": ("<", I2_RATE_MAX_EXCLUSIVE),
            "overshoot_quantiles": ("has", ["max", "p50", "p90", "p99"]),
        }[suffix]
    if gate_id == "I3":
        return {
            "inherited_G6": ("==", "PASS"),
            "absolute_gradient_records": ("==", I3_GRADIENT_UPDATES),
            "G6_absolute_cross_check": ("==", True),
        }[name]
    if gate_id == "I4":
        return "==", "PASS"
    if gate_id == "I5":
        return {
            "inherited_G8": ("==", "PASS"),
            "all_registered_strata": (
                "==",
                ["overall", "change", "turn", "other"],
            ),
            "self_mode_overall_delta": ("<=", -1e-6),
        }[name]
    suffix = name.rsplit(".", 1)[1]
    return {
        "inherited_G9": ("==", "PASS"),
        "nonfinite_reset_count": ("==", 0),
        "range_violation_count": ("==", 0),
        "range_observation_count": ("==", I6_RANGE_OBSERVATIONS),
        "reconstruction_rows": ("==", I6_RECONSTRUCTION_ROWS),
        "reconstruction_error_max": ("<=", I6_RECONSTRUCTION_MAX),
    }[suffix]


class IBR1GateContractError(RuntimeError):
    """Raised when gate/seal input is malformed or has been tampered with."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1GateContractError(message)


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        # Detached scalar tensors/arrays are useful in pure telemetry tests.
        if hasattr(value, "item"):
            try:
                value = value.item()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                raise IBR1GateContractError(f"{label} must be a scalar numeric") from exc
        if isinstance(value, bool) or not isinstance(value, Real):
            raise IBR1GateContractError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise IBR1GateContractError(f"{label} is nonfinite")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise IBR1GateContractError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise IBR1GateContractError(f"{label} must be nonnegative")
    return result


def _valid_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for i, lchar in enumerate(left, 1):
        current = [i]
        for j, rchar in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (lchar != rchar),
                )
            )
        previous = current
    return previous[-1]


def _reject_nested_seal_escalation(value: Any, label: str, path: str = "$") -> None:
    """Reject authorization/internal-test aliases at arbitrary nesting depth.

    Authority receipts already perform an equivalent check.  Repeating it here
    is deliberate: a gate or seal must not become an alternate serialization
    path that weakens the policy.  We reject the exact spelling, common
    punctuation/case aliases, and short spelling variants of the three frozen
    policy fields.
    """

    targets = {
        "formaltrainingauthorized": "formal_training_authorized",
        "internaltestopened": "internal_test_opened",
        "internaltest": "internal_test",
    }
    forbidden_aliases = {
        "formaltrainingallowed",
        "allowformaltraining",
        "formalrunauthorized",
        "formalrunallowed",
        "allowformalrun",
        "trainingformalauthorized",
        "openinternaltest",
        "internaltestopen",
        "internaltestallowed",
        "allowinternaltest",
        "internaltestenabled",
        "enableinternaltest",
        "internaltestaccess",
        "accessinternaltest",
    }
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            _require(isinstance(raw_key, str), f"{label} has a non-string key at {path}")
            key = raw_key
            normalized = "".join(character for character in key.lower() if character.isalnum())
            matched: str | None = normalized if normalized in targets else None
            semantic_forbidden = normalized in forbidden_aliases or (
                "formal" in normalized
                and ("train" in normalized or "run" in normalized)
                and (
                    "author" in normalized
                    or "allow" in normalized
                    or "enable" in normalized
                    or "permit" in normalized
                )
            ) or (
                "internal" in normalized
                and "test" in normalized
                and (
                    "open" in normalized
                    or "allow" in normalized
                    or "enable" in normalized
                    or "access" in normalized
                    or "permit" in normalized
                )
            )
            suspicious_family = (
                ("formal" in normalized or "forml" in normalized)
                and ("train" in normalized or "authorized" in normalized)
            ) or (
                ("internal" in normalized or "internl" in normalized)
                and ("test" in normalized or "opened" in normalized)
            )
            if matched is None and suspicious_family:
                candidates = [
                    target
                    for target in targets
                    if (
                        _levenshtein(normalized, target) <= 2
                        or (len(normalized) >= 12 and target.startswith(normalized))
                        or (len(normalized) >= 12 and normalized.startswith(target))
                    )
                ]
                if candidates:
                    matched = min(
                        candidates,
                        key=lambda target: _levenshtein(normalized, target),
                    )
            if matched is not None:
                expected = targets[matched]
                allowed = (
                    key == expected
                    and (
                        (expected == "formal_training_authorized" and item is False)
                        or (expected == "internal_test_opened" and item is False)
                        or (expected == "internal_test" and item == "sealed")
                    )
                )
                _require(
                    allowed,
                    f"{label} contains suspicious or authorizing field at {path}.{key}",
                )
            elif semantic_forbidden:
                raise IBR1GateContractError(
                    f"{label} contains suspicious or authorizing field at {path}.{key}"
                )
            _reject_nested_seal_escalation(item, label, f"{path}.{key}")
    elif _is_sequence(value):
        for index, item in enumerate(value):
            _reject_nested_seal_escalation(item, label, f"{path}[{index}]")


def _canonical(value: Any, label: str) -> bytes:
    _reject_nested_seal_escalation(value, label)
    try:
        # Use the authority implementation so the byte-level contract remains
        # identical to all existing IBR1 receipts.
        return canonical_json_bytes(value)
    except Exception as exc:  # noqa: BLE001 - normalize foreign contract errors
        raise IBR1GateContractError(f"{label} is not finite canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: Mapping[str, Any], label: str) -> dict[str, Any]:
    _require("receipt_payload_sha256" not in payload, f"{label} already has a payload SHA")
    document = dict(payload)
    document["receipt_payload_sha256"] = _sha256_bytes(_canonical(document, label))
    return document


def _verify_self_hash(document: Mapping[str, Any], label: str) -> str:
    _require(isinstance(document, Mapping), f"{label} must be a mapping")
    payload = dict(document)
    stored = _valid_sha256(payload.pop("receipt_payload_sha256", None), f"{label} payload SHA")
    _require(stored == _sha256_bytes(_canonical(payload, label)), f"{label} payload SHA mismatch")
    return stored


def _sealed_mapping(
    value: Any,
    *,
    label: str,
    analysis_class: str | None = None,
    require_self_hash: bool = False,
    require_architecture: bool = False,
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    document = dict(value)
    _reject_nested_seal_escalation(document, label)
    if analysis_class is not None:
        _require(document.get("analysis_class") == analysis_class, f"{label} analysis_class drifted")
    _require(document.get("family_id") == IBR1_FAMILY_ID, f"{label} family drifted")
    if require_architecture:
        _require(
            document.get("architecture_lock") == IBR1_ARCHITECTURE_LOCK,
            f"{label} architecture lock drifted",
        )
    _require(
        document.get("internal_test") == "sealed"
        and document.get("internal_test_opened") is False,
        f"{label} internal-test policy drifted",
    )
    _require(document.get("formal_training_authorized") is not True, f"{label} authorizes formal training")
    if require_self_hash:
        _verify_self_hash(document, label)
    else:
        _canonical(document, label)
    return document


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _close(left: float, right: float, *, rel: float = 1e-6, abs_tol: float = 1e-9) -> bool:
    return math.isclose(left, right, rel_tol=rel, abs_tol=abs_tol)


def _flatten_numeric(value: Any, label: str) -> tuple[float, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        return (_finite_float(value, label),)
    if hasattr(value, "detach") and hasattr(value, "numel"):
        try:
            return _flatten_numeric(value.detach().cpu().reshape(-1).tolist(), label)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise IBR1GateContractError(f"{label} tensor cannot be materialized") from exc
    if hasattr(value, "tolist") and not isinstance(value, Mapping):
        try:
            materialized = value.tolist()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            raise IBR1GateContractError(f"{label} array cannot be materialized") from exc
        if materialized is not value:
            return _flatten_numeric(materialized, label)
    if _is_sequence(value):
        result: list[float] = []
        for index, item in enumerate(value):
            result.extend(_flatten_numeric(item, f"{label}[{index}]")
            )
        _require(bool(result), f"{label} has zero numeric support")
        return tuple(result)
    raise IBR1GateContractError(f"{label} must be numeric or a numeric sequence")


def _coerce_scalar(value: Any, label: str) -> float:
    values = _flatten_numeric(value, label)
    _require(len(values) == 1, f"{label} must be scalar")
    return values[0]


def _check(passed: bool, *, observed: Any, comparator: str, threshold: Any) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "observed": observed,
        "comparator": comparator,
        "threshold": threshold,
    }


@dataclass(frozen=True)
class IBR1GateReceipt:
    """In-memory receipt for one IBR1 gate."""

    gate_id: str
    passed: bool
    checks: Mapping[str, Mapping[str, Any]]
    metrics: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    contract: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "analysis_class": IBR1_GATE_RECEIPT_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "gate_id": self.gate_id,
            "valid_input": True,
            "passed": bool(self.passed),
            "status": "PASS" if self.passed else "FAIL",
            "decision": "PASS" if self.passed else "STOP",
            "checks": {name: dict(value) for name, value in self.checks.items()},
            "metrics": dict(self.metrics),
            "thresholds": dict(self.thresholds),
            "contract": dict(self.contract),
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
        return _self_hash(payload, f"IBR1 {self.gate_id} receipt")


def _make_receipt(
    gate_id: str,
    *,
    checks: Mapping[str, Mapping[str, Any]],
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> IBR1GateReceipt:
    _require(gate_id in IBR1_GATE_IDS, f"unknown IBR1 gate {gate_id!r}")
    _canonical(checks, f"IBR1 {gate_id} checks")
    _canonical(metrics, f"IBR1 {gate_id} metrics")
    return IBR1GateReceipt(
        gate_id=gate_id,
        passed=all(bool(value.get("passed")) for value in checks.values()),
        checks=dict(checks),
        metrics=dict(metrics),
        thresholds=dict(thresholds),
        contract=dict(contract),
    )


def _coerce_f2_receipt(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, F2GateReceipt):
        return value.to_dict()
    _require(isinstance(value, Mapping), f"{label} must be an F2 GateReceipt or mapping")
    document = dict(value)
    _require(document.get("analysis_class") == "f2_preformal_smoke_gate", f"{label} class drifted")
    _require(document.get("valid_input") is True, f"{label} input is not valid")
    _require(document.get("formal_training_authorized") is not True, f"{label} authorizes formal training")
    _require(document.get("gate_id") in {"G6", "G7", "G8", "G9"}, f"{label} gate id drifted")
    checks = _mapping(document.get("checks"), f"{label} checks")
    _require(bool(checks), f"{label} checks must be nonempty")
    expected_pass = all(
        isinstance(check, Mapping) and check.get("passed") is True
        for check in checks.values()
    )
    _require(document.get("passed") is expected_pass, f"{label} pass/check inconsistency")
    _require(
        document.get("status") == ("PASS" if expected_pass else "FAIL")
        and document.get("decision") == ("GO" if expected_pass else "STOP"),
        f"{label} status/decision inconsistency",
    )
    _canonical(document, label)
    return document


def _coerce_g6(value: Any, index: int) -> G6Update:
    if isinstance(value, G6Update):
        return value
    _require(isinstance(value, Mapping), f"G6 update {index} must be a mapping")
    required = ("u_pre", "aux_reachable", "track_reachable")
    _require(all(name in value for name in required), f"G6 update {index} is incomplete")
    return G6Update(
        u_pre=value["u_pre"],
        aux_reachable=value["aux_reachable"],
        track_reachable=value["track_reachable"],
        cosine_total_track=value.get("cosine_total_track"),
        signed_projection=value.get("signed_projection"),
        aux_track_ratio=value.get("aux_track_ratio"),
        per_aux_ratios=value.get("per_aux_ratios"),
    )


def _coerce_g7(value: Any, index: int) -> G7Update:
    if isinstance(value, G7Update):
        return value
    _require(isinstance(value, Mapping), f"G7 update {index} must be a mapping")
    required = (
        "u_pre",
        "per_method_over_base",
        "total_method_over_base",
        "abs_tanh_method_scales",
    )
    _require(all(name in value for name in required), f"G7 update {index} is incomplete")
    return G7Update(
        u_pre=value["u_pre"],
        per_method_over_base=value["per_method_over_base"],
        total_method_over_base=value["total_method_over_base"],
        abs_tanh_method_scales=value["abs_tanh_method_scales"],
        r_prev=value.get("r_prev"),
        abs_tanh_s_prev=value.get("abs_tanh_s_prev"),
    )


def _pick_alias(primary: Any, aliases: Mapping[str, Any], label: str) -> Any:
    present = [value for value in (primary, *aliases.values()) if value is not None]
    _require(bool(present), f"{label} is required")
    first = present[0]
    _require(all(value is first or value == first for value in present[1:]), f"conflicting {label} aliases")
    return first


def _arm_sidecars(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, Mapping):
        # Accept either family-arm keys or the explicit update labels.
        candidates = list(value.values()) if value and all(isinstance(item, Mapping) for item in value.values()) else []
    elif _is_sequence(value):
        candidates = list(value)
    else:
        raise IBR1GateContractError("update-0 checkpoint sidecars must be a mapping or sequence")
    result: dict[str, dict[str, Any]] = {}
    for index, candidate in enumerate(candidates):
        document = _sealed_mapping(
            candidate,
            label=f"update-0 checkpoint sidecar[{index}]",
            analysis_class=CHECKPOINT_SIDECAR_CLASS,
            require_architecture=True,
        )
        arm = document.get("family_arm")
        _require(arm in (IBR1_CTRL, IBR1_SELF), f"checkpoint sidecar[{index}] has an unknown family arm")
        _require(arm not in result, f"duplicate update-0 checkpoint sidecar for {arm}")
        result[arm] = document
    _require(set(result) == {IBR1_CTRL, IBR1_SELF}, "I1 requires one update-0 sidecar per arm")
    return result


def _live_update0_sidecars(
    project_root: str | Path,
    values: Mapping[str, str | Path] | Sequence[str | Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    root = Path(project_root).expanduser().resolve()
    _require(root.is_dir(), "I1 project_root is not a directory")
    paths = list(values.values()) if isinstance(values, Mapping) else list(values)
    _require(len(paths) == 2, "I1 requires two update-0 checkpoint sidecar paths")
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    resolved_paths: set[Path] = set()
    for index, value in enumerate(paths):
        path, document, binding = _load_bound_artifact(
            root,
            value,
            label=f"I1 live update-0 checkpoint sidecar[{index}]",
            analysis_class=CHECKPOINT_SIDECAR_CLASS,
            require_self_hash=False,
            require_architecture=True,
        )
        _require(path not in resolved_paths, "I1 update-0 sidecar paths must be distinct")
        resolved_paths.add(path)
        _require(set(document) == _SIDECAR_KEYS, f"I1 live sidecar {path} schema drifted")
        arm = document.get("family_arm")
        _require(arm in (IBR1_CTRL, IBR1_SELF), f"I1 live sidecar {path} has an unknown arm")
        _require(document.get("u_pre") == 0, f"I1 live sidecar {path} is not update 0")
        _require(arm not in documents, f"duplicate I1 live update-0 sidecar for {arm}")
        documents[str(arm)] = document
        bindings[str(arm)] = binding
    _require(
        set(documents) == {IBR1_CTRL, IBR1_SELF},
        "I1 live sidecars do not cover CTRL/SELF update 0",
    )
    return documents, bindings


def _portable_project_relative_path(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path string")
    _require("\\" not in value, f"{label} must use portable forward slashes")
    pure = PurePosixPath(value)
    _require(
        not pure.is_absolute()
        and pure.as_posix() == value
        and all(part not in ("", ".", "..") and ":" not in part for part in pure.parts),
        f"{label} must be a normalized project-root-relative path",
    )
    return value


def _validate_receipt_artifact_binding(
    value: Any,
    *,
    label: str,
    analysis_class: str,
    expected_document: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the exact standard receipt binding, optionally against bytes."""

    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    binding = dict(value)
    required = {
        "path",
        "sha256",
        "receipt_payload_sha256",
        "analysis_class",
    }
    _require(set(binding) == required, f"{label} keys differ from the standard receipt binding")
    path_value = _portable_project_relative_path(binding.get("path"), f"{label}.path")
    file_sha = _valid_sha256(binding.get("sha256"), f"{label}.sha256")
    payload_sha = _valid_sha256(
        binding.get("receipt_payload_sha256"),
        f"{label}.receipt_payload_sha256",
    )
    _require(binding.get("analysis_class") == analysis_class, f"{label} analysis_class drifted")
    if expected_document is not None:
        _require(
            expected_document.get("analysis_class") == analysis_class,
            f"{label} expected document class drifted",
        )
        _require(
            expected_document.get("receipt_payload_sha256") == payload_sha,
            f"{label} payload SHA differs from the receipt document",
        )
    if project_root is not None:
        root = Path(project_root).expanduser().resolve()
        _require(root.is_dir(), f"{label} project_root is not a directory")
        path = _rooted_path(root, path_value, label)
        observed = _load_canonical_json(path, label)
        _require(
            expected_document is None or observed == dict(expected_document),
            f"{label} bytes differ from the supplied receipt document",
        )
        _require(_sha256_bytes(path.read_bytes()) == file_sha, f"{label} file SHA mismatch")
        _verify_self_hash(observed, label)
        _require(observed.get("receipt_payload_sha256") == payload_sha, f"{label} live payload SHA mismatch")
    return {
        "path": path_value,
        "sha256": file_sha,
        "receipt_payload_sha256": payload_sha,
        "analysis_class": analysis_class,
    }


def _binding_from_live_receipt(
    project_root: str | Path,
    receipt_path: str | Path,
    *,
    label: str,
    analysis_class: str,
    expected_document: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    _require(root.is_dir(), f"{label} project_root is not a directory")
    path = _rooted_path(root, receipt_path, label)
    observed = _load_canonical_json(path, label)
    _require(observed == dict(expected_document), f"{label} bytes differ from the supplied document")
    payload_sha = _verify_self_hash(observed, label)
    binding = {
        "path": _relative_path(root, path),
        "sha256": _sha256_bytes(path.read_bytes()),
        "receipt_payload_sha256": payload_sha,
        "analysis_class": analysis_class,
    }
    return _validate_receipt_artifact_binding(
        binding,
        label=label,
        analysis_class=analysis_class,
        expected_document=expected_document,
        project_root=root,
    )


def evaluate_i1(
    final_assembly: Mapping[str, Any] | None = None,
    cal_numeric_evidence: Mapping[str, Any] | None = None,
    update0_checkpoint_sidecars: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
    *,
    final_assembly_authority: Mapping[str, Any] | None = None,
    cal_evidence: Mapping[str, Any] | None = None,
    checkpoint_sidecars: Mapping[str, Mapping[str, Any]] | Sequence[Mapping[str, Any]] | None = None,
    final_assembly_binding: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    final_assembly_receipt_path: str | Path | None = None,
    update0_checkpoint_sidecar_paths: Mapping[str, str | Path]
    | Sequence[str | Path]
    | None = None,
    non_authority: bool = False,
) -> IBR1GateReceipt:
    """Evaluate final authority, CAL structure, and update-0 identity.

    Authority-bearing use requires ``project_root``, the live final receipt
    path, and the two live update-0 sidecar paths.  Pure in-memory inspection
    is available only with ``non_authority=True`` and cannot be promoted into
    a combined authoritative gate receipt.
    """

    _require(isinstance(non_authority, bool), "I1 non_authority must be boolean")

    assembly = _sealed_mapping(
        _pick_alias(final_assembly, {"final_assembly_authority": final_assembly_authority}, "final assembly"),
        label="final IBR1 assembly",
        analysis_class=ASSEMBLY_RECEIPT_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    cal = _sealed_mapping(
        _pick_alias(cal_numeric_evidence, {"cal_evidence": cal_evidence}, "CAL numeric evidence"),
        label="IBR1 CAL numeric evidence",
        analysis_class=CAL_NUMERIC_EVIDENCE_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    sidecars = _arm_sidecars(
        _pick_alias(
            update0_checkpoint_sidecars,
            {"checkpoint_sidecars": checkpoint_sidecars},
            "update-0 checkpoint sidecars",
        )
    )

    authority_mode = not non_authority
    live_sidecar_bindings: dict[str, dict[str, Any]] = {}
    if authority_mode:
        _require(
            project_root is not None
            and final_assembly_receipt_path is not None
            and update0_checkpoint_sidecar_paths is not None,
            "I1 authority requires project_root, final_assembly_receipt_path, "
            "and update0_checkpoint_sidecar_paths",
        )
        live_sidecars, live_sidecar_bindings = _live_update0_sidecars(
            project_root,
            update0_checkpoint_sidecar_paths,
        )
        _require(
            live_sidecars == sidecars,
            "I1 live update-0 sidecar bytes differ from the supplied sidecar documents",
        )
    else:
        _require(
            update0_checkpoint_sidecar_paths is None,
            "I1 non-authority mode does not accept authoritative sidecar paths",
        )

    _require(assembly.get("phase") == ASSEMBLY_PHASE_FINAL, "I1 final assembly is not final phase")
    _require(assembly.get("candidate_cap") == 1, "I1 candidate cap drifted")
    _require(isinstance(assembly.get("lambda_freeze_binding"), Mapping), "I1 final assembly has no lambda freeze")
    _require(cal.get("support") == "CAL", "I1 CAL support drifted")

    zero = _mapping(cal.get("zero_init_persistence"), "I1 CAL zero-init evidence")
    post = _mapping(cal.get("post_decode_range"), "I1 CAL post-decode evidence")
    recon = _mapping(cal.get("realized_delta_reconstruction"), "I1 CAL reconstruction evidence")
    prev_free = _mapping(cal.get("prev_free_observation_graph"), "I1 CAL prev-free evidence")

    cal_rows = _nonnegative_int(cal.get("rows"), "I1 CAL rows")
    cal_dtype = cal.get("geometry_dtype")
    zero_rows = _nonnegative_int(zero.get("checked_rows"), "I1 zero-init checked rows")
    zero_cells = _nonnegative_int(zero.get("checked_cells"), "I1 zero-init checked cells")
    zero_shape = tuple(zero.get("per_row_shape", ())) if _is_sequence(zero.get("per_row_shape")) else ()
    post_rows = _nonnegative_int(post.get("checked_rows"), "I1 post-decode checked rows")
    post_cells = _nonnegative_int(post.get("checked_cells"), "I1 post-decode checked cells")
    post_shape = tuple(post.get("per_row_shape", ())) if _is_sequence(post.get("per_row_shape")) else ()
    recon_rows = _nonnegative_int(recon.get("checked_rows"), "I1 reconstruction checked rows")
    recon_cells = _nonnegative_int(recon.get("checked_cells"), "I1 reconstruction checked cells")
    recon_shape = tuple(recon.get("per_row_shape", ())) if _is_sequence(recon.get("per_row_shape")) else ()
    zero_failures = _nonnegative_int(zero.get("failures"), "I1 zero-init failures")
    post_violations = _nonnegative_int(post.get("violations"), "I1 post-decode violations")
    post_abs_max = _finite_float(post.get("abs_max"), "I1 post-decode abs max")
    recon_failures = _nonnegative_int(recon.get("failures"), "I1 reconstruction failures")
    recon_max = _finite_float(recon.get("error_max"), "I1 reconstruction error max")
    prev_failures = _nonnegative_int(prev_free.get("failures"), "I1 prev-free failures")
    _require(post_abs_max >= 0.0, "I1 post-decode abs max must be nonnegative")
    _require(recon_max >= 0.0, "I1 reconstruction error max must be nonnegative")

    if authority_mode:
        expected_assembly_binding = _binding_from_live_receipt(
            project_root,
            final_assembly_receipt_path,
            label="I1 final assembly receipt",
            analysis_class=ASSEMBLY_RECEIPT_CLASS,
            expected_document=assembly,
        )
        if final_assembly_binding is not None:
            supplied_assembly_binding = _validate_receipt_artifact_binding(
                final_assembly_binding,
                label="I1 final assembly binding",
                analysis_class=ASSEMBLY_RECEIPT_CLASS,
                expected_document=assembly,
                project_root=project_root,
            )
            _require(
                supplied_assembly_binding == expected_assembly_binding,
                "I1 supplied final assembly binding differs from live receipt bytes",
            )
    elif final_assembly_binding is None:
        first_sidecar = sidecars[IBR1_CTRL]
        expected_assembly_binding = _validate_receipt_artifact_binding(
            first_sidecar.get("final_assembly_receipt"),
            label="I1 non-authority final assembly binding",
            analysis_class=ASSEMBLY_RECEIPT_CLASS,
            expected_document=assembly,
        )
    else:
        expected_assembly_binding = _validate_receipt_artifact_binding(
            final_assembly_binding,
            label="I1 final assembly binding",
            analysis_class=ASSEMBLY_RECEIPT_CLASS,
            expected_document=assembly,
        )

    sidecar_bindings: list[Mapping[str, Any]] = []
    tensor_shas: list[str] = []
    sidecar_checks: dict[str, bool] = {}
    for arm, sidecar in sidecars.items():
        _require(sidecar.get("u_pre") == 0, f"I1 {arm} update-0 sidecar clock drifted")
        expected_engine = "S-CTRL" if arm == IBR1_CTRL else "S-SELF"
        sidecar_checks[f"update0_sidecar.{arm}.engine_arm"] = sidecar.get("engine_arm") == expected_engine
        tensor_shas.append(_valid_sha256(sidecar.get("checkpoint_tensor_sha256"), f"I1 {arm} tensor SHA"))
        binding = _validate_receipt_artifact_binding(
            sidecar.get("final_assembly_receipt"),
            label=f"I1 {arm} final assembly binding",
            analysis_class=ASSEMBLY_RECEIPT_CLASS,
            expected_document=assembly,
            project_root=project_root,
        )
        sidecar_bindings.append(binding)
        sidecar_checks[f"update0_sidecar.{arm}.final_binding"] = (
            binding == expected_assembly_binding
        )

    if sidecar_bindings:
        first_binding = dict(sidecar_bindings[0])
        binding_equal = all(dict(binding) == first_binding for binding in sidecar_bindings[1:])
    else:
        binding_equal = False
    sidecar_checks["update0_sidecars.same_final_binding"] = binding_equal
    sidecar_checks["update0_sidecars.same_tensor_sha"] = len(set(tensor_shas)) == 1

    checks: dict[str, Mapping[str, Any]] = {
        "final_assembly_authority": _check(
            assembly.get("phase") == ASSEMBLY_PHASE_FINAL and assembly.get("candidate_cap") == 1,
            observed={"phase": assembly.get("phase"), "candidate_cap": assembly.get("candidate_cap")},
            comparator="==",
            threshold={"phase": ASSEMBLY_PHASE_FINAL, "candidate_cap": 1},
        ),
        "cal_rows": _check(cal_rows == I1_CAL_ROWS, observed=cal_rows, comparator="==", threshold=I1_CAL_ROWS),
        "cal_fp32": _check(cal_dtype == "torch.float32", observed=cal_dtype, comparator="==", threshold="torch.float32"),
        "zero_init": _check(
            zero_rows == I1_CAL_ROWS and zero_cells == I1_CAL_CELLS and zero_shape == I1_CONTROLLED_SHAPE and zero_failures == 0,
            observed={"rows": zero_rows, "cells": zero_cells, "shape": list(zero_shape), "failures": zero_failures},
            comparator="frozen_zero_init_contract",
            threshold={"rows": I1_CAL_ROWS, "cells": I1_CAL_CELLS, "shape": list(I1_CONTROLLED_SHAPE), "failures": 0},
        ),
        "post_decode_range": _check(
            post_rows == I1_CAL_ROWS and post_cells == I1_CAL_CELLS and post_shape == I1_CONTROLLED_SHAPE and post_violations == 0 and post_abs_max <= 1.0,
            observed={"rows": post_rows, "cells": post_cells, "shape": list(post_shape), "violations": post_violations, "abs_max": post_abs_max},
            comparator="rows/cells/shape/violations/abs_max",
            threshold={"rows": I1_CAL_ROWS, "cells": I1_CAL_CELLS, "shape": list(I1_CONTROLLED_SHAPE), "violations": 0, "abs_max": 1.0},
        ),
        "reconstruction": _check(
            recon_rows == I1_CAL_ROWS and recon_cells == I1_CAL_CELLS and recon_shape == I1_CONTROLLED_SHAPE and recon_failures == 0 and recon_max <= I1_RECONSTRUCTION_MAX,
            observed={"rows": recon_rows, "cells": recon_cells, "shape": list(recon_shape), "failures": recon_failures, "error_max": recon_max},
            comparator="rows/cells/shape/failures/error_max",
            threshold={"rows": I1_CAL_ROWS, "cells": I1_CAL_CELLS, "shape": list(I1_CONTROLLED_SHAPE), "failures": 0, "error_max": I1_RECONSTRUCTION_MAX},
        ),
        "prev_free_observation": _check(prev_failures == 0, observed=prev_failures, comparator="==", threshold=0),
        "update0_checkpoint_tensor_identity": _check(all(sidecar_checks.values()), observed=sidecar_checks, comparator="all", threshold=True),
    }
    return _make_receipt(
        I1_GATE_ID,
        checks=checks,
        metrics={
            "cal_rows": cal_rows,
            "cal_geometry_dtype": cal_dtype,
            "cal_zero_init_failures": zero_failures,
            "cal_post_decode_violations": post_violations,
            "cal_post_decode_abs_max": post_abs_max,
            "cal_reconstruction_failures": recon_failures,
            "cal_reconstruction_error_max": recon_max,
            "cal_prev_free_failures": prev_failures,
            "update0_checkpoint_tensor_sha256": tensor_shas,
            "final_assembly_receipt_payload_sha256": assembly.get("receipt_payload_sha256"),
        },
        thresholds={
            "cal_rows": I1_CAL_ROWS,
            "cal_cells": I1_CAL_CELLS,
            "controlled_shape": list(I1_CONTROLLED_SHAPE),
            "post_decode_abs_max": 1.0,
            "reconstruction_error_max": I1_RECONSTRUCTION_MAX,
        },
        contract={
            "both_arms": "AND",
            "authority": "final assembly + update-0 sidecars",
            "evidence_mode": "filesystem_bound" if authority_mode else "non_authority",
            "live_evidence": (
                {
                    "final_assembly": expected_assembly_binding,
                    "update0_checkpoint_sidecars": live_sidecar_bindings,
                }
                if authority_mode
                else None
            ),
            "formal_training_authorized": False,
            "internal_test": "sealed",
        },
    )


def _compact_i2_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    arms = _mapping(summary.get("arms"), "I2 summary arms")
    compact: dict[str, Any] = {}
    for arm in (IBR1_CTRL, IBR1_SELF):
        item = _mapping(arms.get(arm), f"I2 summary {arm}")
        compact[arm] = {
            key: item.get(key)
            for key in (
                "rows",
                "I2_any_axis_violation_count",
                "I2_any_axis_denominator",
                "I2_any_axis_violation_rate",
                "I2_pass",
                "axis_violation_counts",
                "horizon_violation_counts",
                "row_within_update_violation_counts",
                "positive_boundary_counts",
                "negative_boundary_counts",
                "overshoot_all_axis_cells",
                "overshoot_violating_only_descriptive",
                "geometry_reconstruction_error_max",
                "telescoping_reconstruction_error_max",
            )
        }
    return compact


def _validate_i2_raw_prebound_fields(raw: Sequence[Mapping[str, Any]]) -> None:
    """Re-derive every pre-bound mask and overshoot from the source scalar."""

    for row_index, record in enumerate(raw):
        prebound = record.get("additive_prebound_fy")
        mask = record.get("prebound_violation_mask")
        overshoot = record.get("prebound_overshoot_fy")
        _require(_is_sequence(prebound) and len(prebound) == 8, f"I2 row {row_index} prebound shape drifted")
        _require(_is_sequence(mask) and len(mask) == 8, f"I2 row {row_index} violation-mask shape drifted")
        _require(_is_sequence(overshoot) and len(overshoot) == 8, f"I2 row {row_index} overshoot shape drifted")
        for horizon in range(8):
            pre_row = prebound[horizon]
            mask_row = mask[horizon]
            overshoot_row = overshoot[horizon]
            _require(_is_sequence(pre_row) and len(pre_row) == 2, f"I2 row {row_index} prebound[{horizon}] shape drifted")
            _require(_is_sequence(mask_row) and len(mask_row) == 2, f"I2 row {row_index} mask[{horizon}] shape drifted")
            _require(_is_sequence(overshoot_row) and len(overshoot_row) == 2, f"I2 row {row_index} overshoot[{horizon}] shape drifted")
            for axis in range(2):
                value = _finite_float(pre_row[axis], f"I2 row {row_index} additive_prebound_fy[{horizon}][{axis}]")
                try:
                    binary32_value = struct.unpack("<f", struct.pack("<f", value))[0]
                except (OverflowError, struct.error) as exc:
                    raise IBR1GateContractError(
                        f"I2 row {row_index} additive_prebound_fy[{horizon}][{axis}] is not binary32"
                    ) from exc
                _require(
                    binary32_value == value,
                    f"I2 row {row_index} additive_prebound_fy[{horizon}][{axis}] is not exact binary32 telemetry",
                )
                expected_mask = abs(binary32_value) > 1.0
                observed_mask = mask_row[axis]
                _require(
                    isinstance(observed_mask, bool) and observed_mask is expected_mask,
                    f"I2 row {row_index} prebound_violation_mask[{horizon}][{axis}] differs from additive_prebound_fy",
                )
                expected_overshoot = max(
                    struct.unpack(
                        "<f",
                        struct.pack("<f", abs(binary32_value) - 1.0),
                    )[0],
                    0.0,
                )
                observed_overshoot = overshoot_row[axis]
                _require(
                    isinstance(observed_overshoot, Real)
                    and not isinstance(observed_overshoot, bool),
                    f"I2 row {row_index} prebound_overshoot_fy[{horizon}][{axis}] must be numeric",
                )
                observed_overshoot_float = _finite_float(
                    observed_overshoot,
                    f"I2 row {row_index} prebound_overshoot_fy[{horizon}][{axis}]",
                )
                _require(
                    observed_overshoot_float == expected_overshoot,
                    f"I2 row {row_index} prebound_overshoot_fy[{horizon}][{axis}] differs from additive_prebound_fy",
                )


def evaluate_i2(
    training_records: Sequence[Mapping[str, Any]],
    training_summary: Mapping[str, Any],
) -> IBR1GateReceipt:
    """Recompute I2 from raw branch-2 rows and require exact summary identity."""

    _require(_is_sequence(training_records), "I2 training_records must be a sequence")
    _sealed_mapping(training_summary, label="I2 training summary", analysis_class="ibr1_training_geometry_summary")
    raw = deepcopy(list(training_records))
    for index, record in enumerate(raw):
        _require(isinstance(record, Mapping), f"I2 training_records[{index}] must be a mapping")
        _canonical(record, f"I2 training_records[{index}]")
    _validate_i2_raw_prebound_fields(raw)
    collector = GeometryCollector(expected_training_rows_per_arm=I2_ROWS_PER_ARM)
    collector.training_records = raw
    try:
        recomputed = collector._validate_training()
    except Exception as exc:  # noqa: BLE001 - diagnostics contract is fail-closed here
        raise IBR1GateContractError(f"I2 raw training geometry failed validation: {exc}") from exc
    supplied = dict(training_summary)
    _require(
        _canonical(recomputed, "I2 recomputed summary") == _canonical(supplied, "I2 supplied summary"),
        "I2 supplied training summary differs from exact raw-row recomputation",
    )
    arm_metrics = _compact_i2_summary(recomputed)
    checks: dict[str, Mapping[str, Any]] = {}
    for arm in (IBR1_CTRL, IBR1_SELF):
        item = arm_metrics[arm]
        checks[f"{arm}.row_cardinality"] = _check(item["rows"] == I2_ROWS_PER_ARM, observed=item["rows"], comparator="==", threshold=I2_ROWS_PER_ARM)
        checks[f"{arm}.denominator"] = _check(item["I2_any_axis_denominator"] == I2_ROWS_PER_ARM * 8, observed=item["I2_any_axis_denominator"], comparator="==", threshold=I2_ROWS_PER_ARM * 8)
        checks[f"{arm}.violation_rate"] = _check(item["I2_any_axis_violation_rate"] < I2_RATE_MAX_EXCLUSIVE, observed=item["I2_any_axis_violation_rate"], comparator="<", threshold=I2_RATE_MAX_EXCLUSIVE)
        quantiles = _mapping(item["overshoot_all_axis_cells"], f"I2 {arm} quantiles")
        checks[f"{arm}.overshoot_quantiles"] = _check(
            all(key in quantiles and _finite_float(quantiles[key], f"I2 {arm} {key}") >= 0.0 for key in ("max", "p50", "p90", "p99")),
            observed=quantiles,
            comparator="has",
            threshold=["max", "p50", "p90", "p99"],
        )
    return _make_receipt(
        I2_GATE_ID,
        checks=checks,
        metrics={
            "raw_records": len(raw),
            "summary_exact_match": True,
            "summary_sha256": canonical_json_sha256(recomputed),
            "arms": arm_metrics,
        },
        thresholds={"rows_per_arm": I2_ROWS_PER_ARM, "denominator_per_arm": I2_ROWS_PER_ARM * 8, "violation_rate_max_exclusive": I2_RATE_MAX_EXCLUSIVE},
        contract={
            "quantity": "any-axis abs(additive_prebound_fy)>1 per horizon",
            "raw_summary_binding": "recompute then exact canonical equality",
            "overshoot_universe": "256*8*2 axis cells per arm",
            "both_arms": "AND",
        },
    )


def _finite_named_map(
    value: Any,
    *,
    label: str,
    nonnegative: bool = False,
) -> dict[str, float]:
    mapping = _mapping(value, label)
    _require(set(mapping) == set(IBR1_AUX_COMPONENTS), f"{label} auxiliary names drifted")
    result: dict[str, float] = {}
    for name in IBR1_AUX_COMPONENTS:
        numeric = _finite_float(mapping[name], f"{label}.{name}")
        if nonnegative:
            _require(numeric >= 0.0, f"{label}.{name} must be nonnegative")
        result[name] = numeric
    return result


def _bool_named_map(value: Any, label: str) -> dict[str, bool]:
    mapping = _mapping(value, label)
    _require(set(mapping) == set(IBR1_AUX_COMPONENTS), f"{label} auxiliary names drifted")
    result: dict[str, bool] = {}
    for name in IBR1_AUX_COMPONENTS:
        _require(isinstance(mapping[name], bool), f"{label}.{name} must be boolean")
        result[name] = bool(mapping[name])
    return result


def _diagnostic_cosine(
    dot: float,
    left_norm: float,
    right_norm: float,
) -> float:
    if left_norm <= DIAGNOSTIC_EPS or right_norm <= DIAGNOSTIC_EPS:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _require_close(observed: float, expected: float, label: str) -> None:
    _require(
        _close(observed, expected),
        f"{label} drifted: observed {observed!r}, expected {expected!r}",
    )


def evaluate_i3(
    g6_updates: Sequence[G6Update | Mapping[str, Any]],
    gradient_geometry: Mapping[str, Any],
) -> IBR1GateReceipt:
    """Evaluate frozen F2 G6 and cross-check every absolute diagnostic row."""

    _require(_is_sequence(g6_updates), "I3 g6_updates must be a sequence")
    normalized_updates = tuple(
        _coerce_g6(value, index) for index, value in enumerate(g6_updates)
    )
    try:
        inherited = evaluate_g6(normalized_updates, block_mode="bstar")
    except Exception as exc:  # noqa: BLE001 - normalize the frozen evaluator error
        raise IBR1GateContractError(f"I3 inherited G6 input is invalid: {exc}") from exc
    inherited_document = _coerce_f2_receipt(inherited, "I3 inherited G6 receipt")
    _require(inherited_document.get("gate_id") == "G6", "I3 inherited receipt is not G6")

    geometry = _sealed_mapping(
        gradient_geometry,
        label="I3 gradient geometry",
        analysis_class="ibr1_gradient_geometry",
    )
    _require(geometry.get("deciding_arm") == IBR1_CTRL, "I3 gradient geometry deciding arm drifted")
    records_value = geometry.get("records")
    _require(_is_sequence(records_value), "I3 gradient geometry records must be a sequence")
    records = tuple(records_value)
    _require(len(records) == I3_GRADIENT_UPDATES, "I3 requires exactly 128 gradient records")
    _require(len(normalized_updates) == len(records), "I3 G6/diagnostic cardinality differs")

    required_scalars = (
        "track_grad_norm",
        "weighted_aux_grad_norm",
        "total_grad_norm",
        "weighted_aux_track_dot",
        "weighted_aux_track_cosine",
        "weighted_aux_signed_projection",
        "actual_ratio_denominator",
        "per_aux_aggregate_discrepancy_norm",
        "per_aux_aggregate_rounding_bound_norm",
    )
    cross_checked = 0
    for index, (raw_record, update) in enumerate(zip(records, normalized_updates)):
        _require(isinstance(raw_record, Mapping), f"I3 gradient record {index} must be a mapping")
        record = dict(raw_record)
        _canonical(record, f"I3 gradient record {index}")
        _require(
            record.get("u_pre") == index
            and record.get("arm") == IBR1_CTRL
            and record.get("engine_arm") == "S-CTRL"
            and record.get("grad_accum") == I3_GRAD_ACCUM,
            f"I3 gradient record {index} clock/arm/accumulator drifted",
        )
        _require(update.u_pre == index, f"I3 G6 update {index} clock drifted")
        _require(isinstance(update.aux_reachable, bool), f"I3 G6[{index}] aux_reachable must be boolean")
        _require(isinstance(update.track_reachable, bool), f"I3 G6[{index}] track_reachable must be boolean")

        scalars = {
            name: _finite_float(record.get(name), f"I3[{index}].{name}")
            for name in required_scalars
        }
        track_norm = scalars["track_grad_norm"]
        aux_norm = scalars["weighted_aux_grad_norm"]
        total_norm = scalars["total_grad_norm"]
        aux_track_dot = scalars["weighted_aux_track_dot"]
        _require(
            track_norm >= 0.0 and aux_norm >= 0.0 and total_norm >= 0.0,
            f"I3 gradient record {index} has a negative norm",
        )
        _require(
            scalars["per_aux_aggregate_discrepancy_norm"] >= 0.0
            and scalars["per_aux_aggregate_rounding_bound_norm"] >= 0.0,
            f"I3 gradient record {index} has a negative reconstruction norm",
        )
        _require(
            scalars["per_aux_aggregate_discrepancy_norm"]
            <= scalars["per_aux_aggregate_rounding_bound_norm"] + 1e-12,
            f"I3 gradient record {index} per-aux aggregate reconstruction failed",
        )
        cauchy_slack = 1e-8 * max(1.0, aux_norm * track_norm)
        _require(
            abs(aux_track_dot) <= aux_norm * track_norm + cauchy_slack,
            f"I3 gradient record {index} violates the dot-product norm bound",
        )
        total_sq = track_norm * track_norm + aux_norm * aux_norm + 2.0 * aux_track_dot
        _require(total_sq >= -1e-9 * max(1.0, total_norm * total_norm), f"I3 gradient record {index} has impossible total norm")
        _require_close(total_norm * total_norm, max(0.0, total_sq), f"I3[{index}] total norm identity")

        weighted = _finite_named_map(
            record.get("per_aux_weighted_grad_norm"),
            label=f"I3[{index}].per_aux_weighted_grad_norm",
            nonnegative=True,
        )
        raw = _finite_named_map(
            record.get("per_aux_raw_grad_norm_derived_from_frozen_lambda"),
            label=f"I3[{index}].per_aux_raw_grad_norm_derived_from_frozen_lambda",
            nonnegative=True,
        )
        _finite_named_map(
            record.get("per_aux_cosine_to_track"),
            label=f"I3[{index}].per_aux_cosine_to_track",
        )
        _finite_named_map(
            record.get("per_aux_signed_projection_to_track"),
            label=f"I3[{index}].per_aux_signed_projection_to_track",
        )
        per_aux_below_eps = _bool_named_map(
            record.get("per_aux_norm_below_eps"),
            f"I3[{index}].per_aux_norm_below_eps",
        )
        for name in IBR1_AUX_COMPONENTS:
            _require_close(
                raw[name] * float(IBR1_FROZEN_AUX_COEFFICIENTS[name]),
                weighted[name],
                f"I3[{index}] frozen-lambda norm reconstruction for {name}",
            )
            _require(
                per_aux_below_eps[name] == (weighted[name] <= DIAGNOSTIC_EPS),
                f"I3[{index}] near-zero flag drift for {name}",
            )

        for field, expected in (
            ("track_norm_below_eps", track_norm <= DIAGNOSTIC_EPS),
            ("weighted_aux_norm_below_eps", aux_norm <= DIAGNOSTIC_EPS),
            ("total_norm_below_eps", total_norm <= DIAGNOSTIC_EPS),
        ):
            _require(isinstance(record.get(field), bool), f"I3[{index}].{field} must be boolean")
            _require(record[field] == expected, f"I3[{index}].{field} drifted")

        denominator = max(track_norm, DIAGNOSTIC_EPS)
        _require_close(scalars["actual_ratio_denominator"], denominator, f"I3[{index}] ratio denominator")
        expected_aux_cosine = _diagnostic_cosine(aux_track_dot, aux_norm, track_norm)
        expected_aux_projection = 0.0 if track_norm <= DIAGNOSTIC_EPS else aux_track_dot / track_norm
        _require_close(scalars["weighted_aux_track_cosine"], expected_aux_cosine, f"I3[{index}] aux cosine")
        _require_close(scalars["weighted_aux_signed_projection"], expected_aux_projection, f"I3[{index}] aux projection")

        expected_aux_reachable = aux_norm > 0.0
        expected_track_reachable = track_norm > 0.0
        _require(update.aux_reachable == expected_aux_reachable, f"I3 G6[{index}] aux reachability differs from diagnostics")
        _require(update.track_reachable == expected_track_reachable, f"I3 G6[{index}] track reachability differs from diagnostics")

        total_track_dot = aux_track_dot + track_norm * track_norm
        expected_cosine = total_track_dot / (
            max(total_norm, DIAGNOSTIC_EPS) * max(track_norm, DIAGNOSTIC_EPS)
        )
        expected_cosine = max(-1.0, min(1.0, expected_cosine))
        expected_projection = (total_track_dot / denominator) * I3_GRAD_ACCUM
        expected_ratio = aux_norm / denominator
        if index < 8:
            _require(
                update.cosine_total_track is None
                and update.signed_projection is None
                and update.aux_track_ratio is None
                and update.per_aux_ratios is None,
                f"I3 G6[{index}] emits forbidden pre-window geometry",
            )
        else:
            supplied_cosine = _finite_float(update.cosine_total_track, f"I3 G6[{index}] cosine")
            supplied_projection = _finite_float(update.signed_projection, f"I3 G6[{index}] projection")
            supplied_ratio = _finite_float(update.aux_track_ratio, f"I3 G6[{index}] aux/track ratio")
            _require(update.per_aux_ratios is None, f"I3 G6[{index}] mixes B* and per-aux ratios")
            _require_close(supplied_cosine, expected_cosine, f"I3 G6[{index}] total/track cosine")
            _require_close(supplied_projection, expected_projection, f"I3 G6[{index}] signed projection")
            _require_close(supplied_ratio, expected_ratio, f"I3 G6[{index}] weighted aux/track ratio")
        cross_checked += 1

    inherited_checks = _mapping(inherited_document.get("checks"), "I3 inherited G6 checks")
    required_inherited_checks = {
        "aux_reachability",
        "track_reachability",
        "zero_grad_clock",
        "cosine_median",
        "positive_projection",
        "aux_track_ratio_median",
    }
    _require(required_inherited_checks.issubset(inherited_checks), "I3 inherited G6 checks are incomplete")
    checks = {
        "inherited_G6": _check(
            inherited_document.get("passed") is True,
            observed=inherited_document.get("status"),
            comparator="==",
            threshold="PASS",
        ),
        "absolute_gradient_records": _check(
            cross_checked == I3_GRADIENT_UPDATES,
            observed=cross_checked,
            comparator="==",
            threshold=I3_GRADIENT_UPDATES,
        ),
        "G6_absolute_cross_check": _check(True, observed=True, comparator="==", threshold=True),
    }
    return _make_receipt(
        I3_GATE_ID,
        checks=checks,
        metrics={
            "records": cross_checked,
            "inherited_G6": inherited_document,
            "gradient_geometry_sha256": canonical_json_sha256(geometry),
        },
        thresholds={
            "aux_reachable_updates_min": I3_AUX_REACHABLE_MIN,
            "track_reachable_updates_min": I3_TRACK_REACHABLE_MIN,
            "cosine_total_track_median_min": I3_COSINE_MEDIAN_MIN,
            "positive_signed_projection_min": I3_POSITIVE_PROJECTION_MIN,
            "weighted_aux_over_track_median_max": I3_RATIO_MAX,
        },
        contract={
            "arm": IBR1_CTRL,
            "clock": "u_pre=0..127",
            "grad_accum": I3_GRAD_ACCUM,
            "F2_projection_conversion": "average-gradient projection * grad_accum",
            "F2_evaluator": "evaluate_g6(block_mode='bstar')",
        },
    )


def evaluate_i4(
    ctrl_g7_updates: Sequence[G7Update | Mapping[str, Any]],
    self_g7_updates: Sequence[G7Update | Mapping[str, Any]],
) -> IBR1GateReceipt:
    """Evaluate inherited G7 independently for CTRL and SELF, then AND."""

    arm_inputs = {IBR1_CTRL: ctrl_g7_updates, IBR1_SELF: self_g7_updates}
    arm_receipts: dict[str, dict[str, Any]] = {}
    checks: dict[str, Mapping[str, Any]] = {}
    for arm, updates in arm_inputs.items():
        _require(_is_sequence(updates), f"I4 {arm} G7 updates must be a sequence")
        normalized = tuple(_coerce_g7(value, index) for index, value in enumerate(updates))
        try:
            inherited = evaluate_g7(normalized)
        except Exception as exc:  # noqa: BLE001
            raise IBR1GateContractError(f"I4 {arm} inherited G7 input is invalid: {exc}") from exc
        document = _coerce_f2_receipt(inherited, f"I4 {arm} inherited G7 receipt")
        _require(document.get("gate_id") == "G7", f"I4 {arm} inherited receipt is not G7")
        arm_receipts[arm] = document
        checks[f"{arm}.inherited_G7"] = _check(
            document.get("passed") is True,
            observed=document.get("status"),
            comparator="==",
            threshold="PASS",
        )
        metrics = _mapping(document.get("metrics"), f"I4 {arm} G7 metrics")
        _require(metrics.get("updates") == 128, f"I4 {arm} G7 clock/cardinality drifted")
    return _make_receipt(
        I4_GATE_ID,
        checks=checks,
        metrics={"arms": arm_receipts},
        thresholds={"registry": "frozen F2 G7"},
        contract={"both_arms": "AND", "clock": "u_pre=0..127", "F2_evaluator": "evaluate_g7"},
    )


def evaluate_i5(
    s_self_update0: Mapping[str, Any],
    s_self_update128: Mapping[str, Any],
    s_ctrl_update128: Mapping[str, Any],
) -> IBR1GateReceipt:
    """Evaluate inherited fixed-support G8 with all four strata retained."""

    try:
        inherited = evaluate_g8(
            s_self_update0=s_self_update0,
            s_self_update128=s_self_update128,
            s_ctrl_update128=s_ctrl_update128,
        )
    except Exception as exc:  # noqa: BLE001
        raise IBR1GateContractError(f"I5 inherited G8 input is invalid: {exc}") from exc
    document = _coerce_f2_receipt(inherited, "I5 inherited G8 receipt")
    _require(document.get("gate_id") == "G8", "I5 inherited receipt is not G8")
    inherited_checks = _mapping(document.get("checks"), "I5 inherited G8 checks")
    required_strata = ("overall", "change", "turn", "other")
    _require(
        all(f"self_improvement.{stratum}" in inherited_checks for stratum in required_strata),
        "I5 inherited G8 is missing a registered self-improvement stratum",
    )
    metrics = _mapping(document.get("metrics"), "I5 inherited G8 metrics")
    counts = _mapping(metrics.get("support_counts"), "I5 G8 support counts")
    _require(set(counts) == set(required_strata), "I5 G8 support strata drifted")
    deltas = _mapping(metrics.get("self_mode_improvement_delta"), "I5 G8 improvement deltas")
    overall_delta = _finite_float(deltas.get("overall"), "I5 G8 overall delta")
    checks = {
        "inherited_G8": _check(document.get("passed") is True, observed=document.get("status"), comparator="==", threshold="PASS"),
        "all_registered_strata": _check(
            set(counts) == set(required_strata),
            observed=list(required_strata),
            comparator="==",
            threshold=list(required_strata),
        ),
        "self_mode_overall_delta": _check(overall_delta <= -1e-6, observed=overall_delta, comparator="<=", threshold=-1e-6),
    }
    return _make_receipt(
        I5_GATE_ID,
        checks=checks,
        metrics={"inherited_G8": document, "registered_strata": list(required_strata)},
        thresholds={"self_mode_overall_delta_max": -1e-6, "registry": "frozen F2 G8"},
        contract={"support": "EVAL-FIX", "all_registered_strata_required": True, "F2_evaluator": "evaluate_g8"},
    )


def _g9_mapping(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    required = {
        "expected_static_resets",
        "observed_static_resets",
        "nonfinite_reset_count",
        "range_violation_count",
        "range_observation_count",
        "reconstruction_errors",
        "first_quartile_self_errors",
        "last_quartile_self_errors",
    }
    _require(required.issubset(value), f"{label} is missing G9 inputs")
    document = dict(value)
    _canonical(document, label)
    document["reconstruction_errors"] = list(
        _reconstruction_error_rows(
            document["reconstruction_errors"],
            f"{label}.reconstruction_errors",
        )
    )
    return document


def _reconstruction_error_rows(value: Any, label: str) -> tuple[float, ...]:
    _require(
        _is_sequence(value),
        f"{label} must be the original one-dimensional row sequence",
    )
    _require(
        len(value) == I6_RECONSTRUCTION_ROWS,
        f"{label} must contain exactly {I6_RECONSTRUCTION_ROWS} row scalars",
    )
    rows: list[float] = []
    for index, item in enumerate(value):
        _require(
            not _is_sequence(item),
            f"{label}[{index}] must be one scalar, not nested support",
        )
        numeric = _coerce_scalar(item, f"{label}[{index}]")
        _require(numeric >= 0.0, f"{label}[{index}] must be nonnegative")
        rows.append(numeric)
    return tuple(rows)


def evaluate_i6(
    ctrl_g9_inputs: Mapping[str, Any],
    self_g9_inputs: Mapping[str, Any],
) -> IBR1GateReceipt:
    """Evaluate inherited G9 per arm and apply IBR1's zero-violation guard."""

    arm_inputs = {
        IBR1_CTRL: _g9_mapping(ctrl_g9_inputs, "I6 CTRL G9 inputs"),
        IBR1_SELF: _g9_mapping(self_g9_inputs, "I6 SELF G9 inputs"),
    }
    arm_metrics: dict[str, Any] = {}
    checks: dict[str, Mapping[str, Any]] = {}
    for arm, inputs in arm_inputs.items():
        try:
            inherited = evaluate_g9(**inputs)
        except Exception as exc:  # noqa: BLE001
            raise IBR1GateContractError(f"I6 {arm} inherited G9 input is invalid: {exc}") from exc
        document = _coerce_f2_receipt(inherited, f"I6 {arm} inherited G9 receipt")
        _require(document.get("gate_id") == "G9", f"I6 {arm} inherited receipt is not G9")
        reconstruction = _reconstruction_error_rows(
            inputs["reconstruction_errors"],
            f"I6 {arm} reconstruction errors",
        )
        reconstruction_max = max(abs(value) for value in reconstruction)
        nonfinite_resets = _nonnegative_int(inputs["nonfinite_reset_count"], f"I6 {arm} nonfinite resets")
        range_violations = _nonnegative_int(inputs["range_violation_count"], f"I6 {arm} range violations")
        range_observations = _nonnegative_int(inputs["range_observation_count"], f"I6 {arm} range observations")
        checks[f"{arm}.inherited_G9"] = _check(document.get("passed") is True, observed=document.get("status"), comparator="==", threshold="PASS")
        checks[f"{arm}.nonfinite_reset_count"] = _check(nonfinite_resets == 0, observed=nonfinite_resets, comparator="==", threshold=0)
        checks[f"{arm}.range_violation_count"] = _check(range_violations == 0, observed=range_violations, comparator="==", threshold=0)
        checks[f"{arm}.range_observation_count"] = _check(range_observations == I6_RANGE_OBSERVATIONS, observed=range_observations, comparator="==", threshold=I6_RANGE_OBSERVATIONS)
        checks[f"{arm}.reconstruction_rows"] = _check(len(reconstruction) == I6_RECONSTRUCTION_ROWS, observed=len(reconstruction), comparator="==", threshold=I6_RECONSTRUCTION_ROWS)
        checks[f"{arm}.reconstruction_error_max"] = _check(reconstruction_max <= I6_RECONSTRUCTION_MAX, observed=reconstruction_max, comparator="<=", threshold=I6_RECONSTRUCTION_MAX)
        arm_metrics[arm] = {
            "inherited_G9": document,
            "nonfinite_reset_count": nonfinite_resets,
            "range_violation_count": range_violations,
            "range_observation_count": range_observations,
            "reconstruction_rows": len(reconstruction),
            "reconstruction_error_max": reconstruction_max,
        }
    return _make_receipt(
        I6_GATE_ID,
        checks=checks,
        metrics={"arms": arm_metrics},
        thresholds={
            "nonfinite_reset_count": 0,
            "range_violation_count": 0,
            "range_observation_count": I6_RANGE_OBSERVATIONS,
            "reconstruction_rows": I6_RECONSTRUCTION_ROWS,
            "reconstruction_error_max": I6_RECONSTRUCTION_MAX,
            "inherited_drift_and_reset_registry": "frozen F2 G9",
        },
        contract={"both_arms": "AND", "range_rate_weakening_forbidden": True, "F2_evaluator": "evaluate_g9"},
    )


def _validate_critical_gate_metrics(
    gate_id: str,
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    _require(set(metrics) == set(_GATE_METRIC_KEYS[gate_id]), f"{label} metric keys drifted")
    if gate_id == "I1":
        for name in (
            "cal_rows",
            "cal_zero_init_failures",
            "cal_post_decode_violations",
            "cal_reconstruction_failures",
            "cal_prev_free_failures",
        ):
            _nonnegative_int(metrics[name], f"{label}.{name}")
        for name in ("cal_post_decode_abs_max", "cal_reconstruction_error_max"):
            _require(
                _finite_float(metrics[name], f"{label}.{name}") >= 0.0,
                f"{label}.{name} must be nonnegative",
            )
        _require(
            isinstance(metrics["cal_geometry_dtype"], str),
            f"{label}.cal_geometry_dtype must be a string",
        )
        tensor_shas = metrics["update0_checkpoint_tensor_sha256"]
        _require(
            _is_sequence(tensor_shas) and len(tensor_shas) == 2,
            f"{label} must carry two update-0 tensor SHAs",
        )
        for index, sha in enumerate(tensor_shas):
            _valid_sha256(sha, f"{label}.update0_checkpoint_tensor_sha256[{index}]")
        _valid_sha256(
            metrics["final_assembly_receipt_payload_sha256"],
            f"{label}.final_assembly_receipt_payload_sha256",
        )
        return
    if gate_id == "I2":
        _require(metrics["raw_records"] == 2 * I2_ROWS_PER_ARM, f"{label} raw record count drifted")
        _require(metrics["summary_exact_match"] is True, f"{label} summary is not exact")
        _valid_sha256(metrics["summary_sha256"], f"{label}.summary_sha256")
        arms = _mapping(metrics["arms"], f"{label}.arms")
        _require(set(arms) == {IBR1_CTRL, IBR1_SELF}, f"{label} arm metrics drifted")
        return
    if gate_id == "I3":
        _require(metrics["records"] == I3_GRADIENT_UPDATES, f"{label} gradient record count drifted")
        inherited = _coerce_f2_receipt(metrics["inherited_G6"], f"{label}.inherited_G6")
        _require(inherited["gate_id"] == "G6", f"{label} nested inherited gate is not G6")
        _valid_sha256(metrics["gradient_geometry_sha256"], f"{label}.gradient_geometry_sha256")
        return
    if gate_id == "I4":
        arms = _mapping(metrics["arms"], f"{label}.arms")
        _require(set(arms) == {IBR1_CTRL, IBR1_SELF}, f"{label} arm metrics drifted")
        for arm in (IBR1_CTRL, IBR1_SELF):
            inherited = _coerce_f2_receipt(arms[arm], f"{label}.{arm}")
            _require(inherited["gate_id"] == "G7", f"{label}.{arm} is not G7")
        return
    if gate_id == "I5":
        inherited = _coerce_f2_receipt(metrics["inherited_G8"], f"{label}.inherited_G8")
        _require(inherited["gate_id"] == "G8", f"{label} nested inherited gate is not G8")
        _require(
            metrics["registered_strata"] == ["overall", "change", "turn", "other"],
            f"{label} registered strata drifted",
        )
        return
    arms = _mapping(metrics["arms"], f"{label}.arms")
    _require(set(arms) == {IBR1_CTRL, IBR1_SELF}, f"{label} arm metrics drifted")
    for arm in (IBR1_CTRL, IBR1_SELF):
        item = _mapping(arms[arm], f"{label}.{arm}")
        required = {
            "inherited_G9",
            "nonfinite_reset_count",
            "range_violation_count",
            "range_observation_count",
            "reconstruction_rows",
            "reconstruction_error_max",
        }
        _require(set(item) == required, f"{label}.{arm} metric keys drifted")
        inherited = _coerce_f2_receipt(item["inherited_G9"], f"{label}.{arm}.inherited_G9")
        _require(inherited["gate_id"] == "G9", f"{label}.{arm} is not G9")
        for name in (
            "nonfinite_reset_count",
            "range_violation_count",
            "range_observation_count",
            "reconstruction_rows",
        ):
            _nonnegative_int(item[name], f"{label}.{arm}.{name}")
        _require(
            _finite_float(item["reconstruction_error_max"], f"{label}.{arm}.reconstruction_error_max")
            >= 0.0,
            f"{label}.{arm}.reconstruction_error_max must be nonnegative",
        )


def _frozen_check_result(
    *,
    comparator: str,
    observed: Any,
    threshold: Any,
) -> bool:
    if comparator == "==":
        return observed == threshold
    if comparator == "<":
        return _finite_float(observed, "gate check observed") < _finite_float(
            threshold,
            "gate check threshold",
        )
    if comparator == "<=":
        return _finite_float(observed, "gate check observed") <= _finite_float(
            threshold,
            "gate check threshold",
        )
    if comparator == "all":
        return (
            isinstance(observed, Mapping)
            and bool(observed)
            and all(value is True for value in observed.values())
            and threshold is True
        )
    if comparator == "has":
        return (
            isinstance(observed, Mapping)
            and _is_sequence(threshold)
            and all(name in observed for name in threshold)
        )
    if comparator == "frozen_zero_init_contract":
        return isinstance(observed, Mapping) and dict(observed) == dict(threshold)
    if comparator == "rows/cells/shape/violations/abs_max":
        if not isinstance(observed, Mapping) or not isinstance(threshold, Mapping):
            return False
        return (
            observed.get("rows") == threshold.get("rows")
            and observed.get("cells") == threshold.get("cells")
            and observed.get("shape") == threshold.get("shape")
            and observed.get("violations") == threshold.get("violations")
            and _finite_float(observed.get("abs_max"), "post-decode observed max")
            <= _finite_float(threshold.get("abs_max"), "post-decode threshold max")
        )
    if comparator == "rows/cells/shape/failures/error_max":
        if not isinstance(observed, Mapping) or not isinstance(threshold, Mapping):
            return False
        return (
            observed.get("rows") == threshold.get("rows")
            and observed.get("cells") == threshold.get("cells")
            and observed.get("shape") == threshold.get("shape")
            and observed.get("failures") == threshold.get("failures")
            and _finite_float(observed.get("error_max"), "reconstruction observed max")
            <= _finite_float(threshold.get("error_max"), "reconstruction threshold max")
        )
    raise IBR1GateContractError(f"unknown frozen check comparator {comparator!r}")


def _validate_gate_metric_check_consistency(
    gate_id: str,
    checks: Mapping[str, Any],
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    for name, value in checks.items():
        check = _mapping(value, f"{label}.checks.{name}")
        recomputed = _frozen_check_result(
            comparator=str(check["comparator"]),
            observed=check["observed"],
            threshold=check["threshold"],
        )
        _require(
            check["passed"] is recomputed,
            f"{label}.checks.{name} verdict differs from observed/threshold",
        )
    if gate_id == "I1":
        _require(checks["cal_rows"]["observed"] == metrics["cal_rows"], f"{label} CAL row metric drifted")
        _require(checks["cal_fp32"]["observed"] == metrics["cal_geometry_dtype"], f"{label} CAL dtype metric drifted")
        _require(
            checks["post_decode_range"]["observed"]["violations"]
            == metrics["cal_post_decode_violations"]
            and checks["post_decode_range"]["observed"]["abs_max"]
            == metrics["cal_post_decode_abs_max"],
            f"{label} post-decode metrics drifted",
        )
        _require(
            checks["reconstruction"]["observed"]["failures"]
            == metrics["cal_reconstruction_failures"]
            and checks["reconstruction"]["observed"]["error_max"]
            == metrics["cal_reconstruction_error_max"],
            f"{label} reconstruction metrics drifted",
        )
        return
    if gate_id == "I2":
        arms = _mapping(metrics["arms"], f"{label}.arms")
        for arm in (IBR1_CTRL, IBR1_SELF):
            item = _mapping(arms[arm], f"{label}.{arm}")
            _require(
                checks[f"{arm}.row_cardinality"]["observed"] == item["rows"]
                and checks[f"{arm}.denominator"]["observed"]
                == item["I2_any_axis_denominator"]
                and checks[f"{arm}.violation_rate"]["observed"]
                == item["I2_any_axis_violation_rate"],
                f"{label}.{arm} check/metric drifted",
            )
        return
    if gate_id == "I3":
        _require(
            checks["absolute_gradient_records"]["observed"] == metrics["records"]
            and checks["inherited_G6"]["observed"]
            == metrics["inherited_G6"]["status"],
            f"{label} G6 check/metric drifted",
        )
        return
    if gate_id == "I4":
        for arm in (IBR1_CTRL, IBR1_SELF):
            _require(
                checks[f"{arm}.inherited_G7"]["observed"]
                == metrics["arms"][arm]["status"],
                f"{label}.{arm} G7 check/metric drifted",
            )
        return
    if gate_id == "I5":
        _require(
            checks["inherited_G8"]["observed"] == metrics["inherited_G8"]["status"]
            and checks["all_registered_strata"]["observed"]
            == metrics["registered_strata"],
            f"{label} G8 check/metric drifted",
        )
        return
    for arm in (IBR1_CTRL, IBR1_SELF):
        item = metrics["arms"][arm]
        _require(
            checks[f"{arm}.inherited_G9"]["observed"]
            == item["inherited_G9"]["status"]
            and checks[f"{arm}.nonfinite_reset_count"]["observed"]
            == item["nonfinite_reset_count"]
            and checks[f"{arm}.range_violation_count"]["observed"]
            == item["range_violation_count"]
            and checks[f"{arm}.range_observation_count"]["observed"]
            == item["range_observation_count"]
            and checks[f"{arm}.reconstruction_rows"]["observed"]
            == item["reconstruction_rows"]
            and checks[f"{arm}.reconstruction_error_max"]["observed"]
            == item["reconstruction_error_max"],
            f"{label}.{arm} G9 check/metric drifted",
        )


def _coerce_ibr1_gate_receipt(value: Any, label: str) -> dict[str, Any]:
    document = value.to_dict() if isinstance(value, IBR1GateReceipt) else value
    receipt = _sealed_mapping(
        document,
        label=label,
        analysis_class=IBR1_GATE_RECEIPT_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    gate_id = receipt.get("gate_id")
    _require(gate_id in IBR1_GATE_IDS, f"{label} has an unknown gate id")
    required_top_level = {
        "schema_version",
        "analysis_class",
        "family_id",
        "architecture_lock",
        "gate_id",
        "valid_input",
        "passed",
        "status",
        "decision",
        "checks",
        "metrics",
        "thresholds",
        "contract",
        "formal_training_authorized",
        "internal_test",
        "internal_test_opened",
        "receipt_payload_sha256",
    }
    _require(set(receipt) == required_top_level, f"{label} top-level keys drifted")
    _require(receipt.get("schema_version") == 1, f"{label} schema version drifted")
    _require(receipt.get("valid_input") is True, f"{label} is not valid input")
    _require(receipt.get("formal_training_authorized") is False, f"{label} formal policy drifted")
    checks = _mapping(receipt.get("checks"), f"{label} checks")
    _require(set(checks) == set(_GATE_CHECK_NAMES[str(gate_id)]), f"{label} check names drifted")
    for name, check_value in checks.items():
        check = _mapping(check_value, f"{label}.checks.{name}")
        _require(
            set(check) == {"passed", "observed", "comparator", "threshold"},
            f"{label}.checks.{name} schema drifted",
        )
        _require(isinstance(check["passed"], bool), f"{label}.checks.{name}.passed must be boolean")
        comparator, threshold = _expected_check_signature(str(gate_id), name)
        _require(
            check["comparator"] == comparator and check["threshold"] == threshold,
            f"{label}.checks.{name} frozen signature drifted",
        )
    thresholds = _mapping(receipt.get("thresholds"), f"{label} thresholds")
    _require(
        set(thresholds) == set(_GATE_THRESHOLD_KEYS[str(gate_id)])
        and dict(thresholds) == _expected_gate_thresholds(str(gate_id)),
        f"{label} frozen thresholds drifted",
    )
    metrics = _mapping(receipt.get("metrics"), f"{label} metrics")
    _validate_critical_gate_metrics(str(gate_id), metrics, f"{label}.metrics")
    _validate_gate_metric_check_consistency(
        str(gate_id),
        checks,
        metrics,
        label,
    )
    _require(isinstance(receipt.get("contract"), Mapping), f"{label} contract must be a mapping")
    _require(bool(receipt["contract"]), f"{label} contract must be nonempty")
    expected_pass = all(
        isinstance(check, Mapping) and check.get("passed") is True
        for check in checks.values()
    )
    _require(receipt.get("passed") is expected_pass, f"{label} pass/check inconsistency")
    _require(
        receipt.get("status") == ("PASS" if expected_pass else "FAIL")
        and receipt.get("decision") == ("PASS" if expected_pass else "STOP"),
        f"{label} status/decision inconsistency",
    )
    return receipt


def _require_i1_filesystem_authority(
    receipt: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    _require(receipt.get("gate_id") == I1_GATE_ID, f"{label} is not I1")
    contract = _mapping(receipt.get("contract"), f"{label} contract")
    _require(
        contract.get("evidence_mode") == "filesystem_bound",
        f"{label} is non-authority and cannot enter an authoritative combined receipt",
    )
    evidence = _mapping(contract.get("live_evidence"), f"{label} live evidence")
    _require(
        set(evidence) == {"final_assembly", "update0_checkpoint_sidecars"},
        f"{label} live evidence schema drifted",
    )
    final_binding = _validate_receipt_artifact_binding(
        evidence.get("final_assembly"),
        label=f"{label} final assembly evidence",
        analysis_class=ASSEMBLY_RECEIPT_CLASS,
    )
    sidecars = _mapping(
        evidence.get("update0_checkpoint_sidecars"),
        f"{label} update-0 sidecar evidence",
    )
    _require(
        set(sidecars) == {IBR1_CTRL, IBR1_SELF},
        f"{label} update-0 sidecar evidence does not cover both arms",
    )
    normalized_sidecars: dict[str, dict[str, Any]] = {}
    paths: set[str] = set()
    for arm in (IBR1_CTRL, IBR1_SELF):
        binding = _mapping(sidecars[arm], f"{label} {arm} sidecar evidence")
        _require(
            set(binding) == {"path", "sha256", "analysis_class"},
            f"{label} {arm} sidecar evidence schema drifted",
        )
        path = _portable_project_relative_path(binding.get("path"), f"{label} {arm} sidecar path")
        _require(path not in paths, f"{label} sidecar evidence paths must be distinct")
        paths.add(path)
        normalized_sidecars[arm] = {
            "path": path,
            "sha256": _valid_sha256(binding.get("sha256"), f"{label} {arm} sidecar SHA"),
            "analysis_class": CHECKPOINT_SIDECAR_CLASS,
        }
        _require(
            binding.get("analysis_class") == CHECKPOINT_SIDECAR_CLASS,
            f"{label} {arm} sidecar evidence class drifted",
        )
    return {
        "final_assembly": final_binding,
        "update0_checkpoint_sidecars": normalized_sidecars,
    }


def build_ibr1_combined_gate_receipt(
    *receipts: IBR1GateReceipt | Mapping[str, Any] | Sequence[IBR1GateReceipt | Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine exactly I1-I6 while permanently withholding formal authority."""

    if len(receipts) == 1 and _is_sequence(receipts[0]):
        values = tuple(receipts[0])
    else:
        values = receipts
    by_gate: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(values):
        document = _coerce_ibr1_gate_receipt(value, f"combined IBR1 gate[{index}]")
        gate_id = str(document["gate_id"])
        _require(gate_id not in by_gate, f"duplicate combined IBR1 gate {gate_id}")
        by_gate[gate_id] = document
    _require(set(by_gate) == set(IBR1_GATE_IDS), "combined IBR1 receipt requires exactly I1-I6")
    _require_i1_filesystem_authority(by_gate[I1_GATE_ID], "combined IBR1 I1 gate")
    passed = all(by_gate[gate_id]["passed"] is True for gate_id in IBR1_GATE_IDS)
    next_step = (
        "independent_review_then_new_preregistration"
        if passed
        else "SEAL_STOP"
    )
    payload = {
        "schema_version": 1,
        "analysis_class": IBR1_COMBINED_GATE_RECEIPT_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "valid_input": True,
        "gate_order": list(IBR1_GATE_IDS),
        "mechanism_pass": passed,
        "passed": passed,
        "status": "PASS" if passed else "FAIL",
        "decision": "MECHANISM_PASS" if passed else "SEAL_STOP",
        "next_step": next_step,
        "formal_training_authorized": False,
        "same_family_retry_authorized": False,
        "same_family_seed_change_authorized": False,
        "same_family_lambda_change_authorized": False,
        "same_family_decode_change_authorized": False,
        "same_family_gate_change_authorized": False,
        "gates": {gate_id: by_gate[gate_id] for gate_id in IBR1_GATE_IDS},
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    return _self_hash(payload, "combined IBR1 gate receipt")


def _coerce_combined_receipt(value: Any, label: str) -> dict[str, Any]:
    document = _sealed_mapping(
        value,
        label=label,
        analysis_class=IBR1_COMBINED_GATE_RECEIPT_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    _require(document.get("valid_input") is True, f"{label} is not valid input")
    gates = _mapping(document.get("gates"), f"{label} gates")
    _require(list(document.get("gate_order", ())) == list(IBR1_GATE_IDS), f"{label} gate order drifted")
    _require(set(gates) == set(IBR1_GATE_IDS), f"{label} does not contain I1-I6")
    normalized = {
        gate_id: _coerce_ibr1_gate_receipt(gates[gate_id], f"{label}.{gate_id}")
        for gate_id in IBR1_GATE_IDS
    }
    _require_i1_filesystem_authority(normalized[I1_GATE_ID], f"{label}.I1")
    expected_pass = all(normalized[gate_id]["passed"] is True for gate_id in IBR1_GATE_IDS)
    _require(
        document.get("passed") is expected_pass
        and document.get("mechanism_pass") is expected_pass,
        f"{label} mechanism verdict is inconsistent with I1-I6",
    )
    _require(
        document.get("status") == ("PASS" if expected_pass else "FAIL")
        and document.get("decision")
        == ("MECHANISM_PASS" if expected_pass else "SEAL_STOP"),
        f"{label} status/decision is inconsistent with its mechanism verdict",
    )
    _require(document.get("formal_training_authorized") is False, f"{label} formal policy drifted")
    expected_next = "independent_review_then_new_preregistration" if expected_pass else "SEAL_STOP"
    _require(document.get("next_step") == expected_next, f"{label} next-step policy drifted")
    for field in (
        "same_family_retry_authorized",
        "same_family_seed_change_authorized",
        "same_family_lambda_change_authorized",
        "same_family_decode_change_authorized",
        "same_family_gate_change_authorized",
    ):
        _require(document.get(field) is False, f"{label} weakens {field}")
    return document


def build_ibr1_candidate_lock_receipt() -> dict[str, Any]:
    """Build the single-candidate seed-0 lock required before smoke."""

    return _self_hash(
        {
            "schema_version": 1,
            "analysis_class": IBR1_CANDIDATE_LOCK_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "candidate": "IBR1 normalized cumulative bounded residual",
            "candidate_cap": 1,
            "candidate_index": 0,
            "seed": 0,
            "package": "SA-Hstar",
            "formal_training_authorized": False,
            "same_family_retry_authorized": False,
            "same_family_seed_selection_authorized": False,
            "same_family_lambda_tuning_authorized": False,
            "same_family_decode_tuning_authorized": False,
            "same_family_gate_tuning_authorized": False,
            "lock_state": "sealed",
            "internal_test": "sealed",
            "internal_test_opened": False,
        },
        "IBR1 candidate lock",
    )


def _verify_candidate_lock(value: Any, label: str = "IBR1 candidate lock") -> dict[str, Any]:
    document = _sealed_mapping(
        value,
        label=label,
        analysis_class=IBR1_CANDIDATE_LOCK_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    _require(
        document.get("candidate_cap") == 1
        and document.get("candidate_index") == 0
        and document.get("seed") == 0
        and document.get("package") == "SA-Hstar"
        and document.get("formal_training_authorized") is False
        and document.get("lock_state") == "sealed",
        f"{label} identity drifted",
    )
    for field in (
        "same_family_retry_authorized",
        "same_family_seed_selection_authorized",
        "same_family_lambda_tuning_authorized",
        "same_family_decode_tuning_authorized",
        "same_family_gate_tuning_authorized",
    ):
        _require(document.get(field) is False, f"{label} weakens {field}")
    return document


def _exclusive_write_json(path: Path, document: Mapping[str, Any], label: str) -> str:
    payload = _canonical(dict(document), label) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
    except FileExistsError as exc:
        raise IBR1GateContractError(f"refusing to overwrite {label}: {path}") from exc
    except OSError as exc:
        raise IBR1GateContractError(f"cannot write {label}: {path}") from exc
    return _sha256_bytes(payload)


def _freeze_document(
    output: str | Path,
    document: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    destination = Path(output).expanduser().resolve()
    file_sha = _exclusive_write_json(destination, document, label)
    binding: dict[str, Any] = {
        "path": str(destination),
        "sha256": file_sha,
        "analysis_class": document.get("analysis_class"),
    }
    if "receipt_payload_sha256" in document:
        binding["receipt_payload_sha256"] = document["receipt_payload_sha256"]
    return binding


def freeze_ibr1_gate_receipt(
    output: str | Path,
    receipt: IBR1GateReceipt | Mapping[str, Any],
) -> dict[str, Any]:
    document = _coerce_ibr1_gate_receipt(receipt, "IBR1 gate receipt")
    return _freeze_document(output, document, label=f"IBR1 {document['gate_id']} receipt")


def freeze_ibr1_combined_gate_receipt(
    output: str | Path,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    document = _coerce_combined_receipt(receipt, "combined IBR1 gate receipt")
    return _freeze_document(output, document, label="combined IBR1 gate receipt")


def freeze_ibr1_candidate_lock_receipt(output: str | Path) -> dict[str, Any]:
    document = build_ibr1_candidate_lock_receipt()
    return _freeze_document(output, document, label="IBR1 candidate lock")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = path.read_bytes()
        value = json.loads(payload.decode("utf-8"), parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IBR1GateContractError(f"cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must contain a JSON object")
    _require(payload == _canonical(value, label) + b"\n", f"{label} is not canonical JSON plus LF")
    return value


def _load_canonical_jsonl(
    path: Path,
    *,
    label: str,
    expected_records: int,
) -> tuple[dict[str, Any], ...]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise IBR1GateContractError(f"cannot read {label}: {path}") from exc
    _require(payload.endswith(b"\n"), f"{label} must end in LF")
    lines = payload.splitlines()
    _require(len(lines) == expected_records, f"{label} record count drifted")
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        _require(bool(line), f"{label}[{index}] is empty")
        try:
            value = json.loads(
                line.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise IBR1GateContractError(
                f"cannot read {label}[{index}]"
            ) from exc
        _require(isinstance(value, dict), f"{label}[{index}] must be a JSON object")
        _require(
            line == _canonical(value, f"{label}[{index}]"),
            f"{label}[{index}] is not canonical JSON",
        )
        records.append(value)
    return tuple(records)


def _g6_updates_from_gradient_geometry(
    gradient_geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    records = gradient_geometry.get("records")
    _require(_is_sequence(records), "diagnostics gradient geometry records are missing")
    updates: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        record = _mapping(value, f"diagnostics gradient record {index}")
        track_norm = _finite_float(
            record.get("track_grad_norm"),
            f"diagnostics gradient record {index} track norm",
        )
        aux_norm = _finite_float(
            record.get("weighted_aux_grad_norm"),
            f"diagnostics gradient record {index} weighted auxiliary norm",
        )
        total_norm = _finite_float(
            record.get("total_grad_norm"),
            f"diagnostics gradient record {index} total norm",
        )
        aux_track_dot = _finite_float(
            record.get("weighted_aux_track_dot"),
            f"diagnostics gradient record {index} auxiliary/track dot",
        )
        update: dict[str, Any] = {
            "u_pre": index,
            "aux_reachable": aux_norm > 0.0,
            "track_reachable": track_norm > 0.0,
        }
        if index >= 8:
            denominator = max(track_norm, DIAGNOSTIC_EPS)
            total_track_dot = aux_track_dot + track_norm * track_norm
            cosine = total_track_dot / (
                max(total_norm, DIAGNOSTIC_EPS) * max(track_norm, DIAGNOSTIC_EPS)
            )
            update.update(
                {
                    "cosine_total_track": max(-1.0, min(1.0, cosine)),
                    "signed_projection": (
                        total_track_dot / denominator
                    )
                    * I3_GRAD_ACCUM,
                    "aux_track_ratio": aux_norm / denominator,
                }
            )
        updates.append(update)
    return tuple(updates)


def _validate_diagnostics_bundle(
    project_root: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Verify bytes, recompute summaries, and rebuild diagnostics-backed gates."""

    _require(
        manifest_path.name == DIAGNOSTICS_MANIFEST_FILENAME,
        "diagnostics manifest filename drifted",
    )
    required_manifest_keys = {
        "schema_version",
        "analysis_class",
        "family_id",
        "architecture_lock",
        "artifacts",
        "lifecycle_bindings",
        "manifest_written_after_all_bound_artifacts",
        "formal_training_authorized",
        "internal_test",
        "internal_test_opened",
        "receipt_payload_sha256",
    }
    _require(set(manifest) == required_manifest_keys, "diagnostics manifest schema drifted")
    _require(manifest.get("schema_version") == 1, "diagnostics manifest version drifted")
    artifacts = _mapping(manifest.get("artifacts"), "diagnostics manifest artifacts")
    expected = {
        TRAINING_GEOMETRY_FILENAME: ("jsonl", EXPECTED_TRAINING_RECORDS, None),
        EVAL_GEOMETRY_FILENAME: ("jsonl", EXPECTED_EVAL_RECORDS, None),
        GRADIENT_GEOMETRY_FILENAME: (
            "json",
            EXPECTED_GRADIENT_RECORDS,
            "ibr1_gradient_geometry",
        ),
        OPTIMIZER_GEOMETRY_FILENAME: (
            "json",
            EXPECTED_OPTIMIZER_RECORDS,
            "ibr1_optimizer_geometry",
        ),
        DIAGNOSTICS_SUMMARY_FILENAME: (
            "json",
            None,
            "ibr1_diagnostics_summary",
        ),
    }
    _require(set(artifacts) == set(expected), "diagnostics artifact set drifted")
    verified_artifacts: dict[str, Any] = {}
    loaded_jsonl: dict[str, tuple[dict[str, Any], ...]] = {}
    loaded_json: dict[str, dict[str, Any]] = {}
    for filename, (artifact_format, record_count, analysis_class) in expected.items():
        entry = _mapping(artifacts[filename], f"diagnostics artifact entry {filename}")
        expected_keys = {"filename", "sha256", "bytes", "format"}
        if record_count is not None:
            expected_keys.add("records")
        _require(set(entry) == expected_keys, f"diagnostics artifact entry {filename} schema drifted")
        _require(entry.get("filename") == filename, f"diagnostics artifact entry {filename} filename drifted")
        _require(entry.get("format") == artifact_format, f"diagnostics artifact {filename} format drifted")
        if record_count is not None:
            _require(entry.get("records") == record_count, f"diagnostics artifact {filename} declared records drifted")
        declared_bytes = _nonnegative_int(entry.get("bytes"), f"diagnostics artifact {filename} bytes")
        declared_sha = _valid_sha256(entry.get("sha256"), f"diagnostics artifact {filename} SHA")
        artifact_path = (manifest_path.parent / filename).resolve()
        _require(
            artifact_path.parent == manifest_path.parent
            and artifact_path.is_relative_to(project_root),
            f"diagnostics artifact {filename} escapes the controlled directory",
        )
        _require(artifact_path.is_file(), f"diagnostics artifact {filename} is missing")
        payload = artifact_path.read_bytes()
        _require(len(payload) == declared_bytes, f"diagnostics artifact {filename} byte count drifted")
        _require(_sha256_bytes(payload) == declared_sha, f"diagnostics artifact {filename} SHA drifted")

        if artifact_format == "jsonl":
            loaded_jsonl[filename] = _load_canonical_jsonl(
                artifact_path,
                label=f"diagnostics artifact {filename}",
                expected_records=int(record_count),
            )
        else:
            document = _load_canonical_json(
                artifact_path,
                f"diagnostics artifact {filename}",
            )
            normalized = _sealed_mapping(
                document,
                label=f"diagnostics artifact {filename}",
                analysis_class=str(analysis_class),
                require_self_hash=filename == DIAGNOSTICS_SUMMARY_FILENAME,
                require_architecture=filename == DIAGNOSTICS_SUMMARY_FILENAME,
            )
            loaded_json[filename] = normalized
            if record_count is not None:
                records = normalized.get("records")
                _require(
                    _is_sequence(records) and len(records) == record_count,
                    f"diagnostics artifact {filename} content cardinality drifted",
                )
            else:
                _require(
                    normalized.get("formal_training_authorized") is False
                    and isinstance(normalized.get("training_geometry"), Mapping)
                    and isinstance(normalized.get("eval_geometry"), Mapping),
                    "diagnostics summary schema drifted",
                )
        artifact_binding: dict[str, Any] = {
            "path": _relative_path(project_root, artifact_path),
            "sha256": declared_sha,
            "bytes": declared_bytes,
            "format": artifact_format,
        }
        if record_count is not None:
            artifact_binding["records"] = record_count
        verified_artifacts[filename] = artifact_binding

    summary_document = loaded_json[DIAGNOSTICS_SUMMARY_FILENAME]
    expected_summary_keys = {
        "schema_version",
        "analysis_class",
        "family_id",
        "architecture_lock",
        "training_geometry",
        "eval_geometry",
        "engineering_fail_closed",
        "formal_training_authorized",
        "internal_test",
        "internal_test_opened",
        "receipt_payload_sha256",
    }
    _require(
        set(summary_document) == expected_summary_keys,
        "diagnostics summary schema drifted",
    )
    _require(
        summary_document.get("engineering_fail_closed") is False,
        "diagnostics summary reports an engineering failure",
    )
    training_summary = _mapping(
        summary_document.get("training_geometry"),
        "diagnostics training summary",
    )
    eval_summary = _mapping(
        summary_document.get("eval_geometry"),
        "diagnostics EVAL summary",
    )
    recomputed_i2 = evaluate_i2(
        loaded_jsonl[TRAINING_GEOMETRY_FILENAME],
        training_summary,
    ).to_dict()
    eval_collector = GeometryCollector()
    eval_collector.eval_records = deepcopy(
        list(loaded_jsonl[EVAL_GEOMETRY_FILENAME])
    )
    try:
        recomputed_eval_summary = eval_collector._validate_eval()
    except Exception as exc:  # noqa: BLE001 - normalize diagnostics failures
        raise IBR1GateContractError(
            f"diagnostics raw EVAL geometry failed validation: {exc}"
        ) from exc
    _require(
        _canonical(recomputed_eval_summary, "recomputed diagnostics EVAL summary")
        == _canonical(eval_summary, "supplied diagnostics EVAL summary"),
        "diagnostics EVAL summary differs from exact raw-row recomputation",
    )
    gradient_geometry = loaded_json[GRADIENT_GEOMETRY_FILENAME]
    recomputed_i3 = evaluate_i3(
        _g6_updates_from_gradient_geometry(gradient_geometry),
        gradient_geometry,
    ).to_dict()

    lifecycle = _mapping(
        manifest.get("lifecycle_bindings"),
        "diagnostics lifecycle bindings",
    )
    _require(
        set(lifecycle) == _LIFECYCLE_BINDING_KEYS,
        "diagnostics lifecycle binding keys drifted",
    )
    for name, value in lifecycle.items():
        binding = _mapping(value, f"diagnostics lifecycle binding {name}")
        _require(bool(binding), f"diagnostics lifecycle binding {name} is empty")
        _canonical(binding, f"diagnostics lifecycle binding {name}")
        if "verified" in binding:
            _require(
                binding.get("verified") is True,
                f"diagnostics lifecycle binding {name} is not verified",
            )
    return (
        {
            "directory": _relative_path(project_root, manifest_path.parent),
            "artifacts": verified_artifacts,
            "lifecycle_bindings_sha256": canonical_json_sha256(lifecycle),
            "recomputed_training_geometry_sha256": canonical_json_sha256(
                training_summary
            ),
            "recomputed_eval_geometry_sha256": canonical_json_sha256(
                recomputed_eval_summary
            ),
            "recomputed_i2_receipt_payload_sha256": recomputed_i2[
                "receipt_payload_sha256"
            ],
            "recomputed_i3_receipt_payload_sha256": recomputed_i3[
                "receipt_payload_sha256"
            ],
        },
        {I2_GATE_ID: recomputed_i2, I3_GATE_ID: recomputed_i3},
    )


def _rooted_path(project_root: Path, value: str | Path, label: str) -> Path:
    raw = Path(value).expanduser()
    path = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise IBR1GateContractError(f"{label} lies outside project_root: {path}") from exc
    return path


def _relative_path(project_root: Path, path: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise IBR1GateContractError(f"artifact lies outside project_root: {path}") from exc


def _artifact_binding(
    project_root: Path,
    path: Path,
    document: Mapping[str, Any],
) -> dict[str, Any]:
    binding: dict[str, Any] = {
        "path": _relative_path(project_root, path),
        "sha256": _sha256_bytes(path.read_bytes()),
        "analysis_class": document.get("analysis_class"),
    }
    if "receipt_payload_sha256" in document:
        binding["receipt_payload_sha256"] = _verify_self_hash(
            document,
            f"artifact {path.name}",
        )
    return binding


def _load_bound_artifact(
    project_root: Path,
    value: str | Path,
    *,
    label: str,
    analysis_class: str,
    require_self_hash: bool,
    require_architecture: bool = False,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = _rooted_path(project_root, value, label)
    document = _load_canonical_json(path, label)
    normalized = _sealed_mapping(
        document,
        label=label,
        analysis_class=analysis_class,
        require_self_hash=require_self_hash,
        require_architecture=require_architecture,
    )
    return path, normalized, _artifact_binding(project_root, path, normalized)


def _checkpoint_sidecar_artifacts(
    project_root: Path,
    values: Mapping[str, str | Path] | Sequence[str | Path],
    *,
    expected_final_binding: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = list(values.values()) if isinstance(values, Mapping) else list(values)
    _require(len(paths) == 4, "result seal requires four checkpoint sidecars")
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    update0_tensor_shas: list[str] = []
    for index, value in enumerate(paths):
        path, document, binding = _load_bound_artifact(
            project_root,
            value,
            label=f"IBR1 checkpoint sidecar[{index}]",
            analysis_class=CHECKPOINT_SIDECAR_CLASS,
            require_self_hash=False,
            require_architecture=True,
        )
        _require(set(document) == _SIDECAR_KEYS, f"checkpoint sidecar {path} schema drifted")
        arm = document.get("family_arm")
        u_pre = document.get("u_pre")
        _require(arm in (IBR1_CTRL, IBR1_SELF), f"checkpoint sidecar {path} has an unknown arm")
        _require(u_pre in (0, 128), f"checkpoint sidecar {path} has an unknown update")
        expected_engine = "S-CTRL" if arm == IBR1_CTRL else "S-SELF"
        _require(document.get("engine_arm") == expected_engine, f"checkpoint sidecar {path} engine arm drifted")
        key = f"{arm}:update{u_pre}"
        _require(key not in documents, f"duplicate checkpoint sidecar identity {key}")
        _require(
            dict(_mapping(document.get("final_assembly_receipt"), f"{key} assembly binding"))
            == dict(expected_final_binding),
            f"checkpoint sidecar {key} binds a different final assembly",
        )
        tensor_sha = _valid_sha256(document.get("checkpoint_tensor_sha256"), f"{key} tensor SHA")
        _valid_sha256(document.get("model_source_sha256"), f"{key} model source SHA")
        source_sha = _mapping(document.get("source_sha256"), f"{key} source SHA mapping")
        _require(bool(source_sha), f"{key} source SHA mapping is empty")
        for source_name, source_value in source_sha.items():
            _require(isinstance(source_name, str) and bool(source_name), f"{key} source name is invalid")
            _valid_sha256(source_value, f"{key} source SHA {source_name}")
        _require(isinstance(document.get("state_schema"), Mapping), f"{key} state schema is malformed")
        _require(isinstance(document.get("snapshot_policy"), Mapping), f"{key} snapshot policy is malformed")
        checkpoint_name = _portable_project_relative_path(
            document.get("checkpoint_file"),
            f"{key} checkpoint_file",
        )
        _require(
            PurePosixPath(checkpoint_name).name == checkpoint_name,
            f"{key} checkpoint_file must be one adjacent filename",
        )
        checkpoint_path = (path.parent / checkpoint_name).resolve()
        _require(
            checkpoint_path.parent == path.parent
            and checkpoint_path.is_relative_to(project_root),
            f"{key} checkpoint file escapes the controlled directory",
        )
        checkpoint_file_sha = _valid_sha256(
            document.get("checkpoint_file_sha256"),
            f"{key} checkpoint file SHA",
        )
        _require(checkpoint_path.is_file(), f"{key} checkpoint file is missing")
        _require(
            _sha256_bytes(checkpoint_path.read_bytes()) == checkpoint_file_sha,
            f"{key} checkpoint file SHA drifted",
        )
        if u_pre == 0:
            update0_tensor_shas.append(tensor_sha)
        documents[key] = document
        binding["checkpoint_file"] = checkpoint_name
        binding["checkpoint_file_path"] = _relative_path(
            project_root,
            checkpoint_path,
        )
        binding["checkpoint_file_sha256"] = checkpoint_file_sha
        bindings[key] = binding
    required = {
        f"{IBR1_CTRL}:update0",
        f"{IBR1_CTRL}:update128",
        f"{IBR1_SELF}:update0",
        f"{IBR1_SELF}:update128",
    }
    _require(set(documents) == required, "checkpoint sidecars do not cover CTRL/SELF update0/update128")
    _require(len(set(update0_tensor_shas)) == 1, "update-0 checkpoint tensor SHAs differ between arms")
    return documents, bindings


def _gate_receipt_artifacts(
    project_root: Path,
    values: Mapping[str, str | Path] | Sequence[str | Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    paths = list(values.values()) if isinstance(values, Mapping) else list(values)
    _require(len(paths) == len(IBR1_GATE_IDS), "result seal requires six IBR1 gate receipt paths")
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(paths):
        path, raw, binding = _load_bound_artifact(
            project_root,
            value,
            label=f"IBR1 gate receipt[{index}]",
            analysis_class=IBR1_GATE_RECEIPT_CLASS,
            require_self_hash=True,
            require_architecture=True,
        )
        document = _coerce_ibr1_gate_receipt(raw, f"IBR1 gate receipt {path.name}")
        gate_id = str(document["gate_id"])
        _require(gate_id not in documents, f"duplicate bound gate receipt {gate_id}")
        documents[gate_id] = document
        bindings[gate_id] = binding
    _require(set(documents) == set(IBR1_GATE_IDS), "bound gate receipts do not cover I1-I6")
    return documents, bindings


def _count_receipt_artifact(
    project_root: Path,
    value: str | Path,
    *,
    checkpoint_sidecars: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _rooted_path(project_root, value, "paired runner count receipt")
    document = _load_canonical_json(path, "paired runner count receipt")
    _canonical(document, "paired runner count receipt")
    required_keys = {
        "schema_version",
        "analysis_class",
        "architecture_lock",
        "checkpoint_init_sha256",
        "rows_per_arm",
        "optimizer_updates_per_arm",
        "grad_accum",
        "warmup",
        "loss",
        "expected_static_resets",
        "arms",
        "passed",
        "status",
        "decision",
    }
    _require(set(document) == required_keys, "paired runner count receipt schema drifted")
    _require(
        document.get("schema_version") == 1
        and document.get("analysis_class") == F2_COUNT_RECEIPT_CLASS
        and document.get("architecture_lock") == F2_ARCHITECTURE_LOCK,
        "paired runner count receipt identity drifted",
    )
    _require(
        document.get("rows_per_arm") == 256
        and document.get("optimizer_updates_per_arm") == 128
        and document.get("grad_accum") == 2
        and document.get("warmup") == "u_pre<16"
        and document.get("loss") == "L_aux+0.5*L1+0.5*L2",
        "paired runner count receipt frozen run contract drifted",
    )
    _require(
        _nonnegative_int(
            document.get("expected_static_resets"),
            "paired runner expected static resets",
        )
        == _FROZEN_SMOKE_STATIC_RESETS,
        "paired runner expected static resets differ from frozen SMK-TRAIN count",
    )
    arms = _mapping(document.get("arms"), "paired runner count receipt arms")
    _require(set(arms) == {"S-CTRL", "S-SELF"}, "paired runner count receipt arm set drifted")
    count_names = {
        "rows",
        "feature_forwards",
        "aux_forwards",
        "head_forwards",
        "track_loss_calls",
        "backward_calls",
        "optimizer_steps",
        "controller_steps",
        "static_resets",
        "nonfinite_resets",
        "branch1_logged_rows",
        "branch2_logged_rows",
        "branch2_self_rows",
        "g6_updates",
        "g7_updates",
        "g9_transitions",
        "expert_future_leak_count",
        "self_state_expert_overwrite_count",
    }
    common = {
        "rows": 256,
        "feature_forwards": 256,
        "aux_forwards": 256,
        "head_forwards": 512,
        "track_loss_calls": 512,
        "backward_calls": 256,
        "optimizer_steps": 128,
        "controller_steps": 256,
        "static_resets": _FROZEN_SMOKE_STATIC_RESETS,
        "nonfinite_resets": 0,
        "branch1_logged_rows": 256,
        "g7_updates": 128,
        "g9_transitions": 256,
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }
    arm_specific = {
        "S-CTRL": {
            "branch2_logged_rows": 256,
            "branch2_self_rows": 0,
            "g6_updates": 128,
        },
        "S-SELF": {
            "branch2_logged_rows": 32,
            "branch2_self_rows": 224,
            "g6_updates": 0,
        },
    }
    for arm in ("S-CTRL", "S-SELF"):
        counts = _mapping(arms[arm], f"paired runner counts {arm}")
        _require(set(counts) == count_names, f"paired runner counts {arm} schema drifted")
        expected_counts = {**common, **arm_specific[arm]}
        for name, expected_value in expected_counts.items():
            _require(
                _nonnegative_int(counts.get(name), f"paired runner {arm}.{name}")
                == expected_value,
                f"paired runner {arm}.{name} drifted",
            )
    init_sha = _valid_sha256(
        document.get("checkpoint_init_sha256"),
        "paired runner checkpoint-init SHA",
    )
    update0_shas = {
        checkpoint_sidecars[f"{arm}:update0"]["checkpoint_tensor_sha256"]
        for arm in (IBR1_CTRL, IBR1_SELF)
    }
    _require(update0_shas == {init_sha}, "count receipt checkpoint identity differs from update-0 sidecars")
    _require(
        document.get("passed") is True
        and document.get("status") == "PASS"
        and document.get("decision") == "GO",
        "paired runner count receipt verdict drifted",
    )
    return document, _artifact_binding(project_root, path, document)


def _frozen_eval_stratum_positions() -> dict[str, tuple[int, ...]]:
    positions: dict[str, tuple[int, ...]] = {
        "overall": tuple(range(FROZEN_EVAL_ROWS))
    }
    for stratum, ranges in _FROZEN_EVAL_STRATUM_POSITION_RANGES.items():
        expanded = tuple(
            position
            for start, stop in ranges
            for position in range(start, stop + 1)
        )
        _require(
            len(expanded) == _FROZEN_EVAL_STRATUM_COUNTS[stratum]
            and len(set(expanded)) == len(expanded)
            and tuple(sorted(expanded)) == expanded
            and all(0 <= position < FROZEN_EVAL_ROWS for position in expanded),
            f"frozen EVAL {stratum} strata mapping drifted",
        )
        positions[stratum] = expanded
    _require(
        set(positions) == set(_FROZEN_EVAL_STRATUM_COUNTS)
        and all(
            len(positions[stratum]) == count
            for stratum, count in _FROZEN_EVAL_STRATUM_COUNTS.items()
        ),
        "frozen EVAL strata registry drifted",
    )
    return positions


def _eval_phase_summary(
    value: Any,
    label: str,
    row_losses: Sequence[float],
) -> dict[str, Any]:
    summary = _mapping(value, f"{label} summary")
    _require(
        set(summary) == {"accumulator", "means", "counts"}
        and summary.get("accumulator") == "IEEE-754 binary64 math.fsum",
        f"{label} summary schema drifted",
    )
    strata = ("overall", "change", "turn", "other")
    means = _mapping(summary.get("means"), f"{label} summary means")
    counts = _mapping(summary.get("counts"), f"{label} summary counts")
    _require(
        set(means) == set(strata) and set(counts) == set(strata),
        f"{label} summary strata drifted",
    )
    supplied = {
        "accumulator": "IEEE-754 binary64 math.fsum",
        "means": {
            stratum: _finite_float(means[stratum], f"{label} {stratum} mean")
            for stratum in strata
        },
        "counts": {
            stratum: _nonnegative_int(
                counts[stratum], f"{label} {stratum} count"
            )
            for stratum in strata
        },
    }
    _require(
        all(mean >= 0.0 for mean in supplied["means"].values()),
        f"{label} summary mean must be nonnegative",
    )

    positions = _frozen_eval_stratum_positions()
    recomputed = {
        "accumulator": "IEEE-754 binary64 math.fsum",
        "means": {
            stratum: math.fsum(row_losses[position] for position in positions[stratum])
            / len(positions[stratum])
            for stratum in strata
        },
        "counts": {
            stratum: len(positions[stratum])
            for stratum in strata
        },
    }
    _require(
        supplied == recomputed,
        f"{label} summary differs from row_losses and frozen EVAL raw strata",
    )
    return recomputed


def _eval_phase_receipt_artifacts(
    project_root: Path,
    values: Mapping[str, str | Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    _require(isinstance(values, Mapping), "result seal EVAL phase receipts must be a phase-path mapping")
    expected_phases = {phase.phase: phase for phase in IBR1_EVAL_PHASES}
    _require(set(values) == set(expected_phases), "EVAL phase receipt paths do not cover the frozen six phases")
    documents: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    resolved_paths: set[Path] = set()
    reference_counts: dict[str, int] | None = None
    for phase_name, phase in expected_phases.items():
        path = _rooted_path(project_root, values[phase_name], f"EVAL phase receipt {phase_name}")
        _require(path not in resolved_paths, "EVAL phase receipt paths must be distinct")
        resolved_paths.add(path)
        document = _load_canonical_json(path, f"EVAL phase receipt {phase_name}")
        _canonical(document, f"EVAL phase receipt {phase_name}")
        required = {
            "schema_version",
            "analysis_class",
            "architecture_lock",
            "phase",
            "snapshot",
            "family_id",
            "family_arm",
            "engine_arm",
            "checkpoint_u_pre",
            "support",
            "rows",
            "mode",
            "static_resets",
            "controller_config",
            "eval_mode_contract",
            "row_losses",
            "summary",
        }
        _require(set(document) == required, f"EVAL phase receipt {phase_name} schema drifted")
        expected_u_pre = 0 if phase.snapshot.startswith("update0_") else 128
        _require(
            _nonnegative_int(
                document.get("schema_version"),
                f"EVAL phase {phase_name} schema version",
            )
            == 1
            and document.get("analysis_class") == F2_EVAL_PHASE_RECEIPT_CLASS
            and document.get("architecture_lock") == F2_ARCHITECTURE_LOCK
            and document.get("phase") == phase_name
            and document.get("snapshot") == phase.snapshot
            and document.get("family_id") == IBR1_FAMILY_ID
            and document.get("family_arm") == phase.family_arm
            and document.get("engine_arm")
            == FAMILY_TO_ENGINE_ARM[phase.family_arm]
            and _nonnegative_int(
                document.get("checkpoint_u_pre"),
                f"EVAL phase {phase_name} checkpoint u_pre",
            )
            == expected_u_pre
            and document.get("support") == "EVAL-FIX"
            and _nonnegative_int(
                document.get("rows"), f"EVAL phase {phase_name} rows"
            )
            == FROZEN_EVAL_ROWS
            and document.get("mode") == phase.mode,
            f"EVAL phase receipt {phase_name} identity drifted",
        )
        resets = _mapping(document.get("static_resets"), f"EVAL phase {phase_name} static resets")
        _require(set(resets) == {"expected", "observed"}, f"EVAL phase {phase_name} reset schema drifted")
        expected_resets = _nonnegative_int(resets.get("expected"), f"EVAL phase {phase_name} expected resets")
        observed_resets = _nonnegative_int(resets.get("observed"), f"EVAL phase {phase_name} observed resets")
        _require(
            expected_resets == _FROZEN_EVAL_STATIC_RESETS
            and observed_resets == _FROZEN_EVAL_STATIC_RESETS,
            f"EVAL phase {phase_name} static resets differ from frozen EVAL-FIX count",
        )
        controller = _mapping(document.get("controller_config"), f"EVAL phase {phase_name} controller config")
        eval_contract = _mapping(document.get("eval_mode_contract"), f"EVAL phase {phase_name} mode contract")
        _require(
            _canonical(controller, f"EVAL phase {phase_name} controller config")
            == _canonical(
                _FROZEN_CONTROLLER_CONFIG,
                "frozen F2 controller config",
            ),
            f"EVAL phase {phase_name} controller config differs from frozen DEFAULT_CONFIG",
        )
        _require(
            _canonical(eval_contract, f"EVAL phase {phase_name} mode contract")
            == _canonical(
                _FROZEN_EVAL_MODE_CONTRACT,
                "frozen F2 EVAL mode contract",
            ),
            f"EVAL phase {phase_name} mode contract differs from frozen EVAL_MODE_CONTRACT",
        )
        losses_value = document.get("row_losses")
        _require(_is_sequence(losses_value) and len(losses_value) == FROZEN_EVAL_ROWS, f"EVAL phase {phase_name} row-loss cardinality drifted")
        row_losses = tuple(
            _finite_float(loss, f"EVAL phase {phase_name} row loss {index}")
            for index, loss in enumerate(losses_value)
        )
        _require(all(loss >= 0.0 for loss in row_losses), f"EVAL phase {phase_name} has a negative row loss")
        summary = _eval_phase_summary(document.get("summary"), f"EVAL phase {phase_name}", row_losses)
        if reference_counts is None:
            reference_counts = dict(summary["counts"])
        else:
            _require(summary["counts"] == reference_counts, "EVAL phase fixed-support counts differ")
        documents[phase_name] = document
        summaries[phase_name] = summary
        bindings[phase_name] = _artifact_binding(project_root, path, document)

    snapshot_inputs = {
        "s_self_update0": {
            "logged": summaries["u0_self_logged"],
            "self": summaries["u0_self_self"],
        },
        "s_self_update128": {
            "logged": summaries["u128_self_logged"],
            "self": summaries["u128_self_self"],
        },
        "s_ctrl_update128": {
            "logged": summaries["u128_ctrl_logged"],
            "self": summaries["u128_ctrl_self"],
        },
    }
    recomputed_i5 = evaluate_i5(**snapshot_inputs).to_dict()
    return documents, bindings, recomputed_i5


def _validate_eval_guard_phase_identity(eval_guard: Mapping[str, Any]) -> None:
    expected = {phase.phase: phase for phase in IBR1_EVAL_PHASES}
    _require(eval_guard.get("phase_order") == list(expected), "EVAL guard phase order drifted")
    _require(
        eval_guard.get("rows_per_phase") == FROZEN_EVAL_ROWS
        and eval_guard.get("total_predictor_calls") == len(expected) * FROZEN_EVAL_ROWS
        and eval_guard.get("expected_total_predictor_calls")
        == len(expected) * FROZEN_EVAL_ROWS,
        "EVAL guard phase cardinality drifted",
    )
    phases = eval_guard.get("phases")
    _require(_is_sequence(phases) and len(phases) == len(expected), "EVAL guard phase receipts drifted")
    for index, (phase_name, phase) in enumerate(expected.items()):
        item = _mapping(phases[index], f"EVAL guard phase {phase_name}")
        _require(
            item.get("phase") == phase_name
            and item.get("snapshot") == phase.snapshot
            and item.get("family_id") == IBR1_FAMILY_ID
            and item.get("family_arm") == phase.family_arm
            and item.get("mode") == phase.mode
            and item.get("rows") == FROZEN_EVAL_ROWS
            and item.get("bytes_equal_expected_binding") is True,
            f"EVAL guard phase {phase_name} identity drifted",
        )
        indices = item.get("ordered_original_indices")
        _require(_is_sequence(indices) and len(indices) == FROZEN_EVAL_ROWS, f"EVAL guard phase {phase_name} ordered indices drifted")
        normalized_indices = [
            _nonnegative_int(value, f"EVAL guard phase {phase_name} index {position}")
            for position, value in enumerate(indices)
        ]
        observed_indices_sha = _sha256_bytes(
            _canonical(
                normalized_indices,
                f"EVAL guard phase {phase_name} indices",
            )
        )
        _require(
            len(set(normalized_indices)) == FROZEN_EVAL_ROWS
            and observed_indices_sha
            == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
            and _valid_sha256(
                item.get("ordered_original_indices_sha256"),
                f"EVAL guard phase {phase_name} ordered-index SHA",
            )
            == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
            f"EVAL guard phase {phase_name} ordered-index authority drifted",
        )


def _build_ibr1_result_seal(
    project_root: str | Path,
    *,
    expected_pass: bool,
    final_assembly_receipt_path: str | Path,
    candidate_lock_receipt_path: str | Path,
    checkpoint_sidecar_paths: Mapping[str, str | Path] | Sequence[str | Path],
    count_receipt_path: str | Path,
    eval_guard_receipt_path: str | Path,
    eval_phase_receipt_paths: Mapping[str, str | Path],
    diagnostics_manifest_path: str | Path,
    gate_receipt_paths: Mapping[str, str | Path] | Sequence[str | Path],
    combined_gate_receipt_path: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).expanduser().resolve()
    _require(root.is_dir(), f"project_root is not a directory: {root}")

    final_path, final_assembly, final_binding = _load_bound_artifact(
        root,
        final_assembly_receipt_path,
        label="final IBR1 assembly receipt",
        analysis_class=ASSEMBLY_RECEIPT_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    _require(
        final_assembly.get("phase") == ASSEMBLY_PHASE_FINAL
        and final_assembly.get("candidate_cap") == 1
        and final_assembly.get("formal_training_authorized") is False,
        "result seal final assembly identity drifted",
    )

    candidate_path, candidate_lock, candidate_binding = _load_bound_artifact(
        root,
        candidate_lock_receipt_path,
        label="IBR1 candidate lock receipt",
        analysis_class=IBR1_CANDIDATE_LOCK_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    _verify_candidate_lock(candidate_lock)

    sidecar_documents, sidecar_bindings = _checkpoint_sidecar_artifacts(
        root,
        checkpoint_sidecar_paths,
        expected_final_binding=final_binding,
    )
    _count_receipt, count_binding = _count_receipt_artifact(
        root,
        count_receipt_path,
        checkpoint_sidecars=sidecar_documents,
    )

    eval_path, eval_guard, eval_binding = _load_bound_artifact(
        root,
        eval_guard_receipt_path,
        label="IBR1 EVAL guard receipt",
        analysis_class=EVAL_GUARD_RECEIPT_CLASS,
        require_self_hash=False,
    )
    _require(eval_guard.get("formal_training_authorized") is False, "EVAL guard formal policy drifted")
    _require(
        dict(_mapping(eval_guard.get("final_assembly_receipt"), "EVAL guard final assembly binding"))
        == final_binding,
        "EVAL guard binds a different final assembly",
    )
    _require(
        eval_guard.get("all_phase_mapping_bytes_identical") is True
        and eval_guard.get("all_phase_mapping_sha256_identical") is True
        and eval_guard.get("all_phase_mappings_equal_expected_binding") is True,
        "EVAL guard did not complete all fixed-order checks",
    )
    _validate_eval_guard_phase_identity(eval_guard)
    _eval_phase_documents, eval_phase_bindings, recomputed_i5 = (
        _eval_phase_receipt_artifacts(root, eval_phase_receipt_paths)
    )

    diagnostics_path, diagnostics, diagnostics_binding = _load_bound_artifact(
        root,
        diagnostics_manifest_path,
        label="IBR1 diagnostics manifest",
        analysis_class="ibr1_diagnostics_manifest",
        require_self_hash=True,
        require_architecture=True,
    )
    _require(
        diagnostics.get("formal_training_authorized") is False
        and diagnostics.get("manifest_written_after_all_bound_artifacts") is True,
        "diagnostics manifest completion/formal policy drifted",
    )
    diagnostics_bundle, diagnostics_gate_documents = _validate_diagnostics_bundle(
        root,
        diagnostics_path,
        diagnostics,
    )
    diagnostics_binding["bundle"] = diagnostics_bundle

    gate_documents, gate_bindings = _gate_receipt_artifacts(root, gate_receipt_paths)
    for gate_id in (I2_GATE_ID, I3_GATE_ID):
        _require(
            gate_documents[gate_id] == diagnostics_gate_documents[gate_id],
            f"bound {gate_id} receipt differs from diagnostics raw-data recomputation",
        )
    _require(
        gate_documents[I5_GATE_ID] == recomputed_i5,
        "bound I5 receipt differs from the six live EVAL phase receipts",
    )
    i1_live_evidence = _require_i1_filesystem_authority(
        gate_documents[I1_GATE_ID],
        "bound I1 gate",
    )
    _require(
        i1_live_evidence["final_assembly"] == final_binding,
        "bound I1 gate final assembly evidence differs from the seal authority",
    )
    expected_update0_bindings = {
        arm: {
            key: sidecar_bindings[f"{arm}:update0"][key]
            for key in ("path", "sha256", "analysis_class")
        }
        for arm in (IBR1_CTRL, IBR1_SELF)
    }
    _require(
        i1_live_evidence["update0_checkpoint_sidecars"]
        == expected_update0_bindings,
        "bound I1 gate update-0 sidecar evidence differs from live seal artifacts",
    )
    combined_path, combined_raw, combined_binding = _load_bound_artifact(
        root,
        combined_gate_receipt_path,
        label="combined IBR1 gate receipt",
        analysis_class=IBR1_COMBINED_GATE_RECEIPT_CLASS,
        require_self_hash=True,
        require_architecture=True,
    )
    combined = _coerce_combined_receipt(combined_raw, "combined IBR1 gate receipt")
    _require(combined.get("mechanism_pass") is expected_pass, "combined IBR1 verdict differs from requested seal type")
    combined_gates = _mapping(combined.get("gates"), "combined IBR1 gate documents")
    _require(
        all(dict(combined_gates[gate_id]) == gate_documents[gate_id] for gate_id in IBR1_GATE_IDS),
        "combined IBR1 receipt differs from the six bound gate receipts",
    )

    del final_path, candidate_path, eval_path, diagnostics_path, combined_path
    del _count_receipt, _eval_phase_documents
    analysis_class = IBR1_PASS_SEAL_CLASS if expected_pass else IBR1_NEGATIVE_SEAL_CLASS
    payload = {
        "schema_version": IBR1_SEAL_SCHEMA_VERSION,
        "analysis_class": analysis_class,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "run": {
            "valid_input": True,
            "engineering_failure": False,
            "mechanism_pass": expected_pass,
            "scientific_negative_result": not expected_pass,
            "status": "PASS" if expected_pass else "FAIL",
            "decision": "MECHANISM_PASS" if expected_pass else "SEAL_STOP",
        },
        "gate_outcomes": {
            "combined": {
                "passed": expected_pass,
                "status": combined["status"],
                "decision": combined["decision"],
                "receipt_payload_sha256": combined["receipt_payload_sha256"],
            },
            "gates": {
                gate_id: {
                    "passed": gate_documents[gate_id]["passed"],
                    "status": gate_documents[gate_id]["status"],
                    "receipt_payload_sha256": gate_documents[gate_id]["receipt_payload_sha256"],
                }
                for gate_id in IBR1_GATE_IDS
            },
        },
        "evidence": {
            "final_assembly": final_binding,
            "candidate_lock": candidate_binding,
            "checkpoint_sidecars": sidecar_bindings,
            "count_receipt": count_binding,
            "eval_guard": eval_binding,
            "eval_phase_receipts": eval_phase_bindings,
            "diagnostics_manifest": diagnostics_binding,
            "gate_receipts": gate_bindings,
            "combined_gate_receipt": combined_binding,
        },
        "next_step": (
            "independent_review_then_new_preregistration"
            if expected_pass
            else "SEAL_STOP"
        ),
        "formal_training_authorized": False,
        "same_family_retry_authorized": False,
        "same_family_seed_change_authorized": False,
        "same_family_lambda_change_authorized": False,
        "same_family_decode_change_authorized": False,
        "same_family_gate_change_authorized": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    return _self_hash(payload, "IBR1 result seal")


def build_ibr1_pass_seal(
    project_root: str | Path,
    *,
    final_assembly_receipt_path: str | Path,
    candidate_lock_receipt_path: str | Path,
    checkpoint_sidecar_paths: Mapping[str, str | Path] | Sequence[str | Path],
    count_receipt_path: str | Path,
    eval_guard_receipt_path: str | Path,
    eval_phase_receipt_paths: Mapping[str, str | Path],
    diagnostics_manifest_path: str | Path,
    gate_receipt_paths: Mapping[str, str | Path] | Sequence[str | Path],
    combined_gate_receipt_path: str | Path,
) -> dict[str, Any]:
    """Build an IBR1 mechanism-PASS seal; this never authorizes formal runs."""

    return _build_ibr1_result_seal(
        project_root,
        expected_pass=True,
        final_assembly_receipt_path=final_assembly_receipt_path,
        candidate_lock_receipt_path=candidate_lock_receipt_path,
        checkpoint_sidecar_paths=checkpoint_sidecar_paths,
        count_receipt_path=count_receipt_path,
        eval_guard_receipt_path=eval_guard_receipt_path,
        eval_phase_receipt_paths=eval_phase_receipt_paths,
        diagnostics_manifest_path=diagnostics_manifest_path,
        gate_receipt_paths=gate_receipt_paths,
        combined_gate_receipt_path=combined_gate_receipt_path,
    )


def build_ibr1_negative_result_seal(
    project_root: str | Path,
    *,
    final_assembly_receipt_path: str | Path,
    candidate_lock_receipt_path: str | Path,
    checkpoint_sidecar_paths: Mapping[str, str | Path] | Sequence[str | Path],
    count_receipt_path: str | Path,
    eval_guard_receipt_path: str | Path,
    eval_phase_receipt_paths: Mapping[str, str | Path],
    diagnostics_manifest_path: str | Path,
    gate_receipt_paths: Mapping[str, str | Path] | Sequence[str | Path],
    combined_gate_receipt_path: str | Path,
) -> dict[str, Any]:
    """Build the irreversible no-retry/no-tuning scientific FAIL seal."""

    return _build_ibr1_result_seal(
        project_root,
        expected_pass=False,
        final_assembly_receipt_path=final_assembly_receipt_path,
        candidate_lock_receipt_path=candidate_lock_receipt_path,
        checkpoint_sidecar_paths=checkpoint_sidecar_paths,
        count_receipt_path=count_receipt_path,
        eval_guard_receipt_path=eval_guard_receipt_path,
        eval_phase_receipt_paths=eval_phase_receipt_paths,
        diagnostics_manifest_path=diagnostics_manifest_path,
        gate_receipt_paths=gate_receipt_paths,
        combined_gate_receipt_path=combined_gate_receipt_path,
    )


def freeze_ibr1_result_seal(
    project_root: str | Path,
    output: str | Path,
    *,
    expected_pass: bool,
    final_assembly_receipt_path: str | Path,
    candidate_lock_receipt_path: str | Path,
    checkpoint_sidecar_paths: Mapping[str, str | Path] | Sequence[str | Path],
    count_receipt_path: str | Path,
    eval_guard_receipt_path: str | Path,
    eval_phase_receipt_paths: Mapping[str, str | Path],
    diagnostics_manifest_path: str | Path,
    gate_receipt_paths: Mapping[str, str | Path] | Sequence[str | Path],
    combined_gate_receipt_path: str | Path,
) -> dict[str, Any]:
    """Build and exclusively freeze a PASS or negative IBR1 result seal."""

    document = _build_ibr1_result_seal(
        project_root,
        expected_pass=expected_pass,
        final_assembly_receipt_path=final_assembly_receipt_path,
        candidate_lock_receipt_path=candidate_lock_receipt_path,
        checkpoint_sidecar_paths=checkpoint_sidecar_paths,
        count_receipt_path=count_receipt_path,
        eval_guard_receipt_path=eval_guard_receipt_path,
        eval_phase_receipt_paths=eval_phase_receipt_paths,
        diagnostics_manifest_path=diagnostics_manifest_path,
        gate_receipt_paths=gate_receipt_paths,
        combined_gate_receipt_path=combined_gate_receipt_path,
    )
    root = Path(project_root).expanduser().resolve()
    destination = _rooted_path(root, output, "IBR1 result seal output")
    result = _freeze_document(destination, document, label="IBR1 result seal")
    result["path"] = _relative_path(root, destination)
    result["mechanism_pass"] = expected_pass
    result["formal_training_authorized"] = False
    return result


def freeze_ibr1_pass_seal(
    project_root: str | Path,
    output: str | Path,
    **artifacts: Any,
) -> dict[str, Any]:
    return freeze_ibr1_result_seal(
        project_root,
        output,
        expected_pass=True,
        **artifacts,
    )


def freeze_ibr1_negative_result_seal(
    project_root: str | Path,
    output: str | Path,
    **artifacts: Any,
) -> dict[str, Any]:
    return freeze_ibr1_result_seal(
        project_root,
        output,
        expected_pass=False,
        **artifacts,
    )


def _paths_from_evidence(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    def path_of(value: Any, label: str) -> str:
        return str(_mapping(value, label).get("path"))

    checkpoint_bindings = _mapping(evidence.get("checkpoint_sidecars"), "seal checkpoint sidecars")
    eval_phase_bindings = _mapping(
        evidence.get("eval_phase_receipts"),
        "seal EVAL phase receipts",
    )
    gate_bindings = _mapping(evidence.get("gate_receipts"), "seal gate receipts")
    return {
        "final_assembly_receipt_path": path_of(evidence.get("final_assembly"), "seal final assembly"),
        "candidate_lock_receipt_path": path_of(evidence.get("candidate_lock"), "seal candidate lock"),
        "checkpoint_sidecar_paths": {
            key: path_of(value, f"seal checkpoint sidecar {key}")
            for key, value in checkpoint_bindings.items()
        },
        "count_receipt_path": path_of(evidence.get("count_receipt"), "seal count receipt"),
        "eval_guard_receipt_path": path_of(evidence.get("eval_guard"), "seal EVAL guard"),
        "eval_phase_receipt_paths": {
            key: path_of(value, f"seal EVAL phase receipt {key}")
            for key, value in eval_phase_bindings.items()
        },
        "diagnostics_manifest_path": path_of(evidence.get("diagnostics_manifest"), "seal diagnostics manifest"),
        "gate_receipt_paths": {
            key: path_of(value, f"seal gate receipt {key}")
            for key, value in gate_bindings.items()
        },
        "combined_gate_receipt_path": path_of(evidence.get("combined_gate_receipt"), "seal combined receipt"),
    }


def verify_ibr1_result_seal(
    project_root: str | Path,
    seal_path: str | Path,
) -> dict[str, Any]:
    """Rebuild a result seal from live artifact bytes and require exact match."""

    root = Path(project_root).expanduser().resolve()
    path = _rooted_path(root, seal_path, "IBR1 result seal")
    observed = _load_canonical_json(path, "IBR1 result seal")
    analysis_class = observed.get("analysis_class")
    _require(
        analysis_class in (IBR1_PASS_SEAL_CLASS, IBR1_NEGATIVE_SEAL_CLASS),
        "IBR1 result seal class is invalid",
    )
    sealed = _sealed_mapping(
        observed,
        label="IBR1 result seal",
        analysis_class=str(analysis_class),
        require_self_hash=True,
        require_architecture=True,
    )
    expected_pass = analysis_class == IBR1_PASS_SEAL_CLASS
    run = _mapping(sealed.get("run"), "IBR1 result seal run")
    _require(
        run.get("valid_input") is True
        and run.get("engineering_failure") is False
        and run.get("mechanism_pass") is expected_pass
        and run.get("scientific_negative_result") is (not expected_pass),
        "IBR1 result seal run verdict drifted",
    )
    _require(sealed.get("formal_training_authorized") is False, "IBR1 seal authorizes formal training")
    evidence = _mapping(sealed.get("evidence"), "IBR1 result seal evidence")
    rebuilt = _build_ibr1_result_seal(
        root,
        expected_pass=expected_pass,
        **_paths_from_evidence(evidence),
    )
    _require(sealed == rebuilt, "IBR1 result seal differs from live bound artifact bytes")
    return sealed


__all__ = [
    "IBR1_CANDIDATE_LOCK_CLASS",
    "IBR1_COMBINED_GATE_RECEIPT_CLASS",
    "IBR1_GATE_IDS",
    "IBR1_GATE_RECEIPT_CLASS",
    "IBR1_NEGATIVE_SEAL_CLASS",
    "IBR1_PASS_SEAL_CLASS",
    "IBR1GateContractError",
    "IBR1GateReceipt",
    "build_ibr1_candidate_lock_receipt",
    "build_ibr1_combined_gate_receipt",
    "build_ibr1_negative_result_seal",
    "build_ibr1_pass_seal",
    "evaluate_i1",
    "evaluate_i2",
    "evaluate_i3",
    "evaluate_i4",
    "evaluate_i5",
    "evaluate_i6",
    "freeze_ibr1_candidate_lock_receipt",
    "freeze_ibr1_combined_gate_receipt",
    "freeze_ibr1_gate_receipt",
    "freeze_ibr1_negative_result_seal",
    "freeze_ibr1_pass_seal",
    "freeze_ibr1_result_seal",
    "verify_ibr1_result_seal",
]
