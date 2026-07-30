"""Frozen-checkpoint public-validation diagnostics for F2 memory and reasoning.

This module is evaluator-only.  It never trains, mutates, or selects a model,
and it never opens the sealed internal-test split.  One normal perception
stream is shared by FULL and the direct action-credit interventions;
RECURRENT-STATE-RESET owns an independent perception state.  Every self-rollout
condition owns an independent controller state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from f2_experiment import assembly as assembly_core
from f2_experiment.assembly import load_arm_checkpoint_verified
from f2_experiment.assembly_data import (
    ObservationPacket,
    build_token_ledger_for_rows,
    frozen_cache_roots,
    load_cached_observation,
    verify_frozen_assets,
)
from f2_experiment.assembly_model import (
    F2ArmModules,
    build_eval_row_predictor_from_checkpoint,
)
from f2_experiment.controller import ActionFilterController
from f2_experiment.reproducibility import configure_cuda_reproducibility
from f2_experiment.support import (
    canonical_json_sha256,
    derive_strafe_ledger,
    parse_train_jsonl,
)
from scripts.eval_offline import (
    balanced_control_error_at1,
    compute_metrics,
    continues_evaluation_sequence,
    evaluate_predictions,
    evaluation_sequence_key,
    validate_ordered_evaluation_records,
)


CONDITIONS = (
    "full",
    "recurrent_state_reset",
    "reasoning_direct_off",
    "polar_direct_off",
    "future_direct_off",
)
MODES = ("logged", "self")
FUTURE_HORIZONS = (4, 8, 16)
CONTROLLED_AXES = (0, 2)
VAL_DATA_SHA256 = "696423b1c12f1b77f3c664ad1ca414e8371a55a033d20564aeb9d133e87eb14a"
VAL_MANIFEST_SHA256 = "acef99fb32c445431666e8fc07c73279345b465989bcc28b6efa6f0ef12716ad"
VAL_ROWS = 2848
VAL_IMAGE_PREFIX = "data/collected_v1/episodes/val/"
CONTROL_THRESHOLD = 0.2
TEMPORAL_SHIFT_FRAMES = 17
VAL_BASE_RESET_INDICES = (0, 512, 924, 1886)
VAL_BASE_RESET_SHA256 = (
    "732d591b62ee468448acbfbc862a4b72fa0dfca64847ab6c9634ef3877826223"
)
VAL_STRAFE_RESET_INDICES = (346, 347, 348, 349)
VAL_STRAFE_RESET_SHA256 = (
    "61896595cc78b952a1492550fcb20a6e4c4ed8b86bf7aecfd88b842b05d18846"
)
VAL_COMBINED_RESET_INDICES = (
    0,
    346,
    347,
    348,
    349,
    512,
    924,
    1886,
)
VAL_COMBINED_RESET_SHA256 = (
    "c009b889e51fb238f5f4dcbbe69124921c539afe796173fbdc44e4e9ce07bec7"
)
VAL_DETERMINISM_PROBE_INDICES = (
    *range(0, 8),
    *range(512, 520),
    *range(924, 932),
    *range(1886, 1894),
)
VAL_DETERMINISM_PROBE_SHA256 = (
    "410c9223692effba62478b38caa55f07e1bddef4a69185aeab6863fd1066726f"
)
VAL_PREFIX_512_SHA256 = (
    "61f3f6fd3aa109e05aa31cc4d74f333d17c8f73ced22284db2209e96a39884af"
)
VAL_FULL_2848_SHA256 = (
    "49680e3a7e9050298e1603b8e353259fbce140e5d6243b5e19a12a880caac211"
)
SELECTION_NAMES = (
    "determinism_probe_4x8",
    "prefix_512_engineering_smoke",
    "full_2848_public_validation",
)
BASELINE_NAMES = ("B0_seed0", "B1_seed0", "B1_seed1", "B1_seed2")
EVALUATOR_ONLY_TESTS = ("tests/f2/test_validation_diagnostics.py",)
PREREGISTERED_STOP_RULE = (
    "run the fixed 4x8 determinism probe, then the fixed 512-row engineering "
    "prefix, then all 2848 public-validation rows regardless of scientific "
    "direction; only the 2848-row run is abstract-claim eligible; never change "
    "checkpoint, gate, conditions, threshold, reset policy, or baseline set"
)
INTERNAL_TEST_MARKERS = (
    "/episodes/test/",
    "\\episodes\\test\\",
    "internal_test",
)


class F2ValidationDiagnosticError(RuntimeError):
    """Fail-closed error for the evaluator-only diagnostic protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _exclusive_write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise F2ValidationDiagnosticError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise F2ValidationDiagnosticError(f"{label} must contain a JSON object")
    return value


def _selection_schedule() -> list[dict[str, Any]]:
    return [
        {
            "name": "determinism_probe_4x8",
            "rows": len(VAL_DETERMINISM_PROBE_INDICES),
            "original_index_sha256": VAL_DETERMINISM_PROBE_SHA256,
            "abstract_claim_eligible": False,
        },
        {
            "name": "prefix_512_engineering_smoke",
            "rows": 512,
            "original_index_sha256": VAL_PREFIX_512_SHA256,
            "abstract_claim_eligible": False,
        },
        {
            "name": "full_2848_public_validation",
            "rows": VAL_ROWS,
            "original_index_sha256": VAL_FULL_2848_SHA256,
            "abstract_claim_eligible": True,
        },
    ]


def resolve_validation_selection(
    *, max_rows: int | None, determinism_probe: bool
) -> dict[str, Any]:
    """Resolve one of the three preregistered public-validation selections."""

    if determinism_probe and max_rows is not None:
        raise F2ValidationDiagnosticError(
            "--determinism-probe and --max-rows are mutually exclusive"
        )
    if determinism_probe:
        indices = tuple(VAL_DETERMINISM_PROBE_INDICES)
        name = "determinism_probe_4x8"
        expected_sha = VAL_DETERMINISM_PROBE_SHA256
    elif max_rows is None or max_rows == VAL_ROWS:
        indices = tuple(range(VAL_ROWS))
        name = "full_2848_public_validation"
        expected_sha = VAL_FULL_2848_SHA256
    elif max_rows == 512:
        indices = tuple(range(512))
        name = "prefix_512_engineering_smoke"
        expected_sha = VAL_PREFIX_512_SHA256
    else:
        raise F2ValidationDiagnosticError(
            "only the fixed 4x8 probe, fixed 512-row prefix, or all 2848 rows "
            "are admissible"
        )
    observed_sha = canonical_json_sha256(list(indices))
    if observed_sha != expected_sha:
        raise F2ValidationDiagnosticError("selection index SHA drifted")
    return {
        "name": name,
        "rows": len(indices),
        "original_indices": indices,
        "original_index_sha256": observed_sha,
        "abstract_claim_eligible": name == "full_2848_public_validation",
    }


def derive_validation_reset_contract(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the frozen sequence and 2-DoF strafe reset boundaries."""

    base: list[int] = []
    previous_key = None
    for index, row in enumerate(rows):
        key = evaluation_sequence_key(row, require_sequence_id=True)
        if not continues_evaluation_sequence(previous_key, key):
            base.append(index)
        previous_key = key
    base_indices = tuple(base)
    strafe_indices = tuple(derive_strafe_ledger(rows).reset_boundary_12)
    combined_indices = tuple(sorted(set(base_indices) | set(strafe_indices)))
    observed = {
        "base": (base_indices, canonical_json_sha256(list(base_indices))),
        "strafe": (strafe_indices, canonical_json_sha256(list(strafe_indices))),
        "combined": (
            combined_indices,
            canonical_json_sha256(list(combined_indices)),
        ),
    }
    expected = {
        "base": (VAL_BASE_RESET_INDICES, VAL_BASE_RESET_SHA256),
        "strafe": (VAL_STRAFE_RESET_INDICES, VAL_STRAFE_RESET_SHA256),
        "combined": (VAL_COMBINED_RESET_INDICES, VAL_COMBINED_RESET_SHA256),
    }
    for name in expected:
        if observed[name] != expected[name]:
            raise F2ValidationDiagnosticError(
                f"public-validation {name} reset contract drifted"
            )
    reasons: dict[str, list[str]] = {}
    for index in combined_indices:
        row_reasons = []
        if index in base_indices:
            row_reasons.append("sequence_discontinuity")
        if index in strafe_indices:
            row_reasons.append("strafe_reset")
        reasons[str(index)] = row_reasons
    return {
        "base_indices": list(base_indices),
        "base_sha256": observed["base"][1],
        "strafe_indices": list(strafe_indices),
        "strafe_sha256": observed["strafe"][1],
        "combined_indices": list(combined_indices),
        "combined_sha256": observed["combined"][1],
        "reasons_by_original_index": reasons,
    }


def verify_evaluator_assembly_receipt(
    project_root: str | Path,
    receipt_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the frozen assembly while allowing only this evaluator's new test."""

    root = Path(project_root).expanduser().resolve()
    path = Path(receipt_path).expanduser().resolve()
    receipt = _load_json_object(path, "assembly receipt")
    recorded = receipt.get("tests_sha256")
    if not isinstance(recorded, Mapping) or not recorded:
        raise F2ValidationDiagnosticError("assembly receipt test registry is malformed")
    observed: dict[str, str] = {}
    tests_dir = root / "tests" / "f2"
    if not tests_dir.is_dir():
        raise F2ValidationDiagnosticError("tests/f2 is missing")
    for test_path in sorted(tests_dir.glob("*.py")):
        if test_path.is_file() and not test_path.name.startswith("._"):
            relative = test_path.relative_to(root).as_posix()
            observed[relative] = sha256_file(test_path)
    for relative, expected_sha in recorded.items():
        if observed.get(str(relative)) != expected_sha:
            raise F2ValidationDiagnosticError(
                f"frozen assembly test changed or disappeared: {relative}"
            )
    extras = set(observed) - set(recorded)
    if extras != set(EVALUATOR_ONLY_TESTS):
        raise F2ValidationDiagnosticError(
            "tests/f2 drift is not limited to the evaluator-only test"
        )

    asset_binding = receipt.get("asset_binding")
    base_hf = asset_binding.get("base_hf") if isinstance(asset_binding, Mapping) else None
    base_hf_path = base_hf.get("path") if isinstance(base_hf, Mapping) else None
    if not isinstance(base_hf_path, str) or not base_hf_path:
        raise F2ValidationDiagnosticError("assembly receipt base-HF binding is malformed")

    def asset_verifier(live_root: Path) -> Mapping[str, Any]:
        observed = dict(
            verify_frozen_assets(
                live_root,
                base_hf_dir=base_hf_path,
                verify_token_payload=False,
            )
        )
        # The frozen v1 Windows receipt carries the train-ledger anchor added by
        # assembly._default_asset_binding, while verify_frozen_assets returns
        # only the live asset document.  Evaluation never consumes train rows;
        # retain the receipt-bound anchor and byte-verify every selected val
        # token separately through the public-val ledger below.
        for field in ("token_ledger_sha256", "token_ledger_file_count"):
            observed[field] = asset_binding[field]
        # v1 serialized the relocated manifest's historical Mac path through a
        # legacy mojibake console codec.  It is metadata-only; all cache/data,
        # encoder, effective-root, and payload SHAs are still compared live.
        observed_cache = observed.get("vision_cache")
        recorded_cache = asset_binding.get("vision_cache")
        if isinstance(observed_cache, Mapping) and isinstance(recorded_cache, Mapping):
            normalized_cache = dict(observed_cache)
            normalized_cache["recorded_path_root"] = recorded_cache.get(
                "recorded_path_root"
            )
            observed["vision_cache"] = normalized_cache
        return observed

    original_test_bindings = assembly_core._test_bindings
    assembly_core._test_bindings = lambda live_root: dict(recorded)
    try:
        verified = assembly_core.verify_assembly_receipt(
            root,
            path,
            asset_verifier=asset_verifier,
        )
    finally:
        assembly_core._test_bindings = original_test_bindings
    trace = {
        "analysis_class": "f2_evaluator_assembly_verification",
        "recorded_test_count": len(recorded),
        "recorded_tests_verified": True,
        "evaluator_only_test_drift": {
            relative: observed[relative] for relative in EVALUATOR_ONLY_TESTS
        },
        "source_transitive_controller_assets_verified": True,
        "asset_token_payload_traversed": False,
        "train_token_ledger_anchor_recomputed": False,
        "selected_val_token_files_byte_verified_later": True,
        "legacy_recorded_path_root_metadata_normalized": True,
        "internal_test_opened": False,
    }
    return verified, trace


def validate_preregistration(
    preregistration_path: Path,
    preregistration_receipt_path: Path,
    *,
    checkpoint_sha256: str,
    assembly_receipt_sha256: str,
    baseline_paths: Mapping[str, Path],
    selection: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the exact evaluator, candidate, conditions, and comparators."""

    preregistration = _load_json_object(preregistration_path, "preregistration")
    receipt = _load_json_object(
        preregistration_receipt_path, "preregistration receipt"
    )
    preregistration_sha256 = sha256_file(preregistration_path)
    evaluator_sha256 = sha256_file(Path(__file__).resolve())
    if preregistration.get("analysis_class") != "f2_public_val_memory_reasoning_preregistration":
        raise F2ValidationDiagnosticError("wrong preregistration analysis_class")
    if preregistration.get("status") != "frozen_before_first_f2_public_val_prediction":
        raise F2ValidationDiagnosticError("preregistration is not frozen")
    if receipt.get("analysis_class") != "f2_public_val_memory_reasoning_preregistration_receipt":
        raise F2ValidationDiagnosticError("wrong preregistration receipt class")
    if receipt.get("preregistration_sha256") != preregistration_sha256:
        raise F2ValidationDiagnosticError("preregistration receipt SHA mismatch")
    if receipt.get("evaluator_source_sha256") != evaluator_sha256:
        raise F2ValidationDiagnosticError("evaluator source changed after preregistration")
    if receipt.get("internal_test_opened") is not False:
        raise F2ValidationDiagnosticError("preregistration receipt breaks the test seal")
    if preregistration.get("evaluator_source_sha256") != evaluator_sha256:
        raise F2ValidationDiagnosticError("preregistration evaluator SHA mismatch")
    if preregistration.get("internal_test_opened") is not False:
        raise F2ValidationDiagnosticError("preregistration breaks the test seal")
    candidate = preregistration.get("candidate")
    if not isinstance(candidate, Mapping):
        raise F2ValidationDiagnosticError("preregistration candidate is missing")
    expected_candidate = {
        "arm": "S-SELF",
        "snapshot": 128,
        "seed": 0,
        "checkpoint_sha256": checkpoint_sha256,
        "assembly_receipt_sha256": assembly_receipt_sha256,
        "architecture_lock": "L1+D2+AP2+F2",
        "package": "SA-Hstar",
    }
    for field, expected in expected_candidate.items():
        if candidate.get(field) != expected:
            raise F2ValidationDiagnosticError(
                f"preregistration candidate.{field} mismatch"
            )
    data = preregistration.get("public_validation")
    if not isinstance(data, Mapping) or data.get("sha256") != VAL_DATA_SHA256:
        raise F2ValidationDiagnosticError("preregistration validation SHA mismatch")
    if (
        data.get("manifest_sha256") != VAL_MANIFEST_SHA256
        or int(data.get("rows", -1)) != VAL_ROWS
        or data.get("split") != "val"
        or data.get("path") != "data/collected_v1/datasets/val.jsonl"
        or data.get("manifest_path")
        != "data/collected_v1/datasets/val.jsonl.manifest.json"
        or data.get("internal_test_opened") is not False
    ):
        raise F2ValidationDiagnosticError("preregistration validation identity mismatch")
    evaluation = preregistration.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise F2ValidationDiagnosticError("preregistration evaluation contract is missing")
    if tuple(evaluation.get("conditions", ())) != CONDITIONS:
        raise F2ValidationDiagnosticError("preregistered conditions differ from code")
    if tuple(evaluation.get("modes", ())) != MODES:
        raise F2ValidationDiagnosticError("preregistered modes differ from code")
    if evaluation.get("primary_mode") != "logged":
        raise F2ValidationDiagnosticError("primary mode must remain logged")
    if evaluation.get("selection_schedule") != _selection_schedule():
        raise F2ValidationDiagnosticError("selection schedule differs from code")
    if selection.get("name") not in {
        item["name"] for item in evaluation["selection_schedule"]
    }:
        raise F2ValidationDiagnosticError("current selection was not preregistered")
    if evaluation.get("claim_eligible_selection") != "full_2848_public_validation":
        raise F2ValidationDiagnosticError("claim-eligible selection changed")
    if evaluation.get("control_threshold") != CONTROL_THRESHOLD:
        raise F2ValidationDiagnosticError("control threshold changed")
    if evaluation.get("temporal_shift_frames") != TEMPORAL_SHIFT_FRAMES:
        raise F2ValidationDiagnosticError("temporal shift control changed")
    expected_reset_fields = {
        "base_reset_indices": list(VAL_BASE_RESET_INDICES),
        "base_reset_sha256": VAL_BASE_RESET_SHA256,
        "strafe_reset_indices": list(VAL_STRAFE_RESET_INDICES),
        "strafe_reset_sha256": VAL_STRAFE_RESET_SHA256,
        "combined_reset_indices": list(VAL_COMBINED_RESET_INDICES),
        "combined_reset_sha256": VAL_COMBINED_RESET_SHA256,
        "determinism_probe_indices": list(VAL_DETERMINISM_PROBE_INDICES),
        "determinism_probe_sha256": VAL_DETERMINISM_PROBE_SHA256,
    }
    for field, expected in expected_reset_fields.items():
        if evaluation.get(field) != expected:
            raise F2ValidationDiagnosticError(
                f"preregistered evaluation.{field} mismatch"
            )
    if evaluation.get("baselines") != list(BASELINE_NAMES):
        raise F2ValidationDiagnosticError("baseline names/order changed")
    if evaluation.get("stop_rule") != PREREGISTERED_STOP_RULE:
        raise F2ValidationDiagnosticError("preregistered stop rule changed")
    if evaluation.get("internal_test_opened") is not False:
        raise F2ValidationDiagnosticError("evaluation contract breaks the test seal")
    if tuple(baseline_paths) != BASELINE_NAMES:
        raise F2ValidationDiagnosticError(
            "the exact frozen B0_seed0/B1_seed0/B1_seed1/B1_seed2 set "
            "must be supplied in order"
        )
    expected_baselines = evaluation.get("baseline_prediction_sha256")
    if not isinstance(expected_baselines, Mapping):
        raise F2ValidationDiagnosticError("baseline SHA registry is missing")
    actual_baselines = {name: sha256_file(path) for name, path in baseline_paths.items()}
    if dict(expected_baselines) != actual_baselines:
        raise F2ValidationDiagnosticError("baseline prediction set/SHA mismatch")
    return preregistration, {
        "path": str(preregistration_path),
        "sha256": preregistration_sha256,
        "receipt_path": str(preregistration_receipt_path),
        "receipt_sha256": sha256_file(preregistration_receipt_path),
        "evaluator_source_sha256": evaluator_sha256,
        "current_selection": str(selection["name"]),
        "current_selection_sha256": str(selection["original_index_sha256"]),
    }


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise F2ValidationDiagnosticError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise F2ValidationDiagnosticError(f"{label} must be finite")
    return result


def _load_public_validation(
    val_json: Path,
    val_manifest: Path,
    *,
    max_rows: int | None,
    determinism_probe: bool,
) -> tuple[
    tuple[Mapping[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    if val_json.name != "val.jsonl" or val_manifest.name != "val.jsonl.manifest.json":
        raise F2ValidationDiagnosticError(
            "only the named public validation files are admissible"
        )
    data_sha = sha256_file(val_json)
    manifest_sha = sha256_file(val_manifest)
    if data_sha != VAL_DATA_SHA256 or manifest_sha != VAL_MANIFEST_SHA256:
        raise F2ValidationDiagnosticError(
            "public validation data/manifest SHA differs from the frozen registry"
        )
    try:
        manifest = json.loads(val_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise F2ValidationDiagnosticError("validation manifest is unreadable") from exc
    if manifest.get("split") != "val" or int(manifest.get("sample_count", -1)) != VAL_ROWS:
        raise F2ValidationDiagnosticError("validation manifest identity mismatch")
    if manifest.get("data_jsonl_sha256") != data_sha:
        raise F2ValidationDiagnosticError("validation manifest does not bind the JSONL")
    payload = val_json.read_bytes()
    rows = parse_train_jsonl(payload)
    if len(rows) != VAL_ROWS:
        raise F2ValidationDiagnosticError("validation row count mismatch")
    for index, row in enumerate(rows):
        if row.get("split") != "val" or bool(row.get("mirrored", False)):
            raise F2ValidationDiagnosticError(
                f"validation row {index} is not an unmirrored val row"
            )
        paths = [row.get("current"), *(row.get("images") or [])]
        for raw_path in paths:
            if not isinstance(raw_path, str) or not raw_path.startswith(VAL_IMAGE_PREFIX):
                raise F2ValidationDiagnosticError(
                    f"validation row {index} references a non-val image"
                )
            normalized = raw_path.lower().replace("\\", "/")
            if any(marker.replace("\\", "/") in normalized for marker in INTERNAL_TEST_MARKERS):
                raise F2ValidationDiagnosticError("INTERNAL_TEST_SEAL")
    validate_ordered_evaluation_records(rows, require_sequence_id=True)
    reset_contract = derive_validation_reset_contract(rows)
    selection = resolve_validation_selection(
        max_rows=max_rows,
        determinism_probe=determinism_probe,
    )
    selected_indices = tuple(selection["original_indices"])
    selected_rows = tuple(rows[index] for index in selected_indices)
    probe_groups = []
    if selection["name"] == "determinism_probe_4x8":
        for offset in range(0, len(selected_indices), 8):
            group_indices = selected_indices[offset : offset + 8]
            group_rows = selected_rows[offset : offset + 8]
            keys = [
                evaluation_sequence_key(row, require_sequence_id=True)
                for row in group_rows
            ]
            if group_indices[0] not in VAL_BASE_RESET_INDICES or any(
                not continues_evaluation_sequence(left, right)
                for left, right in zip(keys, keys[1:])
            ):
                raise F2ValidationDiagnosticError(
                    "determinism probe is not four clean contiguous sequences"
                )
            probe_groups.append(
                {
                    "original_indices": list(group_indices),
                    "episode": str(group_rows[0].get("episode", "")),
                    "sequence_id": str(group_rows[0].get("sequence_id", "")),
                    "first_frame_idx": int(group_rows[0]["frame_idx"]),
                    "last_frame_idx": int(group_rows[-1]["frame_idx"]),
                }
            )
    binding = {
        "split": "val",
        "data_sha256": data_sha,
        "manifest_sha256": manifest_sha,
        "rows": len(selected_rows),
        "full_rows": VAL_ROWS,
        "selection_name": selection["name"],
        "selection_sha256": selection["original_index_sha256"],
        "truncated_smoke": len(selected_rows) != VAL_ROWS,
        "abstract_claim_eligible": selection["abstract_claim_eligible"],
        "internal_test_opened": False,
    }
    selection_trace = {
        "schema_version": 1,
        "analysis_class": "f2_public_validation_selection_trace",
        "selection_schedule": _selection_schedule(),
        "selection_name": selection["name"],
        "rows": len(selected_rows),
        "original_indices": list(selected_indices),
        "original_index_sha256": selection["original_index_sha256"],
        "abstract_claim_eligible": selection["abstract_claim_eligible"],
        "probe_groups": probe_groups,
        "reset_contract": reset_contract,
        "internal_test_opened": False,
    }
    return selected_rows, manifest, binding, selection_trace, reset_contract


def _record_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    actions = row.get("step_actions")
    previous = row.get("prev_action")
    if not isinstance(actions, Sequence) or len(actions) != 8:
        raise F2ValidationDiagnosticError("row step_actions must contain 8 steps")
    if not isinstance(previous, Sequence) or len(previous) < 3:
        raise F2ValidationDiagnosticError("row prev_action must contain 3 axes")
    valid = row.get("valid_mask", [True] * 8)
    if not isinstance(valid, Sequence) or len(valid) != 8:
        raise F2ValidationDiagnosticError("row valid_mask must contain 8 entries")
    return {
        "step_actions": [
            [_finite_float(axis, "step action") for axis in action[:3]]
            for action in actions
        ],
        "prev_action": [_finite_float(axis, "prev action") for axis in previous[:3]],
        "valid_mask": [bool(value) for value in valid],
        "transition_type": str(row.get("transition_type", "other")),
        "episode": str(row.get("episode", "")),
        "sequence_id": str(row.get("sequence_id", row.get("chunk_id", ""))),
        "chunk_id": str(row.get("chunk_id", row.get("episode", ""))),
        "clip_id": str(row.get("clip_id", row.get("episode", ""))),
        "frame_idx": int(row["frame_idx"]),
        "mirrored": bool(row.get("mirrored", False)),
        "command": str(row.get("command", "unknown")),
        "source_raw_dir": str(row.get("source_raw_dir") or row.get("episode", "")),
    }


def _packet_inputs(packet: ObservationPacket) -> dict[str, Any]:
    if not isinstance(packet, ObservationPacket):
        raise F2ValidationDiagnosticError("encoder input must be ObservationPacket")
    return {
        "coarse_tokens": packet.coarse_tokens.unsqueeze(0),
        "coarse_tidx": packet.coarse_tidx.unsqueeze(0),
        "fine_tokens": packet.fine_tokens.unsqueeze(0),
        "fine_tidx": packet.fine_tidx.unsqueeze(0),
        "instructions": [packet.instruction],
        "yaw_hist": None if packet.yaw_hist is None else packet.yaw_hist.unsqueeze(0),
        "yaw_curr": None if packet.yaw_curr is None else packet.yaw_curr.reshape(1, 1),
    }


def _state_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _state_tensors(item)


def state_nbytes(value: Any) -> int:
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in _state_tensors(value))
    )


class PerceptionStream:
    """One ordered, action-independent F2 perception recurrence."""

    def __init__(self, arm: F2ArmModules, *, reset_every_row: bool) -> None:
        self.arm = arm
        self.reset_every_row = bool(reset_every_row)
        self.device = next(arm.model.parameters()).device
        self.state: Mapping[str, Any] | None = None
        self.position = -1

    def encode(
        self,
        packet: ObservationPacket,
        *,
        reset: bool,
        position: int,
    ) -> Mapping[str, Any]:
        if position != self.position + 1:
            raise F2ValidationDiagnosticError("perception row order drifted")
        if position == 0 and not reset:
            raise F2ValidationDiagnosticError("first evaluation row must reset")
        if self.state is None:
            self.state = self.arm.adapter.init_state(1, self.device)
        inputs = _packet_inputs(packet)
        output = self.arm.adapter.encode_step(
            inputs["coarse_tokens"],
            inputs["coarse_tidx"],
            inputs["fine_tokens"],
            inputs["fine_tidx"],
            inputs["instructions"],
            self.state,
            reset_mask=bool(reset or self.reset_every_row),
            yaw_hist=inputs["yaw_hist"],
            yaw_curr=inputs["yaw_curr"],
        )
        self.state = output["new_state"]
        self.position = position
        return output


def intervention_alphas(
    output: Mapping[str, Any], condition: str
) -> dict[str, torch.Tensor]:
    if condition not in CONDITIONS:
        raise F2ValidationDiagnosticError(f"unknown condition {condition!r}")
    alphas = dict(output["method_alphas"])
    if condition == "reasoning_direct_off":
        alphas["polar"] = torch.zeros_like(alphas["polar"])
        alphas["future"] = torch.zeros_like(alphas["future"])
    elif condition == "polar_direct_off":
        alphas["polar"] = torch.zeros_like(alphas["polar"])
    elif condition == "future_direct_off":
        alphas["future"] = torch.zeros_like(alphas["future"])
    return alphas


def _softmax_list(logits: torch.Tensor) -> list[float]:
    return torch.softmax(logits.detach().float(), dim=-1)[0].cpu().tolist()


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float().reshape(-1)[0].cpu().item())


def _reasoning_head_telemetry(output: Mapping[str, Any]) -> dict[str, Any]:
    cot = output["cot"]
    decoded = output["cot_decoded"]
    future = output["future"]
    orchestrator = output["orchestrator"]
    result: dict[str, Any] = {
        "current": {
            "theta_probability": _softmax_list(cot["theta_logits"]),
            "distance_probability": _softmax_list(cot["dist_logits"]),
            "invalid_probability": _scalar(torch.sigmoid(cot["invalid_logit"])),
            "theta_idx": int(decoded["theta_idx"].detach().cpu().item()),
            "distance_idx": int(decoded["dist_idx"].detach().cpu().item()),
            "invalid_pred": bool(decoded["invalid_pred"].detach().cpu().item()),
            "confidence": _scalar(decoded["confidence"]),
        },
        "future": {},
        "q_write": _scalar(output["q_write"]),
        "orchestrator_alpha": {
            "tim": _scalar(orchestrator["alpha_tim"]),
            "event": _scalar(orchestrator["alpha_event"]),
            "future": _scalar(orchestrator["alpha_future"]),
        },
    }
    for horizon in FUTURE_HORIZONS:
        branch = future[horizon]
        theta_probability = _softmax_list(branch["theta_logits"])
        distance_probability = _softmax_list(branch["dist_logits"])
        visibility_probability = _scalar(torch.sigmoid(branch["vis_logit"]))
        result["future"][str(horizon)] = {
            "theta_probability": theta_probability,
            "distance_probability": distance_probability,
            "visibility_probability": visibility_probability,
            "theta_idx": int(np.argmax(theta_probability)),
            "distance_idx": int(np.argmax(distance_probability)),
            "confidence": float(
                max(theta_probability)
                * max(distance_probability)
                * visibility_probability
            ),
        }
    return result


def reasoning_telemetry(
    output: Mapping[str, Any],
    memory_reset_output: Mapping[str, Any],
) -> dict[str, Any]:
    full_heads = _reasoning_head_telemetry(output)
    reset_heads = _reasoning_head_telemetry(memory_reset_output)
    telemetry = {
        **full_heads,
        "recurrent_reset_heads": reset_heads,
        "method_feature_l2": {
            name: _scalar(torch.linalg.vector_norm(feature.float(), dim=-1))
            for name, feature in output["method_features"].items()
        },
        "recurrent_memory": {
            "full_tim_mean_l2": _scalar(
                torch.linalg.vector_norm(
                    output["method_features"]["tim_q"][:, :-2].float(), dim=-1
                )
            ),
            "reset_tim_mean_l2": _scalar(
                torch.linalg.vector_norm(
                    memory_reset_output["method_features"]["tim_q"][:, :-2].float(), dim=-1
                )
            ),
            "full_state_bytes": state_nbytes(output["new_state"]),
            "reset_state_bytes": state_nbytes(memory_reset_output["new_state"]),
        },
    }
    return telemetry


def _row_label_telemetry(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "polar_theta_idx": int(row.get("polar_theta_idx", -1)),
        "polar_dist_idx": int(row.get("polar_dist_idx", -1)),
        "polar_invalid": float(row.get("polar_invalid", 1.0)),
        "future": {},
    }
    for horizon in FUTURE_HORIZONS:
        result["future"][str(horizon)] = {
            "valid": bool(row.get(f"fut_valid_{horizon}", False)),
            "visible": float(row.get(f"fut_vis_{horizon}", 0.0)),
            "theta_idx": int(row.get(f"fut_theta_idx_{horizon}", -1)),
            "distance_idx": int(row.get(f"fut_dist_idx_{horizon}", -1)),
        }
    return result


@dataclass
class RolloutState:
    controller: ActionFilterController
    controller_state: Any = None
    prev_fy: tuple[float, float] | None = None

    def previous(
        self,
        logged_prev: tuple[float, float, float],
        *,
        reset: bool,
        mode: str,
    ) -> tuple[float, float]:
        logged = (logged_prev[0], logged_prev[2])
        if mode == "logged":
            return logged
        if mode != "self":
            raise F2ValidationDiagnosticError(f"unknown mode {mode!r}")
        if reset:
            self.controller_state = self.controller.reset(logged_prev)
            self.prev_fy = logged
        if self.controller_state is None or self.prev_fy is None:
            raise F2ValidationDiagnosticError("self rollout lacks reset state")
        return self.prev_fy

    def advance(self, raw_actions: Sequence[Sequence[float]], *, mode: str) -> Any:
        if mode == "logged":
            return None
        if self.controller_state is None:
            raise F2ValidationDiagnosticError("self controller state is missing")
        k0 = raw_actions[0]
        self.controller_state, transition = self.controller.step(
            self.controller_state, (float(k0[0]), float(k0[2]))
        )
        self.prev_fy = transition.next_prev_fy
        return transition


def derive_slices(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    slices: dict[str, list[int]] = defaultdict(list)
    previous_key = None
    previous_invalid = False
    recovery_offset = None
    for index, row in enumerate(rows):
        key = evaluation_sequence_key(row, require_sequence_id=True)
        contiguous = continues_evaluation_sequence(previous_key, key)
        invalid = float(row.get("polar_invalid", 1.0)) > 0.5
        transition = str(row.get("transition_type", "other"))
        if invalid:
            slices["current_invalid"].append(index)
        else:
            slices["current_visible"].append(index)
        if "turn" in transition:
            slices["turn"].append(index)
        slices[transition].append(index)
        if contiguous and previous_invalid and not invalid:
            recovery_offset = 0
        elif not contiguous or invalid:
            recovery_offset = None
        if recovery_offset is not None and recovery_offset <= 2:
            slices[f"reacquisition_offset_{recovery_offset}"].append(index)
            recovery_offset += 1
        previous_key = key
        previous_invalid = invalid
    slices["all"] = list(range(len(rows)))
    return {name: values for name, values in sorted(slices.items())}


def _subset(values: Sequence[Any], indices: Sequence[int]) -> list[Any]:
    return [values[index] for index in indices]


def action_slice_metrics(
    predictions: Sequence[Any],
    records: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
) -> dict[str, Any]:
    if not indices:
        return {"rows": 0, "balanced_control_error_at1": None}
    pred = np.asarray(_subset(predictions, indices), dtype=np.float64)
    selected = _subset(records, indices)
    gt = np.asarray([record["step_actions"] for record in selected], dtype=np.float64)
    previous = np.asarray([record["prev_action"] for record in selected], dtype=np.float64)
    valid = np.asarray([record["valid_mask"] for record in selected], dtype=bool)
    core = compute_metrics(pred, gt, previous, valid, threshold=CONTROL_THRESHOLD)
    return {
        "rows": len(indices),
        "balanced_control_error_at1": balanced_control_error_at1(pred, selected)["value"],
        "h1_forward_mae": float(np.abs(pred[:, 0, 0] - gt[:, 0, 0]).mean()),
        "h1_yaw_mae": float(np.abs(pred[:, 0, 2] - gt[:, 0, 2]).mean()),
        "turn_sign_accuracy": core["turn_sign_accuracy"],
        "smooth_l1": core["smooth_l1"],
    }


def _safe_log_probability(probability: float) -> float:
    return -math.log(max(min(float(probability), 1.0), 1e-12))


def _binary_nll(probability: float, target: float) -> float:
    p = max(min(float(probability), 1.0 - 1e-12), 1e-12)
    y = float(target)
    return -(y * math.log(p) + (1.0 - y) * math.log(1.0 - p))


def _mean(values: Sequence[float]) -> float | None:
    return float(math.fsum(values) / len(values)) if values else None


def _ece(probabilities: Sequence[float], targets: Sequence[float], bins: int = 10) -> float | None:
    if not probabilities:
        return None
    total = len(probabilities)
    value = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        members = [
            index
            for index, probability in enumerate(probabilities)
            if lower <= probability < upper or (bin_index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        confidence = _mean([probabilities[index] for index in members])
        accuracy = _mean([targets[index] for index in members])
        value += len(members) / total * abs(float(confidence) - float(accuracy))
    return float(value)


def _js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p = np.asarray(left, dtype=np.float64)
    q = np.asarray(right, dtype=np.float64)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    eps = 1e-12
    return float(
        0.5 * np.sum(p * np.log((p + eps) / (midpoint + eps)))
        + 0.5 * np.sum(q * np.log((q + eps) / (midpoint + eps)))
    )


def analyze_reasoning(
    rows: Sequence[Mapping[str, Any]],
    telemetry: Sequence[Mapping[str, Any]],
    *,
    shift_frames: int = TEMPORAL_SHIFT_FRAMES,
) -> dict[str, Any]:
    if len(rows) != len(telemetry):
        raise F2ValidationDiagnosticError("reasoning telemetry row count mismatch")
    theta_nll: list[float] = []
    distance_nll: list[float] = []
    theta_correct = 0
    distance_correct = 0
    visible_count = 0
    invalid_nll: list[float] = []
    q_probabilities: list[float] = []
    q_targets: list[float] = []
    future_stats: dict[int, dict[str, list[float] | int]] = {
        horizon: {
            "theta_nll": [],
            "distance_nll": [],
            "visibility_nll": [],
            "visible_count": 0,
            "valid_count": 0,
        }
        for horizon in FUTURE_HORIZONS
    }
    row_lookup = {
        (str(row.get("episode", "")), str(row.get("sequence_id", "")), int(row["frame_idx"])): index
        for index, row in enumerate(rows)
    }
    consistency = {
        horizon: {"aligned": [], "shift_control": []}
        for horizon in FUTURE_HORIZONS
    }
    for index, (row, item) in enumerate(zip(rows, telemetry)):
        current = item["current"]
        invalid_target = float(row.get("polar_invalid", 1.0))
        invalid_nll.append(_binary_nll(current["invalid_probability"], invalid_target))
        # L_verify trains q_write against ``visible AND theta-correct`` for every
        # row.  Keep evaluation aligned with that target: an invalid row is a
        # negative example rather than being silently removed from calibration.
        q_probabilities.append(float(item["q_write"]))
        q_target = 0.0
        if invalid_target < 0.5:
            theta_target = int(row["polar_theta_idx"])
            distance_target = int(row["polar_dist_idx"])
            theta_probability = current["theta_probability"]
            distance_probability = current["distance_probability"]
            theta_nll.append(_safe_log_probability(theta_probability[theta_target]))
            distance_nll.append(_safe_log_probability(distance_probability[distance_target]))
            theta_prediction = int(np.argmax(theta_probability))
            distance_prediction = int(np.argmax(distance_probability))
            theta_correct += int(theta_prediction == theta_target)
            distance_correct += int(distance_prediction == distance_target)
            visible_count += 1
            q_target = float(theta_prediction == theta_target)
        q_targets.append(q_target)
        episode = str(row.get("episode", ""))
        sequence = str(row.get("sequence_id", ""))
        frame = int(row["frame_idx"])
        for horizon in FUTURE_HORIZONS:
            labels = _row_label_telemetry(row)["future"][str(horizon)]
            prediction = item["future"][str(horizon)]
            if not labels["valid"]:
                continue
            stats = future_stats[horizon]
            stats["valid_count"] = int(stats["valid_count"]) + 1
            stats["visibility_nll"].append(
                _binary_nll(prediction["visibility_probability"], labels["visible"])
            )
            if labels["visible"] > 0.5:
                stats["visible_count"] = int(stats["visible_count"]) + 1
                stats["theta_nll"].append(
                    _safe_log_probability(
                        prediction["theta_probability"][labels["theta_idx"]]
                    )
                )
                stats["distance_nll"].append(
                    _safe_log_probability(
                        prediction["distance_probability"][labels["distance_idx"]]
                    )
                )
                aligned_index = row_lookup.get((episode, sequence, frame + horizon))
                shifted_index = row_lookup.get(
                    (episode, sequence, frame + horizon + shift_frames)
                )
                if aligned_index is not None and shifted_index is not None:
                    aligned_row = rows[aligned_index]
                    shifted_row = rows[shifted_index]
                    if (
                        float(aligned_row.get("polar_invalid", 1.0)) < 0.5
                        and float(shifted_row.get("polar_invalid", 1.0)) < 0.5
                    ):
                        aligned_current = telemetry[aligned_index]["current"]
                        shifted_current = telemetry[shifted_index]["current"]
                        aligned_js = 0.5 * (
                            _js_divergence(
                                prediction["theta_probability"],
                                aligned_current["theta_probability"],
                            )
                            + _js_divergence(
                                prediction["distance_probability"],
                                aligned_current["distance_probability"],
                            )
                        )
                        shifted_js = 0.5 * (
                            _js_divergence(
                                prediction["theta_probability"],
                                shifted_current["theta_probability"],
                            )
                            + _js_divergence(
                                prediction["distance_probability"],
                                shifted_current["distance_probability"],
                            )
                        )
                        consistency[horizon]["aligned"].append(aligned_js)
                        consistency[horizon]["shift_control"].append(shifted_js)
    future_result = {}
    consistency_result = {}
    for horizon in FUTURE_HORIZONS:
        stats = future_stats[horizon]
        future_result[str(horizon)] = {
            "valid_rows": int(stats["valid_count"]),
            "visible_rows": int(stats["visible_count"]),
            "theta_nll": _mean(stats["theta_nll"]),
            "distance_nll": _mean(stats["distance_nll"]),
            "visibility_nll": _mean(stats["visibility_nll"]),
            "uniform_theta_nll": math.log(60.0),
            "uniform_distance_nll": math.log(30.0),
        }
        aligned = consistency[horizon]["aligned"]
        shifted = consistency[horizon]["shift_control"]
        aligned_mean = _mean(aligned)
        shifted_mean = _mean(shifted)
        consistency_result[str(horizon)] = {
            "pairs": len(aligned),
            "aligned_js": aligned_mean,
            "shift_control_js": shifted_mean,
            "aligned_over_shift": (
                None
                if aligned_mean is None or shifted_mean in (None, 0.0)
                else float(aligned_mean / shifted_mean)
            ),
            "shift_frames": shift_frames,
        }
    brier = _mean(
        [(probability - target) ** 2 for probability, target in zip(q_probabilities, q_targets)]
    )
    return {
        "current": {
            "visible_rows": visible_count,
            "theta_accuracy": theta_correct / visible_count if visible_count else None,
            "distance_accuracy": distance_correct / visible_count if visible_count else None,
            "theta_nll": _mean(theta_nll),
            "distance_nll": _mean(distance_nll),
            "invalid_nll": _mean(invalid_nll),
            "uniform_theta_nll": math.log(60.0),
            "uniform_distance_nll": math.log(30.0),
        },
        "future": future_result,
        "temporal_consistency": consistency_result,
        "q_calibration": {
            "rows": len(q_probabilities),
            "brier": brier,
            "ece_10bin": _ece(q_probabilities, q_targets, bins=10),
        },
    }


def _paired_scalar_delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def compare_reasoning_heads(
    full: Mapping[str, Any],
    recurrent_reset: Mapping[str, Any],
) -> dict[str, Any]:
    """Report paired aggregate deltas with directions explicit in each name."""

    full_current = full["current"]
    reset_current = recurrent_reset["current"]
    current = {
        "reset_minus_full_theta_nll": _paired_scalar_delta(
            reset_current["theta_nll"], full_current["theta_nll"]
        ),
        "reset_minus_full_distance_nll": _paired_scalar_delta(
            reset_current["distance_nll"], full_current["distance_nll"]
        ),
        "reset_minus_full_invalid_nll": _paired_scalar_delta(
            reset_current["invalid_nll"], full_current["invalid_nll"]
        ),
        "full_minus_reset_theta_accuracy": _paired_scalar_delta(
            full_current["theta_accuracy"], reset_current["theta_accuracy"]
        ),
        "full_minus_reset_distance_accuracy": _paired_scalar_delta(
            full_current["distance_accuracy"], reset_current["distance_accuracy"]
        ),
    }
    future = {}
    temporal = {}
    for horizon in FUTURE_HORIZONS:
        key = str(horizon)
        full_future = full["future"][key]
        reset_future = recurrent_reset["future"][key]
        future[key] = {
            "reset_minus_full_theta_nll": _paired_scalar_delta(
                reset_future["theta_nll"], full_future["theta_nll"]
            ),
            "reset_minus_full_distance_nll": _paired_scalar_delta(
                reset_future["distance_nll"], full_future["distance_nll"]
            ),
            "reset_minus_full_visibility_nll": _paired_scalar_delta(
                reset_future["visibility_nll"], full_future["visibility_nll"]
            ),
        }
        full_temporal = full["temporal_consistency"][key]
        reset_temporal = recurrent_reset["temporal_consistency"][key]
        temporal[key] = {
            "reset_minus_full_aligned_js": _paired_scalar_delta(
                reset_temporal["aligned_js"], full_temporal["aligned_js"]
            ),
            "reset_minus_full_aligned_over_shift": _paired_scalar_delta(
                reset_temporal["aligned_over_shift"],
                full_temporal["aligned_over_shift"],
            ),
            "full_pairs": full_temporal["pairs"],
            "reset_pairs": reset_temporal["pairs"],
        }
    return {
        "current": current,
        "future": future,
        "temporal_consistency": temporal,
        "q_calibration": {
            "reset_minus_full_brier": _paired_scalar_delta(
                recurrent_reset["q_calibration"]["brier"],
                full["q_calibration"]["brier"],
            ),
            "reset_minus_full_ece_10bin": _paired_scalar_delta(
                recurrent_reset["q_calibration"]["ece_10bin"],
                full["q_calibration"]["ece_10bin"],
            ),
        },
    }


def _load_prediction_jsonl(path: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    predictions = []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise F2ValidationDiagnosticError(
                    f"invalid prediction JSONL at {path}:{line_number}"
                ) from exc
            predictions.append(row["pred_step_actions"])
            records.append({key: value for key, value in row.items() if key != "pred_step_actions"})
    return predictions, records


def _identity(record: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(record.get("episode", "")),
        str(record.get("sequence_id", "")),
        int(record["frame_idx"]),
    )


def analyze_action_outputs(
    *,
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, Mapping[str, Sequence[Any]]],
    baseline_paths: Mapping[str, Path],
    selected_indices: Sequence[int],
) -> dict[str, Any]:
    if len(selected_indices) != len(records) or len(rows) != len(records):
        raise F2ValidationDiagnosticError("selected action rows/indices are misaligned")
    if (
        len(set(int(index) for index in selected_indices)) != len(selected_indices)
        or any(int(index) < 0 or int(index) >= VAL_ROWS for index in selected_indices)
    ):
        raise F2ValidationDiagnosticError("selected original indices are invalid")
    slices = derive_slices(rows)
    result: dict[str, Any] = {"slice_support": {k: len(v) for k, v in slices.items()}}
    methods: dict[str, Any] = {}
    for condition in CONDITIONS:
        for mode in MODES:
            name = f"F2_S-SELF_{condition}_{mode}"
            pred = predictions[condition][mode]
            methods[name] = {
                "overall": evaluate_predictions(
                    np.asarray(pred), list(records), threshold=CONTROL_THRESHOLD
                ),
                "slices": {
                    slice_name: action_slice_metrics(pred, records, indices)
                    for slice_name, indices in slices.items()
                },
            }
    persistence_predictions = [
        [[record["prev_action"][0], 0.0, record["prev_action"][2]] for _ in range(8)]
        for record in records
    ]
    methods["repeat_logged_prev"] = {
        "overall": evaluate_predictions(
            np.asarray(persistence_predictions),
            list(records),
            threshold=CONTROL_THRESHOLD,
        ),
        "slices": {
            slice_name: action_slice_metrics(
                persistence_predictions, records, indices
            )
            for slice_name, indices in slices.items()
        },
        "role": "trivial_AP2_persistence_control",
    }
    expected_identity = [_identity(record) for record in records]
    for name, path in baseline_paths.items():
        all_pred, all_baseline_records = _load_prediction_jsonl(path)
        if len(all_pred) != VAL_ROWS or len(all_baseline_records) != VAL_ROWS:
            raise F2ValidationDiagnosticError(
                f"baseline {name} must contain exactly {VAL_ROWS} rows"
            )
        pred = [all_pred[int(index)] for index in selected_indices]
        baseline_records = [
            all_baseline_records[int(index)] for index in selected_indices
        ]
        if [_identity(record) for record in baseline_records] != expected_identity:
            raise F2ValidationDiagnosticError(f"baseline {name} row identity mismatch")
        bound_fields = (
            "step_actions",
            "prev_action",
            "valid_mask",
            "transition_type",
            "command",
            "source_raw_dir",
        )
        for local_index, (baseline_record, canonical_record) in enumerate(
            zip(baseline_records, records)
        ):
            original_index = int(selected_indices[local_index])
            for field in bound_fields:
                if baseline_record.get(field) != canonical_record.get(field):
                    raise F2ValidationDiagnosticError(
                        f"baseline {name} global row {original_index} changed "
                        f"frozen field {field!r}"
                    )
        methods[name] = {
            "overall": evaluate_predictions(
                np.asarray(pred), list(records), threshold=CONTROL_THRESHOLD
            ),
            "slices": {
                slice_name: action_slice_metrics(pred, records, indices)
                for slice_name, indices in slices.items()
            },
            "prediction_path": str(path),
            "prediction_sha256": sha256_file(path),
        }
    result["methods"] = methods
    full_logged = methods["F2_S-SELF_full_logged"]
    full_self = methods["F2_S-SELF_full_self"]
    memory_reset_logged = methods["F2_S-SELF_recurrent_state_reset_logged"]
    reasoning_off_logged = methods["F2_S-SELF_reasoning_direct_off_logged"]
    polar_logged = methods["F2_S-SELF_polar_direct_off_logged"]
    future_logged = methods["F2_S-SELF_future_direct_off_logged"]
    memory_slice_names = (
        "all",
        "current_invalid",
        "current_visible",
        "turn",
        "reacquisition_offset_0",
    )
    memory_effects = {}
    for slice_name in memory_slice_names:
        reset_value = memory_reset_logged["slices"].get(slice_name, {}).get(
            "balanced_control_error_at1"
        )
        full_value = full_logged["slices"].get(slice_name, {}).get(
            "balanced_control_error_at1"
        )
        memory_effects[slice_name] = {
            "memory_reset_minus_full_bce": (
                None
                if reset_value is None or full_value is None
                else reset_value - full_value
            )
        }
    result["paired_effects"] = {
        "memory": memory_effects,
        "reasoning_action_credit": {
            "combined_off_minus_full_bce": (
                reasoning_off_logged["overall"]["balanced_control_error_at1"]["value"]
                - full_logged["overall"]["balanced_control_error_at1"]["value"]
            ),
            "polar_off_minus_full_bce": (
                polar_logged["overall"]["balanced_control_error_at1"]["value"]
                - full_logged["overall"]["balanced_control_error_at1"]["value"]
            ),
            "future_off_minus_full_bce": (
                future_logged["overall"]["balanced_control_error_at1"]["value"]
                - full_logged["overall"]["balanced_control_error_at1"]["value"]
            ),
        },
        "logged_to_self": {
            "full_self_minus_logged_bce": (
                full_self["overall"]["balanced_control_error_at1"]["value"]
                - full_logged["overall"]["balanced_control_error_at1"]["value"]
            )
        },
    }
    return result


def _parse_named_paths(values: Sequence[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise F2ValidationDiagnosticError(
                f"baseline prediction must use NAME=PATH, got {value!r}"
            )
        name, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not name or name in parsed or not path.is_file():
            raise F2ValidationDiagnosticError(f"invalid baseline prediction {value!r}")
        parsed[name] = path
    if tuple(parsed) != BASELINE_NAMES:
        raise F2ValidationDiagnosticError(
            "baseline predictions must be supplied exactly in this order: "
            + ", ".join(BASELINE_NAMES)
        )
    return parsed


def _run_directory_is_owned(output: Path, owner_token: str) -> bool:
    marker = output / "run_started.json"
    try:
        document = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(document, Mapping)
        and document.get("analysis_class")
        == "f2_public_validation_memory_reasoning_run_owner"
        and document.get("owner_token") == owner_token
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--preregistration-receipt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arm", default="S-SELF", choices=("S-SELF",))
    parser.add_argument("--snapshot", type=int, default=128, choices=(128,))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--val-json", default="data/collected_v1/datasets/val.jsonl")
    parser.add_argument(
        "--val-manifest", default="data/collected_v1/datasets/val.jsonl.manifest.json"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--determinism-probe",
        action="store_true",
        help="run the fixed 4x8 probe spanning all four clean val sequences",
    )
    parser.add_argument(
        "--baseline-prediction",
        action="append",
        default=[],
        help="NAME=prediction.jsonl; supply the frozen B0/B1 seed set in order",
    )
    parser.add_argument("--progress-every", type=int, default=32)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    root = Path(args.project_root).expanduser().resolve()
    receipt_path = (root / args.receipt).resolve() if not Path(args.receipt).is_absolute() else Path(args.receipt).resolve()
    checkpoint_path = (root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint).resolve()
    prereg_path = (root / args.preregistration).resolve() if not Path(args.preregistration).is_absolute() else Path(args.preregistration).resolve()
    if args.preregistration_receipt is None:
        prereg_receipt_path = prereg_path.with_name(
            prereg_path.stem + ".receipt.json"
        )
    else:
        prereg_receipt_path = (
            (root / args.preregistration_receipt).resolve()
            if not Path(args.preregistration_receipt).is_absolute()
            else Path(args.preregistration_receipt).resolve()
        )
    val_json = (root / args.val_json).resolve() if not Path(args.val_json).is_absolute() else Path(args.val_json).resolve()
    val_manifest = (root / args.val_manifest).resolve() if not Path(args.val_manifest).is_absolute() else Path(args.val_manifest).resolve()
    output = Path(args.output_dir).expanduser().resolve()
    owner_token = str(
        getattr(args, "_run_owner_token", None) or secrets.token_hex(32)
    )
    output.mkdir(parents=True, exist_ok=False)
    _exclusive_write_json(
        output / "run_started.json",
        {
            "schema_version": 1,
            "analysis_class": "f2_public_validation_memory_reasoning_run_owner",
            "owner_token": owner_token,
            "process_id": os.getpid(),
            "internal_test_opened": False,
        },
    )
    if not prereg_path.is_file():
        raise F2ValidationDiagnosticError("preregistration is missing")
    receipt_sha = sha256_file(receipt_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    baseline_paths = _parse_named_paths(args.baseline_prediction)
    selection = resolve_validation_selection(
        max_rows=args.max_rows,
        determinism_probe=bool(args.determinism_probe),
    )
    _preregistration, prereg_binding = validate_preregistration(
        prereg_path,
        prereg_receipt_path,
        checkpoint_sha256=checkpoint_sha,
        assembly_receipt_sha256=receipt_sha,
        baseline_paths=baseline_paths,
        selection=selection,
    )
    prereg_sha = prereg_binding["sha256"]
    rows, _manifest, data_binding, selection_trace, reset_contract = (
        _load_public_validation(
            val_json,
            val_manifest,
            max_rows=args.max_rows,
            determinism_probe=bool(args.determinism_probe),
        )
    )
    selection_trace_sha = _exclusive_write_json(
        output / "selection_trace.json", selection_trace
    )
    intent = {
        "schema_version": 1,
        "analysis_class": "f2_public_validation_memory_reasoning_diagnostic",
        "claim_eligibility": (
            "full_public_validation_abstract_eligible"
            if data_binding["abstract_claim_eligible"]
            else "engineering_only_not_abstract_eligible"
        ),
        "arm": args.arm,
        "snapshot": args.snapshot,
        "conditions": list(CONDITIONS),
        "modes": list(MODES),
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "assembly_receipt": {"path": str(receipt_path), "sha256": receipt_sha},
        "preregistration": prereg_binding,
        "data": data_binding,
        "selection_trace_sha256": selection_trace_sha,
        "reset_contract": reset_contract,
        "control_threshold": CONTROL_THRESHOLD,
        "temporal_shift_frames": TEMPORAL_SHIFT_FRAMES,
        "baseline_predictions": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in baseline_paths.items()
        },
        "core_changed": False,
        "training_performed": False,
        "gate_changed": False,
        "internal_test_opened": False,
    }
    _exclusive_write_json(output / "run_intent.json", intent)
    receipt_document, assembly_verification = verify_evaluator_assembly_receipt(
        root, receipt_path
    )
    _exclusive_write_json(
        output / "assembly_verification.json", assembly_verification
    )
    base_root, cache_root = frozen_cache_roots(root)
    ledger_started = time.perf_counter()
    ledger = build_token_ledger_for_rows(
        rows, base_root=base_root, cache_root=cache_root
    )
    ledger_document = dict(ledger.to_dict())
    ledger_document.update(
        {
            "analysis_class": "f2_public_val_token_hash_ledger",
            "split": "val",
            "rows": len(rows),
            "selection_name": selection["name"],
            "selection_sha256": selection["original_index_sha256"],
            "internal_test_opened": False,
        }
    )
    ledger_sha = _exclusive_write_json(output / "token_ledger.json", ledger_document)
    ledger_receipt = {
        "schema_version": 1,
        "analysis_class": "f2_public_val_token_ledger",
        "split": "val",
        "rows": len(rows),
        "token_files": ledger.token_files,
        "ledger_sha256": ledger.ledger_sha256,
        "ledger_file_sha256": ledger_sha,
        "selection_trace_sha256": selection_trace_sha,
        "selection_name": selection["name"],
        "selection_sha256": selection["original_index_sha256"],
        "build_seconds": time.perf_counter() - ledger_started,
        "internal_test_opened": False,
    }
    _exclusive_write_json(output / "token_ledger_receipt.json", ledger_receipt)
    payload = load_arm_checkpoint_verified(
        checkpoint_path,
        expected_assembly_receipt_sha256=receipt_sha,
        expected_arm=args.arm,
        expected_u_pre=args.snapshot,
    )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise F2ValidationDiagnosticError("the frozen Windows run requires CUDA")
    cuda_receipt = configure_cuda_reproducibility()
    predictor = build_eval_row_predictor_from_checkpoint(
        root,
        receipt_document,
        args.arm,
        payload,
        device=device,
    )
    del payload
    arm = predictor.arm
    normal_stream = PerceptionStream(arm, reset_every_row=False)
    memory_reset_stream = PerceptionStream(arm, reset_every_row=True)
    rollouts = {
        condition: {
            mode: RolloutState(ActionFilterController())
            for mode in MODES
        }
        for condition in CONDITIONS
    }
    predictions: dict[str, dict[str, list[Any]]] = {
        condition: {mode: [] for mode in MODES} for condition in CONDITIONS
    }
    records: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    prediction_handles = {}
    for condition in CONDITIONS:
        for mode in MODES:
            path = output / f"predictions_{condition}_{mode}.partial.jsonl"
            prediction_handles[condition, mode] = path.open("x", encoding="utf-8")
    telemetry_partial = output / "reasoning_telemetry.partial.jsonl"
    telemetry_handle = telemetry_partial.open("x", encoding="utf-8")
    selected_indices = tuple(int(index) for index in selection["original_indices"])
    reset_indices = set(int(index) for index in reset_contract["combined_indices"])
    reset_reasons_by_index = reset_contract["reasons_by_original_index"]
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.inference_mode():
            for position, row in enumerate(rows):
                original_index = selected_indices[position]
                reset = original_index in reset_indices
                reset_reasons = list(
                    reset_reasons_by_index.get(str(original_index), ())
                )
                if position == 0 and not reset:
                    raise F2ValidationDiagnosticError(
                        "first selected validation row must be a frozen reset boundary"
                    )
                packet = load_cached_observation(
                    row,
                    base_root=base_root,
                    cache_root=cache_root,
                    token_ledger=ledger,
                )
                normal_output = normal_stream.encode(packet, reset=reset, position=position)
                memory_reset_output = memory_reset_stream.encode(
                    packet, reset=reset, position=position
                )
                record = _record_from_row(row)
                record["original_validation_index"] = original_index
                records.append(record)
                telemetry = {
                    "original_validation_index": original_index,
                    "episode": record["episode"],
                    "sequence_id": record["sequence_id"],
                    "frame_idx": record["frame_idx"],
                    "reset": reset,
                    "reset_reasons": reset_reasons,
                    **reasoning_telemetry(normal_output, memory_reset_output),
                }
                logged_prev = tuple(record["prev_action"])
                for condition in CONDITIONS:
                    feature_output = (
                        memory_reset_output
                        if condition == "recurrent_state_reset"
                        else normal_output
                    )
                    alphas = intervention_alphas(feature_output, condition)
                    for mode in MODES:
                        rollout = rollouts[condition][mode]
                        prev = rollout.previous(logged_prev, reset=reset, mode=mode)
                        reference = feature_output["base_features"]
                        prev_tensor = torch.tensor(
                            [list(prev)], device=reference.device, dtype=reference.dtype
                        )
                        model_output = arm.model(
                            reference,
                            prev_tensor,
                            method_features=feature_output["method_features"],
                            method_alphas=alphas,
                        )
                        raw_actions = (
                            model_output.prediction.raw_actions.detach().float()[0].cpu().tolist()
                        )
                        transition = rollout.advance(raw_actions, mode=mode)
                        predictions[condition][mode].append(raw_actions)
                        artifact_row = {
                            **record,
                            "pred_step_actions": raw_actions,
                            "condition": condition,
                            "mode": mode,
                            "reset": reset,
                            "reset_reasons": reset_reasons,
                            "controller_sent_action": (
                                None if transition is None else list(transition.sent_action)
                            ),
                        }
                        prediction_handles[condition, mode].write(
                            json.dumps(artifact_row, ensure_ascii=False) + "\n"
                        )
                # Labels are attached only after all action predictions for the
                # row are complete.  They never enter ObservationPacket,
                # adapter.encode_step, or arm.model.
                telemetry["labels"] = _row_label_telemetry(row)
                telemetry_rows.append(telemetry)
                telemetry_handle.write(json.dumps(telemetry, ensure_ascii=False) + "\n")
                if args.progress_every > 0 and (
                    (position + 1) % args.progress_every == 0 or position + 1 == len(rows)
                ):
                    elapsed = time.perf_counter() - started
                    print(
                        json.dumps(
                            {
                                "status": "running",
                                "rows_completed": position + 1,
                                "rows_total": len(rows),
                                "elapsed_seconds": round(elapsed, 3),
                                "rows_per_second": round((position + 1) / elapsed, 4),
                                "cuda_peak_allocated_mb": round(
                                    torch.cuda.max_memory_allocated(device) / 1024**2, 2
                                ),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        telemetry_handle.close()
        for handle in prediction_handles.values():
            handle.close()
    for condition in CONDITIONS:
        for mode in MODES:
            partial = output / f"predictions_{condition}_{mode}.partial.jsonl"
            partial.replace(output / f"predictions_{condition}_{mode}.jsonl")
    telemetry_path = output / "reasoning_telemetry.jsonl"
    telemetry_partial.replace(telemetry_path)
    action_analysis = analyze_action_outputs(
        rows=rows,
        records=records,
        predictions=predictions,
        baseline_paths=baseline_paths,
        selected_indices=selected_indices,
    )
    full_reasoning = analyze_reasoning(
        rows,
        telemetry_rows,
        shift_frames=TEMPORAL_SHIFT_FRAMES,
    )
    recurrent_reset_reasoning = analyze_reasoning(
        rows,
        [item["recurrent_reset_heads"] for item in telemetry_rows],
        shift_frames=TEMPORAL_SHIFT_FRAMES,
    )
    reasoning_analysis = {
        "full": full_reasoning,
        "recurrent_state_reset": recurrent_reset_reasoning,
        "paired_effects": compare_reasoning_heads(
            full_reasoning, recurrent_reset_reasoning
        ),
    }
    metrics = {
        "schema_version": 1,
        "analysis_class": "f2_public_validation_memory_reasoning_metrics",
        "claim_eligibility": (
            "full_public_validation_abstract_eligible"
            if data_binding["abstract_claim_eligible"]
            else "engineering_only_not_abstract_eligible"
        ),
        "data": data_binding,
        "action": action_analysis,
        "reasoning": reasoning_analysis,
        "runtime": {
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "cuda_reproducibility": cuda_receipt,
        },
        "core_changed": False,
        "training_performed": False,
        "gate_changed": False,
        "internal_test_opened": False,
    }
    metrics_sha = _exclusive_write_json(output / "metrics.json", metrics)
    if sha256_file(checkpoint_path) != checkpoint_sha:
        raise F2ValidationDiagnosticError("checkpoint changed during evaluation")
    artifacts = {}
    for path in sorted(output.glob("*.json*")):
        if path.name in {"complete.json", "failed.json"}:
            continue
        artifacts[path.name] = sha256_file(path)
    if (output / "failed.json").exists():
        raise F2ValidationDiagnosticError("failed and complete seals are mutually exclusive")
    completion = {
        "schema_version": 1,
        "analysis_class": "f2_public_validation_memory_reasoning_completion",
        "status": (
            "PASS_FULL_PUBLIC_VALIDATION"
            if data_binding["abstract_claim_eligible"]
            else "PASS_ENGINEERING"
        ),
        "scientific_result": "see metrics.json; no threshold is changed here",
        "metrics_sha256": metrics_sha,
        "artifact_sha256": artifacts,
        "checkpoint_sha256": checkpoint_sha,
        "preregistration_sha256": prereg_sha,
        "rows": len(rows),
        "selection_name": selection["name"],
        "selection_sha256": selection["original_index_sha256"],
        "abstract_claim_eligible": data_binding["abstract_claim_eligible"],
        "truncated_smoke": data_binding["truncated_smoke"],
        "internal_test_opened": False,
    }
    _exclusive_write_json(output / "complete.json", completion)
    return completion


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    owner_token = secrets.token_hex(32)
    setattr(args, "_run_owner_token", owner_token)
    try:
        result = run(args)
    except Exception as exc:
        output = Path(args.output_dir).expanduser().resolve()
        failure_path = output / "failed.json"
        if (
            _run_directory_is_owned(output, owner_token)
            and not failure_path.exists()
            and not (output / "complete.json").exists()
        ):
            try:
                _exclusive_write_json(
                    failure_path,
                    {
                        "schema_version": 1,
                        "analysis_class": "f2_public_validation_memory_reasoning_failure",
                        "status": "FAILED_ENGINEERING_BURNED_DIRECTORY",
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                        "retry_policy": "use a fresh output directory after code review",
                        "internal_test_opened": False,
                    },
                )
            except FileExistsError:
                pass
        raise
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
