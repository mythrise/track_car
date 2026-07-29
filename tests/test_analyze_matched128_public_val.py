# ruff: noqa: E402 -- import the script module after adding scripts/ to sys.path.
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import analyze_matched128_public_val as target


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True) + "\n").encode("utf-8")


def _public_row(index=0):
    return {
        "step_actions": [[0.0, 0.0, 0.0] for _ in range(8)],
        "prev_action": [0.0, 0.0, 0.0],
        "valid_mask": [True] * 8,
        "transition_type": "steady",
        "episode": "episode-public",
        "sequence_id": "sequence-public",
        "chunk_id": "chunk-public",
        "clip_id": "clip-public",
        "frame_idx": index,
        "mirrored": False,
        "command": "follow",
        "source_raw_dir": "data/collected_v1/episodes/val/episode-public",
    }


def _prediction_row(source, *, index=0, method="matched_B0_seed0"):
    waypoints = np.zeros((1, 8, 3), dtype=np.float64)
    waypoints[0, :, 0] = np.arange(1, 9, dtype=np.float64) * 0.1
    actions = target.waypoints_to_step_actions(waypoints)[0]
    rolling = method == "matched_B1_seed0"
    reset = not rolling or index in target.BASE_RESET_INDICES
    return {
        **target.prediction_identity(source),
        "original_validation_index": index,
        "pred_waypoints": waypoints[0].tolist(),
        "pred_step_actions": actions.tolist(),
        "method": method,
        "state_mode": "rolling" if rolling else "stateless",
        "reset": reset,
        "reset_reasons": (
            ["sequence_discontinuity"]
            if rolling and reset
            else (["stateless_method"] if not rolling else [])
        ),
    }


def _write_valid_completion_run(
    run_dir,
    *,
    preregistration_sha256="b" * 64,
    evaluator_source_sha256="a" * 64,
    runtime_dependencies=None,
):
    runtime_dependencies = runtime_dependencies or {"dependency": "frozen"}
    run_dir.mkdir(parents=True)
    for filename in target.METHOD_FILES.values():
        (run_dir / filename).write_bytes(b"{}\n")
    metrics = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": target.EVALUATION_CLASS,
        "selection_name": "determinism_probe_4x8",
        "selection_rows": len(target.PROBE_INDICES),
        "method_metrics": {
            "matched_B0_seed0": {
                "checkpoint_sha256": target.B0_CHECKPOINT_SHA256,
                "metrics": {"intentionally_untrusted": -999.0},
            },
            "matched_B1_seed0": {
                "checkpoint_sha256": target.B1_CHECKPOINT_SHA256,
                "metrics": {"intentionally_untrusted": -999.0},
            },
        },
        "internal_test_opened": False,
    }
    (run_dir / "metrics.json").write_bytes(_json_bytes(metrics))
    run_intent = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": target.EVALUATION_CLASS,
        "selection_name": "determinism_probe_4x8",
        "selection_rows": len(target.PROBE_INDICES),
        "selection_sha256": target.PROBE_SELECTION_SHA256,
        "preregistration_path": str(target.EXPECTED_PREREGISTRATION_PATH.resolve()),
        "preregistration_sha256": preregistration_sha256,
        "evaluator_source_sha256": evaluator_source_sha256,
        "runtime_dependencies": runtime_dependencies,
        "determinism_probe_gate": None,
        "public_validation": {
            "path": str(target.PUBLIC_VAL_PATH.resolve()),
            "sha256": target.VAL_SHA256,
            "manifest_path": str(target.PUBLIC_VAL_MANIFEST_PATH.resolve()),
            "manifest_sha256": target.VAL_MANIFEST_SHA256,
            "full_rows": target.VAL_ROWS,
            "internal_test_opened": False,
        },
        "internal_test_opened": False,
        "checkpoint_selection_performed": False,
    }
    (run_dir / "run_intent.json").write_bytes(_json_bytes(run_intent))
    (run_dir / "run_started.json").write_bytes(
        _json_bytes(
            {
                "schema_version": target.SCHEMA_VERSION,
                "analysis_class": target.EVALUATION_CLASS,
                "started_utc": "2026-07-22T00:00:00Z",
            }
        )
    )
    artifacts = {
        name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
        for name in target.COMPLETION_ARTIFACT_BASENAMES
    }
    completion = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": "matched128_public_validation_completion_v1",
        "status": "PASS_DETERMINISM_PROBE",
        "selection_name": "determinism_probe_4x8",
        "selection_rows": len(target.PROBE_INDICES),
        "selection_sha256": target.PROBE_SELECTION_SHA256,
        "artifact_sha256": artifacts,
        "checkpoint_sha256": {
            "matched_B0_seed0": target.B0_CHECKPOINT_SHA256,
            "matched_B1_seed0": target.B1_CHECKPOINT_SHA256,
        },
        "preregistration_sha256": preregistration_sha256,
        "evaluator_source_sha256": evaluator_source_sha256,
        "determinism_probe_gate": None,
        "internal_test_opened": False,
        "completed_utc": "2026-07-22T00:01:00Z",
    }
    (run_dir / "complete.json").write_bytes(_json_bytes(completion))
    return runtime_dependencies


def test_relative_reduction():
    assert target.relative_reduction(0.05, 0.10) == 0.5
    assert target.relative_reduction(0.05, 0.0) is None
    assert target.relative_reduction(None, 0.1) is None


def test_primary_bootstrap_requires_frozen_status_and_all_replicates():
    effect = {
        "status": target.BOOTSTRAP_STATUS,
        "requested_replicates": target.BOOTSTRAP_REPLICATES,
        "valid_replicates": target.BOOTSTRAP_REPLICATES,
        "ci95": [-0.2, -0.1],
    }
    assert target.complete_primary_bootstrap_effect(effect) is True
    for field, value in (
        ("status", "UNBOUND_STATUS"),
        ("requested_replicates", target.BOOTSTRAP_REPLICATES - 1),
        ("valid_replicates", target.BOOTSTRAP_REPLICATES - 1),
        ("ci95", [-0.2, 0.0]),
    ):
        mutated = dict(effect)
        mutated[field] = value
        assert target.complete_primary_bootstrap_effect(mutated) is False


def test_frozen_probe_selection_hash():
    assert len(target.PROBE_INDICES) == 32
    assert (
        target.canonical_json_sha256(list(target.PROBE_INDICES))
        == target.PROBE_SELECTION_SHA256
    )


@pytest.mark.parametrize("payload", [b'{"x": NaN}', b'{"x": 1, "x": 2}'])
def test_json_parser_rejects_nonfinite_values_and_duplicate_keys(payload):
    with pytest.raises(target.MatchedAnalysisError, match="valid UTF-8 JSON"):
        target.parse_json_bytes(payload, "unsafe JSON")


def test_method_files_use_evaluator_machine_keys_and_separate_display_names():
    assert target.METHOD_FILES == {
        "matched_B0_seed0": "predictions_matched_B0_seed0.jsonl",
        "matched_B1_seed0": "predictions_matched_B1_seed0.jsonl",
    }
    assert target.METHOD_DISPLAY_NAMES == {
        "matched_B0_seed0": "TrackVLA_B0_matched128_seed0",
        "matched_B1_seed0": "TrackVLAPlusPlusLite_B1_matched128_seed0",
    }


def test_verify_probes_rejects_test_jsonl_before_any_read(monkeypatch):
    reads = []

    def forbidden_read(path):
        reads.append(path)
        raise AssertionError("no bytes may be read before sealed-path rejection")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    args = argparse.Namespace(
        preregistration_json=str(target.EXPECTED_PREREGISTRATION_PATH),
        expected_preregistration_sha256="0" * 64,
        val_json=str(target.PROJECT_ROOT / "data/collected_v1/datasets/test.jsonl"),
        val_manifest=str(target.PUBLIC_VAL_MANIFEST_PATH),
        probe_a_dir=str(target.PROBE_A_DIR),
        probe_b_dir=str(target.PROBE_B_DIR),
        output=str(target.PROBE_COMPARISON_PATH),
    )
    with pytest.raises(target.MatchedAnalysisError, match="sealed internal-test"):
        target.verify_probes(args)
    assert reads == []


def test_analyze_full_rejects_test_jsonl_before_any_read(monkeypatch):
    reads = []

    def forbidden_read(path):
        reads.append(path)
        raise AssertionError("no bytes may be read before sealed-path rejection")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    args = argparse.Namespace(
        preregistration_json=str(target.EXPECTED_PREREGISTRATION_PATH),
        expected_preregistration_sha256="0" * 64,
        val_json=str(target.PROJECT_ROOT / "data/collected_v1/datasets/test.jsonl"),
        val_manifest=str(target.PUBLIC_VAL_MANIFEST_PATH),
        probe_a_dir=str(target.PROBE_A_DIR),
        probe_b_dir=str(target.PROBE_B_DIR),
        full_run_dir=str(target.FULL_RUN_DIR),
        probe_comparison=str(target.PROBE_COMPARISON_PATH),
        f2_prediction=str(target.F2_PREDICTION_PATH),
        f2_completion=str(target.F2_COMPLETION_PATH),
        f2_analysis=str(target.F2_ANALYSIS_PATH),
        f2_metrics=str(target.F2_METRICS_PATH),
        output=str(target.ANALYSIS_OUTPUT_PATH),
        row_output=str(target.ROW_OUTPUT_PATH),
    )
    with pytest.raises(target.MatchedAnalysisError, match="sealed internal-test"):
        target.analyze_full(args)
    assert reads == []


def test_probe_directories_must_be_distinct_fixed_paths():
    with pytest.raises(target.MatchedAnalysisError, match="distinct exact directories"):
        target.resolve_probe_dirs(target.PROBE_A_DIR, target.PROBE_A_DIR)


def test_completion_rejects_parent_escape_artifact_before_artifact_read(tmp_path):
    run_dir = tmp_path / "probe_a_4x8"
    run_dir.mkdir()
    artifacts = {
        name: "0" * 64
        for name in target.COMPLETION_ARTIFACT_BASENAMES
        if name != "metrics.json"
    }
    artifacts["../metrics.json"] = "0" * 64
    completion = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": "matched128_public_validation_completion_v1",
        "status": "PASS_DETERMINISM_PROBE",
        "selection_name": "determinism_probe_4x8",
        "selection_rows": len(target.PROBE_INDICES),
        "selection_sha256": target.PROBE_SELECTION_SHA256,
        "artifact_sha256": artifacts,
        "checkpoint_sha256": {
            "matched_B0_seed0": target.B0_CHECKPOINT_SHA256,
            "matched_B1_seed0": target.B1_CHECKPOINT_SHA256,
        },
        "preregistration_sha256": "b" * 64,
        "evaluator_source_sha256": "a" * 64,
        "determinism_probe_gate": None,
        "internal_test_opened": False,
        "completed_utc": "2026-07-22T00:01:00Z",
    }
    (run_dir / "complete.json").write_bytes(_json_bytes(completion))
    with pytest.raises(target.MatchedAnalysisError, match="artifact name is unsafe"):
        target.validate_completion(
            run_dir,
            expected_run_dir=run_dir,
            expected_status="PASS_DETERMINISM_PROBE",
            expected_selection="determinism_probe_4x8",
            expected_rows=len(target.PROBE_INDICES),
            expected_selection_sha256=target.PROBE_SELECTION_SHA256,
            preregistration_sha256="b" * 64,
            evaluator_source_sha256="a" * 64,
            runtime_dependencies={},
            expected_probe_gate=None,
        )


def test_completion_revalidates_bound_artifacts_from_same_bytes(tmp_path):
    run_dir = tmp_path / "probe_a_4x8"
    runtime_dependencies = _write_valid_completion_run(run_dir)
    completion, verified, payloads = target.validate_completion(
        run_dir,
        expected_run_dir=run_dir,
        expected_status="PASS_DETERMINISM_PROBE",
        expected_selection="determinism_probe_4x8",
        expected_rows=len(target.PROBE_INDICES),
        expected_selection_sha256=target.PROBE_SELECTION_SHA256,
        preregistration_sha256="b" * 64,
        evaluator_source_sha256="a" * 64,
        runtime_dependencies=runtime_dependencies,
        expected_probe_gate=None,
    )
    assert completion["status"] == "PASS_DETERMINISM_PROBE"
    assert set(payloads) == target.COMPLETION_ARTIFACT_BASENAMES
    assert verified["metrics.json"] == hashlib.sha256(payloads["metrics.json"]).hexdigest()


def test_completion_rejects_failed_and_complete_coexistence(tmp_path):
    run_dir = tmp_path / "probe_a_4x8"
    runtime_dependencies = _write_valid_completion_run(run_dir)
    (run_dir / "failed.json").write_bytes(_json_bytes({"status": "FAILED"}))
    with pytest.raises(target.MatchedAnalysisError, match="both failed.json and complete.json"):
        target.validate_completion(
            run_dir,
            expected_run_dir=run_dir,
            expected_status="PASS_DETERMINISM_PROBE",
            expected_selection="determinism_probe_4x8",
            expected_rows=len(target.PROBE_INDICES),
            expected_selection_sha256=target.PROBE_SELECTION_SHA256,
            preregistration_sha256="b" * 64,
            evaluator_source_sha256="a" * 64,
            runtime_dependencies=runtime_dependencies,
            expected_probe_gate=None,
        )


def test_sha_bound_dynamic_import_rejects_dependency_drift_before_execution(
    tmp_path, monkeypatch
):
    dependency = tmp_path / "analyze_f2_public_val_outputs.py"
    dependency.write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(target, "SCRIPTS_ROOT", tmp_path)
    monkeypatch.setattr(target, "F2_ANALYZER_PATH", dependency)
    monkeypatch.setattr(target, "_F2_ANALYZER", None)
    monkeypatch.setattr(target, "_F2_ANALYZER_SOURCE_SHA256", None)
    preregistration = {"analyzer_runtime": target.analyzer_runtime_binding()}
    dependency.write_text('raise RuntimeError("must not execute")\n', encoding="utf-8")
    with pytest.raises(target.MatchedAnalysisError, match="dependency/runtime changed"):
        target.activate_f2_analyzer(preregistration)


def test_prediction_schema_accepts_exact_machine_method_contract():
    source = _public_row()
    prediction = _prediction_row(source)
    rows = target.validate_matched_prediction_rows(
        _json_bytes(prediction),
        [source],
        [0],
        method="matched_B0_seed0",
        label="synthetic B0",
    )
    assert rows == [prediction]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(episode="changed"), "identity mismatch"),
        (lambda row: row.update(mirrored=0), "identity mismatch"),
        (lambda row: row.update(reset=False), "method/state/reset"),
        (lambda row: row.update(method="TrackVLA_B0_matched128_seed0"), "method/state/reset"),
        (
            lambda row: row["pred_waypoints"][0].__setitem__(0, 9.0),
            "waypoint/action conversion",
        ),
        (
            lambda row: row["pred_waypoints"][0].__setitem__(0, "0.1"),
            "prediction tensor contract",
        ),
    ],
)
def test_prediction_schema_rejects_identity_reset_method_or_waypoint_mutation(
    mutation, message
):
    source = _public_row()
    prediction = _prediction_row(source)
    mutation(prediction)
    with pytest.raises(target.MatchedAnalysisError, match=message):
        target.validate_matched_prediction_rows(
            _json_bytes(prediction),
            [source],
            [0],
            method="matched_B0_seed0",
            label="mutated B0",
        )


def test_full_probe_replay_compares_complete_prediction_row_bytes():
    full_rows = [
        {"pred_step_actions": [[0.0, 0.0, 0.0]], "reset": True},
        {"pred_step_actions": [[1.0, 0.0, 0.0]], "reset": False},
    ]
    full_payload = b"".join(_json_bytes(row) for row in full_rows)
    target.require_selected_jsonl_bytes(
        full_payload, _json_bytes(full_rows[1]), [1], label="matched_B1_seed0"
    )
    forged_probe_row = dict(full_rows[1])
    forged_probe_row["reset"] = True
    with pytest.raises(target.MatchedAnalysisError, match="differ bytewise"):
        target.require_selected_jsonl_bytes(
            full_payload,
            _json_bytes(forged_probe_row),
            [1],
            label="matched_B1_seed0",
        )


def test_forged_probe_comparison_is_rejected_after_live_recomputation():
    live = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": target.PROBE_CLASS,
        "status": "PASS_BYTE_IDENTICAL_4X8",
        "methods": {"matched_B0_seed0": {"sha256": "a" * 64}},
    }
    live["canonical_payload_sha256"] = target.canonical_json_sha256(live)
    forged = dict(live)
    forged["methods"] = {"matched_B0_seed0": {"sha256": "f" * 64}}
    forged_without_canonical = dict(forged)
    forged_without_canonical.pop("canonical_payload_sha256")
    forged["canonical_payload_sha256"] = target.canonical_json_sha256(
        forged_without_canonical
    )
    with pytest.raises(target.MatchedAnalysisError, match="live recomputation"):
        target.validate_probe_comparison_payload(_json_bytes(forged), live)
