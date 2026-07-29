# ruff: noqa: E402 -- import the script module after adding scripts/ to sys.path.
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import eval_matched128_public_val as target


def test_frozen_selection_hashes_and_sizes():
    probe = target.selection_indices("determinism_probe_4x8")
    full = target.selection_indices("full_2848_public_validation")
    assert len(probe) == 32
    assert len(full) == 2848
    assert target.canonical_json_sha256(list(probe)) == target.PROBE_SELECTION_SHA256
    assert target.canonical_json_sha256(list(full)) == target.FULL_SELECTION_SHA256


def test_waypoints_to_step_actions_inverts_straight_motion():
    waypoints = np.zeros((1, 8, 3), dtype=np.float64)
    waypoints[0, :, 0] = np.arange(1, 9, dtype=np.float64) * 0.1
    actions = target.waypoints_to_step_actions(waypoints, dt=0.1)
    np.testing.assert_allclose(actions[0, :, 0], 1.0)
    np.testing.assert_allclose(actions[0, :, 1:], 0.0)


def test_rejects_internal_test_path(tmp_path):
    path = tmp_path / "internal_test" / "artifact.json"
    with pytest.raises(target.MatchedPublicValError, match="sealed"):
        target.reject_sealed_path(path, "artifact")


def test_unknown_selection_fails_closed():
    with pytest.raises(target.MatchedPublicValError, match="unsupported selection"):
        target.selection_indices("prefix_512")


def test_test_jsonl_is_rejected_before_any_file_read(monkeypatch):
    args = target.build_parser().parse_args(
        [
            "--run-dir",
            str(target.MATCHED_EVAL_ROOT / "full_2848"),
            "--selection",
            "full_2848_public_validation",
            "--preregistration-json",
            str(target.EXPECTED_PREREGISTRATION_PATH),
            "--expected-preregistration-sha256",
            "0" * 64,
            "--val-json",
            str(target.PROJECT_ROOT / "data/collected_v1/datasets/test.jsonl"),
        ]
    )
    opened = []

    def forbidden_read_bytes(self):
        opened.append(str(self))
        raise AssertionError("no file bytes may be read before the sealed-path rejection")

    def forbidden_open(self, *args, **kwargs):
        opened.append(str(self))
        raise AssertionError("no file may be opened before the sealed-path rejection")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read_bytes)
    monkeypatch.setattr(Path, "open", forbidden_open)
    with pytest.raises(target.MatchedPublicValError, match="sealed"):
        target.run(args)
    assert opened == []


def test_checkpoint_sha_is_checked_before_torch_load(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-frozen-checkpoint")
    called = False

    def forbidden_torch_load(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not see unverified bytes")

    monkeypatch.setattr(target.torch, "load", forbidden_torch_load)
    with pytest.raises(target.MatchedPublicValError, match="SHA mismatch"):
        target.load_checkpoint(checkpoint, "0" * 64)
    assert called is False


def test_runtime_dependency_drift_fails_closed(monkeypatch):
    monkeypatch.setattr(target, "runtime_dependency_binding", lambda: {"tree": "new"})
    with pytest.raises(target.MatchedPublicValError, match="changed before forward"):
        target.require_runtime_dependencies({"tree": "frozen"}, "before forward")


def test_native_windows_peak_stats_prime_allocator_without_rng():
    events = []

    class FakeCuda:
        def memory_allocated(self, device):
            events.append(("memory_allocated", device))
            return 0

        def synchronize(self, device):
            events.append(("synchronize", device))

        def empty_cache(self):
            events.append(("empty_cache",))

        def reset_peak_memory_stats(self, device):
            events.append(("reset", device))

    class FakeTorch:
        uint8 = "uint8"
        cuda = FakeCuda()

        @staticmethod
        def empty(shape, *, dtype, device):
            events.append(("empty", shape, dtype, device))
            return object()

    device = target.torch.device("cuda:0")
    target.reset_cuda_peak_memory_stats_portably(
        device, torch_module=FakeTorch(), platform_name="nt"
    )
    assert events == [
        ("memory_allocated", device),
        ("empty", (1,), "uint8", device),
        ("synchronize", device),
        ("empty_cache",),
        ("reset", device),
    ]


def test_full_gate_is_required(tmp_path, monkeypatch):
    gate = tmp_path / "determinism_probe_comparison.json"
    monkeypatch.setattr(target, "EXPECTED_PROBE_COMPARISON_PATH", gate)
    with pytest.raises(target.MatchedPublicValError, match="PASS gate is missing"):
        target.validate_determinism_probe_gate(
            gate,
            preregistration={
                "evaluator_source_sha256": "0" * 64,
                "analyzer_source_sha256": "1" * 64,
                "runtime_dependencies": {},
            },
            preregistration_sha256="2" * 64,
            raw_rows=[],
        )


def test_forged_probe_gate_canonical_hash_is_rejected(tmp_path, monkeypatch):
    gate = tmp_path / "determinism_probe_comparison.json"
    gate.write_text(
        json.dumps({"canonical_payload_sha256": "0" * 64}), encoding="utf-8"
    )
    monkeypatch.setattr(target, "EXPECTED_PROBE_COMPARISON_PATH", gate)
    with pytest.raises(target.MatchedPublicValError, match="canonical SHA mismatch"):
        target.validate_determinism_probe_gate(
            gate,
            preregistration={
                "evaluator_source_sha256": "0" * 64,
                "analyzer_source_sha256": "1" * 64,
                "runtime_dependencies": {},
            },
            preregistration_sha256="2" * 64,
            raw_rows=[],
        )


def test_completion_artifact_parent_escape_is_rejected(tmp_path):
    run_dir = tmp_path / "probe_a"
    run_dir.mkdir()
    artifacts = {name: "0" * 64 for name in target.COMPLETION_ARTIFACT_BASENAMES}
    artifacts["../metrics.json"] = artifacts.pop("metrics.json")
    completion = {
        "schema_version": target.SCHEMA_VERSION,
        "analysis_class": "matched128_public_validation_completion_v1",
        "status": "PASS_DETERMINISM_PROBE",
        "selection_name": "determinism_probe_4x8",
        "selection_rows": len(target.PROBE_INDICES),
        "selection_sha256": target.PROBE_SELECTION_SHA256,
        "preregistration_sha256": "2" * 64,
        "evaluator_source_sha256": "3" * 64,
        "internal_test_opened": False,
        "checkpoint_sha256": {
            "matched_B0_seed0": target.B0_CHECKPOINT_SHA256,
            "matched_B1_seed0": target.B1_CHECKPOINT_SHA256,
        },
        "artifact_sha256": artifacts,
    }
    (run_dir / "complete.json").write_text(json.dumps(completion), encoding="utf-8")
    with pytest.raises(target.MatchedPublicValError, match="artifact binding"):
        target.validate_probe_completion(
            run_dir,
            preregistration_sha256="2" * 64,
            evaluator_source_sha256="3" * 64,
            runtime_dependencies={},
        )


def test_probe_completion_rejects_failed_and_complete_coexistence(tmp_path):
    run_dir = tmp_path / "probe_a"
    run_dir.mkdir()
    (run_dir / "failed.json").write_text('{"status":"FAILED"}\n', encoding="utf-8")
    (run_dir / "complete.json").write_text('{}\n', encoding="utf-8")
    with pytest.raises(target.MatchedPublicValError, match="both failed.json and complete.json"):
        target.validate_probe_completion(
            run_dir,
            preregistration_sha256="2" * 64,
            evaluator_source_sha256="3" * 64,
            runtime_dependencies={},
        )


def test_val_ledger_hash_and_parse_share_one_read_per_file(tmp_path, monkeypatch):
    ledger_path = tmp_path / "ledger.json"
    receipt_path = tmp_path / "receipt.json"
    entries = {"episodes/val/example.pt": "0" * 64}
    ledger_payload = json.dumps({"entries": entries}, sort_keys=True).encode("utf-8")
    ledger_path.write_bytes(ledger_payload)
    payload_sha256 = "a" * 64
    selection_sha256 = "b" * 64
    receipt_payload = json.dumps(
        {
            "analysis_class": "f2_public_val_token_ledger",
            "internal_test_opened": False,
            "rows": target.VAL_ROWS,
            "token_files": 1,
            "ledger_file_sha256": hashlib.sha256(ledger_payload).hexdigest(),
            "ledger_sha256": payload_sha256,
            "selection_sha256": selection_sha256,
        },
        sort_keys=True,
    ).encode("utf-8")
    receipt_path.write_bytes(receipt_payload)
    monkeypatch.setattr(
        target, "VAL_LEDGER_FILE_SHA256", hashlib.sha256(ledger_payload).hexdigest()
    )
    monkeypatch.setattr(
        target,
        "VAL_LEDGER_RECEIPT_SHA256",
        hashlib.sha256(receipt_payload).hexdigest(),
    )
    monkeypatch.setattr(target, "VAL_LEDGER_PAYLOAD_SHA256", payload_sha256)
    monkeypatch.setattr(target, "VAL_LEDGER_FILES", 1)
    monkeypatch.setattr(target, "FULL_SELECTION_SHA256", selection_sha256)

    class FakeLedger:
        def __init__(self, *, entries):
            self.entries = entries
            self.ledger_sha256 = payload_sha256

    fake_module = SimpleNamespace(
        __file__=str(target.PROJECT_ROOT / "f2_experiment/assembly_data.py"),
        TokenHashLedger=FakeLedger,
    )
    real_import = target.importlib.import_module

    def fake_import(name):
        if name == "f2_experiment.assembly_data":
            return fake_module
        return real_import(name)

    monkeypatch.setattr(target.importlib, "import_module", fake_import)
    real_read_bytes = Path.read_bytes
    counts = {ledger_path.resolve(): 0, receipt_path.resolve(): 0}

    def counted_read_bytes(self):
        resolved = self.resolve()
        if resolved in counts:
            counts[resolved] += 1
        return real_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    token_ledger, binding = target.validate_val_token_ledger(
        ledger_path, receipt_path
    )
    assert token_ledger.entries == entries
    assert binding["ledger_sha256"] == payload_sha256
    assert counts == {ledger_path.resolve(): 1, receipt_path.resolve(): 1}


def test_probe_prediction_schema_rejects_wrong_b1_reset():
    raw_rows = []
    for index in range(max(target.PROBE_INDICES) + 1):
        raw_rows.append(
            {
                "episode": "episodes/val/example",
                "sequence_id": f"sequence_{index // 512}",
                "chunk_id": f"chunk_{index // 512}",
                "clip_id": f"clip_{index // 8}",
                "frame_idx": index,
                "mirrored": False,
                "source_raw_dir": "source",
                "transition_type": "steady",
                "command": "follow",
                "prev_action": [0.0, 0.0, 0.0],
                "step_actions": [[0.0, 0.0, 0.0] for _ in range(8)],
                "valid_mask": [True] * 8,
            }
        )
    predictions = []
    for original_index in target.PROBE_INDICES:
        row = {
            **target.prediction_identity(raw_rows[original_index]),
            "original_validation_index": original_index,
            "pred_waypoints": [[0.0, 0.0, 0.0] for _ in range(8)],
            "pred_step_actions": [[0.0, 0.0, 0.0] for _ in range(8)],
            "method": "matched_B1_seed0",
            "state_mode": "rolling",
            "reset": original_index in target.BASE_RESET_INDICES,
            "reset_reasons": (
                ["sequence_discontinuity"]
                if original_index in target.BASE_RESET_INDICES
                else []
            ),
        }
        predictions.append(row)
    predictions[1]["reset"] = True
    predictions[1]["reset_reasons"] = ["sequence_discontinuity"]
    payload = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8")
        for row in predictions
    )
    with pytest.raises(target.MatchedPublicValError, match="state/reset"):
        target.validate_probe_prediction_rows(
            payload,
            raw_rows,
            method="matched_B1_seed0",
            label="probe B1",
        )
