import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "OpenTrackVLA"
    / "scripts"
    / "train_baseline.py"
)
SPEC = importlib.util.spec_from_file_location("track_car_train_baseline", SCRIPT)
train_baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_baseline)


def write_dataset(tmp_path):
    dataset = tmp_path / "train.jsonl"
    payload = json.dumps({"waypoints": [[0.0, 0.0, 0.0]]}) + "\n"
    dataset.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "sample_count": 1,
        "fps": 10,
        "dt": 0.1,
        "label_mode": "step_action",
        "action_semantics": "arc_turn_v2",
    }
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return dataset


def test_baseline_cli_defaults_to_paper_comparison_history():
    args = train_baseline.parse_args(
        [
            "--train_json",
            "train.jsonl",
            "--base_hf_model_dir",
            "base",
        ]
    )
    assert args.history == 31
    assert args.lr == pytest.approx(2e-5)
    assert args.balance_sampling is False
    assert args.save_optimizer is False


def test_baseline_dataset_binding_accepts_waypoint_records(tmp_path):
    dataset = write_dataset(tmp_path)
    info = train_baseline.inspect_bound_dataset(str(dataset))
    assert info["sample_count"] == 1
    assert info["manifest"]["label_mode"] == "step_action"


def test_baseline_dataset_binding_rejects_content_change(tmp_path):
    dataset = write_dataset(tmp_path)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"waypoints": [[1.0, 0.0, 0.0]]}) + "\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        train_baseline.inspect_bound_dataset(str(dataset))


def test_common_base_training_scope_is_projector_and_planner_only():
    class Base(nn.Module):
        def __init__(self):
            super().__init__()
            self.llm = nn.Linear(2, 2)
            self.proj = nn.Linear(2, 2)
            self.planner = nn.Linear(2, 2)
            self.act_token = nn.Parameter(torch.zeros(1))

    model = Base()
    model.requires_grad_(False)
    model.proj.requires_grad_(True)
    model.planner.requires_grad_(True)
    assert all(p.requires_grad for p in model.proj.parameters())
    assert all(p.requires_grad for p in model.planner.parameters())
    assert not any(p.requires_grad for p in model.llm.parameters())
    assert model.act_token.requires_grad is False


def test_main_failure_records_failed_run_and_reraises(monkeypatch, tmp_path):
    logger = train_baseline.JsonlMetricLogger(tmp_path)
    logger.start_run(
        args={"seed": 0},
        checkpoint_meta={"checkpoint_selection": {"metric": "BCE", "mode": "min"}},
        total_params=1,
        trainable_params=1,
        install_exception_hook=False,
    )

    def fail(_argv=None):
        train_baseline._ACTIVE_METRIC_LOGGER = logger
        raise RuntimeError("training failed")

    monkeypatch.setattr(train_baseline, "_main_impl", fail)
    with pytest.raises(RuntimeError, match="training failed"):
        train_baseline.main([])
    records = [json.loads(line) for line in logger.path.read_text().splitlines()]
    assert records[-1]["phase"] == "run_end"
    assert records[-1]["status"] == "failed"
