import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "OpenTrackVLA"
    / "scripts"
    / "train_pfem.py"
)
SPEC = importlib.util.spec_from_file_location("track_car_train_pfem", SCRIPT)
train_pfem = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_pfem)


def test_ws2_training_cli_defaults():
    args = train_pfem.parse_args(["--train_json", "dummy.jsonl"])
    assert args.lambda_yaw == 2.0
    assert args.aux_delta_vel is False
    assert args.balance_sampling is True


def test_checkpoint_meta_uses_step_action_manifest_and_delta_scale(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "fps": 10,
        "dt": 0.1,
        "label_mode": "step_action",
        "action_semantics": "arc_turn_v2",
        "delta_scale": 1.0,
    }
    Path(str(dataset) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    args = SimpleNamespace(
        train_json=str(dataset),
        n_waypoints=8,
        history=31,
        label_mode=None,
        aux_delta_vel=True,
        lambda_yaw=2.0,
    )
    meta = train_pfem.build_checkpoint_meta(args)
    assert meta["label_mode"] == "step_action"
    assert meta["delta_scale"] == 1.0
    assert meta["aux_delta_vel"] is True
    assert meta["data_manifest_hash"]


def test_checkpoint_meta_rejects_cli_label_conflict(tmp_path):
    dataset = tmp_path / "train.jsonl"
    dataset.write_text("", encoding="utf-8")
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fps": 10,
                "dt": 0.1,
                "label_mode": "step_action",
                "action_semantics": "spin_v1",
                "delta_scale": 1.0,
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        train_json=str(dataset),
        n_waypoints=8,
        history=31,
        label_mode="absolute",
        aux_delta_vel=False,
        lambda_yaw=2.0,
    )
    with pytest.raises(ValueError, match="conflicts"):
        train_pfem.build_checkpoint_meta(args)


def test_balanced_sampler_is_not_used_when_events_already_meet_target():
    class Dataset:
        examples = [
            {"transition_type": "steady_forward"},
            {"transition_type": "turn_onset"},
        ]

        def __len__(self):
            return len(self.examples)

        def get_example(self, index):
            return self.examples[index]

    assert train_pfem.build_training_sampler(Dataset(), enabled=True) is None
