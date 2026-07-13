import hashlib
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


def write_bound_dataset(tmp_path, *, label_mode="step_action", samples=None, manifest_overrides=None):
    dataset = tmp_path / "train.jsonl"
    if samples is None:
        samples = [
            {
                "step_actions": [[1.0, 0.0, 0.0]],
                "prev_action": [0.0, 0.0, 0.0],
                "delta_vel": [[1.0, 0.0, 0.0]],
            }
        ]
    payload = "".join(json.dumps(sample) + "\n" for sample in samples)
    dataset.write_text(payload, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "fps": 10,
        "dt": 0.1,
        "label_mode": label_mode,
        "action_semantics": "arc_turn_v2",
        "delta_scale": 1.0,
        "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "sample_count": len(samples),
    }
    manifest.update(manifest_overrides or {})
    Path(str(dataset) + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset, manifest


def checkpoint_args(dataset, **overrides):
    values = {
        "train_json": str(dataset),
        "n_waypoints": 8,
        "history": 31,
        "label_mode": None,
        "aux_delta_vel": True,
        "lambda_yaw": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ws2_training_cli_defaults():
    args = train_pfem.parse_args(["--train_json", "dummy.jsonl"])
    assert args.lambda_yaw == 2.0
    assert args.aux_delta_vel is False
    assert args.balance_sampling is True


def test_checkpoint_meta_uses_step_action_manifest_and_delta_scale(tmp_path):
    dataset, manifest = write_bound_dataset(tmp_path)
    args = checkpoint_args(dataset)
    meta = train_pfem.build_checkpoint_meta(args)
    assert meta["label_mode"] == "step_action"
    assert meta["delta_scale"] == 1.0
    assert meta["aux_delta_vel"] is True
    assert meta["data_manifest_hash"]
    assert meta["data_jsonl_sha256"] == manifest["data_jsonl_sha256"]
    assert meta["sample_count"] == manifest["sample_count"] == 1


def test_checkpoint_meta_rejects_cli_label_conflict(tmp_path):
    dataset, _manifest = write_bound_dataset(
        tmp_path, manifest_overrides={"action_semantics": "spin_v1"}
    )
    args = checkpoint_args(dataset, label_mode="absolute", aux_delta_vel=False)
    with pytest.raises(ValueError, match="conflicts"):
        train_pfem.build_checkpoint_meta(args)


def test_training_jsonl_sha256_mismatch_is_rejected(tmp_path):
    dataset, _manifest = write_bound_dataset(tmp_path)
    with dataset.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "step_actions": [[0.0, 0.0, 0.0]],
                    "prev_action": [0.0, 0.0, 0.0],
                    "delta_vel": [[0.0, 0.0, 0.0]],
                }
            )
            + "\n"
        )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        train_pfem.build_checkpoint_meta(checkpoint_args(dataset))


def test_training_jsonl_sample_count_mismatch_is_rejected(tmp_path):
    dataset, _manifest = write_bound_dataset(
        tmp_path, manifest_overrides={"sample_count": 2}
    )
    with pytest.raises(ValueError, match="sample_count mismatch"):
        train_pfem.build_checkpoint_meta(checkpoint_args(dataset))


@pytest.mark.parametrize("missing_field", train_pfem.STEP_ACTION_REQUIRED_FIELDS)
def test_step_action_training_rejects_missing_required_fields(tmp_path, missing_field):
    sample = {
        "step_actions": [[1.0, 0.0, 0.0]],
        "prev_action": [0.0, 0.0, 0.0],
        "delta_vel": [[1.0, 0.0, 0.0]],
    }
    del sample[missing_field]
    dataset, _manifest = write_bound_dataset(tmp_path, samples=[sample])
    with pytest.raises(ValueError, match=missing_field):
        train_pfem.build_checkpoint_meta(checkpoint_args(dataset))


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
