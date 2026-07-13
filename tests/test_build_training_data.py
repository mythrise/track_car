import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from data_pipeline import build_training_data as builder


class StubDetector:
    def detect_haar(self, _frame):
        return None

    def detect(self, _frame, haar_result=None):
        assert haar_result is None
        return (0.5, 0.5, 0.25, 0.5), "omdet"


def write_episode(root: Path, *, empty_meta_index=None):
    episode = root / "collected" / "ep001"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "episode": "ep001",
                "instruction": "follow the person",
                "fps": 10,
                "turn_forward_ratio": 0.5,
                "turn_yaw_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )
    for index in range(5):
        cv2.imwrite(str(episode / f"frame_{index:06d}.jpg"), np.zeros((12, 16, 3), dtype=np.uint8))
        meta_path = episode / f"meta_{index:06d}.json"
        if index == empty_meta_index:
            meta_path.write_bytes(b"")
        else:
            meta_path.write_text(
                json.dumps(
                    {
                        "frame": f"frame_{index:06d}.jpg",
                        "timestamp": index * 0.1,
                        "frame_idx": index,
                        "instruction": "follow the person",
                        "episode": "ep001",
                        "command": "forward",
                        "action": [1.0, 0.0, 0.0],
                        "motors": [1900, 1100, 1900, 1100],
                    }
                ),
                encoding="utf-8",
            )
    return episode


def args_for(root: Path, **overrides):
    values = {
        "input": str(root / "collected"),
        "output": str(root / "train.jsonl"),
        "history": 1,
        "n_waypoints": 2,
        "absolute_paths": False,
        "lenient": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_ws2_builder_cli_defaults_to_step_actions_and_turn_mirroring():
    args = builder.parse_args(["--input", "in", "--output", "out.jsonl"])
    assert args.label_mode == "step_action"
    assert args.mirror_augment is True


def test_builder_writes_repo_relative_paths_and_sidecar_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    write_episode(tmp_path)
    samples, manifest = builder.build_dataset(
        args_for(tmp_path),
        detector_factory=lambda device: StubDetector(),
    )
    assert samples
    assert not Path(samples[0]["current"]).is_absolute()
    assert samples[0]["current"].startswith("collected/ep001/")
    assert manifest["path_root"] == "."
    assert manifest["label_mode"] == "absolute"
    assert manifest["action_semantics"] == "arc_turn_v2"
    assert manifest["distance_source"] == "heuristic_bbox"

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["episode"] == "ep001"
    sidecar = Path(str(tmp_path / "train.jsonl") + ".manifest.json")
    assert json.loads(sidecar.read_text(encoding="utf-8"))["schema_version"] == 1


def test_absolute_path_mode_remains_absolute(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"x")
    assert builder.serialize_image_path(image, True) == str(image.resolve())


def test_actions_outside_step_action_range_are_rejected():
    assert builder.meta_to_action({"action": [1.01, 0.0, 0.0]}) is None


def test_empty_meta_fails_closed_before_detector_load(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    write_episode(tmp_path, empty_meta_index=2)
    detector_loaded = False

    def detector_factory(device):
        nonlocal detector_loaded
        detector_loaded = True
        return StubDetector()

    with pytest.raises(builder.DataIntegrityError, match="empty meta"):
        builder.build_dataset(args_for(tmp_path), detector_factory=detector_factory)
    assert detector_loaded is False


def test_timestamp_jitter_over_threshold_fails_closed(tmp_path):
    episode = write_episode(tmp_path)
    timestamps = [0.0, 0.1, 0.2, 0.3, 1.0]
    for index, timestamp in enumerate(timestamps):
        path = episode / f"meta_{index:06d}.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["timestamp"] = timestamp
        path.write_text(json.dumps(meta), encoding="utf-8")
    report = builder.inspect_episode(episode)
    assert any("p95/p50" in problem for problem in report["problems"])


@pytest.mark.parametrize(
    ("prev_yaw", "future_yaws", "expected"),
    [
        (0.0, [0.0, 0.0], "steady_forward"),
        (0.0, [0.0, 1.0], "turn_onset"),
        (1.0, [1.0, 1.0], "sustained_turn"),
        (1.0, [1.0, 0.0], "turn_exit"),
        (1.0, [-1.0, -1.0], "other"),
    ],
)
def test_transition_classification(prev_yaw, future_yaws, expected):
    actions = [[1.0, 0.0, yaw] for yaw in future_yaws]
    assert builder.classify_transition_type([1.0, 0.0, prev_yaw], actions) == expected


def test_step_action_labels_are_per_step_not_cumulative():
    actions = [[1.0, 0.0, 0.5], [0.25, 0.0, -0.5]]
    step_actions, delta_pos, delta_vel = builder.derive_step_labels(
        actions, [0.5, 0.0, 0.0], 0.1
    )
    np.testing.assert_allclose(step_actions, actions)
    np.testing.assert_allclose(delta_pos, [[0.1, 0.0, 0.05], [0.025, 0.0, -0.05]])
    np.testing.assert_allclose(delta_vel, [[0.5, 0.0, 0.5], [-0.75, 0.0, -1.0]])


def test_step_builder_mirrors_only_turn_samples_and_writes_val_split(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    episode = write_episode(tmp_path)
    actions = [
        [1.0, 0.0, 0.0],
        [0.8, 0.0, 1.0],
        [0.8, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ]
    for index, action in enumerate(actions):
        meta_path = episode / f"meta_{index:06d}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["action"] = action
        meta["command"] = "turn_left" if action[2] > 0 else "forward"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
    asymmetric = np.zeros((12, 16, 3), dtype=np.uint8)
    asymmetric[:, :4] = 255
    cv2.imwrite(str(episode / "frame_000001.jpg"), asymmetric)

    args = args_for(
        tmp_path,
        label_mode="step_action",
        mirror_augment=True,
        val_episodes=("ep001",),
        val_output=str(tmp_path / "heldout.jsonl"),
    )
    train_samples, manifest = builder.build_dataset(
        args,
        detector_factory=lambda device: StubDetector(),
    )
    assert train_samples == []
    assert manifest["label_mode"] == "step_action"
    val_lines = [
        json.loads(line)
        for line in (tmp_path / "heldout.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    originals = [sample for sample in val_lines if not sample["mirrored"]]
    mirrors = [sample for sample in val_lines if sample["mirrored"]]
    assert len(originals) == 3
    assert len(mirrors) == 3
    assert mirrors[0]["step_actions"][0][2] == -originals[0]["step_actions"][0][2]
    assert mirrors[0]["prev_action"][2] == -originals[0]["prev_action"][2]
    assert Path(tmp_path / mirrors[0]["current"]).is_file()
    mirrored_frame = cv2.imread(str(tmp_path / mirrors[0]["current"]))
    original_frame = cv2.imread(str(episode / "frame_000001.jpg"))
    np.testing.assert_allclose(mirrored_frame, cv2.flip(original_frame, 1), atol=2)
    val_manifest = json.loads(
        Path(str(tmp_path / "heldout.jsonl") + ".manifest.json").read_text(encoding="utf-8")
    )
    assert val_manifest["statistics"]["mirrored_count"] == 3
    assert val_manifest["statistics"]["sample_count"] == 6
    assert val_manifest["split"] == "val"
    train_manifest = json.loads(
        Path(str(tmp_path / "train.jsonl") + ".manifest.json").read_text(encoding="utf-8")
    )
    assert train_manifest["validation"]["sample_count"] == 6
