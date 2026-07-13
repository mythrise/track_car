import hashlib
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


class HaarStubDetector:
    def detect_haar(self, _frame):
        return (0.5, 0.4, 0.15, 0.1)

    def detect(self, _frame, haar_result=None):
        assert haar_result is not None
        return haar_result, "haar"


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
    assert manifest["distance_source"] == "source_aware_heuristic"
    assert samples[0]["detection_source"] == "omdet"
    assert manifest["statistics"]["detection_source_distribution"] == {"omdet": 3}

    lines = (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["episode"] == "ep001"
    sidecar = Path(str(tmp_path / "train.jsonl") + ".manifest.json")
    sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert sidecar_data["schema_version"] == 1
    assert sidecar_data["sample_count"] == len(lines) == 3
    assert sidecar_data["data_jsonl_sha256"] == hashlib.sha256(
        (tmp_path / "train.jsonl").read_bytes()
    ).hexdigest()


def test_all_samples_from_small_fixture_match_training_schema(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    write_episode(tmp_path)
    builder.build_dataset(
        args_for(tmp_path, label_mode="step_action"),
        detector_factory=lambda device: StubDetector(),
    )
    schema = json.loads(builder.TRAINING_SAMPLE_SCHEMA_PATH.read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert samples
    for sample in samples:
        builder.jsonschema.validate(instance=sample, schema=schema)


def test_absolute_path_mode_remains_absolute(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"x")
    assert builder.serialize_image_path(image, True) == str(image.resolve())


def test_actions_outside_step_action_range_are_rejected():
    assert builder.meta_to_action({"action": [1.01, 0.0, 0.0]}) is None


def test_haar_distance_uses_vertical_position_instead_of_full_body_height(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    write_episode(tmp_path)
    samples, manifest = builder.build_dataset(
        args_for(tmp_path),
        detector_factory=lambda device: HaarStubDetector(),
    )
    assert {sample["detection_source"] for sample in samples} == {"haar"}
    assert {sample["polar_dist_idx"] for sample in samples} != {builder.POLAR_DISTANCE_BINS - 1}
    assert manifest["statistics"]["detection_source_distribution"] == {"haar": 3}
    assert manifest["statistics"]["polar_by_detection_source"]["haar"][
        "max_distance_bin_rate"
    ] == 0.0


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


def test_mirrored_polar_theta_is_recomputed_from_mirrored_bbox(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    source = tmp_path / "frame.jpg"
    cv2.imwrite(str(source), np.zeros((8, 8, 3), dtype=np.uint8))
    cx = 0.63
    theta, _ = builder.bbox_to_polar(cx, 0.5, 8, 8, bbox_h=0.5)
    sample = {
        "images": [],
        "current": str(source),
        "command": "turn_left",
        "prev_action": [1.0, 0.0, 0.5],
        "step_actions": [[1.0, 0.0, 0.5]],
        "bbox": [0.53, 0.25, 0.73, 0.75],
        "polar_theta_idx": builder.discretize_theta(theta),
    }
    mirrored = builder.make_mirrored_sample(
        sample,
        [source],
        tmp_path / "mirrors",
        tmp_path,
        SimpleNamespace(absolute_paths=True),
        fps=10,
    )
    mirrored_cx = (mirrored["bbox"][0] + mirrored["bbox"][2]) / 2.0
    expected_theta, _ = builder.bbox_to_polar(mirrored_cx, 0.5, 8, 8, bbox_h=0.5)
    assert mirrored["polar_theta_idx"] == builder.discretize_theta(expected_theta)


def test_step_builder_does_not_mirror_validation_split(tmp_path, monkeypatch):
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
    assert mirrors == []
    val_manifest = json.loads(
        Path(str(tmp_path / "heldout.jsonl") + ".manifest.json").read_text(encoding="utf-8")
    )
    assert val_manifest["statistics"]["mirrored_count"] == 0
    assert val_manifest["statistics"]["sample_count"] == 3
    assert val_manifest["mirror_augment"] is False
    assert val_manifest["split"] == "val"
    train_manifest = json.loads(
        Path(str(tmp_path / "train.jsonl") + ".manifest.json").read_text(encoding="utf-8")
    )
    assert train_manifest["validation"]["sample_count"] == 3
    assert train_manifest["validation"]["mirror_augment"] is False
