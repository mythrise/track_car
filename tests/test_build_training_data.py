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
