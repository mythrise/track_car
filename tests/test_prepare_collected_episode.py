from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from data_preprocess import prepare_collected_episode as preprocess


def make_episode(root: Path) -> Path:
    episode = root / "test006"
    episode.mkdir()
    (episode / "episode.json").write_text("{}", encoding="utf-8")
    (episode / "frame_000000.jpg").write_bytes(b"jpg")
    (episode / "meta_000000.json").write_text("{}", encoding="utf-8")
    return episode


def make_args(episode: Path, output: Path, **overrides):
    values = {
        "episode": str(episode),
        "output": str(output),
        "history": 31,
        "n_waypoints": 8,
        "label_mode": "step_action",
        "mirror_augment": True,
        "detector_device": "mps",
        "require_omdet": True,
        "processed_episode_dir": None,
        "rotate_180_frames": (),
        "rotate_180_all": False,
        "keep_orientation_frames": (),
        "absolute_paths": False,
        "lenient": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prepare_episode_stages_only_requested_directory(tmp_path, monkeypatch):
    episode = make_episode(tmp_path)
    output = tmp_path / "test006_train.jsonl"

    class Detector:
        using_omdet = True

    detector = Detector()
    requested_devices = []

    def fake_detector(device):
        requested_devices.append(device)
        return detector

    monkeypatch.setattr(
        preprocess,
        "get_default_target_detector",
        fake_detector,
    )

    def fake_builder(args, detector_factory):
        staging_root = Path(args.input)
        children = list(staging_root.iterdir())
        assert [child.name for child in children] == ["test006"]
        assert children[0].is_symlink()
        assert children[0].resolve() == episode.resolve()
        assert args.output == str(output)
        assert args.label_mode == "step_action"
        assert args.val_episodes == ()
        assert detector_factory("cpu") is detector
        return [{"episode": "test006"}], {"sample_count": 1}

    samples, manifest = preprocess.prepare_episode(
        make_args(episode, output),
        builder=fake_builder,
    )
    assert samples == [{"episode": "test006"}]
    assert manifest["sample_count"] == 1
    assert manifest["preprocess"]["source_episode"] == str(episode.resolve())
    assert requested_devices == ["mps"]


def test_prepare_episode_rejects_non_episode_directory(tmp_path):
    with pytest.raises(preprocess.PreprocessError, match="episode.json is missing"):
        preprocess.prepare_episode(make_args(tmp_path, tmp_path / "out.jsonl"))


def test_auto_detector_device_prefers_mps(monkeypatch):
    class Backends:
        class MPS:
            @staticmethod
            def is_available():
                return True

        mps = MPS()

    class Cuda:
        @staticmethod
        def is_available():
            return False

    fake_torch = SimpleNamespace(cuda=Cuda(), backends=Backends())
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert preprocess.resolve_detector_device("auto") == "mps"


def test_materialize_episode_rotates_only_selected_frame(tmp_path):
    episode = make_episode(tmp_path)
    original = np.zeros((4, 5, 3), dtype=np.uint8)
    original[0, 0] = [10, 20, 30]
    cv2.imwrite(str(episode / "frame_000000.jpg"), original)
    destination = tmp_path / "processed" / "test006"

    processed, manifest = preprocess.materialize_episode(
        episode,
        destination,
        (0,),
        "rotate_selected_frames",
    )
    source_image = cv2.imread(str(episode / "frame_000000.jpg"))
    processed_image = cv2.imread(str(processed / "frame_000000.jpg"))
    np.testing.assert_allclose(processed_image, cv2.flip(source_image, -1), atol=12)
    assert manifest["rotate_180_frames"] == [0]
    assert (processed / "preprocess_manifest.json").is_file()


def test_rotate_all_can_keep_already_upright_frames(tmp_path):
    episode = make_episode(tmp_path)
    (episode / "frame_000001.jpg").write_bytes(b"jpg")
    args = make_args(
        episode,
        tmp_path / "out.jsonl",
        rotate_180_all=True,
        keep_orientation_frames=(0,),
    )
    indices, policy = preprocess.resolve_rotation_frames(args, episode)
    assert indices == (1,)
    assert policy == "rotate_all_except_selected_to_upright"


def test_script_path_help_bootstraps_project_imports():
    script = Path(__file__).resolve().parents[1] / "data_preprocess" / "prepare_collected_episode.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
