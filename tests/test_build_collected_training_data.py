import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_preprocess import build_collected_training_data as build
from data_preprocess.prepare_collected_dataset import fingerprint_derived_payload


def write_clean_fixture(tmp_path: Path):
    clean = tmp_path / "clean"
    episode = clean / "episodes" / "train" / "chunk0"
    episode.mkdir(parents=True)
    (episode / "episode.json").write_text(
        json.dumps(
            {
                "episode": "chunk0",
                "sequence_id": "chunk0",
                "instruction": "Follow the person in a black shirt.",
                "instruction_raw": "follow the person in black shirt.",
                "source_provenance": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )
    for index in (2, 3):
        (episode / f"meta_{index:06d}.json").write_text(
            json.dumps(
                {
                    "instruction": "Follow the person in a black shirt.",
                    "instruction_normalized": "Follow the person in a black shirt.",
                    "instruction_raw": "follow the person in black shirt.",
                    "target_group": "black_shirt",
                    "sequence_id": "chunk0",
                    "clip_id": "chunk0__clip000",
                    "source_provenance": {
                        "audit_sha256": "audit",
                        "raw_dir": "test010",
                        "canonical_id": "collected_test010",
                        "source_frame_idx": index,
                        "source_frame_sha256": f"frame{index}",
                        "source_meta_sha256": f"meta{index}",
                        "split": "train",
                    },
                    "cleaning": {
                        "soft_flags": ["motion_stall"] if index == 3 else [],
                        "loss_masks": {
                            "latent_world_dynamics": index != 3,
                            "optical_flow": index != 3,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    payload_hash, payload_count = fingerprint_derived_payload(clean)
    cleaning_manifest = {
        "derived_payload_sha256": payload_hash,
        "counts": {"materialized_payload_file_count": payload_count},
        "instruction_normalization": {"policy": "normalized"},
    }
    (clean / "cleaning_manifest.json").write_text(
        json.dumps(cleaning_manifest), encoding="utf-8"
    )
    return clean, episode, cleaning_manifest


def test_annotate_samples_preserves_prompt_and_separates_mirror_sequence(tmp_path):
    _clean, split_root, _manifest = write_clean_fixture(tmp_path)
    split_root = split_root.parent
    samples = [
        {"episode": "chunk0", "frame_idx": 3, "mirrored": True},
        {"episode": "chunk0", "frame_idx": 3, "mirrored": False},
        {"episode": "chunk0", "frame_idx": 2, "mirrored": False},
    ]
    rows = build.annotate_samples(samples, split_root)
    assert [(row["mirrored"], row["frame_idx"]) for row in rows] == [
        (False, 2),
        (False, 3),
        (True, 3),
    ]
    assert rows[0]["instruction_raw"] == "follow the person in black shirt."
    assert rows[0]["instruction"] == "Follow the person in a black shirt."
    assert rows[1]["world_model_valid"] is False
    assert rows[2]["sequence_id"] == "chunk0__mirror"
    assert rows[2]["clip_id"] == "chunk0__clip000__mirror"


def test_rewrite_dataset_rebinds_hash_and_cleaning_manifest(tmp_path):
    clean, _episode, cleaning_manifest = write_clean_fixture(tmp_path)
    output = tmp_path / "train.jsonl"
    samples = [
        {
            "episode": "chunk0",
            "frame_idx": 2,
            "current": "frame.jpg",
            "images": ["history.jpg"],
            "instruction": "Follow the person in a black shirt.",
            "chunk_id": "chunk0",
            "sequence_id": "chunk0",
            "source_raw_dir": "test010",
            "world_model_valid": True,
            "optical_flow_valid": True,
            "waypoints": [[0.0, 0.0, 0.0]],
            "actions": [[0.0, 0.0, 0.0]],
            "step_actions": [[0.0, 0.0, 0.0]],
            "delta_pos": [[0.0, 0.0, 0.0]],
            "delta_vel": [[0.0, 0.0, 0.0]],
            "prev_action": [0.0, 0.0, 0.0],
            "transition_type": "steady_forward",
            "mirrored": False,
            "action_semantics": "arc_turn_v2",
            "motors": [1500, 1500, 1500, 1500],
            "command": "stop",
            "polar_theta_idx": -1,
            "polar_dist_idx": -1,
            "polar_invalid": 1.0,
            "detection_source": "none",
        }
    ]
    manifest = {"statistics": {}, "sample_count": 1}
    rebound = build.rewrite_bound_dataset(
        output,
        samples,
        manifest,
        split="train",
        clean_root=clean,
        cleaning_manifest=cleaning_manifest,
    )
    expected = hashlib.sha256(output.read_bytes()).hexdigest()
    assert rebound["data_jsonl_sha256"] == expected
    assert (
        rebound["cleaning_payload_sha256"]
        == cleaning_manifest["derived_payload_sha256"]
    )
    assert rebound["statistics"]["raw_episode_count"] == 1
    sidecar = json.loads(Path(str(output) + ".manifest.json").read_text())
    assert sidecar["split"] == "train"
    assert sidecar["data_jsonl_sha256"] == expected


def test_build_all_uses_one_shared_detector_and_train_only_mirroring(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    for split in ("train", "val"):
        (clean / "episodes" / split).mkdir(parents=True)
    payload_hash, payload_count = fingerprint_derived_payload(clean)
    (clean / "cleaning_manifest.json").write_text(
        json.dumps(
            {
                "derived_payload_sha256": payload_hash,
                "counts": {"materialized_payload_file_count": payload_count},
            }
        ),
        encoding="utf-8",
    )
    detector = SimpleNamespace(using_omdet=True)
    loads = []
    monkeypatch.setattr(
        build,
        "get_default_target_detector",
        lambda device: loads.append(device) or detector,
    )

    calls = []

    def fake_builder(args, detector_factory):
        calls.append((Path(args.input).name, args.mirror_augment))
        assert detector_factory("cpu") is detector
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
        Path(str(output) + ".manifest.json").write_text("{}", encoding="utf-8")
        return [], {"statistics": {}, "sample_count": 0}

    args = SimpleNamespace(
        clean_root=str(clean),
        output_dir=str(tmp_path / "out"),
        detector_device="mps",
        require_omdet=True,
        splits=("train", "val"),
        history=31,
        n_waypoints=8,
        label_mode="absolute",
        mirror_train=True,
    )
    result = build.build_all(args, builder=fake_builder)
    assert calls == [("train", True), ("val", False)]
    assert loads == ["mps"]
    assert set(result["splits"]) == {"train", "val"}


def test_build_all_rejects_clean_payload_drift_before_detector_load(
    tmp_path, monkeypatch
):
    clean = tmp_path / "clean"
    (clean / "episodes" / "train").mkdir(parents=True)
    payload_hash, payload_count = fingerprint_derived_payload(clean)
    (clean / "cleaning_manifest.json").write_text(
        json.dumps(
            {
                "derived_payload_sha256": payload_hash,
                "counts": {"materialized_payload_file_count": payload_count},
            }
        ),
        encoding="utf-8",
    )
    (clean / "episodes" / ".DS_Store").write_bytes(b"finder drift")
    monkeypatch.setattr(
        build,
        "get_default_target_detector",
        lambda device: (_ for _ in ()).throw(
            AssertionError("detector must not load")
        ),
    )
    args = SimpleNamespace(
        clean_root=str(clean),
        output_dir=str(tmp_path / "out"),
        detector_device="mps",
        require_omdet=True,
        splits=("train",),
        history=31,
        n_waypoints=8,
        label_mode="absolute",
        mirror_train=True,
    )
    with pytest.raises(build.CollectedBuildError, match="payload failed"):
        build.build_all(args)
