from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from data_preprocess import prepare_collected_dataset as prepare


PURPLE_RAW = "follow the person in purple shirt"
PURPLE_NORMALIZED = "Follow the person in a purple shirt."
BLACK_RAW = "follow the person in black shirt."


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_episode(
    source_root: Path,
    name: str,
    instruction: str,
    frame_count: int,
    *,
    valid_episode_json: bool = True,
) -> Path:
    episode = source_root / name
    episode.mkdir(parents=True)
    summary = {
        "episode": name,
        "instruction": instruction,
        "n_frames": frame_count,
        "width": 8,
        "height": 6,
        "fps": 10,
        "turn_forward_ratio": 0.5,
        "turn_yaw_ratio": 0.5,
    }
    if valid_episode_json:
        (episode / "episode.json").write_text(json.dumps(summary), encoding="utf-8")
    else:
        (episode / "episode.json").write_bytes(b"")
    for index in range(frame_count):
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[0, 0] = [20 + index, 80 + index, 140 + index]
        image[-1, -1] = [200 - index, 100 - index, 30 + index]
        assert cv2.imwrite(str(episode / f"frame_{index:06d}.jpg"), image)
        meta = {
            "frame": f"frame_{index:06d}.jpg",
            "timestamp": 1000.0 + index * 0.1,
            "frame_idx": index,
            "instruction": instruction,
            "episode": name,
            "command": "forward",
            "action": [1.0, 0.0, 0.0],
        }
        (episode / f"meta_{index:06d}.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )
    return episode


def audit_episode(
    raw_dir: str,
    canonical_id: str,
    instruction: str,
    status: str,
    split: str,
    start: int,
    end: int,
    *,
    history: int = 2,
    future: int = 2,
) -> dict:
    frames = end - start + 1
    return {
        "raw_dir": raw_dir,
        "canonical_id": canonical_id,
        "status": status,
        "instruction_raw": instruction,
        "instruction_normalized": instruction,
        "target_group": "purple_shirt" if "purple" in instruction else "black_shirt",
        "declared_frames": frames,
        "image_count": frames,
        "readable_images": frames,
        "meta_count": frames,
        "valid_meta": frames,
        "chunks": [
            {
                "start": start,
                "end": end,
                "frames": frames,
                "raw_samples": max(0, frames - (history + future - 1)),
            }
        ],
        "motion_stall_pair_ranges": [],
        "split": split,
        "issues": ["all accepted images are upside down"],
    }


def write_audit(source_root: Path, path: Path, episodes: list[dict]) -> Path:
    audit = {
        "schema_version": 1,
        "audit_date": "2026-07-15",
        "source_root": str(source_root.resolve()),
        "source_immutable": True,
        "model_window": {
            "history_frames": 2,
            "future_frames": 2,
            "minimum_chunk_frames": 4,
        },
        "hard_cleaning_rules": {},
        "clip_policy": {
            "anchor_block_size": 1,
            "maximum_clip_frames": 4,
            "context_overlap_frames": 3,
        },
        "episodes": episodes,
    }
    path.write_text(json.dumps(audit), encoding="utf-8")
    return path


def write_image_audit(source_root: Path, path: Path, episode_names: list[str]) -> Path:
    value = {
        "schema_version": "image_quality_audit.v1",
        "generated_at": "2026-07-15T00:00:00+08:00",
        "source_root": str(source_root.resolve()),
        "raw_data_modified": False,
        "thresholds": {"blur_hard": "review before any removal"},
        "episodes": {
            name: {
                "blur": {
                    "hard_lt_50_indices": [],
                    "soft_50_to_100_indices": [],
                    "relative_lt_0_25_episode_median_indices": [],
                },
                "brightness": {
                    "soft_dark_indices": [],
                    "soft_overexposed_indices": [],
                    "low_contrast_indices": [],
                },
            }
            for name in episode_names
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def args(source: Path, audit: Path, output: Path, **overrides) -> argparse.Namespace:
    image_audit = write_image_audit(
        source,
        output.parent / f"{output.name}_image_audit.json",
        [path.name for path in source.iterdir() if path.is_dir()],
    )
    values = {
        "source_root": str(source),
        "audit": str(audit),
        "image_quality_audit": str(image_audit),
        "output": str(output),
        "audit_only": False,
        "include_auxiliary": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_materializes_audited_chunk_with_rotation_reindex_and_provenance(tmp_path):
    source = tmp_path / "collected"
    episode = make_episode(source, "test001", PURPLE_RAW, 7)
    audit_path = write_audit(
        source,
        tmp_path / "audit.json",
        [audit_episode("test001", "collected_test001", PURPLE_RAW, "primary", "train", 1, 5)],
    )
    output = tmp_path / "cleaned"
    raw_hash_before = file_hash(episode / "frame_000001.jpg")
    prepare_args = args(source, audit_path, output)
    image_audit_path = Path(prepare_args.image_quality_audit)
    image_audit = json.loads(image_audit_path.read_text())
    image_audit["episodes"]["test001"]["blur"]["hard_lt_50_indices"] = [1]
    image_audit_path.write_text(json.dumps(image_audit), encoding="utf-8")

    manifest = prepare.prepare_dataset(prepare_args)

    chunk_id = "collected_test001__chunk000"
    derived = output / "episodes" / "train" / chunk_id
    assert sorted(path.name for path in derived.glob("frame_*.jpg")) == [
        f"frame_{index:06d}.jpg" for index in range(5)
    ]
    source_image = cv2.imread(str(episode / "frame_000001.jpg"))
    derived_image = cv2.imread(str(derived / "frame_000000.jpg"))
    np.testing.assert_allclose(derived_image, cv2.flip(source_image, -1), atol=15)
    assert file_hash(episode / "frame_000001.jpg") == raw_hash_before

    derived_meta = json.loads((derived / "meta_000000.json").read_text(encoding="utf-8"))
    assert derived_meta["frame_idx"] == 0
    assert derived_meta["frame"] == "frame_000000.jpg"
    assert derived_meta["episode"] == chunk_id
    assert derived_meta["instruction_raw"] == PURPLE_RAW
    assert derived_meta["instruction"] == PURPLE_NORMALIZED
    assert derived_meta["source_provenance"]["source_frame_idx"] == 1
    assert derived_meta["source_provenance"]["source_frame_sha256"] == raw_hash_before
    assert derived_meta["sequence_id"] == chunk_id
    assert "reviewed_laplacian_lt_50_keep" in derived_meta["cleaning"]["soft_flags"]

    derived_episode = json.loads((derived / "episode.json").read_text(encoding="utf-8"))
    assert derived_episode["n_frames"] == 5
    assert derived_episode["instruction_raw"] == PURPLE_RAW
    assert derived_episode["instruction"] == PURPLE_NORMALIZED
    assert manifest["instruction_normalization"]["mapping"][PURPLE_RAW] == PURPLE_NORMALIZED
    assert manifest["counts"]["selected_frame_count"] == 5
    assert manifest["counts"]["selected_anchor_count"] == 2
    assert (output / "audits" / "collected_audit.json").is_file()
    assert (output / "audits" / "image_quality_audit.json").is_file()

    clips = json.loads((output / "clips.json").read_text(encoding="utf-8"))["clips"]
    selected_clips = [row for row in clips if row["selected"]]
    assert all(row["sequence_id"] == chunk_id for row in selected_clips)
    assert selected_clips[1]["tim_reset_at_clip_start"] is False
    assert [row["source_anchor_start"] for row in selected_clips] == [3, 4]
    assert all(row["frame_count"] == 4 for row in selected_clips)
    frame_rows = [json.loads(line) for line in (output / "frame_audit.jsonl").read_text().splitlines()]
    assert next(row for row in frame_rows if row["source_frame_idx"] == 0)["decision"] == (
        "outside_audited_clean_chunks"
    )
    accepted = next(row for row in frame_rows if row["source_frame_idx"] == 1)
    assert accepted["destination_frame_sha256"] is not None


def test_audit_only_and_auxiliary_selection_are_explicit(tmp_path):
    source = tmp_path / "collected"
    make_episode(source, "test001", PURPLE_RAW, 5)
    make_episode(source, "test008", BLACK_RAW, 5, valid_episode_json=False)
    audit_path = write_audit(
        source,
        tmp_path / "audit.json",
        [
            audit_episode("test001", "collected_test001", PURPLE_RAW, "primary", "train", 0, 4),
            audit_episode(
                "test008",
                "collected_test008_salvaged",
                BLACK_RAW,
                "auxiliary_only",
                "auxiliary_ablation_only",
                0,
                4,
            ),
        ],
    )

    primary_output = tmp_path / "primary"
    primary_manifest = prepare.prepare_dataset(args(source, audit_path, primary_output))
    assert primary_manifest["counts"]["selected_episode_count"] == 1
    assert not (primary_output / "episodes" / "auxiliary_ablation_only").exists()
    episodes = json.loads((primary_output / "episodes.json").read_text())["episodes"]
    auxiliary = next(row for row in episodes if row["raw_dir"] == "test008")
    assert auxiliary["selected"] is False
    assert auxiliary["selection_reason"] == "auxiliary_not_requested"

    audit_only_output = tmp_path / "audit_only"
    audit_only_manifest = prepare.prepare_dataset(
        args(
            source,
            audit_path,
            audit_only_output,
            audit_only=True,
            include_auxiliary=True,
        )
    )
    assert audit_only_manifest["counts"]["selected_episode_count"] == 2
    assert audit_only_manifest["derived_payload_sha256"] is None
    assert not (audit_only_output / "episodes").exists()
    assert (audit_only_output / "frame_audit.jsonl").is_file()


def test_matching_output_is_reused_but_changed_source_is_rejected(tmp_path):
    source = tmp_path / "collected"
    episode = make_episode(source, "test001", PURPLE_RAW, 5)
    audit_path = write_audit(
        source,
        tmp_path / "audit.json",
        [audit_episode("test001", "collected_test001", PURPLE_RAW, "primary", "train", 0, 4)],
    )
    output = tmp_path / "cleaned"
    first = prepare.prepare_dataset(args(source, audit_path, output))
    second = prepare.prepare_dataset(args(source, audit_path, output))
    assert second["binding"] == first["binding"]

    payload = output / "episodes" / "train" / "collected_test001__chunk000" / "frame_000000.jpg"
    original_payload = payload.read_bytes()
    payload.write_bytes(original_payload + b"tamper")
    with pytest.raises(prepare.DatasetPreparationError, match="payload failed hash verification"):
        prepare.prepare_dataset(args(source, audit_path, output))
    payload.write_bytes(original_payload)

    (episode / "meta_000004.json").write_text("{}", encoding="utf-8")
    with pytest.raises(prepare.DatasetPreparationError, match="different inputs/options"):
        prepare.prepare_dataset(args(source, audit_path, output))


def test_invalid_audit_accepted_frame_fails_atomically(tmp_path):
    source = tmp_path / "collected"
    episode = make_episode(source, "test001", PURPLE_RAW, 5)
    bad_meta = json.loads((episode / "meta_000002.json").read_text())
    del bad_meta["action"]
    (episode / "meta_000002.json").write_text(json.dumps(bad_meta), encoding="utf-8")
    audit_path = write_audit(
        source,
        tmp_path / "audit.json",
        [audit_episode("test001", "collected_test001", PURPLE_RAW, "primary", "train", 0, 4)],
    )
    output = tmp_path / "cleaned"
    with pytest.raises(prepare.DatasetPreparationError, match="audit accepted an invalid frame"):
        prepare.prepare_dataset(args(source, audit_path, output))
    assert not output.exists()


def test_all_reviewed_prompt_typos_have_exact_normalization():
    expected = {
        "follow the person in purple shirt": "Follow the person in a purple shirt.",
        "follow the person in black pants.": "Follow the person wearing black pants.",
        "follow the person in pblack pants.": "Follow the person wearing black pants.",
        "follow the person in black shirt.": "Follow the person in a black shirt.",
        "follow the person with a black bag.": "Follow the person carrying a black bag.",
        "follow the person in purple shirt with a black bag.": (
            "Follow the person in a purple shirt who is carrying a black bag."
        ),
        "follow the person in green and white3 shirt.": (
            "Follow the person in a green-and-white shirt."
        ),
    }
    assert prepare.INSTRUCTION_NORMALIZATION == expected


def test_script_help_bootstraps_without_training_builder():
    script = Path(__file__).resolve().parents[1] / "data_preprocess" / "prepare_collected_dataset.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "does not invoke OmDet" in result.stdout
