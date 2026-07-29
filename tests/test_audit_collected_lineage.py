from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from data_preprocess import audit_collected_lineage as lineage


RAW = "follow the person in purple shirt"
NORMALIZED = "Follow the person in a purple shirt."


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_groups() -> dict[str, dict]:
    return {
        raw: {
            "raw_instruction": raw,
            "authoritative_normalized": normalized,
            "deprecated_audit_values": set(),
            "raw_dirs": set(),
            "canonical_ids": set(),
            "splits": set(),
            "statuses": set(),
            "selected_frame_count": 0,
            "train_val_sample_count": 0,
            "test_clean_chunk_sample_count": 0,
            "change_categories": list(
                lineage.NORMALIZATION_CHANGE_CATEGORIES[raw]
            ),
        }
        for raw, normalized in lineage.INSTRUCTION_NORMALIZATION.items()
    }


def test_mapping_sha_and_change_categories_cover_authoritative_mapping():
    assert lineage.instruction_mapping_sha256() == (
        "ccfa4c6e2e1dad9d83733cd05f102ee5a26cfbadeb55ee82aa4d5d577d512e73"
    )
    assert set(lineage.NORMALIZATION_CHANGE_CATEGORIES) == set(
        lineage.INSTRUCTION_NORMALIZATION
    )


def test_prompt_audit_deprecates_old_value_without_changing_authority(tmp_path):
    source = tmp_path / "collected"
    clean = tmp_path / "clean"
    source_meta = source / "test001" / "meta_000000.json"
    write_json(
        source / "test001" / "episode.json",
        {"instruction": RAW},
    )
    write_json(source_meta, {"instruction": RAW})

    frame_sha = "a" * 64
    meta_sha = file_sha(source_meta)
    write_json(
        clean / "episodes" / "train" / "chunk_train" / "meta_000000.json",
        {
            "instruction": NORMALIZED,
            "instruction_normalized": NORMALIZED,
            "instruction_raw": RAW,
            "source_provenance": {
                "raw_dir": "test001",
                "source_frame_idx": 0,
                "source_frame_sha256": frame_sha,
                "source_meta_sha256": meta_sha,
                "split": "train",
            },
        },
    )
    audit = {
        "episodes": [
            {
                "raw_dir": "test001",
                "canonical_id": "collected_test001",
                "status": "primary",
                "split": "train",
                "instruction_raw": RAW,
                "instruction_normalized": "follow the person in a purple shirt",
            }
        ]
    }
    episodes = [
        {
            "raw_dir": "test001",
            "canonical_id": "collected_test001",
            "selected": True,
            "split": "train",
            "instruction_raw": RAW,
            "instruction_normalized": NORMALIZED,
        }
    ]
    frame_rows = [
        {
            "selected": True,
            "decision": "accepted_materialized",
            "raw_dir": "test001",
            "source_frame_idx": 0,
            "source_frame_sha256": frame_sha,
            "source_meta_sha256": meta_sha,
            "source_meta": "test001/meta_000000.json",
            "instruction_raw": RAW,
            "instruction_normalized": NORMALIZED,
            "split": "train",
        }
    ]

    report, groups, selected = lineage._validate_prompts(
        source,
        clean,
        audit,
        episodes,
        [],
        frame_rows,
    )

    assert report["deprecated_audit_mismatch_count"] == 1
    assert report["derived_prompt_error_count"] == 0
    assert groups[RAW]["deprecated_audit_values"] == {
        "follow the person in a purple shirt"
    }
    assert set(selected) == {("test001", 0)}


def test_prompt_audit_rejects_episode_missing_from_legacy_audit(tmp_path):
    audit = {
        "episodes": [
            {
                "raw_dir": "test001",
                "instruction_raw": RAW,
                "instruction_normalized": "stale",
            }
        ]
    }
    episodes = [
        {"raw_dir": "test001", "instruction_raw": RAW},
        {"raw_dir": "test002", "instruction_raw": RAW},
    ]
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="collected audit and episodes index raw-dir coverage differ",
    ):
        lineage._validate_prompts(
            tmp_path / "source",
            tmp_path / "clean",
            audit,
            episodes,
            [],
            [],
        )


def valid_split_fixture():
    episodes = [
        {
            "raw_dir": "train_raw",
            "canonical_id": "train_id",
            "split": "train",
            "selected": True,
        },
        {
            "raw_dir": "val_raw",
            "canonical_id": "val_id",
            "split": "val",
            "selected": True,
        },
    ]
    chunks = [
        {
            "chunk_id": "train_chunk",
            "raw_dir": "train_raw",
            "split": "train",
            "selected": True,
            "source_start": 0,
            "source_end": 9,
        },
        {
            "chunk_id": "val_chunk",
            "raw_dir": "val_raw",
            "split": "val",
            "selected": True,
            "source_start": 0,
            "source_end": 9,
        },
    ]
    clips = [
        {
            "clip_id": "train_clip",
            "chunk_id": "train_chunk",
            "selected": True,
            "source_anchor_start": 2,
            "source_anchor_end": 3,
            "anchor_count": 2,
            "source_frame_start": 0,
            "source_frame_end": 5,
        },
        {
            "clip_id": "val_clip",
            "chunk_id": "val_chunk",
            "selected": True,
            "source_anchor_start": 2,
            "source_anchor_end": 3,
            "anchor_count": 2,
            "source_frame_start": 0,
            "source_frame_end": 5,
        },
    ]
    frames = [
        {
            "raw_dir": "train_raw",
            "source_frame_idx": 2,
            "source_frame_sha256": "a" * 64,
            "source_meta_sha256": "b" * 64,
            "split": "train",
        },
        {
            "raw_dir": "val_raw",
            "source_frame_idx": 2,
            "source_frame_sha256": "c" * 64,
            "source_meta_sha256": "d" * 64,
            "split": "val",
        },
    ]
    return episodes, chunks, clips, frames


def test_split_audit_passes_and_rejects_cross_split_source_hash():
    episodes, chunks, clips, frames = valid_split_fixture()
    assert lineage._split_leakage_audit(episodes, chunks, clips, frames)[
        "status"
    ] == "PASS"

    frames[1]["source_frame_sha256"] = frames[0]["source_frame_sha256"]
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="split leakage or range overlap detected",
    ):
        lineage._split_leakage_audit(episodes, chunks, clips, frames)


def dataset_row(split: str, raw_dir: str, chunk_id: str, frame_sha: str, meta_sha: str):
    return {
        "instruction": NORMALIZED,
        "instruction_normalized": NORMALIZED,
        "instruction_raw": RAW,
        "split": split,
        "source_raw_dir": raw_dir,
        "source_frame_idx": 0,
        "source_frame_sha256": frame_sha,
        "source_meta_sha256": meta_sha,
        "chunk_id": chunk_id,
        "frame_idx": 0,
        "mirrored": False,
    }


def sidecar(path: Path, split: str, sample_count: int, *, episode: str | None = None):
    value = {
        "split": split,
        "sample_count": sample_count,
        "data_jsonl_sha256": file_sha(path),
        "prompt_policy": {
            "mapping": lineage.INSTRUCTION_NORMALIZATION,
            "sha256": lineage.instruction_mapping_sha256(),
        },
    }
    if split == "test":
        value["statistics"] = {
            "episode_reports": [{"episode": episode, "samples": sample_count}]
        }
    write_json(path.with_name(path.name + ".manifest.json"), value)


def test_dataset_audit_never_json_decodes_frozen_test(tmp_path):
    clean = tmp_path / "clean"
    datasets = clean / "datasets"
    train_path = datasets / "train.jsonl"
    val_path = datasets / "val.jsonl"
    test_path = datasets / "test.jsonl"

    train_sha, train_meta_sha = "1" * 64, "2" * 64
    val_sha, val_meta_sha = "3" * 64, "4" * 64
    test_sha, test_meta_sha = "5" * 64, "6" * 64
    write_jsonl(
        train_path,
        [dataset_row("train", "train_raw", "train_chunk", train_sha, train_meta_sha)],
    )
    write_jsonl(
        val_path,
        [dataset_row("val", "val_raw", "val_chunk", val_sha, val_meta_sha)],
    )
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_bytes(b"this-is-intentionally-not-json\n")
    sidecar(train_path, "train", 1)
    sidecar(val_path, "val", 1)
    test_manifest_path = test_path.with_name(test_path.name + ".manifest.json")
    test_manifest_path.write_bytes(b"this-manifest-is-also-intentionally-not-json\n")
    test_expectations = {
        "data_jsonl_sha256": file_sha(test_path),
        "manifest_sha256": file_sha(test_manifest_path),
        "sample_count": 1,
    }

    selected_frames = {
        ("train_raw", 0): {
            "instruction_raw": RAW,
            "source_frame_sha256": train_sha,
            "source_meta_sha256": train_meta_sha,
        },
        ("val_raw", 0): {
            "instruction_raw": RAW,
            "source_frame_sha256": val_sha,
            "source_meta_sha256": val_meta_sha,
        },
        ("test_raw", 0): {
            "instruction_raw": RAW,
            "source_frame_sha256": test_sha,
            "source_meta_sha256": test_meta_sha,
        },
    }
    chunks = [
        {
            "selected": True,
            "chunk_id": "train_chunk",
            "raw_dir": "train_raw",
            "split": "train",
        },
        {
            "selected": True,
            "chunk_id": "val_chunk",
            "raw_dir": "val_raw",
            "split": "val",
        },
        {
            "selected": True,
            "chunk_id": "test_chunk",
            "raw_dir": "test_raw",
            "split": "test",
            "raw_sample_count": 1,
        },
    ]
    groups = prompt_groups()

    report = lineage._validate_datasets(
        clean,
        groups,
        selected_frames,
        chunks,
        test_expectations,
    )

    assert report["frozen_test_jsonl_content_parsed"] is False
    assert report["frozen_test_manifest_content_parsed"] is False
    assert report["test"]["sample_count"] == 1
    assert report["test"]["content_policy"].startswith("opaque_sha256")
    assert report["test"]["manifest_content_policy"].startswith("opaque_sha256")
    assert groups[RAW]["train_val_sample_count"] == 2
    assert groups[RAW]["test_clean_chunk_sample_count"] == 1

    wrong_raw = "follow the person in black shirt."
    wrong_normalized = lineage.INSTRUCTION_NORMALIZATION[wrong_raw]
    wrong_row = dataset_row(
        "train", "train_raw", "train_chunk", train_sha, train_meta_sha
    )
    wrong_row.update(
        {
            "instruction_raw": wrong_raw,
            "instruction": wrong_normalized,
            "instruction_normalized": wrong_normalized,
        }
    )
    write_jsonl(train_path, [wrong_row])
    train_sidecar_path = train_path.with_name(train_path.name + ".manifest.json")
    train_sidecar = json.loads(train_sidecar_path.read_text(encoding="utf-8"))
    train_sidecar["data_jsonl_sha256"] = file_sha(train_path)
    write_json(train_sidecar_path, train_sidecar)
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="dataset raw prompt does not match source frame",
    ):
        lineage._validate_datasets(
            clean,
            prompt_groups(),
            selected_frames,
            chunks,
            test_expectations,
        )

    write_jsonl(
        train_path,
        [dataset_row("train", "train_raw", "train_chunk", train_sha, train_meta_sha)],
    )
    train_sidecar["data_jsonl_sha256"] = file_sha(train_path)
    write_json(train_sidecar_path, train_sidecar)

    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="frozen test manifest SHA does not match pre-registration",
    ):
        lineage._validate_datasets(
            clean,
            prompt_groups(),
            selected_frames,
            chunks,
            {**test_expectations, "manifest_sha256": "0" * 64},
        )

    chunks[-1]["split"] = "train"
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="no selected frozen-test clean chunks found",
    ):
        lineage._validate_datasets(
            clean,
            prompt_groups(),
            selected_frames,
            chunks,
            test_expectations,
        )


def test_new_json_writer_refuses_overwrite(tmp_path):
    output = tmp_path / "report.json"
    first_sha = lineage._atomic_write_new_json(output, {"status": "PASS"})
    assert first_sha == file_sha(output)
    with pytest.raises(lineage.CollectedLineageAuditError, match="refusing to overwrite"):
        lineage._atomic_write_new_json(output, {"status": "CHANGED"})


def test_new_json_writer_does_not_overwrite_toctou_racer(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    real_link = lineage.os.link

    def racing_link(source, destination):
        Path(destination).write_text("racer-owned", encoding="utf-8")
        return real_link(source, destination)

    monkeypatch.setattr(lineage.os, "link", racing_link)
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="created concurrently",
    ):
        lineage._atomic_write_new_json(output, {"status": "PASS"})
    assert output.read_text(encoding="utf-8") == "racer-owned"


def test_opaque_line_count_uses_bytes_already_read(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b"")
    assert lineage._opaque_line_count(path) == 0
    path.write_bytes(b"one")
    assert lineage._opaque_line_count(path) == 1
    path.write_bytes(b"one\n")
    assert lineage._opaque_line_count(path) == 1
    path.write_bytes(b"one\ntwo")
    assert lineage._opaque_line_count(path) == 2


def test_portable_path_is_relative_and_rejects_outside_root(tmp_path):
    inside = tmp_path / "data" / "report.json"
    assert lineage._portable_path(inside, tmp_path) == "data/report.json"
    with pytest.raises(
        lineage.CollectedLineageAuditError,
        match="provenance path must be inside",
    ):
        lineage._portable_path(tmp_path.parent / "outside-report.json", tmp_path)


def test_directory_fsync_unsupported_does_not_invalidate_committed_file(
    tmp_path, monkeypatch
):
    output = tmp_path / "report.json"
    real_fsync = lineage.os.fsync
    calls = 0

    def fsync_with_unsupported_directory(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise PermissionError("directory fsync unsupported")
        return real_fsync(descriptor)

    monkeypatch.setattr(lineage.os, "fsync", fsync_with_unsupported_directory)
    written_sha = lineage._atomic_write_new_json(output, {"status": "PASS"})
    assert written_sha == file_sha(output)
    assert calls == 2
