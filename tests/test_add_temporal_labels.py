import hashlib
import json
from pathlib import Path

import pytest

from data_preprocess.add_temporal_labels import (
    TemporalLabelError,
    add_future_labels,
    rewrite_dataset,
)


def test_future_labels_stay_in_chunk_and_mirror_theta():
    rows = [
        {
            "chunk_id": "c0",
            "sequence_id": "c0",
            "frame_idx": 10,
            "mirrored": False,
            "polar_theta_idx": 20,
            "polar_dist_idx": 4,
            "polar_invalid": 0.0,
        },
        {
            "chunk_id": "c0",
            "sequence_id": "c0",
            "frame_idx": 14,
            "mirrored": False,
            "polar_theta_idx": 21,
            "polar_dist_idx": 5,
            "polar_invalid": 0.0,
        },
        {
            "chunk_id": "c0",
            "sequence_id": "c0__mirror",
            "frame_idx": 10,
            "mirrored": True,
            "polar_theta_idx": 39,
            "polar_dist_idx": 4,
            "polar_invalid": 0.0,
        },
    ]
    add_future_labels(rows, horizons=(4, 8))
    assert rows[0]["fut_valid_4"] is True
    assert rows[0]["fut_theta_idx_4"] == 21
    assert rows[0]["fut_dist_idx_4"] == 5
    assert rows[2]["fut_theta_idx_4"] == 38
    assert rows[0]["fut_valid_8"] is False


def test_future_labels_respect_clean_sequence_and_world_model_mask():
    rows = [
        {
            "chunk_id": "shared",
            "sequence_id": "seq0",
            "frame_idx": 0,
            "mirrored": False,
            "polar_theta_idx": 1,
            "polar_dist_idx": 1,
            "polar_invalid": 0.0,
            "world_model_valid": True,
        },
        {
            "chunk_id": "shared",
            "sequence_id": "seq1",
            "frame_idx": 4,
            "mirrored": False,
            "polar_theta_idx": 2,
            "polar_dist_idx": 2,
            "polar_invalid": 0.0,
            "world_model_valid": True,
        },
        {
            "chunk_id": "shared",
            "sequence_id": "seq0",
            "frame_idx": 4,
            "mirrored": False,
            "polar_theta_idx": 3,
            "polar_dist_idx": 3,
            "polar_invalid": 0.0,
            "world_model_valid": False,
        },
    ]
    add_future_labels(rows, horizons=(4,))
    assert rows[0]["fut_valid_4"] is False
    assert rows[0]["fut_theta_idx_4"] == 3


def test_rewrite_dataset_fails_on_tamper_and_updates_aggregate_manifest(tmp_path):
    dataset = tmp_path / "train.jsonl"
    row = {
        "chunk_id": "c0",
        "sequence_id": "c0",
        "frame_idx": 0,
        "mirrored": False,
        "polar_theta_idx": 1,
        "polar_dist_idx": 1,
        "polar_invalid": 0.0,
        "source_raw_dir": "test001",
    }
    dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
    old_hash = hashlib.sha256(dataset.read_bytes()).hexdigest()
    sidecar = Path(str(dataset) + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "split": "train",
                "sample_count": 1,
                "data_jsonl_sha256": old_hash,
                "statistics": {},
            }
        ),
        encoding="utf-8",
    )
    aggregate = tmp_path / "dataset_build_manifest.json"
    aggregate.write_text(
        json.dumps(
            {
                "splits": {
                    "train": {
                        "data_jsonl_sha256": old_hash,
                        "sample_count": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    result = rewrite_dataset(dataset, horizons=(4,))
    assert result["sha256"] != old_hash
    updated_sidecar = json.loads(sidecar.read_text())
    updated_aggregate = json.loads(aggregate.read_text())
    assert updated_sidecar["statistics"]["source_raw_dirs"] == ["test001"]
    assert (
        updated_aggregate["splits"]["train"]["data_jsonl_sha256"]
        == result["sha256"]
    )

    dataset.write_text(dataset.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(TemporalLabelError, match="sha256 mismatch"):
        rewrite_dataset(dataset, horizons=(4,))
