import json
import hashlib
import fcntl
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from data_preprocess import cache_vision_tokens as cache
import cache_gridpool
from experiment_binding import (
    VISION_TOKEN_LAYOUT,
    ExperimentBindingError,
    bind_hf_model_artifact,
    verify_vision_cache,
)


def test_load_dataset_images_deduplicates_and_resolves_manifest_root(tmp_path):
    project = tmp_path / "project"
    data = project / "data"
    episode = data / "processed" / "test006"
    episode.mkdir(parents=True)
    image = episode / "frame_000000.jpg"
    image.write_bytes(b"jpg")
    dataset = data / "train.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "current": "data/processed/test006/frame_000000.jpg",
                "images": ["data/processed/test006/frame_000000.jpg"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(
            {
                "path_root": "..",
                "sample_count": 1,
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    base_root, images, manifest = cache.load_dataset_images(dataset)
    assert base_root == project.resolve()
    assert images == [image.resolve()]
    assert manifest["sample_count"] == 1


def test_load_dataset_images_fails_closed_on_jsonl_hash_mismatch(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"current": "frame.jpg", "images": []}) + "\n",
        encoding="utf-8",
    )
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(
            {
                "path_root": ".",
                "sample_count": 1,
                "data_jsonl_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(cache.VisionCacheError, match="sha256"):
        cache.load_dataset_images(dataset)


def test_load_dataset_images_accepts_null_as_empty_history(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"jpg")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"current": "frame.jpg", "images": None}) + "\n",
        encoding="utf-8",
    )
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(
            {
                "path_root": ".",
                "sample_count": 1,
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    _root, images, _manifest = cache.load_dataset_images(dataset)
    assert images == [image.resolve()]


def test_load_datasets_images_builds_union_and_split_bindings(tmp_path):
    image_a = tmp_path / "a.jpg"
    image_b = tmp_path / "b.jpg"
    image_a.write_bytes(b"a")
    image_b.write_bytes(b"b")
    datasets = []
    for split, image in (("train", image_a), ("val", image_b)):
        dataset = tmp_path / f"{split}.jsonl"
        dataset.write_text(
            json.dumps({"current": image.name, "images": [image_a.name]}) + "\n",
            encoding="utf-8",
        )
        Path(str(dataset) + ".manifest.json").write_text(
            json.dumps(
                {
                    "path_root": ".",
                    "split": split,
                    "sample_count": 1,
                    "data_jsonl_sha256": hashlib.sha256(
                        dataset.read_bytes()
                    ).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        datasets.append(dataset)

    base_root, images, bindings = cache.load_datasets_images(datasets)
    assert base_root == tmp_path.resolve()
    assert images == [image_a.resolve(), image_b.resolve()]
    assert [binding["split"] for binding in bindings] == ["train", "val"]
    assert all(binding["dataset_manifest_sha256"] for binding in bindings)


def test_hash_referenced_tokens_binds_paths_and_payloads(tmp_path):
    base_root = tmp_path / "project"
    image = base_root / "data" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    cache_root = tmp_path / "cache"
    fine, coarse = cache.token_paths(image, base_root, cache_root)
    fine.parent.mkdir(parents=True)
    fine.write_bytes(b"fine")
    coarse.write_bytes(b"coarse")

    digest, count, size = cache.hash_referenced_tokens(
        [image], base_root, cache_root
    )
    assert len(digest) == 64
    assert count == 2
    assert size == len(b"finecoarse")


def test_token_paths_match_tracking_dataset_layout(tmp_path):
    base_root = tmp_path / "project"
    image = base_root / "data" / "processed" / "test006" / "frame_000123.jpg"
    fine, coarse = cache.token_paths(image, base_root, tmp_path / "cache")
    assert fine == tmp_path / "cache/data/processed/test006/frame_000123.jpg_vfine.pt"
    assert coarse == tmp_path / "cache/data/processed/test006/frame_000123.jpg_vcoarse.pt"


def test_token_paths_do_not_collide_across_image_extensions(tmp_path):
    base_root = tmp_path / "project"
    cache_root = tmp_path / "cache"
    jpg = base_root / "frames" / "same.jpg"
    png = base_root / "frames" / "same.png"
    assert cache.token_paths(jpg, base_root, cache_root) != cache.token_paths(
        png, base_root, cache_root
    )


def test_atomic_torch_save_preserves_previous_file_on_failure(tmp_path, monkeypatch):
    destination = tmp_path / "token.pt"
    destination.write_bytes(b"known-good")

    def fail_after_partial_write(_value, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(cache_gridpool.torch, "save", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="interruption"):
        cache_gridpool.atomic_torch_save(object(), destination)
    assert destination.read_bytes() == b"known-good"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_torch_save_compacts_tensor_view_storage(tmp_path):
    destination = tmp_path / "token.pt"
    batched = torch.arange(4 * 64 * 16, dtype=torch.float16).reshape(4, 64, 16)
    view = batched[2]

    assert view.is_contiguous()
    assert view.storage_offset() > 0
    assert view.untyped_storage().nbytes() == batched.untyped_storage().nbytes()

    cache_gridpool.atomic_torch_save(view, destination)
    restored = torch.load(destination, map_location="cpu", weights_only=True)

    assert torch.equal(restored, view)
    assert restored.storage_offset() == 0
    assert restored.untyped_storage().nbytes() == (
        restored.numel() * restored.element_size()
    )


def test_resolve_device_prefers_mps(monkeypatch):
    monkeypatch.setattr(cache.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cache.torch.backends.mps, "is_available", lambda: True)
    assert cache.resolve_device("auto") == "mps"


def test_verify_vision_cache_binds_dataset_and_token_payload(tmp_path):
    project = tmp_path / "project"
    image = project / "frames" / "frame.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    dataset = project / "train.jsonl"
    dataset.write_text(
        json.dumps({"current": "frames/frame.jpg", "images": []}) + "\n",
        encoding="utf-8",
    )
    sidecar = Path(str(dataset) + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "path_root": ".",
                "sample_count": 1,
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    cache_root = tmp_path / "cache"
    fine, coarse = cache.token_paths(image, project, cache_root)
    fine.parent.mkdir(parents=True)
    fine.write_bytes(b"fine")
    coarse.write_bytes(b"coarse")
    payload_hash, file_count, byte_count = cache.hash_referenced_tokens(
        [image], project, cache_root
    )
    dataset_bindings = [
        {
            "dataset_json": str(dataset.resolve()),
            "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            "dataset_manifest_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
            "dataset_sample_count": 1,
        }
    ]
    provenance = {
        "schema_version": 1,
        "path_root": str(project),
        "dataset_bindings": dataset_bindings,
        "union_unique_image_count": 1,
        "producer": {
            "device": "cpu",
            "batch_size": 1,
            "requested_compute_dtype": "float32",
            "compute_dtype": "float32",
            "storage_dtype": "float16",
            "image_size": 384,
            "force_square_resize": True,
        },
        "token_layout": VISION_TOKEN_LAYOUT,
        "encoders": {
            "dinov3": {"path": "/models/dino", "artifact_sha256": "dino"},
            "siglip": {"path": "/models/siglip", "artifact_sha256": "siglip"},
        },
    }
    cache_manifest = cache.make_cache_manifest(
        provenance,
        payload_hash=payload_hash,
        token_file_count=file_count,
        token_byte_count=byte_count,
        nonfinite_retry_count=0,
    )
    (cache_root / "cache_manifest.json").write_text(
        json.dumps(cache_manifest), encoding="utf-8"
    )

    binding = verify_vision_cache(cache_root, [dataset], verify_payload=True)
    assert binding["token_payload_sha256"] == payload_hash
    stale = cache_root / "stale_vfine.pt"
    stale.write_bytes(b"stale")
    with pytest.raises(ExperimentBindingError, match="unreferenced stale token"):
        verify_vision_cache(cache_root, [dataset], verify_payload=True)
    stale.unlink()
    fine.write_bytes(b"tampered")
    with pytest.raises(ExperimentBindingError, match="payload hash mismatch"):
        verify_vision_cache(cache_root, [dataset], verify_payload=True)


def _write_cache_dataset(tmp_path):
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"image")
    dataset = tmp_path / "train.jsonl"
    dataset.write_text(
        json.dumps({"current": image.name, "images": []}) + "\n",
        encoding="utf-8",
    )
    Path(str(dataset) + ".manifest.json").write_text(
        json.dumps(
            {
                "path_root": ".",
                "split": "train",
                "sample_count": 1,
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return dataset, image


def _fake_cache_config(**kwargs):
    return SimpleNamespace(
        image_size=kwargs["image_size"],
        batch_size=kwargs["batch_size"],
        device=kwargs["device"],
        compute_dtype=kwargs["compute_dtype"],
        force_square_resize=True,
        dino_model_name="/models/dino",
        siglip_model_name="/models/siglip",
    )


def _cache_args(dataset, cache_root, **overrides):
    values = {
        "train_json": [str(dataset)],
        "cache_root": str(cache_root),
        "device": "cpu",
        "batch_size": 4,
        "compute_dtype": "float32",
        "log_every": 100,
        "force": False,
        "adopt_existing_partial": False,
        "upgrade_legacy_manifest": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_unbound_partial_cache_requires_explicit_adoption(tmp_path, monkeypatch):
    dataset, image = _write_cache_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    fine, coarse = cache.token_paths(image, tmp_path, cache_root)
    cache_gridpool.atomic_torch_save(torch.zeros(64, 1536).half(), fine)
    cache_gridpool.atomic_torch_save(torch.zeros(4, 1536).half(), coarse)
    original_fine = fine.read_bytes()
    original_coarse = coarse.read_bytes()
    monkeypatch.setattr(cache, "VisionCacheConfig", _fake_cache_config)
    monkeypatch.setattr(cache, "sha256_artifact", lambda path: f"hash:{path}")

    with pytest.raises(cache.VisionCacheError, match="unbound partial"):
        cache.cache_dataset(_cache_args(dataset, cache_root))

    manifest = cache.cache_dataset(
        _cache_args(dataset, cache_root, adopt_existing_partial=True)
    )
    assert manifest["schema_version"] == 3
    assert (cache_root / ".partial_cache_provenance.json").is_file()
    assert fine.read_bytes() == original_fine
    assert coarse.read_bytes() == original_coarse

    with pytest.raises(cache.VisionCacheError, match="disagrees"):
        cache.cache_dataset(
            _cache_args(dataset, cache_root, batch_size=2)
        )


def test_cache_writer_lock_rejects_concurrent_mutation(tmp_path):
    dataset, _image = _write_cache_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    with (cache_root / ".cache_writer.lock").open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(cache.VisionCacheError, match="writer is active"):
            cache.cache_dataset(_cache_args(dataset, cache_root))


def test_schema2_manifest_upgrade_verifies_and_preserves_tokens(tmp_path, monkeypatch):
    dataset, image = _write_cache_dataset(tmp_path)
    cache_root = tmp_path / "cache"
    fine, coarse = cache.token_paths(image, tmp_path, cache_root)
    cache_gridpool.atomic_torch_save(torch.ones(64, 1536).half(), fine)
    cache_gridpool.atomic_torch_save(torch.ones(4, 1536).half(), coarse)
    dataset_sidecar = Path(str(dataset) + ".manifest.json")
    payload_hash, file_count, byte_count = cache.hash_referenced_tokens(
        [image], tmp_path, cache_root
    )
    legacy = {
        "schema_version": 2,
        "dataset_bindings": [
            {
                "dataset_json": str(dataset.resolve()),
                "split": "train",
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "dataset_manifest_sha256": hashlib.sha256(
                    dataset_sidecar.read_bytes()
                ).hexdigest(),
                "dataset_sample_count": 1,
                "unique_image_count": 1,
            }
        ],
        "path_root": str(tmp_path.resolve()),
        "union_unique_image_count": 1,
        "token_file_count": file_count,
        "token_byte_count": byte_count,
        "token_payload_sha256": payload_hash,
        "device": "cpu",
        "batch_size": 4,
        "compute_dtype": "float16",
        "nonfinite_retry_count": 0,
        "dino_model_path": "/models/dino",
        "siglip_model_path": "/models/siglip",
        "dino_model_sha256": "hash:/models/dino",
        "siglip_model_sha256": "hash:/models/siglip",
    }
    cache.atomic_write_json(cache_root / "cache_manifest.json", legacy)
    original = (fine.read_bytes(), coarse.read_bytes())
    monkeypatch.setattr(cache, "VisionCacheConfig", _fake_cache_config)
    monkeypatch.setattr(cache, "sha256_artifact", lambda path: f"hash:{path}")

    with pytest.raises(cache.VisionCacheError, match="explicit provenance upgrade"):
        cache.cache_dataset(
            _cache_args(dataset, cache_root, compute_dtype="float16")
        )
    upgraded = cache.cache_dataset(
        _cache_args(
            dataset,
            cache_root,
            compute_dtype="float16",
            upgrade_legacy_manifest=True,
        )
    )
    assert upgraded["schema_version"] == 3
    assert upgraded["legacy_manifest_sha256"]
    assert (fine.read_bytes(), coarse.read_bytes()) == original
    assert verify_vision_cache(cache_root, [dataset])["cache_provenance_sha256"]


def test_hf_base_artifact_binding_includes_config_and_selected_weights(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"n_waypoints": 8}\n', encoding="utf-8")
    (model / "model.safetensors").write_bytes(b"weights")
    (model / "README.md").write_text("notes", encoding="utf-8")

    first = bind_hf_model_artifact(model)
    assert [entry["path"] for entry in first["files"]] == [
        "config.json",
        "model.safetensors",
    ]
    (model / "README.md").write_text("changed notes", encoding="utf-8")
    assert bind_hf_model_artifact(model)["artifact_sha256"] == first[
        "artifact_sha256"
    ]
    (model / "config.json").write_text('{"n_waypoints": 16}\n', encoding="utf-8")
    assert bind_hf_model_artifact(model)["artifact_sha256"] != first[
        "artifact_sha256"
    ]
