"""Content bindings shared by training, caching, and offline evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath


VISION_CACHE_SCHEMA_VERSION = 3
VISION_CACHE_PROVENANCE_SCHEMA_VERSION = 1
PARTIAL_CACHE_PROVENANCE_FILENAME = ".partial_cache_provenance.json"
VISION_TOKEN_LAYOUT = {
    "schema": "dinov3_siglip_gridpool_v1",
    "filename_schema": "full_image_name_suffix_v2",
    "serialization": "torch_tensor_v1",
    "tower_concat_order": ["dinov3", "siglip"],
    "feature_dim": 1536,
    "levels": {
        "fine": {"token_count": 64},
        "coarse": {"token_count": 4},
    },
}
BASE_MODEL_ARTIFACT_SCHEMA_VERSION = 1


class ExperimentBindingError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_artifact(path: str | Path) -> str:
    """Hash one file or a deterministic directory tree, following file symlinks."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise ExperimentBindingError(f"model artifact does not exist: {root}")
    digest = hashlib.sha256()
    files = [
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != ".DS_Store"
    ]
    if not files:
        raise ExperimentBindingError(f"model artifact directory is empty: {root}")
    for item in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        size = item.stat().st_size
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_weight_index_files(root: Path, index_path: Path) -> list[Path]:
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentBindingError(
            f"invalid HuggingFace weight index: {index_path}"
        ) from exc
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ExperimentBindingError(
            f"HuggingFace weight index has no weight_map: {index_path}"
        )
    files = []
    for relative_value in sorted(set(weight_map.values())):
        if not isinstance(relative_value, str) or not relative_value:
            raise ExperimentBindingError(
                f"HuggingFace weight index has an invalid shard path: {index_path}"
            )
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ExperimentBindingError(
                f"HuggingFace weight shard escapes model directory: {relative_value}"
            )
        shard = (root / relative).resolve()
        try:
            shard.relative_to(root)
        except ValueError as exc:
            raise ExperimentBindingError(
                f"HuggingFace weight shard escapes model directory: {relative_value}"
            ) from exc
        if not shard.is_file():
            raise ExperimentBindingError(
                f"HuggingFace weight shard is missing: {shard}"
            )
        files.append(shard)
    return files


def bind_hf_model_artifact(path: str | Path) -> dict:
    """Bind the inference-relevant files of one local HuggingFace model.

    The digest intentionally excludes README files, download caches, and training
    notes.  It includes the configuration plus the exact weight layout that
    ``from_pretrained`` selects, so a config-only change cannot masquerade as the
    same base model.
    """

    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ExperimentBindingError(
            f"HuggingFace model directory does not exist: {root}"
        )
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ExperimentBindingError(
            f"HuggingFace model config is missing: {config_path}"
        )

    candidates = (
        (root / "model.safetensors.index.json", "safetensors_sharded"),
        (root / "model.safetensors", "safetensors_single"),
        (root / "pytorch_model.bin.index.json", "pytorch_bin_sharded"),
        (root / "pytorch_model.bin", "pytorch_bin_single"),
    )
    selected = next(
        ((candidate, layout) for candidate, layout in candidates if candidate.is_file()),
        None,
    )
    if selected is None:
        raise ExperimentBindingError(
            f"HuggingFace model weights are missing under: {root}"
        )
    selected_path, weight_layout = selected
    if selected_path.name.endswith(".index.json"):
        weight_files = [selected_path, *_safe_weight_index_files(root, selected_path)]
    else:
        weight_files = [selected_path]

    artifact_files = [(config_path, "config")]
    generation_config = root / "generation_config.json"
    if generation_config.is_file():
        artifact_files.append((generation_config, "generation_config"))
    artifact_files.extend(
        (item, "weights_index" if item.name.endswith(".index.json") else "weights")
        for item in weight_files
    )
    entries = []
    seen = set()
    for item, role in sorted(
        artifact_files, key=lambda pair: pair[0].relative_to(root).as_posix()
    ):
        relative = item.relative_to(root).as_posix()
        if relative in seen:
            continue
        seen.add(relative)
        entries.append(
            {
                "path": relative,
                "role": role,
                "size": int(item.stat().st_size),
                "sha256": sha256_file(item),
            }
        )
    manifest = {
        "schema_version": BASE_MODEL_ARTIFACT_SCHEMA_VERSION,
        "format": "huggingface_pretrained",
        "weight_layout": weight_layout,
        "files": entries,
    }
    return {
        **manifest,
        "artifact_sha256": _canonical_json_sha256(manifest),
    }


def _dataset_binding(dataset_path: str | Path) -> tuple[dict, Path]:
    dataset = Path(dataset_path).expanduser().resolve()
    sidecar = Path(str(dataset) + ".manifest.json")
    if not dataset.is_file() or not sidecar.is_file():
        raise ExperimentBindingError(f"dataset/manifest is missing: {dataset}")
    sidecar_bytes = sidecar.read_bytes()
    manifest = json.loads(sidecar_bytes.decode("utf-8"))
    actual_hash = sha256_file(dataset)
    if actual_hash != manifest.get("data_jsonl_sha256"):
        raise ExperimentBindingError(f"dataset JSONL hash mismatch: {dataset}")
    return (
        {
            "dataset_json": str(dataset),
            "data_jsonl_sha256": actual_hash,
            "dataset_manifest_sha256": hashlib.sha256(sidecar_bytes).hexdigest(),
            "dataset_sample_count": int(manifest.get("sample_count", -1)),
        },
        dataset,
    )


def _token_filename(image: Path, level: str) -> str:
    suffix = {"fine": "vfine", "coarse": "vcoarse"}[level]
    return f"{image.name}_{suffix}.pt"


def _relocate_manifest_path(
    value: str,
    *,
    recorded_root: str,
    relocated_root: Path | None,
) -> Path:
    """Resolve one cache-manifest path with an explicit content relocation.

    Vision-cache manifests historically recorded the absolute producer path.
    A byte-identical cache may be moved to another machine, but the mapping is
    accepted only when the caller explicitly supplies the new project root and
    the recorded path lies below the manifest's recorded ``path_root``.
    """

    if relocated_root is None:
        return Path(value).expanduser().resolve()
    if not isinstance(value, str) or not value:
        raise ExperimentBindingError("vision cache contains an empty bound path")
    if not isinstance(recorded_root, str) or not recorded_root:
        raise ExperimentBindingError("vision cache has no recorded path_root")

    normalized_value = value.replace("\\", "/")
    normalized_root = recorded_root.replace("\\", "/").rstrip("/")
    value_path = PurePosixPath(normalized_value)
    root_path = PurePosixPath(normalized_root)
    try:
        relative = value_path.relative_to(root_path)
    except ValueError as exc:
        if value_path.is_absolute() or PureWindowsPath(value).is_absolute():
            raise ExperimentBindingError(
                "vision cache bound path lies outside recorded path_root: "
                f"{value}"
            ) from exc
        relative = value_path
    if any(part in ("", ".", "..") for part in relative.parts):
        raise ExperimentBindingError(
            f"vision cache bound path is not cleanly relative: {value}"
        )
    relocated = relocated_root.resolve()
    candidate = relocated.joinpath(*relative.parts).resolve()
    try:
        candidate.relative_to(relocated)
    except ValueError as exc:
        raise ExperimentBindingError(
            "vision cache relocated path escapes effective path_root: "
            f"{value} -> {candidate}"
        ) from exc
    return candidate


def _cache_images(
    cache_manifest: dict,
    *,
    relocated_root: Path | None = None,
) -> tuple[Path, list[Path]]:
    recorded_root = str(cache_manifest.get("path_root", ""))
    base_root = _relocate_manifest_path(
        recorded_root,
        recorded_root=recorded_root,
        relocated_root=relocated_root,
    )
    if not base_root.is_dir():
        raise ExperimentBindingError(f"cache path_root does not exist: {base_root}")
    unique = {}
    for binding in cache_manifest.get("dataset_bindings") or []:
        dataset = _relocate_manifest_path(
            str(binding.get("dataset_json", "")),
            recorded_root=recorded_root,
            relocated_root=relocated_root,
        )
        if not dataset.is_file():
            raise ExperimentBindingError(f"cache-bound dataset is missing: {dataset}")
        if sha256_file(dataset) != binding.get("data_jsonl_sha256"):
            raise ExperimentBindingError(f"cache-bound dataset hash mismatch: {dataset}")
        sidecar = Path(str(dataset) + ".manifest.json")
        if not sidecar.is_file():
            raise ExperimentBindingError(
                f"cache-bound dataset manifest is missing: {sidecar}"
            )
        if sha256_file(sidecar) != binding.get("dataset_manifest_sha256"):
            raise ExperimentBindingError(
                f"cache-bound dataset manifest hash mismatch: {sidecar}"
            )
        with dataset.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                current = row.get("current")
                history = row.get("images") or []
                if not isinstance(current, str) or not isinstance(history, list):
                    raise ExperimentBindingError(
                        f"invalid cache-bound image record: {dataset}:{line_number}"
                    )
                for value in [current, *history]:
                    raw_image = Path(value)
                    if relocated_root is None and not raw_image.is_absolute():
                        image = (base_root / raw_image).resolve()
                    else:
                        image = _relocate_manifest_path(
                            value,
                            recorded_root=recorded_root,
                            relocated_root=relocated_root,
                        )
                    if not image.is_file():
                        raise ExperimentBindingError(f"cache image is missing: {image}")
                    unique[str(image)] = image
    return base_root, [unique[key] for key in sorted(unique)]


def _hash_cache_tokens(
    cache_root: Path, base_root: Path, images: list[Path]
) -> tuple[str, int, int]:
    expected_tokens = set()
    for image in images:
        try:
            relative_image = image.relative_to(base_root)
        except ValueError as exc:
            raise ExperimentBindingError(
                f"cache image lies outside path_root: {image}"
            ) from exc
        token_dir = cache_root / relative_image.parent
        for level in ("fine", "coarse"):
            expected_tokens.add(token_dir / _token_filename(relative_image, level))
    actual_tokens = set(cache_root.rglob("*_vfine.pt")) | set(
        cache_root.rglob("*_vcoarse.pt")
    )
    unexpected = sorted(actual_tokens - expected_tokens, key=str)
    if unexpected:
        raise ExperimentBindingError(
            f"vision cache contains an unreferenced stale token: {unexpected[0]}"
        )

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for image in images:
        relative_image = image.relative_to(base_root)
        token_dir = cache_root / relative_image.parent
        for level in ("fine", "coarse"):
            token = token_dir / _token_filename(relative_image, level)
            if not token.is_file():
                raise ExperimentBindingError(f"required cache token is missing: {token}")
            relative = token.relative_to(cache_root).as_posix().encode("utf-8")
            size = token.stat().st_size
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(size.to_bytes(8, "big"))
            with token.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            file_count += 1
            byte_count += size
    return digest.hexdigest(), file_count, byte_count


def _validated_cache_provenance(manifest: dict) -> tuple[dict, str]:
    if int(manifest.get("schema_version", -1)) != VISION_CACHE_SCHEMA_VERSION:
        raise ExperimentBindingError(
            "vision cache schema_version must be 3; run "
            "cache_vision_tokens.py --upgrade_legacy_manifest for a schema-2 cache"
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ExperimentBindingError("vision cache provenance object is missing")
    if int(provenance.get("schema_version", -1)) != (
        VISION_CACHE_PROVENANCE_SCHEMA_VERSION
    ):
        raise ExperimentBindingError("vision cache provenance schema is unsupported")
    provenance_sha256 = _canonical_json_sha256(provenance)
    if provenance_sha256 != manifest.get("provenance_sha256"):
        raise ExperimentBindingError("vision cache provenance hash mismatch")
    if provenance.get("token_layout") != VISION_TOKEN_LAYOUT:
        raise ExperimentBindingError("vision cache token layout/schema mismatch")
    producer = provenance.get("producer")
    if not isinstance(producer, dict) or any(
        producer.get(field) in (None, "")
        for field in (
            "device",
            "batch_size",
            "compute_dtype",
            "storage_dtype",
            "image_size",
            "force_square_resize",
        )
    ):
        raise ExperimentBindingError("vision cache producer config is incomplete")
    if producer.get("storage_dtype") != "float16":
        raise ExperimentBindingError("vision cache storage dtype must be float16")
    encoders = provenance.get("encoders")
    if not isinstance(encoders, dict) or any(
        not isinstance(encoders.get(name), dict)
        or encoders[name].get("artifact_sha256") in (None, "")
        or encoders[name].get("path") in (None, "")
        for name in ("dinov3", "siglip")
    ):
        raise ExperimentBindingError("vision cache encoder provenance is incomplete")

    mirrored = {
        "dataset_bindings": provenance.get("dataset_bindings"),
        "path_root": provenance.get("path_root"),
        "union_unique_image_count": provenance.get("union_unique_image_count"),
        "device": producer.get("device"),
        "batch_size": producer.get("batch_size"),
        "compute_dtype": producer.get("compute_dtype"),
        "storage_dtype": producer.get("storage_dtype"),
        "token_layout": provenance.get("token_layout"),
        "dino_model_path": encoders["dinov3"].get("path"),
        "siglip_model_path": encoders["siglip"].get("path"),
        "dino_model_sha256": encoders["dinov3"].get("artifact_sha256"),
        "siglip_model_sha256": encoders["siglip"].get("artifact_sha256"),
    }
    mismatched = [
        field for field, expected in mirrored.items() if manifest.get(field) != expected
    ]
    if mismatched:
        raise ExperimentBindingError(
            "vision cache manifest/provenance disagreement: " + ", ".join(mismatched)
        )
    return provenance, provenance_sha256


def verify_vision_cache(
    cache_root: str | Path,
    required_datasets,
    *,
    verify_payload: bool = True,
    relocated_root: str | Path | None = None,
) -> dict:
    cache_root = Path(cache_root).expanduser().resolve()
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise ExperimentBindingError(f"vision cache manifest is missing: {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    _provenance, provenance_sha256 = _validated_cache_provenance(manifest)
    recorded_root = str(manifest.get("path_root", ""))
    effective_root = (
        Path(relocated_root).expanduser().resolve()
        if relocated_root is not None
        else None
    )
    available = {
        str(
            _relocate_manifest_path(
                str(binding.get("dataset_json", "")),
                recorded_root=recorded_root,
                relocated_root=effective_root,
            )
        ): binding
        for binding in manifest.get("dataset_bindings") or []
    }
    for dataset_path in required_datasets:
        if dataset_path in (None, ""):
            continue
        expected, resolved = _dataset_binding(dataset_path)
        actual = available.get(str(resolved))
        if actual is None or any(
            actual.get(field) != expected[field]
            for field in (
                "data_jsonl_sha256",
                "dataset_manifest_sha256",
                "dataset_sample_count",
            )
        ):
            raise ExperimentBindingError(
                f"vision cache is not bound to current dataset: {resolved}"
            )
    if verify_payload:
        base_root, images = _cache_images(
            manifest, relocated_root=effective_root
        )
        payload_hash, file_count, byte_count = _hash_cache_tokens(
            cache_root, base_root, images
        )
        if (
            payload_hash != manifest.get("token_payload_sha256")
            or file_count != int(manifest.get("token_file_count", -1))
            or byte_count != int(manifest.get("token_byte_count", -1))
        ):
            raise ExperimentBindingError("vision cache token payload hash mismatch")
    return {
        "cache_root": str(cache_root),
        "cache_manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "cache_provenance_sha256": provenance_sha256,
        "token_payload_sha256": manifest.get("token_payload_sha256"),
        "dino_model_sha256": manifest.get("dino_model_sha256"),
        "siglip_model_sha256": manifest.get("siglip_model_sha256"),
        "recorded_path_root": recorded_root,
        "effective_path_root": str(
            effective_root
            if effective_root is not None
            else Path(recorded_root).expanduser().resolve()
        ),
        "path_relocated": effective_root is not None
        and str(effective_root) != str(Path(recorded_root).expanduser().resolve()),
    }
