#!/usr/bin/env python3
"""Local-only model path resolution for offline inference.

The runtime should never silently fall back to Hugging Face downloads. This
module resolves either a normal model directory with config.json, or a copied
Hugging Face cache directory shaped like models--org--repo/snapshots/<hash>/.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


OPENTRACKVLA_ROOT = Path(__file__).resolve().parent
TRACK_CAR_ROOT = OPENTRACKVLA_ROOT.parents[1]


def _has_config(path: Path) -> bool:
    return path.exists() and path.is_dir() and (path / "config.json").exists()


def _snapshot_with_config(path: Path) -> Path | None:
    if _has_config(path):
        return path

    snapshots = path / "snapshots"
    choices = [p for p in snapshots.iterdir() if _has_config(p)] if snapshots.exists() else []
    if not choices:
        nested_configs = list(path.glob("*/snapshots/*/config.json"))
        if not nested_configs:
            return None
        choices = [p.parent for p in nested_configs]
    return sorted(choices, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _hf_cache_root() -> Path:
    hf_home = os.getenv("HF_HOME", "").strip()
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path(os.getenv("HF_HUB_CACHE", "~/.cache/huggingface/hub")).expanduser()


def _cache_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _path_from(value: str | os.PathLike | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser()


def resolve_local_model_path(
    *,
    label: str,
    repo_id: str,
    explicit: str | os.PathLike | None = None,
    env_var: str | None = None,
    candidates: Iterable[str | os.PathLike] = (),
) -> str:
    checked: list[Path] = []

    ordered: list[Path] = []
    if env_var:
        env_path = _path_from(os.getenv(env_var))
        if env_path is not None:
            ordered.append(env_path)

    explicit_path = _path_from(explicit)
    if explicit_path is not None and "/" not in str(explicit_path).replace("\\", "/"):
        ordered.append(explicit_path)
    elif explicit_path is not None and explicit_path.exists():
        ordered.append(explicit_path)

    ordered.extend(Path(p).expanduser() for p in candidates)
    ordered.append(_hf_cache_root() / _cache_dir_name(repo_id))

    # Also support people copying HF cache folders into ckpts_hf directly.
    ordered.append(OPENTRACKVLA_ROOT / "ckpts_hf" / _cache_dir_name(repo_id))

    seen: set[str] = set()
    for path in ordered:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        checked.append(path)
        resolved = _snapshot_with_config(path)
        if resolved is not None:
            return str(resolved.resolve())

    checked_text = "\n  - ".join(str(p) for p in checked)
    raise FileNotFoundError(
        f"{label} local model files were not found. Remote download is disabled.\n"
        f"Expected a directory containing config.json, or a Hugging Face cache "
        f"directory containing snapshots/<hash>/config.json.\n"
        f"Checked:\n  - {checked_text}"
    )


def default_qwen_candidates() -> list[Path]:
    return [
        OPENTRACKVLA_ROOT / "ckpts_hf" / "qwen3-0.6b",
        OPENTRACKVLA_ROOT / "ckpts_hf" / "Qwen3-0.6B",
    ]


def default_siglip_candidates() -> list[Path]:
    return [
        OPENTRACKVLA_ROOT / "ckpts_hf" / "siglip-so400m-patch14-384",
    ]


def default_dinov3_candidates() -> list[Path]:
    return [
        TRACK_CAR_ROOT / "weights" / "modelscope" / "dinov3-vits16-pretrain-lvd1689m",
        OPENTRACKVLA_ROOT / "ckpts_hf" / "dinov3-vits16-pretrain-lvd1689m",
    ]
