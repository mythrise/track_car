#!/usr/bin/env python3
"""Validate the local weight manifest used by the inference pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def check_path(label: str, path_value: str | None, required: bool) -> bool:
    if not path_value:
        status = "missing"
        ok = not required
        print(f"[{status}] {label}: not configured")
        return ok

    path = Path(path_value).expanduser()
    ok = path.exists()
    status = "ok" if ok else ("missing" if required else "optional-missing")
    print(f"[{status}] {label}: {path}")
    return ok or not required


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="weights/weights_manifest.example.json",
        help="Path to a local weights manifest JSON file.",
    )
    ap.add_argument(
        "--allow_missing",
        action="store_true",
        help="Print missing paths but exit successfully. Useful for checking the example manifest.",
    )
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    checks = [
        check_path("opentrackvla_root", manifest.get("opentrackvla_root"), required=True),
        check_path("base_model.local_dir", manifest.get("base_model", {}).get("local_dir"), required=True),
        check_path(
            "pfem_checkpoint.local_path",
            manifest.get("pfem_checkpoint", {}).get("local_path"),
            required=False,
        ),
    ]

    if not all(checks) and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
