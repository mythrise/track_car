#!/usr/bin/env python3
# ruff: noqa: E402
"""Multi-seed public validation evaluator.

Evaluates a single checkpoint (B0 or Harness/F2) on the frozen 2,848-row
public validation set, using the same evaluation infrastructure as
eval_matched128_public_val.py (build_model + evaluate_family + summarize_predictions).

Usage:
    python scripts/eval_single_checkpoint.py \
        --family B0 --seed 1 \
        --checkpoint experiments/.../B0_seed1/baseline_epoch0.pt \
        --output-dir experiments/.../multiseed_eval/B0_seed1

    python scripts/eval_single_checkpoint.py \
        --family Harness --seed 1 \
        --checkpoint experiments/.../F2_seed1/checkpoint_update128_S-SELF.pt \
        --output-dir experiments/.../multiseed_eval/Harness_seed1

For Harness, the checkpoint must be loadable by the same build_model("B1")
logic because SA-Hstar extends TrackVLA++-Lite.  If the model key format
differs, use --harness-as-b1 flag which renames "model" -> "model_state".
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
if _existing not in (None, CUBLAS_WORKSPACE_CONFIG):
    raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG conflict: {_existing!r}")
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
OTV_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(OTV_ROOT))
sys.path.insert(0, str(OTV_ROOT / "scripts"))
sys.path.insert(0, str(SCRIPTS_ROOT))

# Frozen dataset paths
VAL_JSONL = PROJECT_ROOT / "data/collected_v1/datasets/val.jsonl"
VAL_MANIFEST = PROJECT_ROOT / "data/collected_v1/datasets/val.jsonl.manifest.json"
VISION_CACHE = PROJECT_ROOT / "data/collected_v1/vision_cache"
BASE_HF_DIR = Path("E:/AAAI/opentrackvla-qwen06b")
QWEN_PATH = Path("E:/AAAI/resolved_models/Qwen3-0.6B")
VAL_SHA256 = "696423b1c12f1b77f3c664ad1ca414e8371a55a033d20564aeb9d133e87eb14a"
VAL_ROWS = 2848
CLEAN_SEQUENCE_STARTS = {0, 512, 924, 1886}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=["B0", "B1", "Harness"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True, help="Absolute or project-relative path")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--project-root", default=str(PROJECT_ROOT)
    )
    return p.parse_args(argv)


def activate_local_import_paths() -> None:
    for path in (PROJECT_ROOT, OTV_ROOT, OTV_ROOT / "scripts"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_path = ckpt_path.resolve()

    start_utc = _utc_now()
    start_t = time.perf_counter()

    # Configure CUDA determinism
    from f2_experiment.reproducibility import configure_cuda_reproducibility, prepare_cublas_workspace_config
    prepare_cublas_workspace_config()
    cuda_repro = configure_cuda_reproducibility(torch)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"[eval_single] family={args.family} seed={args.seed} device={device}")

    # Verify val.jsonl
    val_sha = _sha256_file(VAL_JSONL)
    assert val_sha == VAL_SHA256, f"val.jsonl SHA mismatch: {val_sha}"
    print(f"[eval_single] val.jsonl SHA OK ({VAL_SHA256[:16]}...)")

    activate_local_import_paths()

    # Import evaluator components from the frozen eval script
    import eval_matched128_public_val as ev
    ev.activate_local_import_paths()

    # Load checkpoint: normalise to {model_state: ...} format
    ckpt_sha = _sha256_file(ckpt_path)
    print(f"[eval_single] checkpoint sha={ckpt_sha[:16]}...")
    raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_ckpt, dict):
        raise RuntimeError(f"checkpoint is not a dict: {ckpt_path}")

    # Normalise key names: train_baseline uses 'model_state'; save_arm_checkpoint uses 'model'
    if "model_state" not in raw_ckpt and "model" in raw_ckpt:
        raw_ckpt = dict(raw_ckpt)
        raw_ckpt["model_state"] = raw_ckpt.pop("model")
        print("[eval_single] normalised checkpoint key: 'model' -> 'model_state'")

    # Load val rows
    raw_rows = [
        json.loads(line)
        for line in VAL_JSONL.read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw_rows) == VAL_ROWS

    # Load val token ledger as a proper TokenHashLedger (required by assembly_data)
    val_ledger_path = (
        PROJECT_ROOT
        / "experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
        / "full_2848/token_ledger.json"
    )
    val_ledger_receipt_path = (
        PROJECT_ROOT
        / "experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
        / "full_2848/token_ledger_receipt.json"
    )
    if val_ledger_path.exists() and val_ledger_receipt_path.exists():
        token_ledger, _ = ev.validate_val_token_ledger(val_ledger_path, val_ledger_receipt_path)
        print(f"[eval_single] loaded frozen val token ledger (TokenHashLedger)")
    else:
        from f2_experiment.assembly_data import build_token_ledger_for_rows
        print("[eval_single] building val token ledger on-the-fly ...")
        token_ledger = build_token_ledger_for_rows(
            raw_rows,
            base_root=root,
            cache_root=VISION_CACHE,
        )

    # Build model
    eval_family = "B0" if args.family == "B0" else "B1"
    model = ev.build_model(
        eval_family,
        raw_ckpt,
        base_hf_model_dir=BASE_HF_DIR,
        qwen_model_path=QWEN_PATH,
        device=device,
    )
    print(f"[eval_single] model loaded (family={eval_family})")

    # Full 2848-row evaluation
    full_indices = list(range(VAL_ROWS))
    with torch.no_grad():
        outputs, runtime = ev.evaluate_family(
            eval_family,
            model,
            full_indices,
            raw_rows,
            image_base_root=root,
            cache_root=VISION_CACHE,
            token_ledger=token_ledger,
            device=device,
        )

    summary = ev.summarize_predictions(outputs)
    h1_wmae = summary["source_macro"]["h1_wmae"]
    all8_wmae = summary["source_macro"]["all8_wmae"]
    elapsed = time.perf_counter() - start_t
    end_utc = _utc_now()

    print(f"\n[eval_single] === Results ===")
    print(f"[eval_single] method={args.family} seed={args.seed}")
    print(f"[eval_single] H1 source-macro wMAE:   {h1_wmae:.8f}")
    print(f"[eval_single] All8 source-macro wMAE: {all8_wmae:.8f}")
    print(f"[eval_single] elapsed: {elapsed:.1f}s")

    result = {
        "schema_version": 1,
        "analysis_class": "multiseed_public_val_result_v1",
        "method": args.family,
        "seed": args.seed,
        "h1_source_macro_wmae": h1_wmae,
        "all8_source_macro_wmae": all8_wmae,
        "val_rows": VAL_ROWS,
        "val_sha256": VAL_SHA256,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "summary": summary,
        "runtime": runtime,
        "cuda_reproducibility": cuda_repro,
        "gpu": gpu_name,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": round(elapsed, 1),
        "internal_test_opened": False,
    }

    result_path = out_dir / f"eval_result_{args.family}_seed{args.seed}.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[eval_single] result written: {result_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
