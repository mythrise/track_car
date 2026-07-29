#!/usr/bin/env python3
# ruff: noqa: E402
"""Multi-seed public-validation evaluator.

Runs inference on the frozen 2,848-row public validation set using a given
checkpoint and reports H1 and All8 source-macro wMAE.  Supports both B0
(no-Harness baseline) and Harness (F2/SA-Hstar) checkpoints.

Usage:
    python scripts/eval_multiseed_public_val.py \
        --method B0 \
        --seed 1 \
        --checkpoint experiments/.../B0_seed1/baseline_epoch0.pt \
        --output-dir experiments/.../multiseed_eval/B0_seed1

    python scripts/eval_multiseed_public_val.py \
        --method Harness \
        --seed 1 \
        --checkpoint experiments/.../F2_seed1/checkpoint_update128_S-SELF.pt \
        --output-dir experiments/.../multiseed_eval/Harness_seed1
"""

from __future__ import annotations

import argparse
import hashlib
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
    raise RuntimeError(
        f"CUBLAS_WORKSPACE_CONFIG conflict: {_existing!r}"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
OTV_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
sys.path.insert(0, str(OTV_ROOT))

# Determinism settings (mirror windows_cuda_deterministic_v1)
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

VAL_JSONL = PROJECT_ROOT / "data/collected_v1/datasets/val.jsonl"
VAL_SHA256 = "696423b1c12f1b77f3c664ad1ca414e8371a55a033d20564aeb9d133e87eb14a"
VISION_CACHE_ROOT = PROJECT_ROOT / "data/collected_v1/vision_cache"
BASE_HF_DIR = Path("E:/AAAI/opentrackvla-qwen06b")
QWEN_PATH = Path("E:/AAAI/resolved_models/Qwen3-0.6B")
RELOCATED_ROOT = PROJECT_ROOT

# Weights for wMAE: forward=1, yaw=2, strafe=0; divisor=3
CONTROL_WEIGHTS = {"forward": 1.0, "strafe": 0.0, "yaw": 2.0}
CONTROL_DIVISOR = 3.0
CLEAN_SEQUENCE_STARTS = {0, 512, 924, 1886}
STRAFE_RESET_INDICES = {346, 347, 348, 349}


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
    p.add_argument("--method", required=True, choices=["B0", "Harness"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--project-root", default=str(PROJECT_ROOT))
    return p.parse_args(argv)


def weighted_mae(
    pred_actions: list[dict],
    target_actions: list[dict],
) -> tuple[float, float]:
    """Compute weighted MAE for H1 and all-8 prediction horizons.

    Returns (h1_wmae, all8_wmae).
    """
    h1_diffs = []
    all8_diffs = []
    weights = []

    for pred, target in zip(pred_actions, target_actions):
        # pred["step_actions"] is [8, 3] shaped list
        pred_steps = pred.get("step_actions", [])
        tgt_steps = target.get("step_actions", [])
        if not pred_steps or not tgt_steps:
            continue

        n = min(len(pred_steps), len(tgt_steps))
        if n == 0:
            continue

        # H1 (first step)
        p0 = pred_steps[0]
        t0 = tgt_steps[0]
        diff_h1 = _step_weighted_mae(p0, t0)
        h1_diffs.append(diff_h1)

        # All-8 (mean over all steps)
        diffs_all = [
            _step_weighted_mae(pred_steps[i], tgt_steps[i])
            for i in range(n)
        ]
        all8_diffs.append(float(np.mean(diffs_all)))
        weights.append(1.0)

    if not h1_diffs:
        return float("nan"), float("nan")

    h1_wmae = float(np.mean(h1_diffs))
    all8_wmae = float(np.mean(all8_diffs))
    return h1_wmae, all8_wmae


def _step_weighted_mae(pred_step, target_step) -> float:
    """Weighted MAE for one timestep [forward, strafe, yaw]."""
    keys = ["forward", "strafe", "yaw"]
    total_w = 0.0
    total_err = 0.0
    for i, key in enumerate(keys):
        w = CONTROL_WEIGHTS[key]
        if w == 0.0:
            continue
        if isinstance(pred_step, (list, tuple)):
            p = float(pred_step[i])
        else:
            p = float(pred_step.get(key, 0.0))
        if isinstance(target_step, (list, tuple)):
            t = float(target_step[i])
        else:
            t = float(target_step.get(key, 0.0))
        total_err += w * abs(p - t)
        total_w += w
    return total_err / CONTROL_DIVISOR


def compute_source_macro_wmae(
    predictions: list[dict],
    targets: list[dict],
) -> tuple[float, float]:
    """Source-macro wMAE: mean-per-source then mean across sources.

    Returns (h1_source_macro_wmae, all8_source_macro_wmae).
    """
    from collections import defaultdict

    by_source_h1: dict[str, list[float]] = defaultdict(list)
    by_source_all8: dict[str, list[float]] = defaultdict(list)

    for pred, target in zip(predictions, targets):
        src = target.get("source_raw_dir", "__unknown__")
        pred_steps = pred.get("step_actions", [])
        tgt_steps = target.get("step_actions", [])
        if not pred_steps or not tgt_steps:
            continue
        n = min(len(pred_steps), len(tgt_steps))
        if n == 0:
            continue

        diff_h1 = _step_weighted_mae(pred_steps[0], tgt_steps[0])
        diffs_all = [
            _step_weighted_mae(pred_steps[i], tgt_steps[i])
            for i in range(n)
        ]
        by_source_h1[src].append(diff_h1)
        by_source_all8[src].append(float(np.mean(diffs_all)))

    if not by_source_h1:
        return float("nan"), float("nan")

    src_h1_means = [float(np.mean(v)) for v in by_source_h1.values()]
    src_all8_means = [float(np.mean(v)) for v in by_source_all8.values()]
    return float(np.mean(src_h1_means)), float(np.mean(src_all8_means))


def run_b0_eval(
    checkpoint_path: Path,
    val_rows: list[dict],
    device: torch.device,
) -> list[dict]:
    """Run B0 (no-Harness baseline) inference on validation rows."""
    from model import DataConfig, JsonTrackingDataset, collate_batch
    from experiment_binding import bind_hf_model_artifact, sha256_artifact, verify_vision_cache
    from local_weights import resolve_local_model_path
    from validation_metrics import waypoints_to_step_actions
    from open_trackvla_hf import OpenTrackVLAForCausalLM

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = ckpt.get("model", ckpt)

    # Build model
    from open_trackvla_hf.configuration_open_trackvla import OpenTrackVLAConfig
    config = OpenTrackVLAConfig.from_pretrained(str(BASE_HF_DIR))
    model = OpenTrackVLAForCausalLM.from_pretrained(str(BASE_HF_DIR))
    model.load_state_dict(model_state, strict=False)
    model = model.to(device).eval()

    predictions = []
    with torch.no_grad():
        for row in val_rows:
            # minimal stateless evaluation
            pred = _infer_b0_row(model, row, device)
            predictions.append(pred)
    return predictions


def _infer_b0_row(model, row, device):
    # Stub - actual inference needs vision cache
    return {"step_actions": row.get("step_actions", [])}


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.project_root).resolve()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    start_utc = _utc_now()
    start_t = time.perf_counter()

    # Verify val.jsonl
    val_sha = _sha256_file(VAL_JSONL)
    assert val_sha == VAL_SHA256, f"val.jsonl SHA mismatch: {val_sha}"
    print(f"[eval_multiseed] val.jsonl SHA verified ({VAL_SHA256[:12]}...)")

    val_rows = [json.loads(line) for line in VAL_JSONL.read_text("utf-8").splitlines() if line.strip()]
    assert len(val_rows) == 2848, f"Expected 2848 val rows, got {len(val_rows)}"
    print(f"[eval_multiseed] loaded {len(val_rows)} validation rows")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    ckpt_sha = _sha256_file(ckpt_path)
    print(f"[eval_multiseed] checkpoint: {ckpt_path.name} sha={ckpt_sha[:12]}...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    # Use the frozen evaluator infrastructure
    print(f"[eval_multiseed] running inference with method={args.method} seed={args.seed} ...")

    # Import the matched public val evaluator to reuse its inference code
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

    # We'll import and use the eval infrastructure from eval_matched128_public_val
    # but with a custom checkpoint path
    from eval_matched128_public_val import (
        activate_local_import_paths,
        load_and_build_b0_model,
        load_and_build_b1_model,
        run_stateless_b0_eval,
        run_rolling_b1_eval,
    )
    activate_local_import_paths()

    if args.method == "B0":
        model = load_and_build_b0_model(ckpt_path, device=device)
        preds, targets = run_stateless_b0_eval(model, val_rows, device=device)
    else:  # Harness
        model = load_and_build_b1_model(ckpt_path, device=device)
        preds, targets = run_rolling_b1_eval(model, val_rows, device=device)

    h1_wmae, all8_wmae = compute_source_macro_wmae(preds, targets)
    elapsed = time.perf_counter() - start_t
    end_utc = _utc_now()

    print(f"[eval_multiseed] H1 source-macro wMAE: {h1_wmae:.8f}")
    print(f"[eval_multiseed] All8 source-macro wMAE: {all8_wmae:.8f}")

    result = {
        "method": args.method,
        "seed": args.seed,
        "h1_source_macro_wmae": h1_wmae,
        "all8_source_macro_wmae": all8_wmae,
        "val_rows": 2848,
        "val_sha256": VAL_SHA256,
        "checkpoint_sha256": ckpt_sha,
        "gpu": gpu_name,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": round(elapsed, 1),
        "internal_test_opened": False,
    }

    (out_dir / "eval_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[eval_multiseed] results written to {out_dir}/eval_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
