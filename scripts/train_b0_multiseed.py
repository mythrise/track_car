#!/usr/bin/env python3
# ruff: noqa: E402
"""B0 no-Harness multi-seed training script.

Replicates the B0 matched-128 training with an arbitrary seed.  Reads the
frozen SMK-TRAIN row indices from support_receipt_v3.json to ensure the same
256 training rows as seed-0, but allows torch/numpy seed to vary.

Does NOT call enforce_matched_args (which locks seed=0 for the official frozen
contract). This is intentional for multi-seed reproducibility experiments.

Usage:
    python scripts/train_b0_multiseed.py \
        --seed 1 \
        --out_dir experiments/.../matched128/B0_seed1
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

# CUDA determinism must come before any torch import.
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
sys.path.insert(0, str(OTV_ROOT / "scripts"))

SUPPORT_RECEIPT = PROJECT_ROOT / "experiments/collected_v1_main/f2_smoke/support_receipt_v3.json"
TRAIN_JSONL = PROJECT_ROOT / "data/collected_v1/datasets/train.jsonl"
CACHE_ROOT = PROJECT_ROOT / "data/collected_v1/vision_cache"
BASE_HF_DIR = Path("E:/AAAI/opentrackvla-qwen06b")
QWEN_PATH = Path("E:/AAAI/resolved_models/Qwen3-0.6B")
TRAIN_SHA = "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36"
SUPPORT_RECEIPT_SHA = "2eb3ef48e2596653205ef6d778f4ca3bb5524a9010d0968ccd7e92deca170d72"
MATCHED_ROWS = 256
MATCHED_UPDATES = 128


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
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--project_root", default=str(PROJECT_ROOT))
    return p.parse_args(argv)


def load_smk_train_indices() -> tuple[int, ...]:
    """Extract the frozen SMK-TRAIN row indices from support_receipt_v3.json."""
    sha = _sha256_file(SUPPORT_RECEIPT)
    assert sha == SUPPORT_RECEIPT_SHA, f"support_receipt_v3 SHA mismatch: {sha}"
    doc = json.loads(SUPPORT_RECEIPT.read_text("utf-8"))
    # Navigate: smoke_plan.smoke.ordered_row_indices
    indices = doc["smoke_plan"]["smoke"]["ordered_row_indices"]
    assert len(indices) == MATCHED_ROWS, f"Expected {MATCHED_ROWS} rows, got {len(indices)}"
    return tuple(indices)


def main(argv=None):
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir = out_dir.resolve()
    seed = args.seed

    # Fail-closed: don't silently overwrite
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"Output dir is not empty (fail-closed): {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    start_utc = _utc_now()
    start_t = time.perf_counter()

    # Verify CUDA
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)

    # Frozen CUDA determinism
    from f2_experiment.reproducibility import configure_cuda_reproducibility, prepare_cublas_workspace_config
    prepare_cublas_workspace_config()
    cuda_repro = configure_cuda_reproducibility(torch)

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[train_b0_multiseed] seed={seed} device={device} gpu={gpu_name}")

    # Verify train.jsonl SHA
    train_sha = _sha256_file(TRAIN_JSONL)
    assert train_sha == TRAIN_SHA, f"train.jsonl SHA mismatch: {train_sha}"
    print(f"[train_b0_multiseed] train.jsonl SHA verified")

    # Load frozen SMK-TRAIN row indices
    row_indices = load_smk_train_indices()
    print(f"[train_b0_multiseed] loaded {len(row_indices)} frozen SMK-TRAIN row indices")

    # Import training modules (same as train_baseline.py)
    from train_baseline import inspect_bound_dataset, load_official_base, set_seed
    from model import DataConfig, JsonTrackingDataset, collate_batch
    from cache_gridpool import atomic_torch_save
    from experiment_binding import bind_hf_model_artifact, sha256_artifact, verify_vision_cache
    from experiment_logging import JsonlMetricLogger
    from local_weights import resolve_local_model_path
    from harness.baseline_adapter import OpenTrackVLABaselineAdapter

    # Re-seed after imports (consistent with original flow)
    set_seed(seed)

    # Build base model
    qwen_path = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=str(QWEN_PATH),
        env_var="QWEN_MODEL_PATH",
        candidates=[str(QWEN_PATH)],
    )
    print(f"[train_b0_multiseed] loading base model from {BASE_HF_DIR} ...")
    base = load_official_base(str(BASE_HF_DIR), str(qwen_path))
    base = base.to(device)

    # Build dataset (full train.jsonl, then subset to frozen indices)
    data_config = DataConfig(
        history=31,
        n_waypoints=8,
        dt=0.1,
        label_mode="absolute",
    )
    train_dataset = JsonTrackingDataset(
        jsonl_path=str(TRAIN_JSONL),
        config=data_config,
        image_base_root=str(PROJECT_ROOT),
        cache_root=str(CACHE_ROOT),
        qwen_model_path=str(qwen_path),
    )
    dataset_info = inspect_bound_dataset(train_dataset)
    print(f"[train_b0_multiseed] full dataset: {dataset_info['sample_count']} rows")

    from torch.utils.data import DataLoader, Subset
    subset = Subset(train_dataset, list(row_indices))
    assert len(subset) == MATCHED_ROWS
    print(f"[train_b0_multiseed] subset: {len(subset)} rows (frozen SMK-TRAIN)")

    loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=False,  # ordered_jsonl
        num_workers=0,
        collate_fn=collate_batch,
    )

    # Set up model for training
    base_model = base
    for param in base_model.parameters():
        param.requires_grad = False
    for param in base_model.base.proj.parameters():
        param.requires_grad = True
    for param in base_model.base.planner.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in base_model.parameters())
    print(f"[train_b0_multiseed] trainable={trainable/1e6:.2f}M / total={total/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in base_model.parameters() if p.requires_grad],
        lr=2e-5,
        weight_decay=1e-4,
    )

    metric_logger = JsonlMetricLogger(out_dir / "metrics.jsonl")

    # Run start receipt
    run_start = {
        "phase": "run_start",
        "config": {
            "seed": seed,
            "batch_size": 2,
            "lr": 2e-5,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "max_optimizer_updates": MATCHED_UPDATES,
            "out_dir": str(out_dir),
            "train_json": str(TRAIN_JSONL),
            "base_hf_model_dir": str(BASE_HF_DIR),
        },
        "matched_multiseed": {
            "ordered_row_indices_sha256": "7073a02c866913903a67438673ec7cf6898574bd7fa9bac891ad6142b563818f",
            "processed_samples": MATCHED_ROWS,
            "optimizer_updates": MATCHED_UPDATES,
            "source_dataset_sha256": TRAIN_SHA,
        },
        "timestamp_utc": start_utc,
    }
    metric_logger.log(run_start)

    # Training loop
    base_model.train()
    optimizer_updates = 0
    processed_samples = 0

    for batch in loader:
        if optimizer_updates >= MATCHED_UPDATES:
            break

        optimizer.zero_grad()
        waypoints = batch["waypoints"].to(device)  # [B, 8, 3]
        loss_result = base_model(
            coarse_tokens=batch["coarse_tokens"].to(device),
            coarse_tidx=batch["coarse_tidx"].to(device),
            fine_tokens=batch["fine_tokens"].to(device),
            fine_tidx=batch["fine_tidx"].to(device),
            instructions=batch["instructions"],
            waypoints=waypoints,
            prev_state=None,
            yaw_hist=batch.get("yaw_hist", torch.zeros(waypoints.shape[0], 31, device=device)),
            yaw_curr=batch.get("yaw_curr", torch.zeros(waypoints.shape[0], 1, device=device)),
            prev_action=None,
        )
        if isinstance(loss_result, dict):
            loss = loss_result.get("L_nav", loss_result.get("loss", loss_result.get("total_loss")))
        else:
            loss = loss_result
        if loss is None:
            loss = sum(v for v in loss_result.values() if isinstance(v, torch.Tensor) and v.requires_grad)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in base_model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        optimizer_updates += 1
        processed_samples += waypoints.shape[0]

        if optimizer_updates % 10 == 0 or optimizer_updates == MATCHED_UPDATES:
            print(
                f"  epoch=0 step={optimizer_updates} loss={float(loss):.4f}",
                flush=True,
            )
            metric_logger.log({
                "phase": "train",
                "optimizer_updates": optimizer_updates,
                "processed_samples": processed_samples,
                "loss": float(loss),
                "grad_norm": float(grad_norm),
            })

    assert optimizer_updates == MATCHED_UPDATES, f"Only completed {optimizer_updates} updates"
    assert processed_samples == MATCHED_ROWS

    # Save checkpoint
    ckpt_path = out_dir / "baseline_epoch0.pt"
    state = {k: v.cpu() for k, v in base_model.state_dict().items() if "llm" not in k}
    checkpoint = {
        "model_state": state,
        "seed": seed,
        "optimizer_updates": optimizer_updates,
        "processed_samples": processed_samples,
    }
    atomic_torch_save(checkpoint, ckpt_path)
    ckpt_sha = _sha256_file(ckpt_path)
    print(f"[train_b0_multiseed] saved {ckpt_path.name} sha={ckpt_sha[:12]}...")

    end_utc = _utc_now()
    elapsed = time.perf_counter() - start_t

    # Run receipt
    receipt = {
        "schema_version": 1,
        "analysis_class": "b0_multiseed_training_v1",
        "method": "B0",
        "architecture_lock": "L1+D2+AP2+F2",
        "seed": seed,
        "updates": optimizer_updates,
        "processed_samples": processed_samples,
        "train_jsonl_sha256": TRAIN_SHA,
        "val_rows": 2848,
        "lambda_freeze_receipt": "N/A (B0 has no aux losses)",
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "support_receipt_sha256": SUPPORT_RECEIPT_SHA,
        "cuda_reproducibility": cuda_repro,
        "gpu": gpu_name,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": round(elapsed, 1),
        "internal_test_opened": False,
        "note": "multi-seed extension; same frozen SMK-TRAIN row indices as seed-0; "
                "only torch/numpy RNG seed changed",
    }
    (out_dir / "run_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train_b0_multiseed] done  seed={seed}  elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
