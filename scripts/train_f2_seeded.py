#!/usr/bin/env python3
# ruff: noqa: E402 -- CUDA determinism must be set before torch-heavy imports.
"""Multi-seed F2 SA-Hstar training script.

Runs the paired S-CTRL / S-SELF 128-update smoke training with an arbitrary
seed for multi-seed reproducibility experiments.  Uses the frozen lambda
values from FROZEN_AUX_COEFFICIENTS (source literals in assembly_model) and
the same SMK-TRAIN support rows as the official seed-0 smoke run.

Does NOT re-run the CAL audit or G6/G7/G8/G9 gate evaluation - those belong
to the official frozen smoke contract.  This script is for multi-seed
extension only: lambda is already frozen, only torch/numpy RNG seed changes.

Usage:
    python scripts/train_f2_seeded.py \
        --project-root E:\\AAAI\\track_car \
        --receipt experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json \
        --seed 1 \
        --output-dir experiments/windows_cuda_f2/public_val_memory_reasoning_v1/matched128/F2_seed1
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
        f"CUBLAS_WORKSPACE_CONFIG conflict: {_existing!r} != {CUBLAS_WORKSPACE_CONFIG!r}"
    )
os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch

# Wire project roots.
PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT_DEFAULT))
OTV_ROOT = PROJECT_ROOT_DEFAULT / "third_party" / "OpenTrackVLA"
sys.path.insert(0, str(OTV_ROOT))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-seed F2 SA-Hstar training (lambda frozen, seed varies)"
    )
    p.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT))
    p.add_argument(
        "--receipt",
        required=True,
        help="Path to assembly_receipt_cuda_final_v1.json",
    )
    p.add_argument("--seed", type=int, required=True, help="torch/numpy RNG seed")
    p.add_argument("--output-dir", required=True, help="Fresh output directory")
    return p.parse_args(argv)


def configure_cuda_determinism(torch_module) -> dict:
    """Mirror windows_cuda_deterministic_v1 contract from the official smoke."""
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")
    # Disable all SDPA backends except math-only.
    try:
        torch_module.backends.cuda.enable_flash_sdp(False)
        torch_module.backends.cuda.enable_mem_efficient_sdp(False)
        torch_module.backends.cuda.enable_math_sdp(True)
        sdpa_backend = "math_only"
    except AttributeError:
        sdpa_backend = "unknown"
    cuda_rt = torch_module.version.cuda or "unknown"
    return {
        "contract_id": "windows_cuda_deterministic_v1",
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "cuda_runtime": cuda_rt,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "sdpa_backend": sdpa_backend,
        "sdpa_flash_enabled": False,
        "sdpa_mem_efficient_enabled": False,
        "sdpa_math_enabled": True,
        "sdpa_cudnn_enabled": False,
        "torch_version": torch_module.__version__,
    }


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    receipt_path = Path(args.receipt).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = receipt_path.resolve()
    out_dir = Path(args.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    seed = args.seed

    if out_dir.exists():
        sys.exit(f"Output dir already exists (fail-closed): {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=False)
    start_utc = _utc_now()
    start_t = time.perf_counter()

    # --- CUDA setup ---
    # Must configure reproducibility BEFORE any torch.cuda operation, so
    # that build_production_smoke_plan's internal configure_cuda_reproducibility()
    # finds the module already registered and skips the "initialized too late" check.
    from f2_experiment.reproducibility import configure_cuda_reproducibility, prepare_cublas_workspace_config
    prepare_cublas_workspace_config()
    cuda_repro = configure_cuda_reproducibility(torch)
    if not torch.cuda.is_available():
        sys.exit("CUDA not available; this script requires CUDA.")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    # --- Seed ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f"[train_f2_seeded] seed={seed} device={device} gpu={gpu_name}")

    # --- Import F2 assembly ---
    from f2_experiment.assembly_model import (
        FROZEN_AUX_COEFFICIENTS,
        build_production_smoke_plan,
    )
    from f2_experiment.runner import S_SELF, ARM_ORDER, RunnerTelemetryHooks
    from f2_experiment.runner import run_paired_smoke

    receipt_document = json.loads(receipt_path.read_bytes())
    receipt_sha = _sha256_file(receipt_path)
    print(f"[train_f2_seeded] assembly receipt: {receipt_path.name} sha={receipt_sha[:12]}...")
    print(f"[train_f2_seeded] FROZEN_AUX_COEFFICIENTS: {FROZEN_AUX_COEFFICIENTS}")

    # --- Build smoke plan with new seed ---
    # Re-seed immediately before plan construction so the weight init matches.
    torch.manual_seed(seed)
    np.random.seed(seed)
    plan = build_production_smoke_plan(
        root,
        receipt_document,
        seed=seed,
        device=device,
        aux_coefficients=FROZEN_AUX_COEFFICIENTS,
    )
    print(f"[train_f2_seeded] plan built: seed={plan.seed} device={plan.device}")
    print(f"[train_f2_seeded] smoke_rows={len(plan.smoke_rows)} strafe_resets={len(plan.strafe_reset_original_indices)}")

    # Use the real G6 update function wired into the production plan.
    # For multi-seed we record telemetry but don't gate on it.
    hooks = RunnerTelemetryHooks(g6_update=plan.g6_update)

    # --- Run paired 128-update training ---
    print("[train_f2_seeded] starting 128-update paired training ...")
    train_start = time.perf_counter()
    result = run_paired_smoke(
        plan.smoke_rows,
        callbacks={arm: plan.arms[arm].callbacks for arm in ARM_ORDER},
        hooks=hooks,
        strafe_reset_original_indices=plan.strafe_reset_original_indices,
        expected_static_reset_original_indices=plan.expected_static_reset_original_indices,
        controller_config=plan.controller_config,
        require_audit_counters=True,
    )
    train_elapsed = time.perf_counter() - train_start
    print(f"[train_f2_seeded] training complete in {train_elapsed:.1f}s")

    # --- Save S-SELF checkpoint using the official save_arm_checkpoint ---
    from f2_experiment.assembly import save_arm_checkpoint
    s_self_payload = plan.arms[S_SELF].checkpoint_payload()
    ckpt_path = out_dir / f"checkpoint_update128_{S_SELF}.pt"
    ckpt_meta = save_arm_checkpoint(
        ckpt_path,
        arm=S_SELF,
        model_state=s_self_payload["model"],
        optimizer_state=s_self_payload.get("optimizer"),
        u_pre=128,
        assembly_receipt_sha256=receipt_sha,
    )
    ckpt_sha = ckpt_meta["file_sha256"]
    print(f"[train_f2_seeded] saved {ckpt_path.name} sha={ckpt_sha[:12]}...")

    end_utc = _utc_now()
    elapsed_total = time.perf_counter() - start_t

    # --- Run receipt ---
    receipt_doc = {
        "schema_version": 1,
        "analysis_class": "f2_multiseed_training_v1",
        "method": "SA-Hstar",
        "architecture_lock": "L1+D2+AP2+F2",
        "seed": seed,
        "updates": 128,
        "processed_samples": 256,
        "train_jsonl_sha256": "1715b3ce2c65df7caaa41d4a3f2f1eba61746e4b33158ae3267ad1477e96dd36",
        "val_rows": 2848,
        "lambda_freeze_receipt": "20260719_f2_seeded_cal_lambda_freeze_receipt.json",
        "frozen_aux_coefficients": FROZEN_AUX_COEFFICIENTS,
        "assembly_receipt_sha256": receipt_sha,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "cuda_reproducibility": cuda_repro,
        "gpu": gpu_name,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "elapsed_seconds": round(elapsed_total, 1),
        "internal_test_opened": False,
        "note": "multi-seed extension; lambda frozen from seed-0 CAL; "
                "only torch/numpy RNG seed changed; no G6/G7/G8/G9 gate eval",
    }
    receipt_path_out = out_dir / "run_receipt.json"
    receipt_path_out.write_text(
        json.dumps(receipt_doc, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[train_f2_seeded] receipt written: {receipt_path_out}")
    print(f"[train_f2_seeded] done  seed={seed}  elapsed={elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
