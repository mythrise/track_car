#!/usr/bin/env python3
# ruff: noqa: E402
"""Exp1: Training-time No-Auxiliary-Head Ablation (seed 0).

Trains SA-Hstar with ALL auxiliary heads disabled at training time:
L_aux is zeroed out before backprop (aux forward pass still runs,
no gradient from polar/future/verify losses). Everything else is
identical to the official Harness seed-0 run:
  - Same frozen assembly receipt
  - Same seed=0 (torch + numpy RNG)
  - Same 128 optimizer updates / 256 processed samples
  - Same train.jsonl (SHA 1715b3ce...)
  - Same frozen lambda values

Purpose: If NoAux H1 > Harness H1, the 61.5% improvement is caused by
the auxiliary learning objectives, not model capacity or other factors.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

S_SELF = "S-SELF"
S_CTRL = "S-CTRL"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="NoAux ablation training (seed 0)")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    parser.add_argument(
        "--receipt",
        default="experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: matched128/NoAux_seed{seed})",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = (
            f"experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
            f"/matched128/NoAux_seed{args.seed}"
        )

    root = Path(args.project_root).resolve()
    receipt_path = root / args.receipt
    out_dir = root / args.output_dir

    start_utc = _utc_now()
    start_t = time.perf_counter()

    seed: int = args.seed
    print(f"[noaux_train] No-Auxiliary-Head Ablation, seed={seed}")

    # --- 1. CUDA reproducibility (must be before any torch.cuda call) ---
    from f2_experiment.reproducibility import (
        configure_cuda_reproducibility,
        prepare_cublas_workspace_config,
    )
    prepare_cublas_workspace_config()
    configure_cuda_reproducibility()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[noaux_train] GPU: {gpu_name}")

    # --- 2. Load receipt ---
    if not receipt_path.exists():
        sys.exit(f"Receipt not found: {receipt_path}")
    receipt_document = json.loads(receipt_path.read_text("utf-8"))
    receipt_sha = _sha256_file(receipt_path)
    print(f"[noaux_train] receipt sha={receipt_sha[:16]}...")

    # --- 3. Verify output dir is clean ---
    if out_dir.exists() and any(out_dir.iterdir()):
        sys.exit(f"Output dir is not empty (fail-closed): {out_dir}")

    # --- 4. Build production smoke plan (seed=0, identical to official run) ---
    print(f"[noaux_train] building smoke plan (seed={seed}) ...")
    from f2_experiment.assembly_model import build_production_smoke_plan
    plan = build_production_smoke_plan(
        root,
        receipt_document,
        seed=seed,
        device=device,
    )
    print(f"[noaux_train] plan built: seed={plan.seed} device={plan.device}")
    print(f"[noaux_train] smoke_rows={len(plan.smoke_rows)}")

    # --- 5. Patch both arms: replace aux_forward with a zero-returning wrapper ---
    # ArmCallbacks and SmokeArmAssembly are both frozen dataclasses;
    # dataclasses.replace() creates a new instance with the swapped field.
    from f2_experiment.runner import AuxForwardResult
    from f2_experiment.assembly_model import SmokeArmAssembly

    new_arms: dict[str, Any] = {}
    for arm_name in (S_CTRL, S_SELF):
        old_assembly = plan.arms[arm_name]
        real_aux_forward = old_assembly.callbacks.aux_forward

        # Wrap: run real aux_forward so _scratch.aux_loss is populated
        # (required by backward's null-check), then return a fresh zero tensor
        # to the runner so L_aux contributes zero to row_loss and no aux
        # gradient enters the backward pass.
        def make_noaux_wrapper(real_fn: Any) -> Any:
            def noaux_aux_forward(
                features: Any, aux_targets: Any, event: Any
            ) -> AuxForwardResult:
                result = real_fn(features, aux_targets, event)
                # _scratch.aux_loss now set (satisfies backward null-check).
                # Return zero so runner computes: row_loss = 0 + 0.5*L1 + 0.5*L2
                return AuxForwardResult(loss=result.loss.new_zeros(()))
            return noaux_aux_forward

        patched_callbacks = dataclasses.replace(
            old_assembly.callbacks,
            aux_forward=make_noaux_wrapper(real_aux_forward),
        )
        new_arms[arm_name] = dataclasses.replace(
            old_assembly,
            callbacks=patched_callbacks,
        )

    # SmokeAssemblyPlan is frozen — replace() creates a new plan with patched arms
    plan = dataclasses.replace(plan, arms=new_arms)
    print("[noaux_train] plan rebuilt with zero-aux arms")

    # --- 6. Run paired 128-update training ---
    from f2_experiment.runner import (
        RunnerTelemetryHooks,
        run_paired_smoke,
        ARM_ORDER,
    )

    hooks = RunnerTelemetryHooks(g6_update=plan.g6_update)

    print("[noaux_train] starting 128-update paired training (L_aux=0 for all rows) ...")
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
    print(f"[noaux_train] training complete in {train_elapsed:.1f}s")

    # --- 7. Save checkpoint ---
    from f2_experiment.assembly import save_arm_checkpoint
    out_dir.mkdir(parents=True, exist_ok=True)

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
    print(f"[noaux_train] checkpoint saved sha={ckpt_sha[:16]}...")

    end_utc = _utc_now()
    total_elapsed = time.perf_counter() - start_t

    # --- 8. Write run receipt ---
    train_sha = hashlib.sha256(
        (root / "data/collected_v1/datasets/train.jsonl").read_bytes()
    ).hexdigest()

    receipt = {
        "schema_version": 1,
        "experiment": "NoAux_ablation",
        "method": "NoAux",
        "seed": 0,
        "architecture_lock": "L1+D2+AP2+F2",
        "aux_heads_disabled": True,
        "aux_loss_value": "0 (zeroed before backprop, forward pass intact)",
        "updates": 128,
        "processed_samples": 256,
        "train_jsonl_sha256": train_sha,
        "val_rows": 2848,
        "lambda_freeze_receipt": "20260719_f2_seeded_cal_lambda_freeze_receipt.json",
        "receipt_sha256": receipt_sha,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "gpu": gpu_name,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "training_elapsed_s": round(train_elapsed, 1),
        "total_elapsed_s": round(total_elapsed, 1),
        "note": (
            "Ablation: same arch as Harness seed-0 but L_aux=0. "
            "Aux heads still in forward pass but produce no gradient. "
            "Purpose: isolate contribution of auxiliary learning objectives."
        ),
    }
    receipt_out = out_dir / "run_receipt.json"
    receipt_out.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[noaux_train] receipt written to {receipt_out}")
    print(f"[noaux_train] DONE total={total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
