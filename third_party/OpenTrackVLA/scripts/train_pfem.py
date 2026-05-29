#!/usr/bin/env python3
"""PFEM-Harness training script.

Usage:
    # Stage 1: train harness on pre-cached visual tokens
    python scripts/train_pfem.py --train_json data/tracking_train.jsonl --epochs 1

    # Stage 2: LoRA finetune (add --lora)
    python scripts/train_pfem.py --train_json data/tracking_train.jsonl --epochs 1 --lora
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import OpenTrackVLA, ModelConfig, JsonTrackingDataset, DataConfig, collate_batch
from harness.harness_wrapper import PFEMHarness


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", type=str, default="ckpts_pfem")
    ap.add_argument("--vision_feat_dim", type=int, default=1536)
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--cache_root", type=str, default=None)
    ap.add_argument(
        "--base_hf_model_dir",
        type=str,
        default=None,
        help="Optional OpenTrackVLA HuggingFace checkpoint directory to initialize the base planner.",
    )
    ap.add_argument("--qwen_model_path", type=str, default=None,
                    help="Optional local Qwen/Qwen3-0.6B directory for offline training.")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.qwen_model_path:
        os.environ["QWEN_MODEL_PATH"] = args.qwen_model_path
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[train_pfem] device={device}")

    # Build base model (frozen LLM). Prefer the official OpenTrackVLA checkpoint
    # when provided, instead of starting from a bare Qwen backbone.
    if args.base_hf_model_dir:
        from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

        print(f"[train_pfem] loading base HF checkpoint: {args.base_hf_model_dir}")
        base_hf_dir = Path(args.base_hf_model_dir)
        hf_config = OpenTrackVLAConfig.from_pretrained(str(base_hf_dir))
        qwen_model_path = os.environ.get("QWEN_MODEL_PATH", "").strip()
        if qwen_model_path:
            hf_config.llm_name = qwen_model_path
        hf_model = OpenTrackVLAForWaypoint(hf_config)
        state_path = base_hf_dir / "model.safetensors"
        if not state_path.exists():
            raise FileNotFoundError(f"Missing OpenTrackVLA HF weights: {state_path}")
        missing, unexpected = hf_model.load_state_dict(load_safetensors(str(state_path)), strict=False)
        print(
            f"[train_pfem] loaded HF state dict: "
            f"{len(missing)} missing, {len(unexpected)} unexpected"
        )
        base = hf_model.model
        args.n_waypoints = int(base.cfg.n_waypoints)
        print(f"[train_pfem] loaded official base planner, n_waypoints={args.n_waypoints}")
    else:
        mcfg = ModelConfig(
            llm_name=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-0.6B"),
            n_waypoints=args.n_waypoints,
            freeze_llm=True,
        )
        base = OpenTrackVLA(mcfg, vision_feat_dim=args.vision_feat_dim)
    base = base.to(device)

    # Wrap with PFEM-Harness
    model = PFEMHarness(base).to(device)

    # Count params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train_pfem] trainable: {trainable/1e6:.2f}M / total: {total/1e6:.2f}M")

    # Dataset
    ds = JsonTrackingDataset(DataConfig(
        train_json=args.train_json,
        n_waypoints=args.n_waypoints,
        history=args.history,
        cache_root=args.cache_root,
    ))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=collate_batch)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4)

    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for batch in loader:
            B = batch["coarse_tokens"].size(0)
            state = model.init_state(B, device)

            out = model.forward_step(
                coarse_tokens=batch["coarse_tokens"],
                coarse_tidx=batch["coarse_tidx"],
                fine_tokens=batch["fine_tokens"],
                fine_tidx=batch["fine_tidx"],
                instructions=batch["instruction"],
                prev_state=state,
            )

            # Build GT dict from batch
            gt = {
                "waypoints": batch["waypoints"].to(device),
                "theta_idx": torch.zeros(B, dtype=torch.long, device=device),
                "dist_idx": torch.zeros(B, dtype=torch.long, device=device),
                "invalid": torch.zeros(B, device=device),
            }
            # If batch has polar labels, use them
            if "polar_theta_idx" in batch:
                gt["theta_idx"] = batch["polar_theta_idx"].to(device)
                gt["dist_idx"] = batch["polar_dist_idx"].to(device)
                gt["invalid"] = batch["polar_invalid"].to(device)

            losses = model.compute_losses(out, gt)
            loss = losses["loss"]

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optim.step()

            epoch_loss += loss.item()
            steps += 1
            if steps % 10 == 0:
                print(f"  epoch {epoch} step {steps}: loss={loss.item():.4f} "
                      f"L_track={losses['L_track']:.4f} L_cot={losses['L_cot']:.4f}")

        avg = epoch_loss / max(1, steps)
        print(f"[epoch {epoch}] avg_loss={avg:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"pfem_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": {k: v for k, v in model.state_dict().items()
                           if not k.startswith("base.llm.")},
            "optimizer_state": optim.state_dict(),
            "loss": avg,
        }, ckpt_path)
        print(f"  saved {ckpt_path}")

    print("[train_pfem] done.")


if __name__ == "__main__":
    main()
