#!/usr/bin/env python3
"""PFEM-Harness training script.

Usage:
    # Stage 1: train harness on pre-cached visual tokens
    python scripts/train_pfem.py --train_json data/tracking_train.jsonl --epochs 1
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import OpenTrackVLA, ModelConfig, JsonTrackingDataset, DataConfig, collate_batch
from harness.harness_wrapper import PFEMHarness
from harness.core.event_sampling import compute_event_sampling_weights, weighted_event_fraction
from local_weights import default_qwen_candidates, resolve_local_model_path


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out_dir", type=str, default="ckpts_pfem")
    ap.add_argument("--vision_feat_dim", type=int, default=1536)
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=8)
    ap.add_argument("--label_mode", choices=("absolute", "step_action"), default=None)
    ap.add_argument("--lambda_yaw", type=float, default=2.0)
    ap.add_argument("--aux_delta_vel", action="store_true")
    ap.add_argument(
        "--balance_sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use transition-event stratified sampling (default: enabled).",
    )
    ap.add_argument("--cache_root", type=str, default=None)
    ap.add_argument(
        "--base_hf_model_dir",
        type=str,
        default=None,
        help="Optional OpenTrackVLA HuggingFace checkpoint directory to initialize the base planner.",
    )
    ap.add_argument("--qwen_model_path", type=str, default=None,
                    help="Optional local Qwen/Qwen3-0.6B directory for offline training.")
    args = ap.parse_args(argv)
    if args.lambda_yaw < 0:
        ap.error("--lambda_yaw must be >= 0")
    if args.aux_delta_vel and args.label_mode == "absolute":
        ap.error("--aux_delta_vel requires --label_mode step_action")
    return args


def _single_manifest_value(value, field):
    if isinstance(value, list):
        unique = list(dict.fromkeys(value))
        if len(unique) != 1:
            raise ValueError(f"Training requires one {field} value, got {unique}")
        return unique[0]
    return value


def build_checkpoint_meta(args):
    manifest_path = Path(str(args.train_json) + ".manifest.json")
    manifest = {}
    manifest_hash = None
    if manifest_path.exists():
        raw = manifest_path.read_bytes()
        manifest_hash = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw.decode("utf-8"))
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError(f"Unsupported data manifest schema: {manifest_path}")
    else:
        print(
            f"!!! [train_pfem] WARNING: no sidecar manifest at {manifest_path}; "
            "using legacy absolute-label metadata defaults"
        )

    fps = _single_manifest_value(manifest.get("fps", 10.0), "fps")
    dt = _single_manifest_value(manifest.get("dt", 1.0 / float(fps)), "dt")
    manifest_label_mode = _single_manifest_value(manifest.get("label_mode", "absolute"), "label_mode")
    if args.label_mode is not None and str(args.label_mode) != str(manifest_label_mode):
        raise ValueError(
            f"--label_mode={args.label_mode} conflicts with manifest label_mode={manifest_label_mode}"
        )
    label_mode = args.label_mode or manifest_label_mode
    args.label_mode = str(label_mode)
    if args.aux_delta_vel and args.label_mode != "step_action":
        raise ValueError("--aux_delta_vel requires step_action labels")
    action_semantics = _single_manifest_value(
        manifest.get("action_semantics", "unknown_legacy"),
        "action_semantics",
    )
    delta_scale = _single_manifest_value(manifest.get("delta_scale", 1.0), "delta_scale")
    return {
        "schema_version": 1,
        "n_waypoints": int(args.n_waypoints),
        "history": int(args.history),
        "dt": float(dt),
        "fps": float(fps),
        "label_mode": str(label_mode),
        "action_semantics": str(action_semantics),
        "delta_scale": float(delta_scale),
        "aux_delta_vel": bool(args.aux_delta_vel),
        "lambda_yaw": float(args.lambda_yaw),
        "data_manifest_hash": manifest_hash,
        "train_args": dict(vars(args)),
    }


def build_training_sampler(dataset, enabled=True):
    if not enabled:
        return None
    transition_types = [
        dataset.get_example(index).get("transition_type", "other")
        for index in range(len(dataset))
    ]
    weights = compute_event_sampling_weights(transition_types)
    if not weights:
        return None
    if all(abs(float(weight) - 1.0) < 1e-12 for weight in weights):
        print("[train_pfem] balanced sampling: dataset already meets the event target")
        return None
    event_fraction = weighted_event_fraction(transition_types, weights)
    max_weight = max(weights)
    print(
        f"[train_pfem] balanced sampling: expected_event_fraction={event_fraction:.3f} "
        f"max_weight={max_weight:.2f}"
    )
    if event_fraction < 0.4:
        print(
            "!!! [train_pfem] WARNING: the 10x sampling cap prevents the current dataset "
            "from reaching the 40% event target"
        )
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ["QWEN_MODEL_PATH"] = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=args.qwen_model_path,
        env_var="QWEN_MODEL_PATH",
        candidates=default_qwen_candidates(),
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train_pfem] device={device}")

    checkpoint_meta = build_checkpoint_meta(args)

    # Build base model (frozen LLM). Prefer the official OpenTrackVLA checkpoint
    # when provided, instead of starting from a bare Qwen backbone.
    if args.base_hf_model_dir:
        from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

        print(f"[train_pfem] loading base HF checkpoint: {args.base_hf_model_dir}")
        base_hf_dir = Path(args.base_hf_model_dir)
        hf_config = OpenTrackVLAConfig.from_pretrained(str(base_hf_dir), local_files_only=True)
        hf_config.llm_name = os.environ["QWEN_MODEL_PATH"]
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
            llm_name=os.environ["QWEN_MODEL_PATH"],
            n_waypoints=args.n_waypoints,
            freeze_llm=True,
        )
        base = OpenTrackVLA(mcfg, vision_feat_dim=args.vision_feat_dim)
    base = base.to(device)
    # The official base checkpoint may be authoritative for n_waypoints.
    checkpoint_meta = build_checkpoint_meta(args)

    # Wrap with PFEM-Harness
    model = PFEMHarness(
        base,
        label_mode=args.label_mode,
        dt=checkpoint_meta["dt"],
        lambda_yaw=args.lambda_yaw,
        aux_delta_vel=args.aux_delta_vel,
    ).to(device)

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
    sampler = build_training_sampler(ds, args.balance_sampling)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )

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
                prev_action=batch["prev_action"].to(device),
            )

            # Build GT dict from batch
            gt = {
                "waypoints": batch["waypoints"].to(device),
                "step_actions": batch["step_actions"].to(device),
                "delta_vel": batch["delta_vel"].to(device),
                "theta_idx": torch.zeros(B, dtype=torch.long, device=device),
                "dist_idx": torch.zeros(B, dtype=torch.long, device=device),
                "invalid": torch.zeros(B, device=device),
                "valid_mask": batch["valid_mask"].to(device),
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
            "meta": checkpoint_meta,
        }, ckpt_path)
        print(f"  saved {ckpt_path}")

    print("[train_pfem] done.")


if __name__ == "__main__":
    main()
