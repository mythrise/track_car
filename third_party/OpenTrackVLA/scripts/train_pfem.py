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
import random
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
from cache_gridpool import atomic_torch_save
from harness.harness_wrapper import PFEMHarness
from harness.core.event_sampling import compute_event_sampling_weights, weighted_event_fraction
from harness.sequence_state import continues_sequence, detach_state, sample_sequence_key
from experiment_binding import (
    bind_hf_model_artifact,
    sha256_artifact,
    verify_vision_cache,
)
from experiment_logging import JsonlMetricLogger
from local_weights import default_qwen_candidates, resolve_local_model_path
from validation_metrics import BalancedControlAccumulator, waypoints_to_step_actions


STEP_ACTION_REQUIRED_FIELDS = ("step_actions", "prev_action", "delta_vel")
_ACTIVE_METRIC_LOGGER = None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_json", required=True)
    ap.add_argument("--val_json", default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--base_lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--max_optimizer_updates", type=int, default=0)
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
    ap.add_argument("--val_cache_root", type=str, default=None)
    ap.add_argument(
        "--base_hf_model_dir",
        type=str,
        default=None,
        help="Optional OpenTrackVLA HuggingFace checkpoint directory to initialize the base planner.",
    )
    ap.add_argument("--qwen_model_path", type=str, default=None,
                    help="Optional local Qwen/Qwen3-0.6B directory for offline training.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--grad_accum_steps",
        type=int,
        default=None,
        help="Defaults to 2 for rolling batch-1 training, otherwise 1.",
    )
    ap.add_argument(
        "--save_optimizer", action=argparse.BooleanOptionalAction, default=False
    )
    ap.add_argument(
        "--state_mode",
        choices=("stateless", "rolling"),
        default="rolling",
        help="Carry TIM/Event state across consecutive clip samples when rolling.",
    )
    ap.add_argument("--disable_cot_loss", action="store_true")
    ap.add_argument("--disable_tim", action="store_true")
    ap.add_argument("--disable_future", action="store_true")
    ap.add_argument("--disable_verifier", action="store_true")
    ap.add_argument("--disable_events", action="store_true")
    ap.add_argument("--disable_orchestrator", action="store_true")
    args = ap.parse_args(argv)
    if args.lambda_yaw < 0:
        ap.error("--lambda_yaw must be >= 0")
    if args.aux_delta_vel and args.label_mode == "absolute":
        ap.error("--aux_delta_vel requires --label_mode step_action")
    if args.state_mode == "rolling" and args.batch_size != 1:
        ap.error("--state_mode rolling currently requires --batch_size 1")
    if args.grad_accum_steps is None:
        args.grad_accum_steps = 2 if args.state_mode == "rolling" else 1
    if args.grad_accum_steps <= 0:
        ap.error("--grad_accum_steps must be positive")
    if args.max_optimizer_updates < 0:
        ap.error("--max_optimizer_updates must be >= 0")
    return args


def _single_manifest_value(value, field):
    if isinstance(value, list):
        unique = list(dict.fromkeys(value))
        if len(unique) != 1:
            raise ValueError(f"Training requires one {field} value, got {unique}")
        return unique[0]
    return value


def inspect_training_jsonl(dataset_path, label_mode):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Training JSONL does not exist: {dataset_path}")
    digest = hashlib.sha256()
    sample_count = 0
    with dataset_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            sample_count += 1
            if label_mode != "step_action":
                continue
            try:
                sample = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid JSON in step_action training sample at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(sample, dict):
                raise ValueError(
                    f"step_action training sample at {dataset_path}:{line_number} must be an object"
                )
            missing = [field for field in STEP_ACTION_REQUIRED_FIELDS if field not in sample]
            if missing:
                raise ValueError(
                    f"step_action training sample at {dataset_path}:{line_number} "
                    f"missing required fields: {', '.join(missing)}"
                )
    return digest.hexdigest(), sample_count


def build_checkpoint_meta(args):
    manifest_path = Path(str(args.train_json) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Training data manifest is required for content binding: {manifest_path}"
        )
    raw = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw.decode("utf-8"))
    if int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(f"Unsupported data manifest schema: {manifest_path}")

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
    expected_data_hash = manifest.get("data_jsonl_sha256")
    expected_sample_count = manifest.get("sample_count")
    if not isinstance(expected_data_hash, str) or not expected_data_hash:
        raise ValueError(f"Data manifest missing data_jsonl_sha256: {manifest_path}")
    if not isinstance(expected_sample_count, int) or expected_sample_count < 0:
        raise ValueError(f"Data manifest missing valid sample_count: {manifest_path}")
    data_jsonl_sha256, sample_count = inspect_training_jsonl(args.train_json, args.label_mode)
    if data_jsonl_sha256 != expected_data_hash:
        raise ValueError(
            f"Training JSONL sha256 mismatch for {args.train_json}: "
            f"manifest={expected_data_hash}, actual={data_jsonl_sha256}"
        )
    if sample_count != expected_sample_count:
        raise ValueError(
            f"Training JSONL sample_count mismatch for {args.train_json}: "
            f"manifest={expected_sample_count}, actual={sample_count}"
        )
    state_mode = str(getattr(args, "state_mode", "stateless"))
    return {
        "schema_version": 1,
        "model_family": "pfem_harness",
        "experiment_id": "H0" if state_mode == "rolling" else "H0-S",
        "n_waypoints": int(args.n_waypoints),
        "history": int(args.history),
        "dt": float(dt),
        "fps": float(fps),
        "label_mode": str(label_mode),
        "action_semantics": str(action_semantics),
        "delta_scale": float(delta_scale),
        "aux_delta_vel": bool(args.aux_delta_vel),
        "lambda_yaw": float(args.lambda_yaw),
        "seed": int(getattr(args, "seed", 0)),
        "state_mode": state_mode,
        "sampling_policy": (
            "weighted_random"
            if bool(getattr(args, "balance_sampling", False))
            and state_mode == "stateless"
            else "ordered_jsonl"
        ),
        "grad_accum_steps": int(getattr(args, "grad_accum_steps", 1)),
        "batch_size": int(getattr(args, "batch_size", 1)),
        "effective_batch_size": int(
            getattr(args, "batch_size", 1) * getattr(args, "grad_accum_steps", 1)
        ),
        "base_lr": float(getattr(args, "base_lr", 2e-5)),
        "head_lr": float(getattr(args, "lr", 3e-4)),
        "weight_decay": float(getattr(args, "weight_decay", 1e-4)),
        "grad_clip": float(getattr(args, "grad_clip", 1.0)),
        "data_manifest_hash": manifest_hash,
        "data_jsonl_sha256": data_jsonl_sha256,
        "sample_count": sample_count,
        "training_source_raw_dirs": list(
            (manifest.get("statistics") or {}).get("source_raw_dirs") or []
        ),
        "train_args": dict(vars(args)),
    }


def inspect_validation_binding(dataset_path):
    if not dataset_path:
        return None
    path = Path(dataset_path)
    manifest_path = Path(str(path) + ".manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Validation manifest is required: {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    actual_hash, rows = inspect_training_jsonl(path, str(manifest.get("label_mode", "absolute")))
    if actual_hash != manifest.get("data_jsonl_sha256"):
        raise ValueError(f"Validation JSONL sha256 mismatch: {path}")
    if rows != int(manifest.get("sample_count", -1)):
        raise ValueError(f"Validation JSONL sample_count mismatch: {path}")
    return {
        "data_manifest_hash": hashlib.sha256(raw).hexdigest(),
        "data_jsonl_sha256": actual_hash,
        "sample_count": rows,
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


def _main_impl():
    global _ACTIVE_METRIC_LOGGER
    args = parse_args()
    if not args.cache_root:
        raise ValueError("--cache_root is required for reproducible training")
    if not args.base_hf_model_dir:
        raise ValueError("--base_hf_model_dir is required for matched PFEM training")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    qwen_path = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=args.qwen_model_path,
        env_var="QWEN_MODEL_PATH",
        candidates=default_qwen_candidates(),
    )
    os.environ["QWEN_MODEL_PATH"] = qwen_path
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[train_pfem] device={device}")

    checkpoint_meta = build_checkpoint_meta(args)
    validation_binding = inspect_validation_binding(args.val_json)
    if validation_binding is not None:
        checkpoint_meta["validation"] = validation_binding
    cache_datasets = [args.train_json]
    if args.val_json and not args.val_cache_root:
        cache_datasets.append(args.val_json)
    cache_info = verify_vision_cache(
        args.cache_root, cache_datasets, verify_payload=True
    )
    checkpoint_meta.update(
        {
            "vision_cache_root": cache_info["cache_root"],
            "vision_cache_manifest_sha256": cache_info[
                "cache_manifest_sha256"
            ],
            "vision_cache_provenance_sha256": cache_info[
                "cache_provenance_sha256"
            ],
            "vision_cache_token_payload_sha256": cache_info[
                "token_payload_sha256"
            ],
            "dino_model_sha256": cache_info["dino_model_sha256"],
            "siglip_model_sha256": cache_info["siglip_model_sha256"],
            "qwen_model_sha256": sha256_artifact(qwen_path),
        }
    )
    if args.val_json and args.val_cache_root:
        checkpoint_meta["validation_vision_cache"] = verify_vision_cache(
            args.val_cache_root, [args.val_json], verify_payload=True
        )

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
    checkpoint_meta["n_waypoints"] = int(args.n_waypoints)
    checkpoint_meta["train_args"] = dict(vars(args))

    # Wrap with PFEM-Harness
    model = PFEMHarness(
        base,
        label_mode=args.label_mode,
        dt=checkpoint_meta["dt"],
        lambda_yaw=args.lambda_yaw,
        aux_delta_vel=args.aux_delta_vel,
        use_cot_loss=not args.disable_cot_loss,
        use_tim=not args.disable_tim,
        use_future=not args.disable_future,
        use_verifier=not args.disable_verifier,
        use_events=not args.disable_events,
        use_orchestrator=not args.disable_orchestrator,
    ).to(device)
    checkpoint_meta["trainable_base_modules"] = ["proj", "planner"]
    checkpoint_meta["context_mode"] = "zero_init_residual_over_base_hidden"
    checkpoint_meta["verifier_delta_mode"] = "differentiable_soft_residual"
    disabled = [
        name
        for name, flag in {
            "cot_loss": args.disable_cot_loss,
            "tim": args.disable_tim,
            "future": args.disable_future,
            "verifier": args.disable_verifier,
            "events": args.disable_events,
            "orchestrator": args.disable_orchestrator,
        }.items()
        if flag
    ]
    checkpoint_meta["disabled_components"] = disabled
    family_id = "H0" if args.state_mode == "rolling" else "H0-S"
    checkpoint_meta["experiment_id"] = (
        family_id if not disabled else family_id + "-ablation:" + ",".join(disabled)
    )
    if args.base_hf_model_dir:
        base_model_binding = bind_hf_model_artifact(args.base_hf_model_dir)
        checkpoint_meta["base_hf_model_dir"] = str(
            Path(args.base_hf_model_dir).expanduser().resolve()
        )
        checkpoint_meta["base_model_sha256"] = base_model_binding[
            "artifact_sha256"
        ]
        checkpoint_meta["base_model_artifact"] = base_model_binding

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
        require_cached_tokens=True,
    ))
    if args.state_mode == "rolling" and args.balance_sampling:
        print("[train_pfem] rolling state requires ordered samples; disabling balanced sampling")
    sampler = build_training_sampler(
        ds, args.balance_sampling and args.state_mode == "stateless"
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None and args.state_mode == "stateless"),
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )
    val_loader = None
    if args.val_json:
        val_ds = JsonTrackingDataset(DataConfig(
            train_json=args.val_json,
            n_waypoints=args.n_waypoints,
            history=args.history,
            cache_root=args.val_cache_root or args.cache_root,
            require_cached_tokens=True,
        ))
        val_loader = DataLoader(
            val_ds,
            batch_size=1 if args.state_mode == "rolling" else args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )

    base_parameters = []
    head_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (base_parameters if name.startswith("base.") else head_parameters).append(
            parameter
        )
    optim = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.base_lr},
            {"params": head_parameters, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    checkpoint_meta["checkpoint_selection"] = {
        "metric": "validation_episode_macro_BCE@1",
        "mode": "min",
        "rule": "strict_improvement_earliest_epoch",
    }

    def optimizer_step(accumulated_batches, context):
        if accumulated_batches < args.grad_accum_steps:
            correction = args.grad_accum_steps / float(accumulated_batches)
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        grad_info = metric_logger.clip_grad_norm_and_check(
            [p for p in model.parameters() if p.requires_grad],
            args.grad_clip,
            context=context,
        )
        optim.step()
        optim.zero_grad(set_to_none=True)
        metric_logger.log({"phase": "optimizer", **context, **grad_info})
        return grad_info

    os.makedirs(args.out_dir, exist_ok=True)
    metric_logger = JsonlMetricLogger(args.out_dir, device=device)
    metric_logger.start_run(
        args=dict(vars(args)),
        checkpoint_meta=checkpoint_meta,
        total_params=total,
        trainable_params=trainable,
    )
    _ACTIVE_METRIC_LOGGER = metric_logger

    def make_ground_truth(batch):
        B = batch["coarse_tokens"].size(0)
        gt = {
            "waypoints": batch["waypoints"].to(device),
            "step_actions": batch["step_actions"].to(device),
            "delta_vel": batch["delta_vel"].to(device),
            "theta_idx": batch["polar_theta_idx"].to(device),
            "dist_idx": batch["polar_dist_idx"].to(device),
            "invalid": batch["polar_invalid"].to(device),
            "valid_mask": batch["valid_mask"].to(device),
        }
        for horizon in (4, 8, 16):
            for field in ("fut_valid", "fut_vis", "fut_theta_idx", "fut_dist_idx"):
                key = f"{field}_{horizon}"
                if key in batch:
                    gt[key] = batch[key].to(device)
        return gt

    @torch.inference_mode()
    def evaluate_validation():
        if val_loader is None:
            return None
        model.eval()
        total = 0.0
        count = 0
        selector = BalancedControlAccumulator()
        state = None
        previous_key = None
        for batch in val_loader:
            B = batch["coarse_tokens"].size(0)
            current_key = sample_sequence_key(batch) if args.state_mode == "rolling" else None
            carry = args.state_mode == "rolling" and current_key is not None and not current_key[2]
            batch_state = state
            if not carry or not continues_sequence(previous_key, current_key):
                batch_state = model.init_state(B, device)
            out = model.forward_step(
                coarse_tokens=batch["coarse_tokens"],
                coarse_tidx=batch["coarse_tidx"],
                fine_tokens=batch["fine_tokens"],
                fine_tidx=batch["fine_tidx"],
                instructions=batch["instruction"],
                prev_state=batch_state,
                yaw_hist=batch["yaw_hist"].to(device),
                yaw_curr=batch["yaw_curr"].to(device),
                prev_action=batch["prev_action"].to(device),
            )
            total += float(model.compute_losses(out, make_ground_truth(batch))["loss"].item())
            count += 1
            predicted_actions = (
                out["step_actions"].detach().cpu().numpy()
                if args.label_mode == "step_action"
                else waypoints_to_step_actions(
                    out["waypoints"].detach().cpu().numpy(),
                    float(checkpoint_meta["dt"]),
                )
            )
            selector.add(
                predicted_actions,
                batch["step_actions"].cpu().numpy(),
                batch["valid_mask"].cpu().numpy(),
                batch["command"],
                batch["source_raw_dir"],
            )
            if carry:
                state = detach_state(out["new_state"])
                previous_key = current_key
        selection = selector.compute()
        if selection["value"] is None:
            raise ValueError("validation BCE@1 has no supported episode-command cells")
        return {
            "selection_bce_at1": float(selection["value"]),
            "family_loss": total / max(1, count),
            "selection_detail": selection,
        }

    def save_checkpoint(
        path, epoch, loss, *, checkpoint_role="epoch", selected_value=None
    ):
        saved_meta = dict(checkpoint_meta)
        saved_meta.update(
            {
                "checkpoint_role": str(checkpoint_role),
                "selection_verified": checkpoint_role == "best_validation",
                "selected_epoch": (
                    int(epoch) if checkpoint_role == "best_validation" else None
                ),
                "selected_value": (
                    float(selected_value)
                    if checkpoint_role == "best_validation"
                    else None
                ),
            }
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": {k: v for k, v in model.state_dict().items()
                           if not k.startswith("base.llm.")},
            "loss": loss,
            "meta": saved_meta,
        }
        if args.save_optimizer:
            checkpoint["optimizer_state"] = optim.state_dict()
        atomic_torch_save(checkpoint, path)

    processed_samples = 0
    optimizer_updates = 0
    global_step = 0
    best_val = float("inf")
    last_epoch = None
    last_average = None
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        epoch_start_samples = processed_samples
        model.train()
        epoch_loss = 0.0
        steps = 0
        rolling_state = None
        previous_key = None
        accumulated_batches = 0
        last_train_record = None
        optim.zero_grad(set_to_none=True)
        for batch in loader:
            B = batch["coarse_tokens"].size(0)
            current_key = (
                sample_sequence_key(batch) if args.state_mode == "rolling" else None
            )
            carry_forward = (
                args.state_mode == "rolling" and current_key is not None and not current_key[2]
            )
            batch_state = rolling_state
            if (
                not carry_forward
                or not continues_sequence(previous_key, current_key)
            ):
                batch_state = model.init_state(B, device)

            out = model.forward_step(
                coarse_tokens=batch["coarse_tokens"],
                coarse_tidx=batch["coarse_tidx"],
                fine_tokens=batch["fine_tokens"],
                fine_tidx=batch["fine_tidx"],
                instructions=batch["instruction"],
                prev_state=batch_state,
                yaw_hist=batch["yaw_hist"].to(device),
                yaw_curr=batch["yaw_curr"].to(device),
                prev_action=batch["prev_action"].to(device),
            )

            losses = model.compute_losses(out, make_ground_truth(batch))
            loss = losses["loss"]
            metric_logger.check_finite_losses(
                losses,
                context={"epoch": int(epoch), "micro_step": int(global_step + 1)},
            )

            (loss / float(args.grad_accum_steps)).backward()
            accumulated_batches += 1
            grad_info = None
            if accumulated_batches == args.grad_accum_steps:
                grad_info = optimizer_step(
                    accumulated_batches,
                    {
                        "epoch": int(epoch),
                        "micro_step": int(global_step + 1),
                        "optimizer_updates": int(optimizer_updates + 1),
                        "processed_samples": int(processed_samples + B),
                    },
                )
                accumulated_batches = 0
                optimizer_updates += 1

            if carry_forward:
                rolling_state = detach_state(out["new_state"])
                previous_key = current_key

            epoch_loss += loss.item()
            steps += 1
            global_step += 1
            processed_samples += int(B)
            last_train_record = {
                "phase": "train",
                "epoch": int(epoch),
                "micro_step": int(global_step),
                "optimizer_updates": int(optimizer_updates),
                "processed_samples": int(processed_samples),
                "loss": float(loss.item()),
                "L_track": float(losses["L_track"]),
                "L_cot": float(losses["L_cot"]),
                "L_future": float(losses["L_future"]),
                "L_verify": float(losses["L_verify"]),
            }
            if grad_info is not None:
                last_train_record.update(grad_info)
            if metric_logger.log_train_step(last_train_record):
                print(f"  epoch {epoch} samples={processed_samples} updates={optimizer_updates}: loss={loss.item():.4f} "
                      f"L_track={losses['L_track']:.4f} L_cot={losses['L_cot']:.4f}")
            if (
                args.max_optimizer_updates
                and optimizer_updates >= args.max_optimizer_updates
            ):
                break

        if accumulated_batches and (
            not args.max_optimizer_updates
            or optimizer_updates < args.max_optimizer_updates
        ):
            grad_info = optimizer_step(
                accumulated_batches,
                {
                    "epoch": int(epoch),
                    "micro_step": int(global_step),
                    "optimizer_updates": int(optimizer_updates + 1),
                    "processed_samples": int(processed_samples),
                    "partial_accumulation_flush": True,
                },
            )
            optimizer_updates += 1
            if last_train_record is not None:
                last_train_record.update(grad_info)
        if last_train_record is not None:
            last_train_record["optimizer_updates"] = int(optimizer_updates)
            if metric_logger.log_train_step(last_train_record, final=True):
                print(
                    f"  epoch {epoch} samples={last_train_record['processed_samples']} "
                    f"updates={optimizer_updates}: loss={last_train_record['loss']:.4f} "
                    f"L_track={last_train_record['L_track']:.4f} "
                    f"L_cot={last_train_record['L_cot']:.4f} [epoch-final]"
                )

        avg = epoch_loss / max(1, steps)
        train_wall_time = time.perf_counter() - epoch_started
        epoch_samples = processed_samples - epoch_start_samples
        print(f"[epoch {epoch}] avg_loss={avg:.4f}")
        metric_logger.log(
            {
                "phase": "epoch",
                "epoch": int(epoch),
                "optimizer_updates": int(optimizer_updates),
                "processed_samples": int(processed_samples),
                "average_loss": float(avg),
                "train_wall_time_s": float(train_wall_time),
                "epoch_samples_per_second": (
                    float(epoch_samples) / train_wall_time
                    if train_wall_time > 0
                    else None
                ),
            }
        )
        last_epoch = int(epoch)
        last_average = float(avg)
        checkpoint_meta["processed_samples"] = int(processed_samples)
        checkpoint_meta["optimizer_updates"] = int(optimizer_updates)

        # Save checkpoint
        ckpt_path = os.path.join(args.out_dir, f"pfem_epoch{epoch}.pt")
        checkpoint_started = time.perf_counter()
        save_checkpoint(ckpt_path, epoch, avg)
        metric_logger.log_checkpoint(
            ckpt_path,
            role="epoch",
            epoch=epoch,
            optimizer_updates=optimizer_updates,
            write_wall_time_s=time.perf_counter() - checkpoint_started,
        )
        print(f"  saved {ckpt_path}")
        validation_started = time.perf_counter()
        validation = evaluate_validation()
        validation_wall_time = time.perf_counter() - validation_started
        if validation is not None:
            val_bce = validation["selection_bce_at1"]
            metric_logger.check_finite_losses(
                {
                    "BCE_at_1": val_bce,
                    "family_loss": validation["family_loss"],
                },
                context={"phase": "validation", "epoch": int(epoch)},
            )
            improved = val_bce < best_val
            print(
                f"[epoch {epoch}] val_BCE@1={val_bce:.6f} "
                f"val_family_loss={validation['family_loss']:.6f}"
            )
            metric_logger.log(
                {
                    "phase": "validation",
                    "epoch": int(epoch),
                    "BCE_at_1": float(val_bce),
                    "family_loss": float(validation["family_loss"]),
                    "selection_detail": validation["selection_detail"],
                    "improved": bool(improved),
                    "validation_wall_time_s": float(validation_wall_time),
                    "validation_samples_per_second": (
                        float(validation_binding["sample_count"])
                        / validation_wall_time
                        if validation_binding is not None
                        and validation_wall_time > 0
                        else None
                    ),
                }
            )
            if improved:
                best_val = val_bce
                checkpoint_meta["best_validation"] = validation
                best_path = os.path.join(args.out_dir, "pfem_best.pt")
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    best_path,
                    epoch,
                    val_bce,
                    checkpoint_role="best_validation",
                    selected_value=val_bce,
                )
                metric_logger.log_checkpoint(
                    best_path,
                    role="best_validation",
                    epoch=epoch,
                    optimizer_updates=optimizer_updates,
                    selected_value=val_bce,
                    write_wall_time_s=time.perf_counter() - checkpoint_started,
                )
        if (
            args.max_optimizer_updates
            and optimizer_updates >= args.max_optimizer_updates
        ):
            break

    print("[train_pfem] done.")
    metric_logger.end_run(
        status="completed",
        summary={
            "final_epoch": last_epoch,
            "final_average_loss": last_average,
            "micro_steps": int(global_step),
            "optimizer_updates": int(optimizer_updates),
            "processed_samples": int(processed_samples),
            "best_validation_BCE_at_1": (
                float(best_val) if np.isfinite(best_val) else None
            ),
        },
    )


def main():
    global _ACTIVE_METRIC_LOGGER
    try:
        return _main_impl()
    except KeyboardInterrupt as exc:
        if _ACTIVE_METRIC_LOGGER is not None and not _ACTIVE_METRIC_LOGGER.ended:
            _ACTIVE_METRIC_LOGGER.end_run(status="interrupted", error=exc)
        raise
    except BaseException as exc:
        if _ACTIVE_METRIC_LOGGER is not None and not _ACTIVE_METRIC_LOGGER.ended:
            _ACTIVE_METRIC_LOGGER.end_run(status="failed", error=exc)
        raise
    finally:
        _ACTIVE_METRIC_LOGGER = None


if __name__ == "__main__":
    main()
