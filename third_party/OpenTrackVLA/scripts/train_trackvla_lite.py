#!/usr/bin/env python3
# ruff: noqa: E402 -- matched CUDA env must be set before torch-heavy imports.
"""Train the paper-structured TrackVLA++-Lite baseline on collected clips."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matched_smoke import prepare_matched_cli_environment

prepare_matched_cli_environment(sys.argv[1:])

import torch
from torch.utils.data import DataLoader

from harness.sequence_state import continues_sequence, detach_state, sample_sequence_key
from harness.trackvla_lite import TrackVLAPlusPlusLite
from cache_gridpool import atomic_torch_save
from experiment_binding import (
    bind_hf_model_artifact,
    sha256_artifact,
    verify_vision_cache,
)
from experiment_logging import JsonlMetricLogger
from local_weights import default_qwen_candidates, resolve_local_model_path
from matched_smoke import (
    assert_matched_counters,
    assert_matched_loader,
    build_scoped_subset,
    configure_matched_cuda,
    enforce_matched_args,
    load_matched_smoke_binding,
)
from model import DataConfig, JsonTrackingDataset, collate_batch
from train_baseline import inspect_bound_dataset, load_official_base, set_seed
from validation_metrics import BalancedControlAccumulator, waypoints_to_step_actions


_ACTIVE_METRIC_LOGGER = None


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", default=None)
    parser.add_argument("--base_hf_model_dir", required=True)
    parser.add_argument("--qwen_model_path", default=None)
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--val_cache_root", default=None)
    parser.add_argument(
        "--matched_support_receipt",
        "--matched_128_support_receipt",
        dest="matched_support_receipt",
        default=None,
    )
    parser.add_argument("--relocated_root", default=None)
    parser.add_argument("--out_dir", default="experiments/b1_trackvla_pp_lite")
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--head_lr", type=float, default=3e-4)
    parser.add_argument("--base_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_optimizer_updates", type=int, default=0)
    parser.add_argument(
        "--save_optimizer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--variant",
        choices=("polar_only", "polar_tim4", "polar_tim16"),
        default="polar_tim4",
    )
    parser.add_argument(
        "--state_mode", choices=("rolling", "stateless"), default="rolling"
    )
    args = parser.parse_args(argv)
    if args.history != 31:
        parser.error(
            "local TrackVLA++-Lite uses 31 coarse history frames + current fine = 32 observations"
        )
    if args.state_mode == "rolling" and args.batch_size != 1:
        parser.error("rolling TrackVLA++-Lite currently requires --batch_size 1")
    if args.state_mode != "rolling":
        parser.error(
            "B1/TrackVLA++-Lite checkpoints are defined as rolling; "
            "use eval_offline --evaluation_tier exploratory "
            "--allow_state_mode_override for a stateless sensitivity run"
        )
    if args.grad_accum_steps <= 0:
        parser.error("--grad_accum_steps must be positive")
    if args.max_steps < 0 or args.max_optimizer_updates < 0:
        parser.error("step/update limits must be >= 0")
    if args.max_steps and args.max_optimizer_updates:
        parser.error("use only one of --max_steps or --max_optimizer_updates")
    return args


def build_dataset(path, cache_root, args):
    return JsonTrackingDataset(
        DataConfig(
            train_json=path,
            n_waypoints=args.n_waypoints,
            history=args.history,
            cache_root=cache_root,
            require_cached_tokens=True,
        )
    )


def build_loader(path, cache_root, args, *, training: bool, dataset=None):
    if dataset is None:
        dataset = build_dataset(path, cache_root, args)
    rolling = args.state_mode == "rolling"
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=bool(training and not rolling),
        num_workers=0,
        collate_fn=collate_batch,
    )


def ground_truth(batch, device):
    return {
        "waypoints": batch["waypoints"].to(device),
        "valid_mask": batch["valid_mask"].to(device),
        "theta_idx": batch["polar_theta_idx"].to(device),
        "dist_idx": batch["polar_dist_idx"].to(device),
        "invalid": batch["polar_invalid"].to(device),
    }


def forward_batch(model, batch, state, device):
    return model.forward_step(
        coarse_tokens=batch["coarse_tokens"].to(device),
        coarse_tidx=batch["coarse_tidx"].to(device),
        fine_tokens=batch["fine_tokens"].to(device),
        fine_tidx=batch["fine_tidx"].to(device),
        instructions=batch["instruction"],
        prev_state=state,
        yaw_hist=batch["yaw_hist"].to(device),
        yaw_curr=batch["yaw_curr"].to(device),
    )


def next_state(model, batch, previous_key, state, device, state_mode):
    current_key = sample_sequence_key(batch)
    carry_forward = state_mode == "rolling" and not current_key[2]
    batch_state = state
    if not carry_forward or not continues_sequence(previous_key, current_key):
        batch_state = model.init_state(batch["coarse_tokens"].size(0), device)
    return current_key, batch_state, carry_forward


@torch.inference_mode()
def evaluate(model, loader, device, state_mode, dt):
    model.eval()
    state = None
    previous_key = None
    total = 0.0
    steps = 0
    selector = BalancedControlAccumulator()
    for batch in loader:
        current_key, batch_state, carry_forward = next_state(
            model, batch, previous_key, state, device, state_mode
        )
        output = forward_batch(model, batch, batch_state, device)
        losses = model.compute_losses(output, ground_truth(batch, device))
        total += float(losses["loss"].item())
        steps += 1
        selector.add(
            waypoints_to_step_actions(
                output["waypoints"].detach().cpu().numpy(), dt
            ),
            batch["step_actions"].cpu().numpy(),
            batch["valid_mask"].cpu().numpy(),
            batch["command"],
            batch["source_raw_dir"],
        )
        if carry_forward:
            state = detach_state(output["new_state"])
            previous_key = current_key
    selection = selector.compute()
    if selection["value"] is None:
        raise ValueError("validation BCE@1 has no supported episode-command cells")
    return {
        "selection_bce_at1": float(selection["value"]),
        "family_loss": total / max(1, steps),
        "selection_detail": selection,
    }


def checkpoint_meta(
    args,
    dataset_info,
    model,
    trainable_params,
    validation_info=None,
    cache_info=None,
    qwen_model_sha256=None,
    base_model_binding=None,
):
    manifest = dataset_info["manifest"]
    fps = manifest.get("fps", 10.0)
    dt = manifest.get("dt", 1.0 / float(fps))
    if isinstance(fps, list) or isinstance(dt, list):
        raise ValueError("TrackVLA++-Lite training requires one fps/dt value")
    base_model_binding = base_model_binding or bind_hf_model_artifact(
        args.base_hf_model_dir
    )
    meta = {
        "schema_version": 1,
        "model_family": "trackvla_pp_lite",
        "experiment_id": {
            "polar_only": "B1-P",
            "polar_tim4": "B1",
            "polar_tim16": "B1-T16",
        }[args.variant],
        "label_mode": "absolute",
        "dataset_label_mode": manifest.get("label_mode"),
        "n_waypoints": int(model.base.cfg.n_waypoints),
        "history": int(args.history),
        "fps": float(fps),
        "dt": float(dt),
        "action_semantics": manifest.get("action_semantics"),
        "data_manifest_hash": dataset_info["manifest_hash"],
        "data_jsonl_sha256": dataset_info["data_hash"],
        "sample_count": dataset_info["sample_count"],
        "training_source_raw_dirs": dataset_info["source_raw_dirs"],
        "base_hf_model_dir": str(Path(args.base_hf_model_dir).expanduser().resolve()),
        "base_model_sha256": base_model_binding["artifact_sha256"],
        "base_model_artifact": base_model_binding,
        "trainable_params": int(trainable_params),
        "seed": int(args.seed),
        "state_mode": args.state_mode,
        "sampling_policy": "ordered_jsonl",
        "batch_size": int(args.batch_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "effective_batch_size": int(args.batch_size * args.grad_accum_steps),
        "base_lr": float(args.base_lr),
        "head_lr": float(args.head_lr),
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
        "paper_reference": "arXiv:2510.07134",
        "local_coarse_history": 31,
        "short_term_observations": 32,
        "trackvla_lite_variant": args.variant,
        "tim_tokens": 0 if model.tim is None else int(model.tim.n_tokens),
        "tim_gate": (
            "none"
            if model.tim is None
            else "confidence_only:C/(C_avg+C);invalid_freezes"
        ),
        "tim_confidence_average": (
            "none"
            if model.tim is None
            else "all_timesteps;invalid_contributes_zero"
        ),
        "polar_representation": "factorized_theta60_dist30_invalid",
        "reason_feedback": "soft_factorized_token_via_zero_init_residual_fusion",
        "loss_formula": "L_traj+0.2*L_reason",
        "text_loss": "omitted_no_local_QA_corpus",
        "strict_reproduction": False,
        "train_args": dict(vars(args)),
    }
    if cache_info is not None:
        meta.update(
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
            }
        )
    if qwen_model_sha256 is not None:
        meta["qwen_model_sha256"] = qwen_model_sha256
    if validation_info is not None:
        meta["validation"] = {
            "data_manifest_hash": validation_info["manifest_hash"],
            "data_jsonl_sha256": validation_info["data_hash"],
            "sample_count": validation_info["sample_count"],
        }
    return meta


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    loss,
    meta,
    *,
    save_optimizer=False,
    checkpoint_role="epoch",
    selected_value=None,
):
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith("base.llm.")
    }
    checkpoint_meta = dict(meta)
    checkpoint_meta.update(
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
        "epoch": int(epoch),
        "model_state": state,
        "loss": float(loss),
        "meta": checkpoint_meta,
    }
    if save_optimizer:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    atomic_torch_save(checkpoint, path)


def _main_impl(argv=None):
    global _ACTIVE_METRIC_LOGGER
    args = parse_args(argv)
    if not args.cache_root:
        raise ValueError("--cache_root is required for reproducible training")
    matched_enabled = bool(args.matched_support_receipt)
    cuda_reproducibility = None
    if matched_enabled:
        enforce_matched_args(args, family="B1")
        # This must precede set_seed(), cuda.is_available(), or model transfer.
        cuda_reproducibility = configure_matched_cuda(torch)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    set_seed(args.seed)
    qwen_path = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=args.qwen_model_path,
        env_var="QWEN_MODEL_PATH",
        candidates=default_qwen_candidates(),
    )
    os.environ["QWEN_MODEL_PATH"] = qwen_path
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[train_trackvla_lite] device={device} state_mode={args.state_mode}")

    dataset_info = inspect_bound_dataset(args.train_json)
    validation_info = inspect_bound_dataset(args.val_json) if args.val_json else None
    matched_binding = (
        load_matched_smoke_binding(
            args.matched_support_receipt,
            train_json=args.train_json,
            dataset_info=dataset_info,
            relocated_root=args.relocated_root,
        )
        if matched_enabled
        else None
    )
    cache_datasets = [args.train_json]
    if args.val_json and not args.val_cache_root:
        cache_datasets.append(args.val_json)
    cache_info = verify_vision_cache(
        args.cache_root,
        cache_datasets,
        verify_payload=not matched_enabled,
        relocated_root=args.relocated_root,
    )
    cache_info["token_payload_verified"] = not matched_enabled
    validation_cache_info = None
    if args.val_json and args.val_cache_root:
        validation_cache_info = verify_vision_cache(
            args.val_cache_root,
            [args.val_json],
            verify_payload=True,
            relocated_root=args.relocated_root,
        )
    qwen_hash = sha256_artifact(qwen_path)
    base_model_binding = bind_hf_model_artifact(args.base_hf_model_dir)
    base = load_official_base(args.base_hf_model_dir, qwen_path)
    args.n_waypoints = int(base.cfg.n_waypoints)
    use_tim = args.variant != "polar_only"
    tim_tokens = 16 if args.variant == "polar_tim16" else 4
    model = TrackVLAPlusPlusLite(
        base,
        expected_history=args.history,
        use_tim=use_tim,
        tim_tokens=tim_tokens,
    ).to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[train_trackvla_lite] trainable={trainable / 1e6:.2f}M / "
        f"total={total_params / 1e6:.2f}M"
    )

    source_train_dataset = build_dataset(args.train_json, args.cache_root, args)
    matched_token_ledger = None
    if matched_binding is not None:
        train_dataset, matched_token_ledger = build_scoped_subset(
            source_train_dataset,
            matched_binding,
            cache_root=args.cache_root,
        )
    else:
        train_dataset = source_train_dataset
    train_loader = build_loader(
        args.train_json,
        args.cache_root,
        args,
        training=True,
        dataset=train_dataset,
    )
    if matched_binding is not None:
        assert_matched_loader(train_loader, family="B1")
    val_loader = (
        build_loader(
            args.val_json,
            args.val_cache_root or args.cache_root,
            args,
            training=False,
        )
        if args.val_json
        else None
    )
    head_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (base_parameters if name.startswith("base.") else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.base_lr},
            {"params": head_parameters, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

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
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metric_logger.log({"phase": "optimizer", **context, **grad_info})
        return grad_info
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_logger = JsonlMetricLogger(output_dir, device=device)
    meta = checkpoint_meta(
        args,
        dataset_info,
        model,
        trainable,
        validation_info=validation_info,
        cache_info=cache_info,
        qwen_model_sha256=qwen_hash,
        base_model_binding=base_model_binding,
    )
    if validation_cache_info is not None:
        meta["validation_vision_cache"] = validation_cache_info
    if matched_binding is not None:
        assert matched_token_ledger is not None
        assert cuda_reproducibility is not None
        meta["matched_128"] = matched_binding.metadata(
            token_ledger=matched_token_ledger,
            cuda_reproducibility=cuda_reproducibility,
        )
        meta["internal_test"] = "sealed"
        meta["internal_test_opened"] = False
        meta["vision_cache_token_payload_verified"] = False
    meta["checkpoint_selection"] = {
        "metric": "validation_episode_macro_BCE@1",
        "mode": "min",
        "rule": "strict_improvement_earliest_epoch",
    }
    metric_logger.start_run(
        args=dict(vars(args)),
        checkpoint_meta=meta,
        total_params=total_params,
        trainable_params=trainable,
    )
    _ACTIVE_METRIC_LOGGER = metric_logger
    best_val = float("inf")
    global_step = 0
    processed_samples = 0
    optimizer_updates = 0
    last_epoch = None
    last_average = None
    for epoch in range(args.epochs):
        epoch_started = time.perf_counter()
        epoch_start_samples = processed_samples
        model.train()
        state = None
        previous_key = None
        total = 0.0
        epoch_steps = 0
        accumulated_batches = 0
        last_train_record = None
        optimizer.zero_grad(set_to_none=True)
        for batch in train_loader:
            current_key, batch_state, carry_forward = next_state(
                model, batch, previous_key, state, device, args.state_mode
            )
            output = forward_batch(model, batch, batch_state, device)
            losses = model.compute_losses(output, ground_truth(batch, device))
            metric_logger.check_finite_losses(
                losses,
                context={"epoch": int(epoch), "micro_step": int(global_step + 1)},
            )
            (losses["loss"] / float(args.grad_accum_steps)).backward()
            accumulated_batches += 1
            grad_info = None
            if accumulated_batches == args.grad_accum_steps:
                grad_info = optimizer_step(
                    accumulated_batches,
                    {
                        "epoch": int(epoch),
                        "micro_step": int(global_step + 1),
                        "optimizer_updates": int(optimizer_updates + 1),
                        "processed_samples": int(
                            processed_samples + batch["coarse_tokens"].size(0)
                        ),
                    },
                )
                accumulated_batches = 0
                optimizer_updates += 1
            if carry_forward:
                state = detach_state(output["new_state"])
                previous_key = current_key
            global_step += 1
            processed_samples += int(batch["coarse_tokens"].size(0))
            epoch_steps += 1
            total += float(losses["loss"].item())
            last_train_record = {
                "phase": "train",
                "epoch": int(epoch),
                "micro_step": int(global_step),
                "optimizer_updates": int(optimizer_updates),
                "processed_samples": int(processed_samples),
                "loss": float(losses["loss"].item()),
                "L_traj": float(losses["L_traj"]),
                "L_reason": float(losses["L_reason"]),
            }
            if grad_info is not None:
                last_train_record.update(grad_info)
            if metric_logger.log_train_step(last_train_record):
                print(
                    f"  epoch={epoch} samples={processed_samples} updates={optimizer_updates} "
                    f"loss={losses['loss'].item():.4f} "
                    f"L_traj={losses['L_traj']:.4f} L_reason={losses['L_reason']:.4f}"
                )
            if args.max_steps and global_step >= args.max_steps:
                break
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
                    f"  epoch={epoch} samples={last_train_record['processed_samples']} "
                    f"updates={optimizer_updates} loss={last_train_record['loss']:.4f} "
                    f"L_traj={last_train_record['L_traj']:.4f} "
                    f"L_reason={last_train_record['L_reason']:.4f} [epoch-final]"
                )
        meta["processed_samples"] = int(processed_samples)
        meta["optimizer_updates"] = int(optimizer_updates)
        if matched_binding is not None:
            assert_matched_counters(processed_samples, optimizer_updates)
        average = total / max(1, epoch_steps)
        train_wall_time = time.perf_counter() - epoch_started
        epoch_samples = processed_samples - epoch_start_samples
        checkpoint_path = output_dir / f"trackvla_lite_epoch{epoch}.pt"
        checkpoint_started = time.perf_counter()
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            epoch,
            average,
            meta,
            save_optimizer=args.save_optimizer,
        )
        metric_logger.log_checkpoint(
            checkpoint_path,
            role="epoch",
            epoch=epoch,
            optimizer_updates=optimizer_updates,
            write_wall_time_s=time.perf_counter() - checkpoint_started,
        )
        print(
            f"[train_trackvla_lite] saved {checkpoint_path} avg_loss={average:.4f}"
        )
        metric_logger.log(
            {
                "phase": "epoch",
                "epoch": int(epoch),
                "optimizer_updates": int(optimizer_updates),
                "processed_samples": int(processed_samples),
                "average_loss": float(average),
                "train_wall_time_s": float(train_wall_time),
                "epoch_samples_per_second": (
                    float(epoch_samples) / train_wall_time
                    if train_wall_time > 0
                    else None
                ),
            }
        )
        last_epoch = int(epoch)
        last_average = float(average)
        if val_loader is not None:
            validation_started = time.perf_counter()
            validation = evaluate(
                model,
                val_loader,
                device,
                args.state_mode,
                float(meta["dt"]),
            )
            validation_wall_time = time.perf_counter() - validation_started
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
                f"[train_trackvla_lite] val_BCE@1={val_bce:.6f} "
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
                        float(validation_info["sample_count"])
                        / validation_wall_time
                        if validation_info is not None and validation_wall_time > 0
                        else None
                    ),
                }
            )
            if improved:
                best_val = val_bce
                meta["best_validation"] = validation
                best_path = output_dir / "trackvla_lite_best.pt"
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch,
                    val_bce,
                    meta,
                    save_optimizer=args.save_optimizer,
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
            (args.max_steps and global_step >= args.max_steps)
            or (
                args.max_optimizer_updates
                and optimizer_updates >= args.max_optimizer_updates
            )
        ):
            break
    run_summary = {
            "final_epoch": last_epoch,
            "final_average_loss": last_average,
            "micro_steps": int(global_step),
            "optimizer_updates": int(optimizer_updates),
            "processed_samples": int(processed_samples),
            "best_validation_BCE_at_1": (
                float(best_val) if torch.isfinite(torch.tensor(best_val)) else None
            ),
        }
    if matched_binding is not None:
        assert_matched_counters(processed_samples, optimizer_updates)
        run_summary.update(
            {
                "matched_128_contract": True,
                "internal_test_opened": False,
            }
        )
    metric_logger.end_run(status="completed", summary=run_summary)


def main(argv=None):
    global _ACTIVE_METRIC_LOGGER
    try:
        return _main_impl(argv)
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
