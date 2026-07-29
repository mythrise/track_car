#!/usr/bin/env python3
# ruff: noqa: E402 -- matched CUDA env must be set before torch-heavy imports.
"""Fine-tune the native no-Harness OpenTrackVLA waypoint planner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matched_smoke import prepare_matched_cli_environment

prepare_matched_cli_environment(sys.argv[1:])

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader, WeightedRandomSampler

from harness.core.event_sampling import compute_event_sampling_weights
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
    parser.add_argument("--out_dir", default="experiments/b0_opentrackvla")
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--max_optimizer_updates", type=int, default=0)
    parser.add_argument(
        "--save_optimizer", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--balance_sampling", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args(argv)
    if args.max_steps < 0 or args.max_optimizer_updates < 0:
        parser.error("step/update limits must be >= 0")
    if args.max_steps and args.max_optimizer_updates:
        parser.error("use only one of --max_steps or --max_optimizer_updates")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_bound_dataset(dataset_path: str) -> dict:
    path = Path(dataset_path)
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"dataset and sidecar manifest are required: {path}, {manifest_path}"
        )
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw.decode("utf-8"))
    expected_hash = manifest.get("data_jsonl_sha256")
    actual_hash = sha256_file(path)
    if expected_hash != actual_hash:
        raise ValueError(
            f"training JSONL sha256 mismatch: manifest={expected_hash}, actual={actual_hash}"
        )
    rows = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows += 1
            record = json.loads(line)
            if "waypoints" not in record:
                raise ValueError(f"sample {line_number} is missing waypoints")
    if rows != int(manifest.get("sample_count", -1)):
        raise ValueError(
            f"training JSONL sample_count mismatch: manifest={manifest.get('sample_count')}, actual={rows}"
        )
    return {
        "manifest": manifest,
        "manifest_hash": hashlib.sha256(manifest_raw).hexdigest(),
        "data_hash": actual_hash,
        "sample_count": rows,
        "source_raw_dirs": list(
            (manifest.get("statistics") or {}).get("source_raw_dirs") or []
        ),
    }


def load_official_base(base_hf_model_dir: str, qwen_model_path: str):
    from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

    base_dir = Path(base_hf_model_dir).expanduser().resolve()
    config = OpenTrackVLAConfig.from_pretrained(str(base_dir), local_files_only=True)
    config.llm_name = qwen_model_path
    wrapper = OpenTrackVLAForWaypoint(config)
    state_path = base_dir / "model.safetensors"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    missing, unexpected = wrapper.load_state_dict(
        load_safetensors(str(state_path)), strict=False
    )
    print(
        f"[train_baseline] loaded official base: {len(missing)} missing, "
        f"{len(unexpected)} unexpected"
    )
    return wrapper.model


def build_sampler(dataset, enabled: bool):
    if not enabled:
        return None
    labels = [
        str(dataset.get_example(index).get("transition_type", "other"))
        for index in range(len(dataset))
    ]
    weights = compute_event_sampling_weights(labels)
    if not weights or all(abs(float(weight) - 1.0) < 1e-12 for weight in weights):
        return None
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def masked_native_loss(model, predicted, target, valid_mask):
    """Original OpenTrackVLA normalized waypoint MSE (without beta scaling)."""

    predicted_norm = predicted
    target_norm = target
    alpha = getattr(model, "alpha_task", None)
    if alpha is not None and predicted.size(-1) >= 2:
        xy_scale = alpha[..., :2].to(predicted).clamp_min(1e-6)
        predicted_norm = predicted.clone()
        target_norm = target.clone()
        predicted_norm[..., :2] = predicted_norm[..., :2] / xy_scale
        target_norm[..., :2] = target_norm[..., :2] / xy_scale
    error = (predicted_norm - target_norm).pow(2)
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1)
    mask = mask.expand_as(error)
    denominator = mask.sum()
    return (
        (error * mask).sum() / denominator
        if denominator.item() > 0
        else error.sum() * 0.0
    )


@torch.inference_mode()
def evaluate(model, loader, device, dt):
    model.eval()
    total = 0.0
    count = 0
    selector = BalancedControlAccumulator()
    for batch in loader:
        predicted = model(
            batch["coarse_tokens"].to(device),
            batch["coarse_tidx"].to(device),
            batch["fine_tokens"].to(device),
            batch["fine_tidx"].to(device),
            batch["instruction"],
            yaw_hist=batch["yaw_hist"].to(device),
            yaw_curr=batch["yaw_curr"].to(device),
        )
        loss = masked_native_loss(
            model,
            predicted,
            batch["waypoints"].to(device),
            batch["valid_mask"].to(device),
        )
        total += float(loss.item()) * predicted.size(0)
        count += predicted.size(0)
        selector.add(
            waypoints_to_step_actions(predicted.detach().cpu().numpy(), dt),
            batch["step_actions"].cpu().numpy(),
            batch["valid_mask"].cpu().numpy(),
            batch["command"],
            batch["source_raw_dir"],
        )
    selection = selector.compute()
    if selection["value"] is None:
        raise ValueError("validation BCE@1 has no supported episode-command cells")
    return {
        "selection_bce_at1": float(selection["value"]),
        "native_mse": total / max(1, count),
        "selection_detail": selection,
    }


def checkpoint_meta(
    args,
    dataset_info,
    model,
    trainable_params: int,
    validation_info=None,
    cache_info=None,
    qwen_model_sha256=None,
    base_model_binding=None,
):
    manifest = dataset_info["manifest"]
    fps = manifest.get("fps", 10.0)
    dt = manifest.get("dt", 1.0 / float(fps))
    if isinstance(fps, list) or isinstance(dt, list):
        raise ValueError("baseline training requires one fps/dt value")
    base_model_binding = base_model_binding or bind_hf_model_artifact(
        args.base_hf_model_dir
    )
    meta = {
        "schema_version": 1,
        "model_family": "opentrackvla_baseline",
        "experiment_id": "B0",
        "label_mode": "absolute",
        "dataset_label_mode": manifest.get("label_mode"),
        "n_waypoints": int(model.cfg.n_waypoints),
        "history": int(args.history),
        "short_term_observations": int(args.history) + 1,
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
        "state_mode": "stateless",
        "sampling_policy": (
            "weighted_random" if args.balance_sampling else "ordered_jsonl"
        ),
        "batch_size": int(args.batch_size),
        "grad_accum_steps": 1,
        "effective_batch_size": int(args.batch_size),
        "base_lr": float(args.lr),
        "head_lr": None,
        "weight_decay": float(args.weight_decay),
        "grad_clip": float(args.grad_clip),
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
        if not key.startswith("llm.")
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
        enforce_matched_args(args, family="B0")
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
    print(f"[train_baseline] device={device}")

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
    model = load_official_base(args.base_hf_model_dir, qwen_path).to(device)
    # Match B1/H0 common-base adaptation: only the visual projector and native
    # trajectory head are updated. TVI/act-token differences must not become a
    # hidden baseline advantage.
    model.requires_grad_(False)
    model.proj.requires_grad_(True)
    model.planner.requires_grad_(True)
    args.n_waypoints = int(model.cfg.n_waypoints)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[train_baseline] trainable={trainable / 1e6:.2f}M / "
        f"total={total_params / 1e6:.2f}M"
    )

    source_train_dataset = JsonTrackingDataset(
        DataConfig(
            train_json=args.train_json,
            n_waypoints=args.n_waypoints,
            history=args.history,
            cache_root=args.cache_root,
            require_cached_tokens=True,
        )
    )
    matched_token_ledger = None
    if matched_binding is not None:
        train_dataset, matched_token_ledger = build_scoped_subset(
            source_train_dataset,
            matched_binding,
            cache_root=args.cache_root,
        )
    else:
        train_dataset = source_train_dataset
    sampler = build_sampler(train_dataset, args.balance_sampling)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        # The matched B0/B1/H0 comparison uses one common chronological JSONL
        # order.  Weighted sampling remains an explicit exploratory option.
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_batch,
    )
    if matched_binding is not None:
        assert_matched_loader(train_loader, family="B0")
    val_loader = None
    if args.val_json:
        val_dataset = JsonTrackingDataset(
            DataConfig(
                train_json=args.val_json,
                n_waypoints=args.n_waypoints,
                history=args.history,
                cache_root=args.val_cache_root or args.cache_root,
                require_cached_tokens=True,
            )
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
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
    meta["trainable_base_modules"] = ["proj", "planner"]
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
        epoch_total = 0.0
        epoch_steps = 0
        last_train_record = None
        for batch in train_loader:
            predicted = model(
                batch["coarse_tokens"].to(device),
                batch["coarse_tidx"].to(device),
                batch["fine_tokens"].to(device),
                batch["fine_tidx"].to(device),
                batch["instruction"],
                yaw_hist=batch["yaw_hist"].to(device),
                yaw_curr=batch["yaw_curr"].to(device),
            )
            nav_loss = masked_native_loss(
                model,
                predicted,
                batch["waypoints"].to(device),
                batch["valid_mask"].to(device),
            )
            loss = float(model.cfg.beta_nav) * nav_loss
            metric_logger.check_finite_losses(
                {"loss": loss, "L_nav": nav_loss},
                context={"epoch": int(epoch), "micro_step": int(global_step + 1)},
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_info = metric_logger.clip_grad_norm_and_check(
                [p for p in model.parameters() if p.requires_grad],
                args.grad_clip,
                context={"epoch": int(epoch), "micro_step": int(global_step + 1)},
            )
            optimizer.step()
            global_step += 1
            processed_samples += int(predicted.size(0))
            optimizer_updates += 1
            epoch_steps += 1
            epoch_total += float(loss.item())
            metric_logger.log(
                {
                    "phase": "optimizer",
                    "epoch": int(epoch),
                    "micro_step": int(global_step),
                    "optimizer_updates": int(optimizer_updates),
                    "processed_samples": int(processed_samples),
                    **grad_info,
                }
            )
            last_train_record = {
                "phase": "train",
                "epoch": int(epoch),
                "micro_step": int(global_step),
                "optimizer_updates": int(optimizer_updates),
                "processed_samples": int(processed_samples),
                "loss": float(loss.item()),
                "L_nav": float(nav_loss.item()),
                **grad_info,
            }
            if metric_logger.log_train_step(last_train_record):
                print(
                    f"  epoch={epoch} step={global_step} loss={loss.item():.4f} "
                    f"L_nav={nav_loss.item():.4f}"
                )
            update_limit = args.max_optimizer_updates or args.max_steps
            if update_limit and optimizer_updates >= update_limit:
                break
        if last_train_record is not None and metric_logger.log_train_step(
            last_train_record, final=True
        ):
            print(
                f"  epoch={epoch} step={last_train_record['micro_step']} "
                f"loss={last_train_record['loss']:.4f} "
                f"L_nav={last_train_record['L_nav']:.4f} [epoch-final]"
            )
        average = epoch_total / max(1, epoch_steps)
        train_wall_time = time.perf_counter() - epoch_started
        epoch_samples = processed_samples - epoch_start_samples
        if matched_binding is not None:
            assert_matched_counters(processed_samples, optimizer_updates)
        meta["processed_samples"] = int(processed_samples)
        meta["optimizer_updates"] = int(optimizer_updates)
        checkpoint_path = output_dir / f"baseline_epoch{epoch}.pt"
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
        print(f"[train_baseline] saved {checkpoint_path} avg_loss={average:.4f}")
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
                model, val_loader, device, float(meta["dt"])
            )
            validation_wall_time = time.perf_counter() - validation_started
            val_bce = validation["selection_bce_at1"]
            metric_logger.check_finite_losses(
                {
                    "BCE_at_1": val_bce,
                    "family_loss": validation["native_mse"],
                },
                context={"phase": "validation", "epoch": int(epoch)},
            )
            improved = val_bce < best_val
            print(
                f"[train_baseline] val_BCE@1={val_bce:.6f} "
                f"val_native_mse={validation['native_mse']:.6f}"
            )
            metric_logger.log(
                {
                    "phase": "validation",
                    "epoch": int(epoch),
                    "BCE_at_1": float(val_bce),
                    "family_loss": float(validation["native_mse"]),
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
                best_path = output_dir / "baseline_best.pt"
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
        update_limit = args.max_optimizer_updates or args.max_steps
        if update_limit and optimizer_updates >= update_limit:
            break
    run_summary = {
            "final_epoch": last_epoch,
            "final_average_loss": last_average,
            "micro_steps": int(global_step),
            "optimizer_updates": int(optimizer_updates),
            "processed_samples": int(processed_samples),
            "best_validation_BCE_at_1": (
                float(best_val) if np.isfinite(best_val) else None
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
