#!/usr/bin/env python3
"""Offline action-space evaluation for absolute and step-action checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENTRACKVLA_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
for path in (PROJECT_ROOT, OPENTRACKVLA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


AXIS_NAMES = ("forward", "strafe", "yaw")


def smooth_l1_values(pred, target):
    difference = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    return np.where(difference < 1.0, 0.5 * difference**2, difference - 0.5)


def waypoints_to_step_actions(waypoints, dt):
    """Invert the shared discrete pose composition for absolute checkpoints."""

    poses = np.asarray(waypoints, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None, ...]
        squeeze = True
    elif poses.ndim == 3:
        squeeze = False
    else:
        raise ValueError("waypoints must have shape (T, 3) or (B, T, 3)")
    if poses.shape[-1] != 3 or float(dt) <= 0:
        raise ValueError("waypoints must have 3 axes and dt must be > 0")

    previous = np.concatenate((np.zeros_like(poses[:, :1]), poses[:, :-1]), axis=1)
    world_delta = poses[..., :2] - previous[..., :2]
    yaw_before = previous[..., 2]
    cos_yaw = np.cos(yaw_before)
    sin_yaw = np.sin(yaw_before)
    forward = (cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]) / float(dt)
    strafe = (-sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]) / float(dt)
    yaw = (poses[..., 2] - previous[..., 2]) / float(dt)
    actions = np.stack((forward, strafe, yaw), axis=-1)
    return actions[0] if squeeze else actions


def transition_event_mask(actions, prev_actions, threshold=0.2):
    sequence = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(prev_actions, dtype=np.float64)
    prior_yaw = np.concatenate((previous[:, None, 2], sequence[:, :-1, 2]), axis=1)
    yaw = sequence[..., 2]
    active = np.abs(yaw) > float(threshold)
    prior_active = np.abs(prior_yaw) > float(threshold)
    sign_flip = active & prior_active & (np.sign(yaw) != np.sign(prior_yaw))
    return (active != prior_active) | sign_flip


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def compute_metrics(pred_actions, gt_actions, prev_actions, valid_mask=None, threshold=0.2):
    pred = np.asarray(pred_actions, dtype=np.float64)
    target = np.asarray(gt_actions, dtype=np.float64)
    previous = np.asarray(prev_actions, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred_actions and gt_actions must share shape (N, T, 3)")
    if previous.shape != (pred.shape[0], 3):
        raise ValueError("prev_actions must have shape (N, 3)")
    mask = (
        np.ones(pred.shape[:2], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != pred.shape[:2]:
        raise ValueError("valid_mask must have shape (N, T)")

    errors = smooth_l1_values(pred, target)
    per_axis = {}
    saturation = {}
    for axis, name in enumerate(AXIS_NAMES):
        values = errors[..., axis][mask]
        per_axis[name] = float(values.mean()) if values.size else None
        saturated = np.abs(pred[..., axis])[mask] > 0.95
        saturation[name] = float(saturated.mean()) if saturated.size else None

    turn_mask = mask & (np.abs(target[..., 2]) > float(threshold))
    sign_correct = np.sign(pred[..., 2]) == np.sign(target[..., 2])
    turn_sign_accuracy = _safe_ratio(np.count_nonzero(sign_correct & turn_mask), np.count_nonzero(turn_mask))

    gt_events = transition_event_mask(target, previous, threshold) & mask
    pred_events = transition_event_mask(pred, previous, threshold) & mask
    true_positive = int(np.count_nonzero(gt_events & pred_events))
    false_positive = int(np.count_nonzero(~gt_events & pred_events & mask))
    false_negative = int(np.count_nonzero(gt_events & ~pred_events & mask))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else 0.0
    )
    saturated_all = np.abs(pred)[np.repeat(mask[..., None], 3, axis=-1)] > 0.95
    return {
        "samples": int(pred.shape[0]),
        "valid_steps": int(np.count_nonzero(mask)),
        "smooth_l1": per_axis,
        "turn_sign_accuracy": turn_sign_accuracy,
        "transition": {
            "precision": precision,
            "recall": recall,
            "f1": float(f1),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        },
        "saturation_rate": {
            "overall": float(saturated_all.mean()) if saturated_all.size else None,
            **saturation,
        },
    }


def evaluate_predictions(pred_actions, records, threshold=0.2):
    if not records:
        raise ValueError("validation dataset is empty")
    gt = np.asarray([record["step_actions"] for record in records], dtype=np.float64)
    previous = np.asarray([record["prev_action"] for record in records], dtype=np.float64)
    valid = np.asarray(
        [record.get("valid_mask", [True] * gt.shape[1]) for record in records],
        dtype=bool,
    )
    transitions = [str(record.get("transition_type", "other")) for record in records]
    result = compute_metrics(pred_actions, gt, previous, valid, threshold)
    result["by_transition_type"] = {}
    for transition_type in sorted(set(transitions)):
        indices = [index for index, value in enumerate(transitions) if value == transition_type]
        result["by_transition_type"][transition_type] = compute_metrics(
            np.asarray(pred_actions)[indices],
            gt[indices],
            previous[indices],
            valid[indices],
            threshold,
        )
    return result


def _parse_named_paths(values):
    runs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        if not name or not path:
            raise ValueError(f"invalid checkpoint spec: {value!r}")
        runs.append((name, Path(path).expanduser().resolve()))
    return runs


def _parse_mode_overrides(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"mode override must be NAME=MODE: {value!r}")
        name, mode = value.split("=", 1)
        if mode not in {"absolute", "step_action"}:
            raise ValueError(f"invalid mode override: {value!r}")
        overrides[name] = mode
    return overrides


def _collect_checkpoint_predictions(checkpoint_path, val_json, args, mode_override=None):
    import torch
    from torch.utils.data import DataLoader

    from inference_pipeline import mac_server
    from model import DataConfig, JsonTrackingDataset, collate_batch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    meta = checkpoint.get("meta", {})
    label_mode = mode_override or str(meta.get("label_mode", "absolute"))
    history = int(meta.get("history", args.history))
    n_waypoints = int(meta.get("n_waypoints", args.n_waypoints))
    dt = float(meta.get("dt", args.dt))
    train_args = meta.get("train_args") if isinstance(meta.get("train_args"), dict) else {}
    aux_delta_vel = bool(meta.get("aux_delta_vel", train_args.get("aux_delta_vel", False)))

    root = mac_server.resolve_opentrackvla_root(args.opentrackvla_root)
    weight_args = SimpleNamespace(
        qwen_model_path=args.qwen_model_path,
        dinov3_model_path=args.dinov3_model_path,
        siglip_model_path=args.siglip_model_path,
        base_hf_model_dir=args.base_hf_model_dir,
    )
    mac_server.configure_default_weight_paths(weight_args, root)
    device = torch.device(args.device or mac_server.default_device())
    model = mac_server.load_model(
        checkpoint,
        device,
        root,
        base_hf_model_dir=weight_args.base_hf_model_dir,
        n_waypoints=n_waypoints,
        label_mode=label_mode,
        control_dt=dt,
        aux_delta_vel=aux_delta_vel,
    )
    dataset = JsonTrackingDataset(
        DataConfig(
            train_json=str(val_json),
            n_waypoints=n_waypoints,
            history=history,
            cache_root=args.cache_root,
            default_dt=dt,
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    predictions = []
    records = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch_size = batch["coarse_tokens"].size(0)
            output = model.forward_step(
                coarse_tokens=batch["coarse_tokens"].to(device),
                coarse_tidx=batch["coarse_tidx"].to(device),
                fine_tokens=batch["fine_tokens"].to(device),
                fine_tidx=batch["fine_tidx"].to(device),
                instructions=batch["instruction"],
                prev_state=model.init_state(batch_size, device),
                prev_action=batch["prev_action"].to(device),
            )
            if label_mode == "step_action":
                batch_predictions = output["step_actions"].detach().cpu().numpy()
            else:
                batch_predictions = waypoints_to_step_actions(
                    output["waypoints"].detach().cpu().numpy(), dt
                )
            predictions.extend(batch_predictions.tolist())
            for index in range(batch_size):
                records.append(
                    {
                        "step_actions": batch["step_actions"][index].tolist(),
                        "prev_action": batch["prev_action"][index].tolist(),
                        "valid_mask": batch["valid_mask"][index].tolist(),
                        "transition_type": batch["transition_type"][index],
                    }
                )
    return label_mode, evaluate_predictions(np.asarray(predictions), records, args.transition_threshold)


def _fmt(value):
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"


def render_comparison_table(results):
    lines = [
        "| run | mode | fwd SmoothL1 | strafe SmoothL1 | yaw SmoothL1 | turn-sign acc | transition F1 | saturation |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in results.items():
        metrics = payload["metrics"]
        lines.append(
            "| " + " | ".join(
                (
                    name,
                    payload["label_mode"],
                    _fmt(metrics["smooth_l1"]["forward"]),
                    _fmt(metrics["smooth_l1"]["strafe"]),
                    _fmt(metrics["smooth_l1"]["yaw"]),
                    _fmt(metrics["turn_sign_accuracy"]),
                    _fmt(metrics["transition"]["f1"]),
                    _fmt(metrics["saturation_rate"]["overall"]),
                )
            ) + " |"
        )
    return "\n".join(lines)


def render_group_tables(results):
    sections = []
    for name, payload in results.items():
        lines = [
            f"### {name} by transition_type",
            "",
            "| transition_type | samples | fwd SmoothL1 | strafe SmoothL1 | yaw SmoothL1 | turn-sign acc | transition F1 | saturation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for group, metrics in payload["metrics"]["by_transition_type"].items():
            lines.append(
                f"| {group} | {metrics['samples']} | {_fmt(metrics['smooth_l1']['forward'])} | "
                f"{_fmt(metrics['smooth_l1']['strafe'])} | {_fmt(metrics['smooth_l1']['yaw'])} | "
                f"{_fmt(metrics['turn_sign_accuracy'])} | "
                f"{_fmt(metrics['transition']['f1'])} | {_fmt(metrics['saturation_rate']['overall'])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", required=True)
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint path or NAME=PATH; repeat for a comparison table.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Optional NAME=absolute|step_action override; otherwise checkpoint meta is used.",
    )
    parser.add_argument("--json_output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--transition_threshold", type=float, default=0.2)
    parser.add_argument("--cache_root", default=None)
    parser.add_argument("--opentrackvla_root", default=None)
    parser.add_argument("--base_hf_model_dir", default=None)
    parser.add_argument("--qwen_model_path", default=None)
    parser.add_argument("--dinov3_model_path", default=None)
    parser.add_argument("--siglip_model_path", default=None)
    return parser


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = build_parser().parse_args(argv)
    runs = _parse_named_paths(args.ckpt)
    overrides = _parse_mode_overrides(args.mode)
    results = {}
    for name, checkpoint_path in runs:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        label_mode, metrics = _collect_checkpoint_predictions(
            checkpoint_path,
            Path(args.val_json).expanduser().resolve(),
            args,
            overrides.get(name),
        )
        results[name] = {
            "checkpoint": str(checkpoint_path),
            "label_mode": label_mode,
            "metrics": metrics,
        }
    print(render_comparison_table(results))
    print()
    print(render_group_tables(results))
    if args.json_output:
        output = Path(args.json_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
