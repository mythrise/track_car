#!/usr/bin/env python3
"""Convert collected car episodes into OpenTrackVLA training JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "car_runtime"))

try:
    from car_hardware import motor_to_action
except ImportError:
    from car_runtime.car_hardware import motor_to_action

from data_pipeline.kinematics import integrate_actions
from data_pipeline.target_detector import get_default_target_detector


SUPPORTED_ACTION_SEMANTICS = {"spin_v1", "arc_turn_v2"}


class DataIntegrityError(RuntimeError):
    """Raised when collected episodes are unsafe to convert in strict mode."""


def load_json(path):
    """Compatibility helper returning ``None`` for missing/empty/bad JSON."""

    data, _ = load_json_diagnostic(path)
    return data


def load_json_diagnostic(path: Path):
    if not path.exists():
        return None, "missing"
    if path.stat().st_size == 0:
        return None, "empty"
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return None, f"malformed: {exc}"
    if not isinstance(value, dict):
        return None, "top-level value is not an object"
    return value, None


def load_meta(ep_dir, frame_idx):
    return load_json(ep_dir / f"meta_{frame_idx:06d}.json")


def first_valid_meta(ep_dir):
    for meta_path in sorted(ep_dir.glob("meta_*.json")):
        meta, error = load_json_diagnostic(meta_path)
        if error is None and meta_to_action(meta) is not None:
            return meta
    return None


def meta_to_action(meta):
    if not isinstance(meta, dict):
        return None
    try:
        if "action" in meta:
            action = [float(value) for value in meta["action"]]
        elif "motors" in meta:
            action = [float(value) for value in motor_to_action(meta["motors"])]
        else:
            return None
    except (TypeError, ValueError, IndexError):
        return None
    if len(action) != 3 or not all(math.isfinite(value) for value in action):
        return None
    return action


def infer_action_semantics(episode_meta: dict) -> str | None:
    explicit = episode_meta.get("action_semantics")
    if explicit is not None:
        return str(explicit) if str(explicit) in SUPPORTED_ACTION_SEMANTICS else None
    # Episodes collected before arc-turn parameters were recorded used the
    # legacy in-place spin controls; newer summaries include turn_yaw_ratio.
    return "arc_turn_v2" if "turn_yaw_ratio" in episode_meta else "spin_v1"


def integrate_waypoints(ep_dir, start_idx, horizon, fps):
    dt = 1.0 / float(fps)
    actions = []
    for offset in range(int(horizon)):
        action = meta_to_action(load_meta(ep_dir, start_idx + offset))
        if action is None:
            return None, None
        actions.append(action)
    return integrate_actions(actions, dt).tolist(), actions


def estimate_target_from_frame(frame, detector=None):
    """Return a normalized person bbox, preferring OmDet-Turbo."""

    active_detector = detector or get_default_target_detector()
    detected, _ = active_detector.detect(frame)
    return detected


def bbox_to_polar(cx, cy, frame_w, frame_h, fov_deg=60, bbox_h=None):
    """Convert normalized bbox geometry to approximate polar coordinates."""

    del frame_w, frame_h
    theta_rad = (float(cx) - 0.5) * math.radians(float(fov_deg))
    if bbox_h is None:
        # Compatibility for external callers; builder always supplies bbox_h.
        dist_est = max(0.5, 3.0 * (1.0 - (float(cy) * 0.8)))
    else:
        # A standing person's apparent height is approximately inverse to
        # distance.  Clamp to the Polar-CoT training bins to avoid extremes.
        dist_est = max(0.6, min(5.0, 1.2 / max(float(bbox_h), 0.05)))
    return theta_rad, dist_est


def discretize_theta(theta_rad, n_theta=60):
    degrees = (math.degrees(theta_rad) + 180.0) % 360.0
    return min(int(degrees * n_theta / 360.0), n_theta - 1)


def discretize_dist(dist_m, n_dist=30, dmin=0.6, dmax=5.0):
    if dist_m <= dmin:
        return 0
    if dist_m >= dmax:
        return n_dist - 1
    return min(int((dist_m - dmin) * n_dist / (dmax - dmin)), n_dist - 1)


def _frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def inspect_episode(ep_dir: Path) -> dict:
    frames = sorted(ep_dir.glob("frame_*.jpg"))
    episode_meta, episode_error = load_json_diagnostic(ep_dir / "episode.json")
    recovered = False
    if episode_meta is None:
        recovered_meta = first_valid_meta(ep_dir)
        if recovered_meta is not None:
            episode_meta = {
                "episode": ep_dir.name,
                "instruction": recovered_meta.get("instruction", "follow the person"),
                "fps": recovered_meta.get("fps", 10),
            }
            recovered = True

    empty_meta = 0
    malformed_meta = 0
    invalid_action_meta = 0
    valid_meta = {}
    timestamps = []
    for meta_path in sorted(ep_dir.glob("meta_*.json")):
        meta, error = load_json_diagnostic(meta_path)
        if error == "empty":
            empty_meta += 1
            continue
        if error is not None:
            malformed_meta += 1
            continue
        action = meta_to_action(meta)
        if action is None:
            invalid_action_meta += 1
            continue
        frame_idx = int(meta.get("frame_idx", _frame_index(meta_path)))
        valid_meta[frame_idx] = meta
        try:
            timestamp = float(meta["timestamp"])
            if math.isfinite(timestamp):
                timestamps.append((frame_idx, timestamp))
            else:
                malformed_meta += 1
        except (KeyError, TypeError, ValueError):
            malformed_meta += 1

    frame_indices = {_frame_index(frame) for frame in frames}
    missing_meta = len(frame_indices - set(valid_meta))
    timestamp_values = [value for _, value in sorted(timestamps)]
    intervals = np.diff(timestamp_values) if len(timestamp_values) >= 2 else np.asarray([])
    positive_intervals = intervals[intervals > 0]
    p50 = float(np.percentile(positive_intervals, 50)) if positive_intervals.size else None
    p95 = float(np.percentile(positive_intervals, 95)) if positive_intervals.size else None
    jitter_ratio = p95 / p50 if p50 and p50 > 0 and p95 is not None else None

    problems = []
    if episode_error is not None:
        problems.append(f"episode.json is {episode_error}")
    if episode_meta is None:
        problems.append("episode metadata could not be recovered")
    if empty_meta:
        problems.append(f"{empty_meta} empty meta files")
    if malformed_meta:
        problems.append(f"{malformed_meta} malformed/timestampless meta files")
    if invalid_action_meta:
        problems.append(f"{invalid_action_meta} meta files with invalid action semantics")
    if missing_meta:
        problems.append(f"{missing_meta} frames without valid meta")
    if jitter_ratio is None:
        problems.append("insufficient valid timestamps")
    elif jitter_ratio > 1.2:
        problems.append(f"timestamp jitter p95/p50={jitter_ratio:.3f} > 1.2")

    fps = None
    semantics = None
    if episode_meta is not None:
        try:
            fps = float(episode_meta.get("fps", 10))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError
        except (TypeError, ValueError):
            fps = None
            problems.append("fps is missing or invalid")
        semantics = infer_action_semantics(episode_meta)
        if semantics is None:
            problems.append(f"unknown action_semantics={episode_meta.get('action_semantics')!r}")

    return {
        "episode": ep_dir.name,
        "episode_dir": ep_dir,
        "episode_meta": episode_meta,
        "episode_recovered": recovered,
        "frames": frames,
        "frame_count": len(frames),
        "valid_meta": valid_meta,
        "valid_meta_count": len(valid_meta),
        "empty_meta": empty_meta,
        "malformed_meta": malformed_meta,
        "invalid_action_meta": invalid_action_meta,
        "missing_meta": missing_meta,
        "timestamp_p50": p50,
        "timestamp_p95": p95,
        "timestamp_jitter_ratio": jitter_ratio,
        "fps": fps,
        "action_semantics": semantics,
        "problems": problems,
    }


def print_integrity_report(report: dict) -> None:
    ratio = report["timestamp_jitter_ratio"]
    ratio_text = "n/a" if ratio is None else f"{ratio:.3f}"
    status = "ERROR" if report["problems"] else "OK"
    recovered = " recovered_episode_json" if report["episode_recovered"] else ""
    print(
        f"[integrity] {report['episode']}: {status}{recovered} "
        f"frames={report['frame_count']} valid_meta={report['valid_meta_count']} "
        f"empty_meta={report['empty_meta']} malformed_meta={report['malformed_meta']} "
        f"missing_meta={report['missing_meta']} p95/p50={ratio_text} "
        f"fps={report['fps']} semantics={report['action_semantics']}"
    )
    for problem in report["problems"]:
        print(f"  - {problem}")


def serialize_image_path(path: Path, absolute_paths: bool, repo_root: Path | None = None) -> str:
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    repo_root = PROJECT_ROOT if repo_root is None else repo_root
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise DataIntegrityError(
            f"Image is outside repo root and cannot use portable relative paths: {resolved}. "
            "Use --absolute_paths to preserve the legacy absolute-path mode."
        ) from exc


def manifest_path_for(output: Path) -> Path:
    return Path(str(output) + ".manifest.json")


def _common_value(values, label: str, lenient: bool):
    unique = sorted(set(values))
    if len(unique) == 1:
        return unique[0]
    message = f"mixed {label} values are unsafe: {unique}"
    if not lenient:
        raise DataIntegrityError(message)
    print(f"!!! [build_training_data] WARNING: {message}")
    return unique


def build_dataset(args, detector_factory=get_default_target_detector):
    input_root = Path(args.input).expanduser().resolve()
    if not input_root.is_dir():
        raise DataIntegrityError(f"input directory does not exist: {input_root}")

    reports = [inspect_episode(path) for path in sorted(input_root.iterdir()) if path.is_dir()]
    for report in reports:
        print_integrity_report(report)
    integrity_errors = [
        f"{report['episode']}: {problem}"
        for report in reports
        for problem in report["problems"]
    ]
    if integrity_errors and not args.lenient:
        raise DataIntegrityError(
            "Strict integrity validation failed:\n  " + "\n  ".join(integrity_errors)
        )
    if integrity_errors:
        print("!!! [build_training_data] LENIENT MODE: continuing after integrity errors")

    usable = [report for report in reports if report["episode_meta"] is not None and report["fps"]]
    if not usable:
        raise DataIntegrityError("no usable episodes found")
    if args.lenient:
        for report in usable:
            if report["action_semantics"] is None:
                fallback = (
                    "arc_turn_v2"
                    if "turn_yaw_ratio" in report["episode_meta"]
                    else "spin_v1"
                )
                print(
                    f"!!! [build_training_data] WARNING: {report['episode']} has unknown semantics; "
                    f"lenient fallback={fallback}"
                )
                report["action_semantics"] = fallback
    fps_value = _common_value([report["fps"] for report in usable], "fps", args.lenient)
    semantics_value = _common_value(
        [report["action_semantics"] for report in usable if report["action_semantics"]],
        "action_semantics",
        args.lenient,
    )

    detector = detector_factory(device="cpu")
    all_samples = []
    command_counts = Counter()
    total_haar_valid = 0
    total_final_valid = 0

    for report in usable:
        ep_dir = report["episode_dir"]
        frames = report["frames"]
        episode_meta = report["episode_meta"]
        instruction = episode_meta.get("instruction", "follow the person")
        fps = float(report["fps"])
        episode_samples = 0
        skipped_bad_meta = 0
        episode_haar_valid = 0
        episode_final_valid = 0

        if len(frames) < args.history + args.n_waypoints + 1:
            print(f"[build] {ep_dir.name}: only {len(frames)} frames; no samples")
            continue

        for t in range(args.history, len(frames) - args.n_waypoints):
            current_frame_path = frames[t]
            frame_idx = _frame_index(current_frame_path)
            meta = report["valid_meta"].get(frame_idx)
            if meta is None:
                skipped_bad_meta += 1
                continue

            images = [
                serialize_image_path(frames[t - args.history + index], args.absolute_paths)
                for index in range(args.history)
            ]
            current = serialize_image_path(current_frame_path, args.absolute_paths)
            frame = cv2.imread(str(current_frame_path))
            haar_detection = detector.detect_haar(frame) if frame is not None else None
            detected, _detection_source = (
                detector.detect(frame, haar_result=haar_detection) if frame is not None else (None, "none")
            )
            if detected is not None:
                cx, cy, bbox_width, bbox_height = detected
                theta_rad, dist_m = bbox_to_polar(
                    cx,
                    cy,
                    frame.shape[1],
                    frame.shape[0],
                    bbox_h=bbox_height,
                )
                theta_idx = discretize_theta(theta_rad)
                dist_idx = discretize_dist(dist_m)
                invalid = 0.0
            else:
                theta_idx = -1
                dist_idx = -1
                invalid = 1.0

            waypoints, actions = integrate_waypoints(ep_dir, frame_idx, args.n_waypoints, fps)
            if waypoints is None:
                skipped_bad_meta += 1
                continue
            if haar_detection is not None:
                episode_haar_valid += 1
            if detected is not None:
                episode_final_valid += 1

            sample = {
                "episode": ep_dir.name,
                "frame_idx": frame_idx,
                "current": current,
                "images": images,
                "instruction": instruction,
                "waypoints": waypoints,
                "actions": actions,
                "motors": meta.get("motors"),
                "command": meta.get("command"),
                "polar_theta_idx": theta_idx,
                "polar_dist_idx": dist_idx,
                "polar_invalid": invalid,
            }
            if detected is not None:
                sample["bbox"] = [
                    cx - bbox_width / 2.0,
                    cy - bbox_height / 2.0,
                    cx + bbox_width / 2.0,
                    cy + bbox_height / 2.0,
                ]

            all_samples.append(sample)
            command_counts[str(meta.get("command"))] += 1
            episode_samples += 1

        total_haar_valid += episode_haar_valid
        total_final_valid += episode_final_valid
        report["samples"] = episode_samples
        report["skipped_bad_meta"] = skipped_bad_meta
        report["haar_valid"] = episode_haar_valid
        report["polar_valid"] = episode_final_valid
        print(
            f"[build] {ep_dir.name}: {len(frames)} frames -> {episode_samples} samples "
            f"({skipped_bad_meta} skipped); Polar valid Haar={episode_haar_valid} "
            f"final={episode_final_valid}"
        )

    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    output_parent = output.resolve().parent
    path_root = os.path.relpath(PROJECT_ROOT.resolve(), output_parent)
    manifest = {
        "schema_version": 1,
        "path_root": path_root,
        "path_mode": "absolute" if args.absolute_paths else "repo_relative",
        "fps": fps_value,
        "dt": (1.0 / fps_value) if isinstance(fps_value, (int, float)) else [1.0 / value for value in fps_value],
        "history": int(args.history),
        "n_waypoints": int(args.n_waypoints),
        "action_semantics": semantics_value,
        "label_mode": "absolute",
        "delta_scale": 1.0,
        "distance_source": "heuristic_bbox",
        "statistics": {
            "sample_count": len(all_samples),
            "episode_count": len(usable),
            "command_distribution": dict(sorted(command_counts.items())),
            "polar_valid": total_final_valid,
            "polar_invalid": len(all_samples) - total_final_valid,
            "polar_valid_rate": total_final_valid / max(1, len(all_samples)),
            "haar_baseline_valid": total_haar_valid,
            "haar_baseline_valid_rate": total_haar_valid / max(1, len(all_samples)),
            "episode_reports": [
                {
                    key: report.get(key)
                    for key in (
                        "episode",
                        "frame_count",
                        "valid_meta_count",
                        "empty_meta",
                        "malformed_meta",
                        "missing_meta",
                        "timestamp_p50",
                        "timestamp_p95",
                        "timestamp_jitter_ratio",
                        "fps",
                        "action_semantics",
                        "samples",
                        "skipped_bad_meta",
                        "haar_valid",
                        "polar_valid",
                    )
                }
                for report in reports
            ],
        },
    }
    manifest_path = manifest_path_for(output)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print(
        f"[build_training_data] wrote {len(all_samples)} samples to {output}; "
        f"manifest={manifest_path}; Polar valid Haar={total_haar_valid}/{max(1, len(all_samples))} "
        f"final={total_final_valid}/{max(1, len(all_samples))}"
    )
    return all_samples, manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Write absolute image paths in JSONL (legacy behavior).",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Downgrade integrity failures to prominent warnings and skip unusable samples.",
    )
    args = parser.parse_args(argv)
    if args.history <= 0:
        parser.error("--history must be > 0")
    if args.n_waypoints <= 0:
        parser.error("--n_waypoints must be > 0")
    return args


def main():
    try:
        build_dataset(parse_args())
    except DataIntegrityError as exc:
        print(f"[build_training_data] ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
