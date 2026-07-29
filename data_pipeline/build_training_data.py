#!/usr/bin/env python3
"""Convert collected car episodes into OpenTrackVLA training JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np

try:
    import jsonschema
except ModuleNotFoundError:
    try:
        from data_pipeline import jsonschema_fallback as jsonschema
    except ModuleNotFoundError:
        import jsonschema_fallback as jsonschema


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
TURN_THRESHOLD = 0.2
POLAR_THETA_BINS = 60
POLAR_DISTANCE_BINS = 30
MAX_PERSON_BBOX_AREA = 0.8
TRAINING_SAMPLE_SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "training_sample.schema.json"


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
    if (
        len(action) != 3
        or not all(math.isfinite(value) for value in action)
        or any(abs(value) > 1.000001 for value in action)
    ):
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


def derive_step_labels(actions, prev_action, dt):
    step_actions = np.asarray(actions, dtype=np.float32)
    previous = np.asarray(prev_action, dtype=np.float32)
    if step_actions.ndim != 2 or step_actions.shape[1] != 3:
        raise ValueError("actions must have shape (T, 3)")
    if previous.shape != (3,):
        raise ValueError("prev_action must have shape (3,)")
    delta_pos = step_actions * float(dt)
    prior = np.concatenate((previous[None, :], step_actions[:-1]), axis=0)
    delta_vel = step_actions - prior
    return step_actions.tolist(), delta_pos.tolist(), delta_vel.tolist()


def classify_transition_type(prev_action, step_actions, threshold=TURN_THRESHOLD):
    previous_yaw = float(prev_action[2])
    yaws = np.asarray(step_actions, dtype=np.float32)[:, 2]
    active = np.abs(yaws) > float(threshold)
    previous_active = abs(previous_yaw) > float(threshold)

    if not previous_active and not bool(active.any()):
        return "steady_forward"

    active_signs = np.sign(yaws[active])
    sign_changed = len(set(active_signs.tolist())) > 1
    if previous_active and active_signs.size:
        sign_changed = sign_changed or bool(np.any(active_signs != np.sign(previous_yaw)))
    transitions = int(np.count_nonzero(active[1:] != active[:-1])) if active.size > 1 else 0

    if not previous_active:
        if bool(active[-1]) and transitions <= 1 and not sign_changed:
            return "turn_onset"
        return "other"
    if bool(active.all()) and not sign_changed:
        return "sustained_turn"
    if not bool(active[-1]) and transitions <= 1 and not sign_changed:
        return "turn_exit"
    return "other"


def contains_turn(step_actions, prev_action=None, threshold=TURN_THRESHOLD):
    previous_turn = (
        prev_action is not None and abs(float(prev_action[2])) > float(threshold)
    )
    return previous_turn or any(
        abs(float(action[2])) > float(threshold) for action in step_actions
    )


def mirror_actions(actions):
    mirrored = np.asarray(actions, dtype=np.float32).copy()
    mirrored[..., 1:] *= -1.0
    return mirrored.tolist()


def mirror_command(command):
    swaps = {
        "turn_left": "turn_right",
        "turn_right": "turn_left",
        "strafe_left": "strafe_right",
        "strafe_right": "strafe_left",
    }
    return swaps.get(command, command)


def validation_output_path(output: Path) -> Path:
    suffix = output.suffix or ".jsonl"
    return output.with_name(f"{output.stem}.val{suffix}")


def _write_mirrored_image(source: Path, destination: Path) -> None:
    frame = cv2.imread(str(source))
    if frame is None:
        raise DataIntegrityError(f"cannot mirror unreadable image: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), cv2.flip(frame, 1)):
        raise DataIntegrityError(f"failed to write mirrored image: {destination}")


def _mirrored_path(source: Path, mirror_root: Path, input_root: Path) -> Path:
    try:
        relative = source.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(source.parent.name) / source.name
    return mirror_root / relative


def make_mirrored_sample(
    sample,
    source_images,
    mirror_root,
    input_root,
    args,
    fps,
    written_images=None,
):
    written_images = set() if written_images is None else written_images
    mirrored_paths = []
    for source in source_images:
        destination = _mirrored_path(source, mirror_root, input_root)
        if destination not in written_images:
            _write_mirrored_image(source, destination)
            written_images.add(destination)
        mirrored_paths.append(serialize_image_path(destination, args.absolute_paths))

    mirrored = dict(sample)
    mirrored["images"] = mirrored_paths[:-1]
    mirrored["current"] = mirrored_paths[-1]
    mirrored["mirrored"] = True
    mirrored["command"] = mirror_command(sample.get("command"))
    mirrored["prev_action"] = mirror_actions([sample["prev_action"]])[0]
    mirrored["step_actions"] = mirror_actions(sample["step_actions"])
    mirrored["actions"] = mirrored["step_actions"]
    mirrored["delta_pos"] = (
        np.asarray(mirrored["step_actions"], dtype=np.float32) / float(fps)
    ).tolist()
    prior = np.asarray([mirrored["prev_action"]] + mirrored["step_actions"][:-1], dtype=np.float32)
    mirrored["delta_vel"] = (
        np.asarray(mirrored["step_actions"], dtype=np.float32) - prior
    ).tolist()
    mirrored["waypoints"] = integrate_actions(mirrored["step_actions"], 1.0 / float(fps)).tolist()
    mirrored["transition_type"] = classify_transition_type(
        mirrored["prev_action"], mirrored["step_actions"]
    )
    if "bbox" in mirrored:
        x0, y0, x1, y1 = mirrored["bbox"]
        mirrored["bbox"] = [1.0 - x1, y0, 1.0 - x0, y1]
    theta_idx = int(mirrored.get("polar_theta_idx", -1))
    if theta_idx >= 0 and "bbox" in mirrored:
        x0, _y0, x1, _y1 = mirrored["bbox"]
        mirrored_theta = theta_from_normalized_cx((x0 + x1) / 2.0)
        mirrored["polar_theta_idx"] = discretize_theta(mirrored_theta)
    return mirrored


def estimate_target_from_frame(frame, detector=None):
    """Return a normalized person bbox, preferring OmDet-Turbo."""

    active_detector = detector or get_default_target_detector()
    detected, _ = active_detector.detect(frame)
    return detected


def is_plausible_person_bbox(detected, max_area=MAX_PERSON_BBOX_AREA):
    if detected is None:
        return False
    try:
        _cx, _cy, width, height = [float(value) for value in detected]
    except (TypeError, ValueError):
        return False
    return (
        0.0 < width <= 1.0
        and 0.0 < height <= 1.0
        and width * height <= float(max_area)
    )


def theta_from_normalized_cx(cx, fov_deg=60):
    return (float(cx) - 0.5) * math.radians(float(fov_deg))


def bbox_to_polar(cx, cy, frame_w, frame_h, fov_deg=60, bbox_h=None):
    """Convert normalized bbox geometry to approximate polar coordinates."""

    del frame_w, frame_h
    theta_rad = theta_from_normalized_cx(cx, fov_deg)
    if bbox_h is None:
        # Haar detects a face rather than a full body, so use its vertical
        # position instead of applying the full-person apparent-height scale.
        dist_est = max(0.5, 3.0 * (1.0 - (float(cy) * 0.8)))
    else:
        # A standing person's apparent height is approximately inverse to
        # distance.  Clamp to the Polar-CoT training bins to avoid extremes.
        dist_est = max(0.6, min(5.0, 1.2 / max(float(bbox_h), 0.05)))
    return theta_rad, dist_est


def discretize_theta(theta_rad, n_theta=POLAR_THETA_BINS):
    degrees = (math.degrees(theta_rad) + 180.0) % 360.0
    return min(int(degrees * n_theta / 360.0), n_theta - 1)


def discretize_dist(dist_m, n_dist=POLAR_DISTANCE_BINS, dmin=0.6, dmax=5.0):
    if dist_m <= dmin:
        return 0
    if dist_m >= dmax:
        return n_dist - 1
    return min(int((dist_m - dmin) * n_dist / (dmax - dmin)), n_dist - 1)


def summarize_polar_by_detection_source(samples):
    grouped = {}
    for sample in samples:
        source = str(sample.get("detection_source", "unknown"))
        group = grouped.setdefault(
            source,
            {"samples": 0, "valid": 0, "invalid": 0, "distance_bins": Counter()},
        )
        group["samples"] += 1
        valid = float(sample.get("polar_invalid", 1.0)) < 0.5
        if valid:
            group["valid"] += 1
            dist_idx = int(sample.get("polar_dist_idx", -1))
            if dist_idx >= 0:
                group["distance_bins"][dist_idx] += 1
        else:
            group["invalid"] += 1

    summary = {}
    for source, group in sorted(grouped.items()):
        valid = group["valid"]
        distance_bins = group["distance_bins"]
        max_bin_count = distance_bins.get(POLAR_DISTANCE_BINS - 1, 0)
        weighted_total = sum(index * count for index, count in distance_bins.items())
        summary[source] = {
            "samples": group["samples"],
            "valid": valid,
            "invalid": group["invalid"],
            "valid_rate": valid / max(1, group["samples"]),
            "distance_bin_distribution": {
                str(index): count for index, count in sorted(distance_bins.items())
            },
            "mean_distance_bin": weighted_total / valid if valid else None,
            "max_distance_bin_count": max_bin_count,
            "max_distance_bin_rate": max_bin_count / max(1, valid),
        }
    return summary


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


def jsonl_sha256_and_row_count(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    row_count = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if raw_line.strip():
                row_count += 1
    return digest.hexdigest(), row_count


def validate_training_sample_subset(samples, rng=None) -> list[int]:
    if not samples:
        return []
    with TRAINING_SAMPLE_SCHEMA_PATH.open(encoding="utf-8") as handle:
        schema = json.load(handle)
    random_source = rng or random.SystemRandom()
    remaining = list(range(1, len(samples)))
    selected = [0]
    if remaining:
        selected.extend(random_source.sample(remaining, min(3, len(remaining))))
    for index in selected:
        try:
            jsonschema.validate(instance=samples[index], schema=schema)
        except jsonschema.ValidationError as exc:
            location = ".".join(str(part) for part in exc.absolute_path) or "<root>"
            raise DataIntegrityError(
                f"training sample {index} failed JSON Schema at {location}: {exc.message}"
            ) from exc
    return selected


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

    reports = [
        inspect_episode(path)
        for path in sorted(input_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
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

    label_mode = getattr(args, "label_mode", "absolute")
    mirror_augment = bool(getattr(args, "mirror_augment", False))
    val_episodes = set(getattr(args, "val_episodes", ()) or ())
    output = Path(args.output).expanduser()
    val_output_arg = getattr(args, "val_output", None)
    val_output = Path(val_output_arg).expanduser() if val_output_arg else validation_output_path(output)
    mirror_root = output.resolve().parent / f".{output.stem}_mirrored_images"

    detector = detector_factory(device="cpu")
    split_samples = {"train": [], "val": []}
    split_haar_valid = {"train": 0, "val": 0}
    mirrored_images_written = set()
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

        if len(frames) < args.history + args.n_waypoints:
            print(f"[build] {ep_dir.name}: only {len(frames)} frames; no samples")
            continue

        for t in range(args.history, len(frames) - args.n_waypoints + 1):
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
            detected, detection_source = (
                detector.detect(frame, haar_result=haar_detection) if frame is not None else (None, "none")
            )
            detection_rejected = None
            if detected is not None and not is_plausible_person_bbox(detected):
                detection_rejected = "bbox_area_or_bounds"
                detected = None
            if detected is not None:
                cx, cy, bbox_width, bbox_height = detected
                theta_rad, dist_m = bbox_to_polar(
                    cx,
                    cy,
                    frame.shape[1],
                    frame.shape[0],
                    bbox_h=bbox_height if detection_source == "omdet" else None,
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
            prev_meta = report["valid_meta"].get(frame_idx - 1)
            prev_action = meta_to_action(prev_meta)
            if prev_action is None:
                skipped_bad_meta += 1
                continue
            step_actions, delta_pos, delta_vel = derive_step_labels(actions, prev_action, 1.0 / fps)
            transition_type = classify_transition_type(prev_action, step_actions)
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
                "step_actions": step_actions,
                "delta_pos": delta_pos,
                "delta_vel": delta_vel,
                "prev_action": prev_action,
                "transition_type": transition_type,
                "mirrored": False,
                "action_semantics": report["action_semantics"],
                "motors": meta.get("motors"),
                "command": meta.get("command"),
                "polar_theta_idx": theta_idx,
                "polar_dist_idx": dist_idx,
                "polar_invalid": invalid,
                "detection_source": detection_source,
            }
            if detection_rejected is not None:
                sample["detection_rejected"] = detection_rejected
            if detected is not None:
                sample["bbox"] = [
                    cx - bbox_width / 2.0,
                    cy - bbox_height / 2.0,
                    cx + bbox_width / 2.0,
                    cy + bbox_height / 2.0,
                ]

            split = "val" if ep_dir.name in val_episodes else "train"
            split_samples[split].append(sample)
            if haar_detection is not None:
                split_haar_valid[split] += 1
            if split == "train" and mirror_augment and contains_turn(step_actions, prev_action):
                source_images = [
                    frames[t - args.history + index]
                    for index in range(args.history)
                ] + [current_frame_path]
                split_samples[split].append(
                    make_mirrored_sample(
                        sample,
                        source_images,
                        mirror_root,
                        input_root,
                        args,
                        fps,
                        mirrored_images_written,
                    )
                )
                if haar_detection is not None:
                    split_haar_valid[split] += 1
            episode_samples += 1

        report["samples"] = episode_samples
        report["skipped_bad_meta"] = skipped_bad_meta
        report["haar_valid"] = episode_haar_valid
        report["polar_valid"] = episode_final_valid
        print(
            f"[build] {ep_dir.name}: {len(frames)} frames -> {episode_samples} samples "
            f"({skipped_bad_meta} skipped); Polar valid Haar={episode_haar_valid} "
            f"final={episode_final_valid}"
        )

    all_samples = split_samples["train"]
    all_output_samples = split_samples["train"] + split_samples["val"]
    spin_turn_samples = sum(
        sample["action_semantics"] == "spin_v1"
        and contains_turn(sample["step_actions"], sample["prev_action"])
        for sample in all_output_samples
    )
    if spin_turn_samples:
        print(
            "!!! [build_training_data] WARNING: spin_v1 turn samples are retained, but final "
            "step-action gains depend on recollecting arc_turn_v2 data "
            f"({spin_turn_samples} turn samples including mirrors)."
        )

    def build_manifest(samples, split_name, dataset_output, data_sha256, row_count):
        output_parent = dataset_output.resolve().parent
        path_root = os.path.relpath(PROJECT_ROOT.resolve(), output_parent)
        split_commands = Counter(str(sample.get("command")) for sample in samples)
        split_transitions = Counter(str(sample.get("transition_type")) for sample in samples)
        split_detection_sources = Counter(
            str(sample.get("detection_source", "unknown")) for sample in samples
        )
        split_polar_valid = sum(float(sample.get("polar_invalid", 1.0)) < 0.5 for sample in samples)
        return {
            "schema_version": 1,
            "data_jsonl_sha256": data_sha256,
            "sample_count": row_count,
            "path_root": path_root,
            "path_mode": "absolute" if args.absolute_paths else "repo_relative",
            "fps": fps_value,
            "dt": (
                (1.0 / fps_value)
                if isinstance(fps_value, (int, float))
                else [1.0 / value for value in fps_value]
            ),
            "history": int(args.history),
            "n_waypoints": int(args.n_waypoints),
            "action_semantics": semantics_value,
            "label_mode": label_mode,
            "delta_scale": 1.0,
            "distance_source": "source_aware_heuristic",
            "distance_source_by_detection": {
                "omdet": "inverse_full_body_bbox_height",
                "haar": "face_bbox_vertical_position",
            },
            "mirror_augment": bool(mirror_augment and split_name == "train"),
            "split": split_name,
            "statistics": {
                "sample_count": len(samples),
                "episode_count": len({sample["episode"] for sample in samples}),
                "mirrored_count": sum(bool(sample.get("mirrored")) for sample in samples),
                "command_distribution": dict(sorted(split_commands.items())),
                "transition_type_distribution": dict(sorted(split_transitions.items())),
                "detection_source_distribution": dict(sorted(split_detection_sources.items())),
                "polar_valid": split_polar_valid,
                "polar_invalid": len(samples) - split_polar_valid,
                "polar_valid_rate": split_polar_valid / max(1, len(samples)),
                "haar_baseline_valid": split_haar_valid[split_name],
                "haar_baseline_valid_rate": split_haar_valid[split_name] / max(1, len(samples)),
                "polar_by_detection_source": summarize_polar_by_detection_source(samples),
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

    def write_split(dataset_output, samples, split_name):
        dataset_output.parent.mkdir(parents=True, exist_ok=True)
        with dataset_output.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
        validate_training_sample_subset(samples)
        data_sha256, row_count = jsonl_sha256_and_row_count(dataset_output)
        if row_count != len(samples):
            raise DataIntegrityError(
                f"JSONL row count changed while writing {dataset_output}: "
                f"expected {len(samples)}, got {row_count}"
            )
        split_manifest = build_manifest(
            samples,
            split_name,
            dataset_output,
            data_sha256,
            row_count,
        )
        sidecar = manifest_path_for(dataset_output)
        with sidecar.open("w", encoding="utf-8") as handle:
            json.dump(split_manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        return split_manifest, sidecar

    manifest, manifest_path = write_split(output, split_samples["train"], "train")
    if val_episodes:
        val_manifest, val_manifest_path = write_split(val_output, split_samples["val"], "val")
        manifest["validation"] = {
            "output": str(val_output),
            "manifest": str(val_manifest_path),
            "sample_count": val_manifest["statistics"]["sample_count"],
            "episodes": sorted(val_episodes),
            "mirror_augment": val_manifest["mirror_augment"],
        }
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")

    output_haar_valid = sum(split_haar_valid.values())
    output_polar_valid = sum(
        float(sample.get("polar_invalid", 1.0)) < 0.5 for sample in all_output_samples
    )
    print(
        f"[build_training_data] wrote {len(all_samples)} train samples to {output}; "
        f"manifest={manifest_path}; Polar valid Haar={output_haar_valid}/{max(1, len(all_output_samples))} "
        f"final={output_polar_valid}/{max(1, len(all_output_samples))}"
    )
    if val_episodes:
        print(f"[build_training_data] wrote {len(split_samples['val'])} val samples to {val_output}")
    return all_samples, manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--label_mode", choices=("step_action", "absolute"), default="step_action")
    parser.add_argument(
        "--mirror_augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Horizontally mirror samples whose horizon contains a turn (default: enabled).",
    )
    parser.add_argument(
        "--val_episodes",
        nargs="*",
        default=(),
        help="Episode directory names to write only to the validation JSONL.",
    )
    parser.add_argument("--val_output", default=None, help="Validation JSONL path; defaults beside --output.")
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
