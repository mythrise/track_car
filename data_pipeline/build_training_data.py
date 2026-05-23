#!/usr/bin/env python3
"""Convert collected car episodes into OpenTrackVLA training format.

Reads data/collected/<episode>/ frames, generates pseudo-labels (polar-CoT,
waypoints from motion), and outputs a JSONL file compatible with model.py's
JsonTrackingDataset.

Usage:
    python data_pipeline/build_training_data.py --input data/collected --output data/car_train.jsonl
"""

import argparse
import cv2
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car_runtime"))

try:
    from car_hardware import motor_to_action
except ImportError:
    from car_runtime.car_hardware import motor_to_action


def estimate_target_from_frame(frame):
    """Simple face/person detection to get target bbox.

    Returns (cx, cy, w, h) normalized or None if not found.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    # Pick largest
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]
    H, W = frame.shape[:2]
    return (x + w / 2) / W, (y + h / 2) / H, w / W, h / H


def bbox_to_polar(cx, cy, frame_w, frame_h, fov_deg=60):
    """Convert normalized bbox center to approximate polar coordinates."""
    x_bias = cx - 0.5
    y_bias = cy - 0.5
    theta_rad = x_bias * math.radians(fov_deg)
    dist_est = max(0.5, 3.0 * (1.0 - (cy * 0.8)))
    return theta_rad, dist_est


def discretize_theta(theta_rad, n_theta=60):
    deg = math.degrees(theta_rad) + 180.0
    deg %= 360.0
    return min(int(deg * n_theta / 360.0), n_theta - 1)


def discretize_dist(dist_m, n_dist=30, dmin=0.6, dmax=5.0):
    if dist_m <= dmin: return 0
    if dist_m >= dmax: return n_dist - 1
    return min(int((dist_m - dmin) * n_dist / (dmax - dmin)), n_dist - 1)


def load_meta(ep_dir, frame_idx):
    meta_path = ep_dir / f"meta_{frame_idx:06d}.json"
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


def meta_to_action(meta):
    if "action" in meta:
        return [float(v) for v in meta["action"]]
    if "motors" in meta:
        return motor_to_action(meta["motors"])
    return [0.0, 0.0, 0.0]


def integrate_waypoints(ep_dir, start_idx, horizon, fps):
    x = y = th = 0.0
    dt = 1.0 / max(float(fps), 1.0)
    waypoints = []
    actions = []
    for k in range(horizon):
        meta = load_meta(ep_dir, start_idx + k)
        vx, vy, wz = meta_to_action(meta)
        x += vx * dt
        y += vy * dt
        th += wz * dt
        actions.append([vx, vy, wz])
        waypoints.append([x, y, th])
    return waypoints, actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--n_waypoints", type=int, default=8)
    args = ap.parse_args()

    episodes = sorted(Path(args.input).iterdir())
    all_samples = []

    for ep_dir in episodes:
        if not ep_dir.is_dir():
            continue
        ep_json = ep_dir / "episode.json"
        if not ep_json.exists():
            continue

        with open(ep_json) as f:
            ep_meta = json.load(f)

        frames = sorted(ep_dir.glob("frame_*.jpg"))
        if len(frames) < args.history + args.n_waypoints + 1:
            print(f"  skip {ep_dir.name}: only {len(frames)} frames")
            continue

        instruction = ep_meta.get("instruction", "follow the person")
        fps = ep_meta.get("fps", 10)

        for t in range(args.history, len(frames) - args.n_waypoints):
            current = str(frames[t])
            images = [str(frames[max(0, t - args.history + i)]) for i in range(args.history)]
            frame_idx = int(frames[t].stem.split("_")[-1])
            meta = load_meta(ep_dir, frame_idx)

            # Detect target in current frame for polar labels
            frame = cv2.imread(current)
            det = estimate_target_from_frame(frame) if frame is not None else None

            if det is not None:
                cx, cy, bw, bh = det
                theta_rad, dist_m = bbox_to_polar(cx, cy, frame.shape[1], frame.shape[0])
                theta_idx = discretize_theta(theta_rad)
                dist_idx = discretize_dist(dist_m)
                invalid = 0.0
            else:
                theta_idx = -1
                dist_idx = -1
                invalid = 1.0

            waypoints, actions = integrate_waypoints(ep_dir, frame_idx, args.n_waypoints, fps)

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
            if det is not None:
                sample["bbox"] = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2]

            all_samples.append(sample)

        print(f"  {ep_dir.name}: {len(frames)} frames → {len(all_samples)} samples total")

    # Write JSONL
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s) + "\n")

    print(f"[build_training_data] wrote {len(all_samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
