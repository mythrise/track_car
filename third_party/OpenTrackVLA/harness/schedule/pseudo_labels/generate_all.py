#!/usr/bin/env python3
"""Generate all pseudo-labels for training data.

Reads a JSONL training file and adds:
- polar_theta_idx, polar_dist_idx, polar_invalid (from bbox + geometry)
- future polar/visibility at Δ∈{4, 8, 16} (from future frames' GT)
- event triggers (from invalid streaks, bbox overlap)

Usage:
    python harness/schedule/pseudo_labels/generate_all.py \
        --input data/car_train.jsonl \
        --output data/car_train_labeled.jsonl
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np


def discretize_theta(theta_rad, n=60):
    deg = (math.degrees(theta_rad) + 180) % 360
    return min(int(deg * n / 360), n - 1)


def discretize_dist(d, n=30, dmin=0.6, dmax=5.0):
    if d <= dmin: return 0
    if d >= dmax: return n - 1
    return min(int((d - dmin) * n / (dmax - dmin)), n - 1)


def bbox_to_polar(bbox, img_w=320, img_h=240, fov=60):
    """bbox = [x0, y0, x1, y1] in normalized coords or pixels."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    if max(bbox) > 1.5:
        cx /= img_w
        cy /= img_h
    x_bias = cx - 0.5
    theta_rad = x_bias * math.radians(fov)
    dist = max(0.5, 3.0 * (1.0 - cy * 0.8))
    return theta_rad, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with open(args.input) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"[pseudo_labels] processing {len(samples)} samples...")

    for i, s in enumerate(samples):
        if "bbox" in s and s["bbox"] is not None:
            theta_rad, dist_m = bbox_to_polar(s["bbox"])
            s["polar_theta_idx"] = discretize_theta(theta_rad)
            s["polar_dist_idx"] = discretize_dist(dist_m)
            s["polar_invalid"] = 0.0
            s["polar_theta_rad"] = theta_rad
            s["polar_dist_m"] = dist_m
        else:
            s.setdefault("polar_theta_idx", -1)
            s.setdefault("polar_dist_idx", -1)
            s.setdefault("polar_invalid", 1.0)

        # Future labels: check ahead, but only within the same episode
        cur_ep = s.get("episode", "")
        for delta in [4, 8, 16]:
            j = i + delta
            same_episode = (j < len(samples) and samples[j].get("episode", "") == cur_ep)
            if same_episode and "bbox" in samples[j] and samples[j]["bbox"] is not None:
                ft, fd = bbox_to_polar(samples[j]["bbox"])
                s[f"fut_theta_{delta}"] = discretize_theta(ft)
                s[f"fut_dist_{delta}"] = discretize_dist(fd)
                s[f"fut_vis_{delta}"] = 1.0
            else:
                s[f"fut_theta_{delta}"] = 0
                s[f"fut_dist_{delta}"] = 0
                s[f"fut_vis_{delta}"] = 0.0

        # Event triggers
        if s.get("polar_invalid", 0) > 0.5:
            # Check if this is start of invalid streak
            if i > 0 and samples[i - 1].get("polar_invalid", 0) < 0.5:
                s["event_occ_start"] = True
        if i > 0 and samples[i - 1].get("polar_invalid", 0) > 0.5 and s.get("polar_invalid", 0) < 0.5:
            theta = s.get("polar_theta_rad", 0)
            s["event_recovery"] = "left" if theta > 0 else "right"

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    n_valid = sum(1 for s in samples if s.get("polar_invalid", 1) < 0.5)
    print(f"[pseudo_labels] done. {n_valid}/{len(samples)} valid polar labels → {args.output}")


if __name__ == "__main__":
    main()
