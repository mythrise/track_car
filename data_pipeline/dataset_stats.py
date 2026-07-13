#!/usr/bin/env python3
"""Report lightweight integrity and label statistics for training JSONL."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def manifest_path_for(dataset_path: Path) -> Path:
    return Path(str(dataset_path) + ".manifest.json")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: sample is not a JSON object")
            yield value


def _flatten_vectors(value, output):
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, float)) for item in value):
            output.append([float(item) for item in value])
        else:
            for item in value:
                _flatten_vectors(item, output)


def _range_stats(vectors):
    if not vectors:
        return {"count": 0, "min": None, "max": None, "per_axis": []}
    width = max(len(vector) for vector in vectors)
    per_axis = []
    flat = []
    for axis in range(width):
        values = [vector[axis] for vector in vectors if axis < len(vector) and math.isfinite(vector[axis])]
        per_axis.append(
            {
                "axis": axis,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        )
        flat.extend(values)
    return {
        "count": len(vectors),
        "min": min(flat) if flat else None,
        "max": max(flat) if flat else None,
        "per_axis": per_axis,
    }


def compute_dataset_stats(dataset_path) -> dict:
    path = Path(dataset_path)
    command_counts = Counter()
    transition_counts = Counter()
    total = 0
    polar_valid = 0
    detection_source_counts = Counter()
    polar_by_source = {}
    deltas = {"delta_pos": [], "delta_vel": []}

    for sample in load_jsonl(path):
        total += 1
        command_counts[str(sample.get("command", "<missing>"))] += 1
        transition_counts[str(sample.get("transition_type", "<missing>"))] += 1
        if float(sample.get("polar_invalid", 1.0)) < 0.5:
            polar_valid += 1
            polar_is_valid = True
        else:
            polar_is_valid = False
        source = str(sample.get("detection_source", "unknown"))
        detection_source_counts[source] += 1
        source_stats = polar_by_source.setdefault(
            source,
            {"samples": 0, "valid": 0, "invalid": 0, "distance_bins": Counter()},
        )
        source_stats["samples"] += 1
        if polar_is_valid:
            source_stats["valid"] += 1
            dist_idx = int(sample.get("polar_dist_idx", -1))
            if dist_idx >= 0:
                source_stats["distance_bins"][dist_idx] += 1
        else:
            source_stats["invalid"] += 1
        for field in deltas:
            if field in sample:
                _flatten_vectors(sample[field], deltas[field])

    manifest = None
    sidecar = manifest_path_for(path)
    if sidecar.exists():
        with sidecar.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

    fps_report = {
        "manifest_fps": manifest.get("fps") if isinstance(manifest, dict) else None,
        "manifest_dt": manifest.get("dt") if isinstance(manifest, dict) else None,
        "episodes": [],
        "consistent": None,
    }
    if isinstance(manifest, dict):
        episode_reports = manifest.get("statistics", {}).get("episode_reports", [])
        episode_fps = [report.get("fps") for report in episode_reports if report.get("fps") is not None]
        fps_report["episodes"] = episode_fps
        fps_report["consistent"] = len({float(value) for value in episode_fps}) <= 1 if episode_fps else None

    polar_source_summary = {}
    for source, source_stats in sorted(polar_by_source.items()):
        valid = source_stats["valid"]
        distance_bins = source_stats["distance_bins"]
        max_bin_count = distance_bins.get(29, 0)
        weighted_total = sum(index * count for index, count in distance_bins.items())
        polar_source_summary[source] = {
            "samples": source_stats["samples"],
            "valid": valid,
            "invalid": source_stats["invalid"],
            "valid_rate": valid / max(1, source_stats["samples"]),
            "distance_bin_distribution": {
                str(index): count for index, count in sorted(distance_bins.items())
            },
            "mean_distance_bin": weighted_total / valid if valid else None,
            "max_distance_bin_count": max_bin_count,
            "max_distance_bin_rate": max_bin_count / max(1, valid),
        }

    return {
        "dataset": str(path),
        "samples": total,
        "command_distribution": dict(sorted(command_counts.items())),
        "transition_type_distribution": dict(sorted(transition_counts.items())),
        "detection_source_distribution": dict(sorted(detection_source_counts.items())),
        "polar": {
            "valid": polar_valid,
            "invalid": total - polar_valid,
            "valid_rate": polar_valid / max(1, total),
        },
        "polar_by_detection_source": polar_source_summary,
        "fps": fps_report,
        "delta_ranges": {field: _range_stats(values) for field, values in deltas.items()},
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Training JSONL path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON only")
    args = parser.parse_args(argv)
    stats = compute_dataset_stats(args.dataset)
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return
    print(f"dataset: {stats['dataset']}")
    print(f"samples: {stats['samples']}")
    print(f"command: {stats['command_distribution']}")
    print(f"transition_type: {stats['transition_type_distribution']}")
    print(f"detection_source: {stats['detection_source_distribution']}")
    print(f"Polar: {stats['polar']}")
    print(f"Polar by detection source: {stats['polar_by_detection_source']}")
    print(f"fps: {stats['fps']}")
    print(f"delta ranges: {stats['delta_ranges']}")


if __name__ == "__main__":
    main()
