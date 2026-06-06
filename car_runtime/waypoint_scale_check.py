#!/usr/bin/env python3
"""Diagnose action, waypoint, and motor scale alignment from a training JSONL.

This script is read-only. It summarizes the normalized actions and cumulative
waypoints in data/car_train.jsonl, then simulates how different waypoint index,
control dt, and motor scale choices map to wheel speed deltas around neutral.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Sequence


ACTION_DIMS = ("forward", "strafe", "yaw")


def parse_csv_ints(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_csv_floats(text: str) -> list[float]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated number")
    return values


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def fmt(value: float) -> str:
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    text_rows = [[str(cell) for cell in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[i]) for row in text_rows)) if text_rows else len(str(header))
        for i, header in enumerate(headers)
    ]
    lines = ["  ".join(str(header).ljust(widths[i]) for i, header in enumerate(headers))]
    lines.append("  ".join("-" * width for width in widths))
    for row in text_rows:
        lines.append("  ".join(row[i].rjust(widths[i]) for i in range(len(widths))))
    return "\n".join(lines)


def as_vector(value: object) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None


def collect_vectors(value: object) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    first = as_vector(value)
    if first is not None and (not value or not isinstance(value[0], list)):
        return [first]
    vectors = []
    for item in value:
        vec = as_vector(item)
        if vec is not None:
            vectors.append(vec)
    return vectors


def load_samples(path: Path) -> tuple[list[dict], Counter[str], Counter[str]]:
    samples: list[dict] = []
    episodes: Counter[str] = Counter()
    commands: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            samples.append(sample)
            episodes[str(sample.get("episode", "<missing>"))] += 1
            commands[str(sample.get("command", "<missing>"))] += 1
    return samples, episodes, commands


def component_stats(vectors: Sequence[Sequence[float]]) -> list[list[str]]:
    rows = []
    for dim, name in enumerate(ACTION_DIMS):
        values = [float(vec[dim]) for vec in vectors if len(vec) > dim]
        abs_values = [abs(value) for value in values]
        rows.append(
            [
                name,
                str(len(values)),
                fmt(mean(values) if values else 0.0),
                fmt(mean(abs_values) if abs_values else 0.0),
                fmt(percentile(abs_values, 50)),
                fmt(percentile(abs_values, 90)),
                fmt(max(abs_values) if abs_values else 0.0),
            ]
        )
    return rows


def wheel_deltas(action_like: Sequence[float], scale: float) -> list[float]:
    forward, strafe, yaw = action_like[:3]
    return [
        (forward + strafe + yaw) * scale,
        (forward - strafe - yaw) * scale,
        (forward - strafe + yaw) * scale,
        (forward + strafe - yaw) * scale,
    ]


def max_abs_wheel_delta(action_like: Sequence[float], scale: float) -> float:
    return max(abs(value) for value in wheel_deltas(action_like, scale))


def summarize_motor(values: Sequence[float]) -> tuple[str, str, str, str]:
    if not values:
        return "0", "0", "0", "0"
    return (
        fmt(mean(values)),
        fmt(percentile(values, 50)),
        fmt(percentile(values, 90)),
        fmt(max(values)),
    )


def vectors_at_index(samples: Sequence[dict], key: str, index: int) -> list[list[float]]:
    vectors = []
    for sample in samples:
        seq = collect_vectors(sample.get(key))
        if index < len(seq):
            vectors.append(seq[index])
    return vectors


def first_vectors(samples: Sequence[dict], key: str) -> list[list[float]]:
    vectors = []
    for sample in samples:
        seq = collect_vectors(sample.get(key))
        if seq:
            vectors.append(seq[0])
    return vectors


def all_vectors(samples: Sequence[dict], key: str) -> list[list[float]]:
    vectors = []
    for sample in samples:
        vectors.extend(collect_vectors(sample.get(key)))
    return vectors


def print_counter(title: str, counter: Counter[str], limit: int = 12) -> None:
    rows = [[name, count] for name, count in counter.most_common(limit)]
    print(f"\n{title}")
    print(table(("name", "count"), rows))
    if len(counter) > limit:
        print(f"... {len(counter) - limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize JSONL action/waypoint scale and simulate motor deltas."
    )
    parser.add_argument("--jsonl", default="data/car_train.jsonl", help="Training JSONL path.")
    parser.add_argument(
        "--control_dt",
        type=float,
        default=0.1,
        help="Seconds per waypoint step used when converting cumulative waypoint to action.",
    )
    parser.add_argument(
        "--indices",
        type=parse_csv_ints,
        default=parse_csv_ints("0,1,3,7"),
        help="Comma-separated waypoint indices to simulate.",
    )
    parser.add_argument(
        "--scales",
        type=parse_csv_floats,
        default=parse_csv_floats("300,400,1000,2000"),
        help="Comma-separated motor scales to simulate.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")
    if args.control_dt <= 0:
        raise ValueError("--control_dt must be positive")

    samples, episodes, commands = load_samples(jsonl_path)
    all_actions = all_vectors(samples, "actions")
    first_actions = first_vectors(samples, "actions")
    all_waypoints = all_vectors(samples, "waypoints")

    action_lengths = Counter(len(collect_vectors(sample.get("actions"))) for sample in samples)
    waypoint_lengths = Counter(len(collect_vectors(sample.get("waypoints"))) for sample in samples)

    print("Waypoint scale check")
    print(f"jsonl: {jsonl_path}")
    print(f"samples: {len(samples)}")
    print(f"all action vectors: {len(all_actions)}")
    print(f"all waypoint vectors: {len(all_waypoints)}")
    print(f"control_dt: {fmt(args.control_dt)} s")
    print(f"indices: {','.join(str(v) for v in args.indices)}")
    print(f"scales: {','.join(fmt(v) for v in args.scales)}")

    print_counter("Episodes", episodes)
    print_counter("Commands", commands)

    print("\nSequence lengths")
    rows = []
    for length in sorted(set(action_lengths) | set(waypoint_lengths)):
        rows.append([length, action_lengths.get(length, 0), waypoint_lengths.get(length, 0)])
    print(table(("length", "actions_samples", "waypoints_samples"), rows))

    print("\nAction scale, first action per sample")
    print(table(("dim", "n", "mean", "mean_abs", "p50_abs", "p90_abs", "max_abs"), component_stats(first_actions)))

    print("\nAction scale, all actions in horizon")
    print(table(("dim", "n", "mean", "mean_abs", "p50_abs", "p90_abs", "max_abs"), component_stats(all_actions)))

    print("\nWaypoint cumulative scale by index")
    waypoint_rows = []
    for index in args.indices:
        vectors = vectors_at_index(samples, "waypoints", index)
        horizon_s = (index + 1) * args.control_dt
        if not vectors:
            waypoint_rows.append([index, fmt(horizon_s), 0, "missing", "", "", "", ""])
            continue
        abs_forward = [abs(vec[0]) for vec in vectors]
        abs_yaw = [abs(vec[2]) for vec in vectors]
        waypoint_rows.append(
            [
                index,
                fmt(horizon_s),
                len(vectors),
                fmt(mean(vec[0] for vec in vectors)),
                fmt(mean(abs_forward)),
                fmt(percentile(abs_forward, 90)),
                fmt(max(abs_forward)),
                fmt(mean(abs_yaw)),
            ]
        )
    print(
        table(
            ("idx", "horizon_s", "n", "mean_x", "mean_abs_x", "p90_abs_x", "max_abs_x", "mean_abs_yaw"),
            waypoint_rows,
        )
    )

    print("\nMotor delta simulation from waypoints")
    print("direct:   wheel_delta = waypoint[index] * motor_scale")
    print("velocity: wheel_delta = waypoint[index] / ((index + 1) * control_dt) * motor_scale")
    motor_rows = []
    for index in args.indices:
        vectors = vectors_at_index(samples, "waypoints", index)
        horizon_s = (index + 1) * args.control_dt
        for scale in args.scales:
            direct = [max_abs_wheel_delta(vec, scale) for vec in vectors]
            velocity = [max_abs_wheel_delta([value / horizon_s for value in vec], scale) for vec in vectors]
            d_mean, d_p50, d_p90, d_max = summarize_motor(direct)
            v_mean, v_p50, v_p90, v_max = summarize_motor(velocity)
            motor_rows.append(
                [
                    index,
                    fmt(horizon_s),
                    fmt(scale),
                    d_mean,
                    d_p50,
                    d_p90,
                    d_max,
                    v_mean,
                    v_p50,
                    v_p90,
                    v_max,
                ]
            )
    print(
        table(
            (
                "idx",
                "horizon_s",
                "scale",
                "direct_mean",
                "direct_p50",
                "direct_p90",
                "direct_max",
                "velocity_mean",
                "velocity_p50",
                "velocity_p90",
                "velocity_max",
            ),
            motor_rows,
        )
    )

    if 1 in args.indices and 300.0 in args.scales:
        vectors = vectors_at_index(samples, "waypoints", 1)
        if vectors:
            direct_forward = [abs(vec[0]) * 300.0 for vec in vectors]
            direct_wheel = [max_abs_wheel_delta(vec, 300.0) for vec in vectors]
            horizon_s = 2.0 * args.control_dt
            velocity_forward = [abs(vec[0]) / horizon_s * 300.0 for vec in vectors]
            velocity_wheel = [
                max_abs_wheel_delta([value / horizon_s for value in vec], 300.0)
                for vec in vectors
            ]
            print("\nwp[1] explanation at scale=300")
            print(
                f"forward-only direct p50 |x| * 300 = {fmt(percentile(direct_forward, 50))}; "
                f"velocity p50 |x| / {fmt(horizon_s)} * 300 = {fmt(percentile(velocity_forward, 50))}"
            )
            print(
                f"wheel direct p50 max_delta = {fmt(percentile(direct_wheel, 50))}; "
                f"wheel velocity p50 max_delta = {fmt(percentile(velocity_wheel, 50))}"
            )


if __name__ == "__main__":
    main()
