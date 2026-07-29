#!/usr/bin/env python3
"""P1: Exploratory post-hoc action-change slice analysis.

Joins matched128_public_val_analysis_v1.rows.jsonl (per-row wMAE by method)
with val.jsonl (action fields) to compute H1 wMAE on four exploratory slices
of rows where the target action differs meaningfully from the previous action.

NOTE: These are EXPLORATORY POST-HOC slices, not preregistered metrics.
The paper must label them "exploratory post-hoc slice analysis".

Slices:
  1. yaw_sign_change   — yaw (axis 0) sign flips vs prev_action
  2. large_fwd_change  — |forward (axis 2 delta)| > 0.3 vs prev_action
  3. stop_to_fwd       — prev forward ≤ 0.2 AND target forward > 0.5
  4. high_h1_diff      — H1 wMAE between logged_prev and target > median

Methods reported: Harness, B0, persistence (repeat_logged_prev).
"""
from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

ROWS_PATH = (
    PROJECT_ROOT
    / "experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
    / "matched128_public_val_v2/analysis/matched128_public_val_analysis_v1.rows.jsonl"
)
VAL_PATH = PROJECT_ROOT / "data/collected_v1/datasets/val.jsonl"
OUTPUT_PATH = (
    PROJECT_ROOT
    / "experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
    / "multiseed_eval/p1_action_change_slices.json"
)

HARNESS_KEY = "Harness_F2_SSELF_update128"
B0_KEY = "TrackVLA_B0_matched128_seed0"
PERSISTENCE_KEY = "repeat_logged_prev"

CONTROL_WEIGHTS = [1.0, 0.0, 2.0]
CONTROL_DIVISOR = 3.0


def _h1_wmae_prev_vs_target(prev: list[float], target_step1: list[float]) -> float:
    """H1 wMAE between previous action and step-1 target (persistence error)."""
    err = sum(
        abs(prev[i] - target_step1[i]) * CONTROL_WEIGHTS[i]
        for i in range(3)
    ) / CONTROL_DIVISOR
    return err


def _source_macro(h1_values: list[float], sources: list[str]) -> float:
    """Source-macro mean identical to F2 analyzer."""
    by_source: dict[str, list[float]] = {}
    for v, s in zip(h1_values, sources):
        by_source.setdefault(s, []).append(v)
    means = [sum(vs) / len(vs) for vs in by_source.values()]
    return sum(means) / len(means) if means else float("nan")


def main() -> None:
    # --- Load rows.jsonl ----
    raw_rows = [
        json.loads(line)
        for line in ROWS_PATH.read_text("utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(raw_rows)} analysis rows")

    # --- Load val.jsonl for action fields ---
    val_rows = [
        json.loads(line)
        for line in VAL_PATH.read_text("utf-8").splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(val_rows)} val rows")

    # --- Merge action fields into analysis rows ---
    joined: list[dict[str, Any]] = []
    for ar in raw_rows:
        ri = ar["row_index"]
        vr = val_rows[ri]
        prev = [float(x) for x in vr["prev_action"][:3]]
        step_actions = [[float(x) for x in step[:3]] for step in vr["step_actions"]]
        target_h1 = step_actions[0]  # horizon-1 target
        valid_mask = [bool(v) for v in vr.get("valid_mask", [True] * 8)]

        persistence_h1 = _h1_wmae_prev_vs_target(prev, target_h1)

        joined.append({
            "row_index": ri,
            "source": ar["source_raw_dir"],
            "prev_action": prev,
            "target_h1": target_h1,
            "step_actions": step_actions,
            "valid_mask": valid_mask,
            "persistence_h1": persistence_h1,
            "harness_h1": ar["methods"][HARNESS_KEY]["h1_wmae"],
            "b0_h1": ar["methods"][B0_KEY]["h1_wmae"],
            "persistence_stored_h1": ar["methods"][PERSISTENCE_KEY]["h1_wmae"],
        })

    # --- Compute persistence H1 median for slice 4 ---
    all_persistence_h1 = [r["persistence_h1"] for r in joined]
    persistence_median = statistics.median(all_persistence_h1)
    print(f"Persistence H1 median: {persistence_median:.6f}")

    # --- Define slices ---
    # Action space encoding: yaw in {-1.0, 0.0, 0.5, 1.0}, fwd in [-0.5, 0.5]
    # yaw sign change: prev and target are nonzero with opposite signs
    def _yaw_sign_change(r: dict[str, Any]) -> bool:
        prev_yaw = r["prev_action"][0]
        target_yaw = r["target_h1"][0]
        return (prev_yaw > 0.1 and target_yaw < -0.1) or (
            prev_yaw < -0.1 and target_yaw > 0.1
        )

    # large forward change: delta > 0.2 on [-0.5, 0.5] scale (~40% of range)
    def _large_fwd_change(r: dict[str, Any]) -> bool:
        delta = abs(r["target_h1"][2] - r["prev_action"][2])
        return delta > 0.2

    # backward/stopped (fwd <= -0.1) transitioning to forward (fwd >= 0.1)
    def _stop_to_fwd(r: dict[str, Any]) -> bool:
        return r["prev_action"][2] <= -0.1 and r["target_h1"][2] >= 0.1

    # any action change: persistence error above median (= above 0 when median=0)
    def _high_h1_diff(r: dict[str, Any]) -> bool:
        return r["persistence_h1"] > persistence_median

    slices = {
        "yaw_sign_change": _yaw_sign_change,
        "large_fwd_change": _large_fwd_change,
        "stop_to_fwd": _stop_to_fwd,
        "high_persistence_error": _high_h1_diff,
        "overall": lambda r: True,  # sanity-check: should match published numbers
    }

    results: dict[str, Any] = {
        "note": (
            "EXPLORATORY POST-HOC SLICES — not preregistered, not abstract-claim eligible. "
            "All results reported regardless of direction."
        ),
        "n_total": len(joined),
        "persistence_h1_median": persistence_median,
        "slices": {},
    }

    print(f"\n{'Slice':<30} {'N':>6}  {'Harness H1':>12}  {'B0 H1':>12}  {'Persist H1':>12}  {'H vs P':>10}")
    print("-" * 90)

    for slice_name, predicate in slices.items():
        subset = [r for r in joined if predicate(r)]
        n = len(subset)
        if n == 0:
            print(f"{slice_name:<30} {'0':>6}  {'—':>12}  {'—':>12}  {'—':>12}  {'—':>10}")
            results["slices"][slice_name] = {"n": 0}
            continue

        sources = [r["source"] for r in subset]
        harness_h1_sm = _source_macro([r["harness_h1"] for r in subset], sources)
        b0_h1_sm = _source_macro([r["b0_h1"] for r in subset], sources)
        persist_h1_sm = _source_macro([r["persistence_h1"] for r in subset], sources)

        # Harness vs persistence: negative = Harness better
        h_vs_p_pct = (harness_h1_sm - persist_h1_sm) / persist_h1_sm * 100

        print(
            f"{slice_name:<30} {n:>6}  "
            f"{harness_h1_sm:>12.8f}  {b0_h1_sm:>12.8f}  "
            f"{persist_h1_sm:>12.8f}  {h_vs_p_pct:>+10.1f}%"
        )

        results["slices"][slice_name] = {
            "n": n,
            "harness_h1_source_macro": harness_h1_sm,
            "b0_h1_source_macro": b0_h1_sm,
            "persistence_h1_source_macro": persist_h1_sm,
            "harness_minus_persistence_pct": round(h_vs_p_pct, 2),
            "harness_better_than_persistence": harness_h1_sm < persist_h1_sm,
        }

    # Overall sanity-check against published numbers
    overall = results["slices"].get("overall", {})
    pub_harness_h1 = 0.05912906
    pub_b0_h1 = 0.15377812
    tol = 1e-5
    harness_ok = math.isfinite(overall.get("harness_h1_source_macro", float("nan"))) and (
        abs(overall["harness_h1_source_macro"] - pub_harness_h1) < tol
    )
    b0_ok = math.isfinite(overall.get("b0_h1_source_macro", float("nan"))) and (
        abs(overall["b0_h1_source_macro"] - pub_b0_h1) < tol
    )
    print(f"\nSanity: Harness overall={'OK' if harness_ok else 'MISMATCH'} "
          f"(got {overall.get('harness_h1_source_macro','?'):.8f}, "
          f"expected {pub_harness_h1})")
    print(f"Sanity: B0 overall={'OK' if b0_ok else 'MISMATCH'} "
          f"(got {overall.get('b0_h1_source_macro','?'):.8f}, "
          f"expected {pub_b0_h1})")
    results["sanity"] = {"harness_matches_published": harness_ok, "b0_matches_published": b0_ok}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
