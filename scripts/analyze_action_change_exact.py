#!/usr/bin/env python3
"""Exp2: Action-Change Focused Metric (exact matching, bootstrap CIs).

Computes per-slice H1 wMAE for rows where the action genuinely changes
(exact discrete equality check, no thresholds).

EXPLORATORY POST-HOC — not preregistered, not abstract-claim eligible.

Slices:
  1. action_change      — H1 target != prev_action (any axis)
  2. yaw_change         — target yaw != prev yaw
  3. forward_change     — target fwd != prev fwd
  4. residual_error     — per-row |target - prev| weighted MAE (persistence gap),
                          compare Harness vs B0 normalized by persistence
  5. persistence_normalized_H1 — H1_method / H1_persistence per row, then mean
"""
from __future__ import annotations

import json
import math
import pathlib
import random
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
    / "multiseed_eval/p2_action_change_metrics.json"
)

HARNESS_KEY = "Harness_F2_SSELF_update128"
B0_KEY = "TrackVLA_B0_matched128_seed0"
PERSISTENCE_KEY = "repeat_logged_prev"

CONTROL_WEIGHTS = [1.0, 0.0, 2.0]
CONTROL_DIVISOR = 3.0

BOOTSTRAP_REPLICATES = 2000
BLOCK_LENGTH = 20
BOOTSTRAP_SEED = 42


def _wmae(a: list[float], b: list[float]) -> float:
    return sum(abs(a[i] - b[i]) * CONTROL_WEIGHTS[i] for i in range(3)) / CONTROL_DIVISOR


def _source_macro(vals: list[float], sources: list[str]) -> float:
    by_src: dict[str, list[float]] = {}
    for v, s in zip(vals, sources):
        by_src.setdefault(s, []).append(v)
    means = [sum(vs) / len(vs) for vs in by_src.values()]
    return sum(means) / len(means) if means else float("nan")


def _moving_block_bootstrap_ci(
    values: list[float],
    n_replicates: int,
    block_length: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """95% CI for the mean via moving-block bootstrap."""
    rng = random.Random(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    observed_mean = sum(values) / n
    replicate_means: list[float] = []
    for _ in range(n_replicates):
        sample: list[float] = []
        while len(sample) < n:
            start = rng.randint(0, n - 1)
            sample.extend(values[start : start + block_length])
        sample = sample[:n]
        replicate_means.append(sum(sample) / len(sample))
    replicate_means.sort()
    lo_idx = int((1 - confidence) / 2 * n_replicates)
    hi_idx = int((1 - (1 - confidence) / 2) * n_replicates)
    return replicate_means[lo_idx], replicate_means[hi_idx]


def main() -> None:
    # --- Load data ---
    ar_lines = ROWS_PATH.read_text("utf-8").splitlines()
    analysis_rows = [json.loads(l) for l in ar_lines if l.strip()]
    val_lines = VAL_PATH.read_text("utf-8").splitlines()
    val_rows = [json.loads(l) for l in val_lines if l.strip()]
    print(f"Loaded {len(analysis_rows)} analysis rows, {len(val_rows)} val rows")

    # --- Join action fields ---
    joined: list[dict[str, Any]] = []
    for ar in analysis_rows:
        ri = ar["row_index"]
        vr = val_rows[ri]
        prev = [float(x) for x in vr["prev_action"][:3]]
        step1 = [float(x) for x in vr["step_actions"][0][:3]]

        harness_h1 = ar["methods"][HARNESS_KEY]["h1_wmae"]
        b0_h1 = ar["methods"][B0_KEY]["h1_wmae"]
        # recompute persistence using raw actions (should match stored)
        persist_h1 = _wmae(prev, step1)

        joined.append({
            "ri": ri,
            "source": ar["source_raw_dir"],
            "prev": prev,
            "step1": step1,
            "harness_h1": harness_h1,
            "b0_h1": b0_h1,
            "persist_h1": persist_h1,
            "stored_persist_h1": ar["methods"][PERSISTENCE_KEY]["h1_wmae"],
        })

    # Sanity: verify persistence recomputation
    n_mismatch = sum(
        1 for r in joined
        if abs(r["persist_h1"] - r["stored_persist_h1"]) > 1e-9
    )
    print(f"Persistence recomputation mismatches: {n_mismatch}/{len(joined)}")

    # --- Define exact-match slice predicates ---
    def _actions_equal(a: list[float], b: list[float]) -> bool:
        return a[0] == b[0] and a[2] == b[2]  # yaw and fwd only (mid axis always 0)

    def _in_slice(name: str, r: dict[str, Any]) -> bool:
        prev, step1 = r["prev"], r["step1"]
        if name == "overall":
            return True
        if name == "action_change":
            return not _actions_equal(prev, step1)
        if name == "action_no_change":
            return _actions_equal(prev, step1)
        if name == "yaw_change":
            return prev[0] != step1[0]
        if name == "forward_change":
            return prev[2] != step1[2]
        return False

    slices = ["overall", "action_change", "action_no_change", "yaw_change", "forward_change"]

    results: dict[str, Any] = {
        "schema_version": 1,
        "analysis_label": "EXPLORATORY_POST_HOC",
        "note": (
            "Exploratory post-hoc action-change slices. Not preregistered. "
            "Action equality uses exact floating-point match on discrete action space "
            "{yaw: {-1,0,0.5,1}, fwd: {-0.5,0,0.5}}. "
            "Bootstrap CIs use moving-block bootstrap (B=2000, block=20)."
        ),
        "n_total": len(joined),
        "bootstrap_params": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_length": BLOCK_LENGTH,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "slices": {},
    }

    print(f"\n{'Slice':<22} {'N':>6}  {'Harness':>10}  {'B0':>10}  {'Persist':>10}  {'H-B0':>8}  {'H-P':>8}")
    print("-" * 85)

    for slice_name in slices:
        subset = [r for r in joined if _in_slice(slice_name, r)]
        n = len(subset)
        if n == 0:
            print(f"{slice_name:<22} {'0':>6}")
            results["slices"][slice_name] = {"n": 0}
            continue

        sources = [r["source"] for r in subset]
        harness_h1_list = [r["harness_h1"] for r in subset]
        b0_h1_list = [r["b0_h1"] for r in subset]
        persist_h1_list = [r["persist_h1"] for r in subset]

        h_sm = _source_macro(harness_h1_list, sources)
        b_sm = _source_macro(b0_h1_list, sources)
        p_sm = _source_macro(persist_h1_list, sources)

        h_vs_b0 = (h_sm - b_sm) / b_sm * 100 if b_sm > 0 else float("nan")
        h_vs_p  = (h_sm - p_sm) / p_sm * 100 if p_sm > 0 else float("nan")

        # Bootstrap CI on Harness row-mean H1
        h_row_mean = sum(harness_h1_list) / n
        lo, hi = _moving_block_bootstrap_ci(
            harness_h1_list, BOOTSTRAP_REPLICATES, BLOCK_LENGTH, BOOTSTRAP_SEED
        )

        print(
            f"{slice_name:<22} {n:>6}  "
            f"{h_sm:>10.6f}  {b_sm:>10.6f}  {p_sm:>10.6f}  "
            f"{h_vs_b0:>+7.1f}%  {h_vs_p:>+7.1f}%"
        )

        results["slices"][slice_name] = {
            "n": n,
            "pct_of_total": round(n / len(joined) * 100, 2),
            "harness": {
                "h1_source_macro": h_sm,
                "h1_row_mean": h_row_mean,
                "h1_bootstrap_ci95": [lo, hi],
            },
            "b0": {"h1_source_macro": b_sm},
            "persistence": {"h1_source_macro": p_sm},
            "harness_vs_b0_pct": round(h_vs_b0, 3),
            "harness_vs_persistence_pct": round(h_vs_p, 3),
            "harness_beats_b0": h_sm < b_sm,
            "harness_beats_persistence": h_sm < p_sm,
        }

    # --- Slice 4: persistence-normalized H1 ---
    print("\n--- persistence_normalized_H1 (H1_method / H1_persistence, only rows where persist > 0) ---")
    changing_rows = [r for r in joined if r["persist_h1"] > 1e-9]
    n_norm = len(changing_rows)
    if n_norm > 0:
        harness_norm = [r["harness_h1"] / r["persist_h1"] for r in changing_rows]
        b0_norm = [r["b0_h1"] / r["persist_h1"] for r in changing_rows]
        h_norm_mean = sum(harness_norm) / n_norm
        b_norm_mean = sum(b0_norm) / n_norm
        h_lo, h_hi = _moving_block_bootstrap_ci(
            harness_norm, BOOTSTRAP_REPLICATES, BLOCK_LENGTH, BOOTSTRAP_SEED
        )
        print(f"  n={n_norm}")
        print(f"  Harness normalized H1 mean: {h_norm_mean:.4f}  (CI95: [{h_lo:.4f}, {h_hi:.4f}])")
        print(f"  B0     normalized H1 mean: {b_norm_mean:.4f}")
        print(f"  < 1.0 means method beats persistence on these rows")

        results["persistence_normalized_h1"] = {
            "analysis_label": "EXPLORATORY_POST_HOC",
            "n_rows_with_nonzero_persistence": n_norm,
            "harness_mean_normalized": h_norm_mean,
            "harness_ci95": [h_lo, h_hi],
            "b0_mean_normalized": b_norm_mean,
            "note": "< 1.0 = method beats persistence on rows where action changes",
        }

    # --- Slice 5: residual error analysis ---
    print("\n--- residual_error (|target - prev| wMAE: how hard each row is) ---")
    residuals = [r["persist_h1"] for r in joined]
    res_mean = sum(residuals) / len(residuals)
    res_median = statistics.median(residuals)
    n_nonzero = sum(1 for v in residuals if v > 1e-9)
    print(f"  Persistence H1 mean={res_mean:.6f} median={res_median:.6f}")
    print(f"  Rows with nonzero action change: {n_nonzero}/{len(joined)} ({n_nonzero/len(joined)*100:.1f}%)")

    results["residual_summary"] = {
        "persistence_h1_mean": res_mean,
        "persistence_h1_median": res_median,
        "n_nonzero_change": n_nonzero,
        "pct_nonzero_change": round(n_nonzero / len(joined) * 100, 2),
    }

    # --- Sanity check ---
    overall = results["slices"].get("overall", {})
    pub_h = 0.05912906
    pub_b = 0.15377812
    h_ok = abs(overall.get("harness", {}).get("h1_source_macro", 999) - pub_h) < 1e-5
    b_ok = abs(overall.get("b0", ).get("h1_source_macro", 999) - pub_b) < 1e-5
    print(f"\nSanity: Harness overall={'OK' if h_ok else 'MISMATCH'} "
          f"(got {overall.get('harness',{}).get('h1_source_macro','?'):.8f})")
    print(f"Sanity: B0 overall={'OK' if b_ok else 'MISMATCH'} "
          f"(got {overall.get('b0',{}).get('h1_source_macro','?'):.8f})")
    results["sanity"] = {"harness_matches_published": h_ok, "b0_matches_published": b_ok}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\nWritten: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
