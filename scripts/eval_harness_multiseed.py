#!/usr/bin/env python3
# ruff: noqa: E402
"""Harness (F2 / SA-Hstar) multi-seed public-validation evaluator.

Runs inference on the frozen 2,848-row public validation set using an
SA-Hstar S-SELF checkpoint trained with an arbitrary seed. Reports H1 and
All8 source-macro wMAE using identical metric math to the official seed-0
evaluation (predictions_full_logged.jsonl + CONTROL_WEIGHTS).

Only "full" condition + "logged" mode is run — that is the primary metric
used for the abstract claim.  Seeds 1/2 results are extensions for the
multi-seed reproducibility check; no new freeze boundary is created.

Usage:
    python scripts/eval_harness_multiseed.py \
        --checkpoint experiments/.../F2_seed1/checkpoint_update128_S-SELF.pt \
        --seed 1 \
        --output-dir experiments/.../multiseed_eval
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

CONTROL_WEIGHTS = np.asarray([1.0, 0.0, 2.0], dtype=np.float64)
CONTROL_DIVISOR = 3.0
VAL_ROWS = 2848
# Combined reset indices (episode boundaries + strafe resets)
COMBINED_RESET_INDICES = frozenset((0, 346, 347, 348, 349, 512, 924, 1886))
# arm name used during training
S_SELF = "S-SELF"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _source_macro(
    values: np.ndarray,
    sources: np.ndarray,
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Source-macro mean of per-row wMAE values (identical to F2 analyzer)."""
    by_source: dict[str, dict[str, Any]] = {}
    means: list[float] = []
    for source in sorted(set(sources.tolist())):
        selected = sources == source
        support = int(selected.sum())
        if support:
            mean = float(values[selected].mean())
            means.append(mean)
            by_source[source] = {"support": support, "mean": mean}
        else:
            by_source[source] = {"support": 0, "mean": None}
    return float(np.mean(means)) if means else float("nan"), by_source


def _weighted_mae_rows(
    predictions: np.ndarray,  # (N, 8, 3)
    targets: np.ndarray,       # (N, 8, 3)
    valid: np.ndarray,         # (N, 8) bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row H1 and All8 wMAE (same formula as F2 analyzer)."""
    errors = (
        np.abs(predictions - targets) * CONTROL_WEIGHTS.reshape(1, 1, 3)
    ).sum(axis=-1) / CONTROL_DIVISOR
    h1 = errors[:, 0]
    all8 = (errors * valid).sum(axis=1) / valid.sum(axis=1)
    return h1, all8


def run_harness_eval(
    checkpoint_path: Path,
    seed: int,
    output_dir: Path,
    project_root: Path,
    method_name: str = "Harness",
) -> dict[str, Any]:
    start_utc = _utc_now()
    start_t = time.perf_counter()

    # --- Determinism setup ---
    from f2_experiment.reproducibility import configure_cuda_reproducibility
    configure_cuda_reproducibility()
    if not torch.cuda.is_available():
        sys.exit("CUDA required for Harness evaluation")
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)

    # --- Locate assembly receipt (same as training) ---
    receipt_path = (
        project_root
        / "experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json"
    )
    if not receipt_path.exists():
        sys.exit(f"Assembly receipt not found: {receipt_path}")
    receipt_document = json.loads(receipt_path.read_text("utf-8"))
    receipt_sha = _sha256_file(receipt_path)
    print(f"[eval_harness] receipt sha={receipt_sha[:16]}...")

    # --- Load checkpoint payload (no SHA pin — multi-seed extension) ---
    ckpt_sha = _sha256_file(checkpoint_path)
    print(f"[eval_harness] checkpoint {checkpoint_path.name} sha={ckpt_sha[:16]}...")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model" not in raw:
        sys.exit("Checkpoint missing 'model' key")
    payload: dict[str, Any] = {
        "model": raw["model"],
        "arm": raw.get("arm", S_SELF),
        "u_pre": raw.get("u_pre", 128),
    }
    # strip assembly_receipt_sha256 so build_eval_row_predictor_from_checkpoint
    # doesn't enforce it (our receipt matches but the payload SHA check is on
    # the init state, not the receipt itself)
    if "assembly_receipt_sha256" in raw:
        payload["assembly_receipt_sha256"] = raw["assembly_receipt_sha256"]

    # --- Build inference arm ---
    print("[eval_harness] building SA-Hstar inference arm (loads 609M base) ...")
    from f2_experiment.assembly_model import build_eval_row_predictor_from_checkpoint
    predictor = build_eval_row_predictor_from_checkpoint(
        project_root,
        receipt_document,
        S_SELF,
        payload,
        device=device,
    )
    arm = predictor.arm
    del payload
    print(f"[eval_harness] arm built on {device}")

    # --- Load frozen val rows ---
    val_jsonl = project_root / "data/collected_v1/datasets/val.jsonl"
    if not val_jsonl.exists():
        sys.exit(f"val.jsonl not found: {val_jsonl}")
    all_rows = [
        json.loads(line) for line in val_jsonl.read_text("utf-8").splitlines()
        if line.strip()
    ]
    if len(all_rows) != VAL_ROWS:
        sys.exit(f"Expected {VAL_ROWS} val rows, got {len(all_rows)}")
    val_sha = hashlib.sha256(val_jsonl.read_bytes()).hexdigest()
    print(f"[eval_harness] loaded {VAL_ROWS} val rows, sha={val_sha[:16]}...")

    # --- Load frozen token ledger ---
    from f2_experiment.assembly_data import (
        load_cached_observation,
        frozen_cache_roots,
    )
    token_ledger_path = (
        project_root
        / "experiments/windows_cuda_f2/public_val_memory_reasoning_v1"
        / "full_2848/token_ledger.json"
    )
    token_ledger_receipt_path = token_ledger_path.with_name("token_ledger_receipt.json")
    if token_ledger_path.exists() and token_ledger_receipt_path.exists():
        from f2_experiment.validation_diagnostics import (
            F2ValidationDiagnosticError,
        )
        try:
            from f2_experiment.assembly_data import TokenHashLedger
            raw_ledger = json.loads(token_ledger_path.read_text("utf-8"))
            token_ledger = TokenHashLedger(entries=raw_ledger)
            print(f"[eval_harness] loaded frozen token ledger ({len(raw_ledger)} entries)")
        except Exception as exc:
            print(f"[eval_harness] WARNING: ledger load via TokenHashLedger failed: {exc}")
            token_ledger = None
    else:
        token_ledger = None

    if token_ledger is None:
        print("[eval_harness] building token ledger on-the-fly ...")
        from f2_experiment.assembly_data import build_token_ledger_for_rows
        _base_root, _cache_root = frozen_cache_roots(project_root)
        token_ledger = build_token_ledger_for_rows(
            all_rows,
            base_root=_base_root,
            cache_root=_cache_root,
        )
        print(f"[eval_harness] token ledger built")

    # --- Rollout state tracking (logged mode only) ---
    from f2_experiment.validation_diagnostics import (
        PerceptionStream,
        intervention_alphas,
    )
    normal_stream = PerceptionStream(arm, reset_every_row=False)

    base_root, cache_root = frozen_cache_roots(project_root)

    all_pred_step_actions: list[list[list[float]]] = []
    all_step_actions: list[list[list[float]]] = []
    all_valid_masks: list[list[bool]] = []
    all_sources: list[str] = []

    print(f"[eval_harness] starting inference on {VAL_ROWS} rows ...")
    inf_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for position, row in enumerate(all_rows):
            reset = position in COMBINED_RESET_INDICES
            if position == 0 and not reset:
                sys.exit("First val row must be a reset boundary")

            packet = load_cached_observation(
                row,
                base_root=base_root,
                cache_root=cache_root,
                token_ledger=token_ledger,
            )
            feature_output = normal_stream.encode(packet, reset=reset, position=position)

            # Logged mode: prev comes from the row's logged_prev_action
            prev_action = row.get("prev_action", [0.5, 0.0, 0.5])
            logged_prev = (float(prev_action[0]), float(prev_action[2]))
            prev_tensor = torch.tensor(
                [list(logged_prev)],
                device=feature_output["base_features"].device,
                dtype=feature_output["base_features"].dtype,
            )

            # Full condition: no intervention alphas zeroed
            alphas = intervention_alphas(feature_output, "full")

            model_output = arm.model(
                feature_output["base_features"],
                prev_tensor,
                method_features=feature_output["method_features"],
                method_alphas=alphas,
            )
            raw_actions = (
                model_output.prediction.raw_actions.detach().float()[0].cpu().tolist()
            )

            all_pred_step_actions.append(raw_actions)
            all_step_actions.append(
                [[float(a) for a in step[:3]] for step in row["step_actions"]]
            )
            all_valid_masks.append(
                [bool(v) for v in row.get("valid_mask", [True] * 8)]
            )
            all_sources.append(str(row.get("source_raw_dir", row.get("episode", ""))))

            if (position + 1) % 256 == 0 or position + 1 == VAL_ROWS:
                elapsed = time.perf_counter() - inf_start
                peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
                print(
                    f"  [{position + 1}/{VAL_ROWS}] "
                    f"{elapsed:.1f}s  {(position+1)/elapsed:.1f} rows/s  "
                    f"peak {peak_mb:.0f}MiB"
                )

    inf_elapsed = time.perf_counter() - inf_start
    print(f"[eval_harness] inference complete in {inf_elapsed:.1f}s")

    # --- Compute wMAE ---
    preds = np.asarray(all_pred_step_actions, dtype=np.float64)   # (N, 8, 3)
    targets = np.asarray(all_step_actions, dtype=np.float64)       # (N, 8, 3)
    valid = np.asarray(all_valid_masks, dtype=np.float64)          # (N, 8)
    sources = np.asarray(all_sources, dtype=object)

    h1, all8 = _weighted_mae_rows(preds, targets, valid)
    h1_source_macro, h1_by_source = _source_macro(h1, sources)
    all8_source_macro, all8_by_source = _source_macro(all8, sources)

    end_utc = _utc_now()
    total_elapsed = time.perf_counter() - start_t

    result = {
        "schema_version": 1,
        "analysis_class": "multiseed_harness_public_val_result_v1",
        "method": "SA-Hstar",
        "seed": seed,
        "h1_source_macro_wmae": h1_source_macro,
        "all8_source_macro_wmae": all8_source_macro,
        "val_rows": VAL_ROWS,
        "val_sha256": val_sha,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": ckpt_sha,
        "receipt_sha256": receipt_sha,
        "assembly_receipt_sha256_from_ckpt": str(raw.get("assembly_receipt_sha256", "")),
        "u_pre": int(raw.get("u_pre", 128)),
        "arm": str(raw.get("arm", S_SELF)),
        "gpu": gpu_name,
        "h1_by_source": h1_by_source,
        "all8_by_source": all8_by_source,
        "start_utc": start_utc,
        "end_utc": end_utc,
        "total_elapsed_s": round(total_elapsed, 1),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"eval_result_{method_name}_seed{seed}.json"
    out_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"\n[eval_harness] RESULT seed={seed}")
    print(f"  H1  source-macro wMAE : {h1_source_macro:.8f}")
    print(f"  All8 source-macro wMAE: {all8_source_macro:.8f}")
    print(f"  -> {out_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SA-Hstar checkpoint on frozen 2848-row public val set"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to S-SELF .pt checkpoint")
    parser.add_argument("--seed", type=int, required=True, help="Seed used during training")
    parser.add_argument("--output-dir", required=True, help="Directory to write eval result JSON")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root")
    parser.add_argument(
        "--method-name", default="Harness",
        help="Method label for output filename: eval_result_{method_name}_seed{seed}.json"
    )
    args = parser.parse_args()

    run_harness_eval(
        checkpoint_path=Path(args.checkpoint).resolve(),
        seed=args.seed,
        output_dir=Path(args.output_dir).resolve(),
        project_root=Path(args.project_root).resolve(),
        method_name=args.method_name,
    )


if __name__ == "__main__":
    main()
