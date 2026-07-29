from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import analyze_f2_public_val_outputs as analysis


def _rows(length: int) -> list[dict[str, object]]:
    return [
        {
            "episode": "episode-0",
            "sequence_id": "sequence-0",
            "source_raw_dir": "source-0",
            "frame_idx": index,
        }
        for index in range(length)
    ]


def test_default_baseline_registry_matches_frozen_comparator_set() -> None:
    assert analysis.DEFAULT_CHECKPOINT_SHA256 == (
        "0d729150d580250852428f7573bade5b158840a0bd3f505c587e85d1a44f1080"
    )
    assert tuple(analysis.DEFAULT_BASELINES) == (
        "B0_seed0",
        "B1_seed0",
        "B1_seed1",
        "B1_seed2",
    )
    for _name, (path, digest) in analysis.DEFAULT_BASELINES.items():
        assert path.startswith(
            "experiments/collected_v1_main/validation_eval_v12/predictions/"
        )
        assert analysis._is_sha256(digest)


def test_moving_block_bootstrap_truncates_final_block_to_sequence_length() -> None:
    rows = _rows(5)
    plan = analysis._bootstrap_plan(
        rows,
        replicates=1000,
        block_length=3,
        seed=20260722,
    )
    assert plan[0]["draw_lengths"] == [3, 2]

    full = np.asarray([0.0, 1.0, 2.0, 10.0, 20.0], dtype=np.float64)
    reference = np.zeros(5, dtype=np.float64)
    mask = np.ones(5, dtype=bool)
    result = analysis.moving_block_effect(full, reference, mask, rows, plan)

    draws = np.asarray(plan[0]["draws"], dtype=np.int64)
    expected = np.empty(draws.shape[0], dtype=np.float64)
    for replicate, (first_start, second_start) in enumerate(draws):
        expected[replicate] = (
            full[first_start : first_start + 3].sum()
            + full[second_start : second_start + 2].sum()
        ) / 5.0

    assert result["valid_replicates"] == 1000
    assert result["bootstrap_mean"] == pytest.approx(float(expected.mean()))
    assert result["ci95"] == pytest.approx(
        [
            float(np.quantile(expected, 0.025, method="linear")),
            float(np.quantile(expected, 0.975, method="linear")),
        ]
    )


@pytest.mark.parametrize(
    "path",
    [
        "data/collected_v1/datasets/test.jsonl",
        "data/collected_v1/episodes/test/predictions.jsonl",
        "experiments/internal_test/run.json",
    ],
)
def test_reject_sealed_path_blocks_internal_test_surface(path: str) -> None:
    with pytest.raises(analysis.F2PublicValAnalysisError, match="sealed"):
        analysis.reject_sealed_path(path, "artifact")


def _write_json(path, value: dict[str, object]) -> str:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return analysis.sha256_file(path)


def _preregistration(analyzer_sha256: str) -> dict[str, object]:
    return {
        "analysis_class": "f2_public_val_memory_reasoning_preregistration",
        "status": "frozen_before_first_f2_public_val_prediction",
        "analyzer_source_sha256": analyzer_sha256,
        "candidate": {
            "checkpoint_sha256": analysis.DEFAULT_CHECKPOINT_SHA256,
        },
        "public_validation": {
            "sha256": analysis.DEFAULT_VAL_SHA256,
            "manifest_sha256": analysis.DEFAULT_MANIFEST_SHA256,
            "rows": analysis.DEFAULT_ROW_COUNT,
            "split": "val",
            "internal_test_opened": False,
        },
        "evaluation": {
            "conditions": list(analysis.CONDITIONS),
            "modes": list(analysis.MODES),
            "claim_eligible_selection": "full_2848_public_validation",
            "baselines": list(analysis.DEFAULT_BASELINES),
            "baseline_prediction_sha256": {
                name: digest
                for name, (_path, digest) in analysis.DEFAULT_BASELINES.items()
            },
            "combined_reset_indices": list(analysis.DEFAULT_RESET_INDICES),
            "combined_reset_sha256": analysis.DEFAULT_COMBINED_RESET_SHA256,
            "control_threshold": analysis.CONTROL_THRESHOLD,
            "analysis": {
                "bootstrap_replicates": analysis.DEFAULT_BOOTSTRAP_REPLICATES,
                "block_length": analysis.DEFAULT_BLOCK_LENGTH,
                "analysis_seed": analysis.DEFAULT_ANALYSIS_SEED,
            },
        },
        "internal_test_opened": False,
    }


def test_preregistration_binds_exact_analyzer_source(tmp_path) -> None:
    preregistration_path = tmp_path / "preregistration.json"
    analyzer_path = tmp_path / "analyzer.py"
    analyzer_path.write_text("# frozen analyzer\n", encoding="utf-8")
    analyzer_sha256 = analysis.sha256_file(analyzer_path)
    preregistration = _preregistration(analyzer_sha256)
    preregistration_sha256 = _write_json(preregistration_path, preregistration)

    _document, binding = analysis.load_preregistration(
        preregistration_path,
        expected_sha256=preregistration_sha256,
        analyzer_source_path=analyzer_path,
    )
    assert binding["analyzer_source_sha256"] == analyzer_sha256

    preregistration["analyzer_source_sha256"] = "0" * 64
    changed_sha256 = _write_json(preregistration_path, preregistration)
    with pytest.raises(
        analysis.F2PublicValAnalysisError,
        match="source changed",
    ):
        analysis.load_preregistration(
            preregistration_path,
            expected_sha256=changed_sha256,
            analyzer_source_path=analyzer_path,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("baselines", ["ARBITRARY"], "baseline names/order"),
        (
            "baseline_prediction_sha256",
            {"B0_seed0": "0" * 64},
            "baseline SHA registry",
        ),
        ("combined_reset_indices", [0], "reset indices"),
        (
            "analysis",
            {
                "bootstrap_replicates": 9999,
                "block_length": analysis.DEFAULT_BLOCK_LENGTH,
                "analysis_seed": analysis.DEFAULT_ANALYSIS_SEED,
            },
            "analysis settings",
        ),
    ],
)
def test_preregistration_rejects_frozen_analysis_tampering(
    tmp_path, field, value, message
) -> None:
    preregistration_path = tmp_path / "preregistration.json"
    analyzer_path = tmp_path / "analyzer.py"
    analyzer_path.write_text("# frozen analyzer\n", encoding="utf-8")
    preregistration = _preregistration(analysis.sha256_file(analyzer_path))
    preregistration["evaluation"][field] = value
    preregistration_sha256 = _write_json(preregistration_path, preregistration)

    with pytest.raises(analysis.F2PublicValAnalysisError, match=message):
        analysis.load_preregistration(
            preregistration_path,
            expected_sha256=preregistration_sha256,
            analyzer_source_path=analyzer_path,
        )


def _analysis_args(**overrides):
    values = {
        "expected_val_sha256": analysis.DEFAULT_VAL_SHA256,
        "expected_manifest_sha256": analysis.DEFAULT_MANIFEST_SHA256,
        "expected_row_count": analysis.DEFAULT_ROW_COUNT,
        "expected_reset_indices": ",".join(
            str(value) for value in analysis.DEFAULT_RESET_INDICES
        ),
        "bootstrap_replicates": analysis.DEFAULT_BOOTSTRAP_REPLICATES,
        "block_length": analysis.DEFAULT_BLOCK_LENGTH,
        "analysis_seed": analysis.DEFAULT_ANALYSIS_SEED,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"expected_val_sha256": "0" * 64}, "expected_val_sha256"),
        ({"expected_manifest_sha256": "0" * 64}, "expected_manifest_sha256"),
        ({"expected_row_count": 2847}, "expected_row_count"),
        ({"expected_reset_indices": "0"}, "expected_reset_indices"),
        ({"bootstrap_replicates": 9999}, "bootstrap_replicates"),
        ({"block_length": 16}, "block_length"),
        ({"analysis_seed": 1}, "analysis_seed"),
    ],
)
def test_frozen_analysis_cli_rejects_contract_overrides(override, message) -> None:
    preregistration = _preregistration("0" * 64)
    with pytest.raises(analysis.F2PublicValAnalysisError, match=message):
        analysis._validate_frozen_analysis_args(
            _analysis_args(**override), preregistration
        )


def test_baseline_cli_rejects_arbitrary_or_reordered_comparators(tmp_path) -> None:
    arbitrary = SimpleNamespace(
        baseline=[f"ARBITRARY={tmp_path / 'arbitrary.jsonl'}"],
        expected_baseline_sha256=[f"ARBITRARY={'0' * 64}"],
    )
    with pytest.raises(analysis.F2PublicValAnalysisError, match="names/order"):
        analysis._baseline_bindings(arbitrary, tmp_path)

    frozen_items = list(analysis.DEFAULT_BASELINES.items())
    reversed_items = list(reversed(frozen_items))
    reordered = SimpleNamespace(
        baseline=[f"{name}={tmp_path / (name + '.jsonl')}" for name, _ in reversed_items],
        expected_baseline_sha256=[
            f"{name}={value[1]}" for name, value in reversed_items
        ],
    )
    with pytest.raises(analysis.F2PublicValAnalysisError, match="names/order"):
        analysis._baseline_bindings(reordered, tmp_path)


def test_baseline_cli_rejects_caller_reported_sha_drift(tmp_path) -> None:
    supplied = SimpleNamespace(
        baseline=[
            f"{name}={tmp_path / (name + '.jsonl')}"
            for name in analysis.DEFAULT_BASELINES
        ],
        expected_baseline_sha256=[
            f"{name}={'0' * 64}" for name in analysis.DEFAULT_BASELINES
        ],
    )
    with pytest.raises(analysis.F2PublicValAnalysisError, match="frozen registry"):
        analysis._baseline_bindings(supplied, tmp_path)


def test_completion_requires_full_claim_eligible_selection(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metrics_path = run_dir / "metrics.json"
    prediction_path = run_dir / "prediction.jsonl"
    metrics_path.write_text("{}\n", encoding="utf-8")
    prediction_path.write_text("{}\n", encoding="utf-8")
    preregistration_sha256 = "1" * 64
    completion = {
        "schema_version": analysis.SCHEMA_VERSION,
        "analysis_class": "f2_public_validation_memory_reasoning_completion",
        "status": "PASS_FULL_PUBLIC_VALIDATION",
        "selection_name": "full_2848_public_validation",
        "selection_sha256": analysis.DEFAULT_FULL_SELECTION_SHA256,
        "checkpoint_sha256": analysis.DEFAULT_CHECKPOINT_SHA256,
        "abstract_claim_eligible": True,
        "truncated_smoke": False,
        "internal_test_opened": False,
        "rows": analysis.DEFAULT_ROW_COUNT,
        "preregistration_sha256": preregistration_sha256,
        "metrics_sha256": analysis.sha256_file(metrics_path),
        "artifact_sha256": {
            metrics_path.name: analysis.sha256_file(metrics_path),
            prediction_path.name: analysis.sha256_file(prediction_path),
        },
    }
    _write_json(run_dir / "complete.json", completion)
    verified, _hashes = analysis._verify_completion(
        run_dir,
        [metrics_path, prediction_path],
        expected_rows=analysis.DEFAULT_ROW_COUNT,
        expected_preregistration_sha256=preregistration_sha256,
    )
    assert verified["status"] == "PASS_FULL_PUBLIC_VALIDATION"

    completion["truncated_smoke"] = True
    _write_json(run_dir / "complete.json", completion)
    with pytest.raises(analysis.F2PublicValAnalysisError, match="truncated"):
        analysis._verify_completion(
            run_dir,
            [metrics_path, prediction_path],
            expected_rows=analysis.DEFAULT_ROW_COUNT,
            expected_preregistration_sha256=preregistration_sha256,
        )
