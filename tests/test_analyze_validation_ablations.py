import copy
import csv
import hashlib
import json

import numpy as np
import pytest

from scripts.analyze_validation_ablations import (
    ANALYSIS_CLASS,
    CELL_FIELDS,
    PUBLICATION_LABEL,
    VALIDATION_EPISODE_IDS,
    ValidationAblationError,
    _bootstrap_counts,
    _bootstrap_distribution,
    analyze_validation_ablations,
    holm_bonferroni,
    load_cells,
    load_contrast_registry,
    main,
    validate_cells,
    validate_contrast_registry,
)


def _sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _registry_payload(iterations=200_000):
    ablations = [
        "H0-noTIM",
        "H0-noFuture",
        "H0-noVerifier",
        "H0-noEventBank",
        "H0-noOrchestrator",
        "H0-S",
        "H0-noPolar",
        "H0-noTurnBalance",
    ]
    methods = ["H0", *ablations]
    return {
        "schema_version": 1,
        "analysis_class": ANALYSIS_CLASS,
        "family_id": "h0_component_ablation_v1",
        "family_size": 8,
        "seed_ids": [0, 1, 2],
        "episode_ids": list(VALIDATION_EPISODE_IDS),
        "primary_metric": "bce_at1",
        "alpha": 0.05,
        "bootstrap_iterations": iterations,
        "analysis_seed": 20260715,
        "expected_optimizer_updates": 6873,
        "expected_processed_samples": 13746,
        "full_method_id": "H0",
        "contrasts": [
            {
                "contrast_id": f"H0_vs_{method}",
                "candidate_id": "H0",
                "reference_id": method,
            }
            for method in ablations
        ],
        "method_contracts": {
            method: {
                "treatment_name": "full" if method == "H0" else method,
                "treatment_config_sha256": _sha(f"treatment:{method}"),
                "state_mode": "stateless" if method == "H0-S" else "rolling",
            }
            for method in methods
        },
        "guardrails": {
            "smooth_l1_forward": {"direction": "lower", "harm_margin": 0.02},
            "smooth_l1_yaw": {"direction": "lower", "harm_margin": 0.02},
            "turn_sign_accuracy": {"direction": "higher", "harm_margin": 0.03},
            "transition_f1": {"direction": "higher", "harm_margin": 0.03},
            "saturation_rate": {"direction": "lower", "harm_margin": 0.02},
        },
    }


def _write_registry(tmp_path, payload=None):
    path = tmp_path / "contrast_registry.json"
    path.write_text(
        json.dumps(payload or _registry_payload(), sort_keys=True), encoding="utf-8"
    )
    return path, load_contrast_registry(path)


def _cells(registry):
    methods = list(registry["method_contracts"])
    shared = {
        field: _sha(field)
        for field in (
            "parent_main_registry_sha256",
            "source_tree_sha256",
            "evaluator_source_sha256",
            "metric_contract_sha256",
            "training_manifest_sha256",
            "training_data_sha256",
            "validation_manifest_sha256",
            "validation_data_sha256",
            "base_model_sha256",
            "qwen_model_sha256",
            "vision_cache_manifest_sha256",
            "vision_cache_provenance_sha256",
            "vision_cache_token_payload_sha256",
            "dino_model_sha256",
            "siglip_model_sha256",
            "fairness_contract_sha256",
            "bce_support_sha256",
        )
    }
    shared["ablation_registry_sha256"] = registry["_registry_sha256"]
    rows = []
    for method_index, method in enumerate(methods):
        # Full H0 is uniformly better than every ablation by 0.08 BCE@1.
        base_bce = 0.20 if method == "H0" else 0.28 + 0.001 * method_index
        for seed in range(3):
            seed_values = [base_bce + 0.01 * seed + 0.002 * episode_index for episode_index in range(3)]
            selected_value = float(np.mean(seed_values))
            run_key = f"{method}:{seed}"
            run_hashes = {
                "checkpoint_sha256": _sha(f"checkpoint:{run_key}"),
                "model_state_sha256": _sha(f"model_state:{run_key}"),
                "training_log_sha256": _sha(f"training_log:{run_key}"),
                "checkpoint_event_sha256": _sha(f"checkpoint_event:{run_key}"),
                "run_end_event_sha256": _sha(f"run_end_event:{run_key}"),
                "selection_detail_sha256": _sha(f"selection_detail:{run_key}"),
                "evaluation_predictions_sha256": _sha(f"predictions:{run_key}"),
                "evaluation_execution_contract_sha256": _sha(
                    f"execution:{method}"
                ),
                "evaluation_result_sha256": _sha(f"eval_result:{run_key}"),
            }
            for episode_index, episode in enumerate(VALIDATION_EPISODE_IDS):
                bce = seed_values[episode_index]
                rows.append(
                    {
                        "schema_version": 2,
                        "analysis_class": ANALYSIS_CLASS,
                        "split": "val",
                        "validation_only": True,
                        "paper_eligible": False,
                        "method_id": method,
                        "seed": seed,
                        "episode": episode,
                        "state_mode": registry["method_contracts"][method]["state_mode"],
                        "treatment_config_sha256": registry["method_contracts"][method]["treatment_config_sha256"],
                        "checkpoint_role": "best_validation",
                        "selection_verified": True,
                        "selected_epoch": 0,
                        "checkpoint_epoch": 0,
                        "selected_value": selected_value,
                        "run_end_status": "completed",
                        "run_end_error_count": 0,
                        "run_end_best_validation_bce": selected_value,
                        "run_end_optimizer_updates": 6873,
                        "run_end_processed_samples": 13746,
                        "run_id": f"run-{run_key}",
                        "bce_at1": bce,
                        "smooth_l1_forward": 0.04 if method == "H0" else 0.045,
                        "smooth_l1_yaw": 0.05 if method == "H0" else 0.055,
                        "turn_sign_accuracy": 0.90 if method == "H0" else 0.89,
                        "transition_f1": 1.0 if method == "H0" else 0.8,
                        "transition_f1_defined": True,
                        "transition_f1_excluded": False,
                        "transition_tp": 1 if method == "H0" else 2,
                        "transition_fp": 0 if method == "H0" else 1,
                        "transition_fn": 0,
                        "transition_zero_fire_collapse": False,
                        "saturation_rate": 0.02 if method == "H0" else 0.025,
                        **shared,
                        **run_hashes,
                    }
                )
    assert len(rows) == 81
    return rows


def _write_jsonl(tmp_path, rows):
    path = tmp_path / "cells.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_end_to_end_writes_validation_only_report_and_normalized_cells(tmp_path):
    registry_path, registry = _write_registry(tmp_path)
    cells_path = _write_jsonl(tmp_path, _cells(registry))
    output_dir = tmp_path / "out"
    report = main(
        [
            "--cells",
            str(cells_path),
            "--contrast_registry",
            str(registry_path),
            "--output_dir",
            str(output_dir),
        ]
    )
    assert report["validation_only"] is True
    assert report["schema_version"] == 2
    assert report["paper_eligible"] is False
    assert report["publication_label"] == PUBLICATION_LABEL
    assert report["family_size"] == 8
    assert report["bootstrap"]["simultaneous_confidence_level"] == pytest.approx(
        0.99375
    )
    assert report["bootstrap"]["quantiles"] == pytest.approx([0.003125, 0.996875])
    assert len(report["contrasts"]) == 8
    assert all(item["decision"]["supported_improvement"] for item in report["contrasts"])
    assert report["transition_f1_evidence"]["overall"] == {
        "cell_count": 81,
        "defined_cell_count": 81,
        "excluded_empty_event_union_cell_count": 0,
        "zero_fire_collapse_cell_count": 0,
        "tp": 153,
        "fp": 72,
        "fn": 0,
    }
    assert (output_dir / "report.json").is_file()
    assert "VAL_ONLY_MODEL_DEVELOPMENT" in (output_dir / "summary.md").read_text()
    assert len((output_dir / "cells.jsonl").read_text().splitlines()) == 81


def test_csv_cells_are_supported(tmp_path):
    _registry_path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    csv_path = tmp_path / "cells.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CELL_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    loaded = load_cells(csv_path)
    validated_registry = validate_contrast_registry(registry)
    assert len(validate_cells(loaded, validated_registry)["rows"]) == 81


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows.pop(), "exactly 81 rows"),
        (lambda rows: rows.append(copy.deepcopy(rows[0])), "exactly 81 rows"),
        (lambda rows: rows[0].update({"split": "test"}), "split=val"),
        (lambda rows: rows[0].update({"episode": "test004"}), "unexpected validation episode"),
        (lambda rows: rows[0].update({"checkpoint_role": "epoch"}), "changes checkpoint_role"),
        (lambda rows: rows[0].update({"run_end_status": "failed"}), "changes run_end_status"),
        (lambda rows: rows[0].update({"validation_data_sha256": _sha("wrong")}), "shared provenance mismatch"),
    ],
)
def test_matrix_and_provenance_gates_fail_closed(tmp_path, mutate, message):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    mutate(rows)
    with pytest.raises(ValidationAblationError, match=message):
        validate_cells(rows, validate_contrast_registry(registry))


def test_run_gate_rejects_non_best_and_reused_checkpoint_provenance(tmp_path):
    _path, registry = _write_registry(tmp_path)
    normalized_registry = validate_contrast_registry(registry)
    rows = _cells(registry)
    for row in rows[:3]:
        row["checkpoint_role"] = "epoch"
    with pytest.raises(ValidationAblationError, match="not checkpoint_role=best_validation"):
        validate_cells(rows, normalized_registry)

    rows = _cells(registry)
    reused = rows[0]["checkpoint_sha256"]
    for row in rows[3:6]:
        row["checkpoint_sha256"] = reused
    with pytest.raises(ValidationAblationError, match="provenance is reused"):
        validate_cells(rows, normalized_registry)


def test_duplicate_cell_and_nonfinite_guardrail_are_rejected(tmp_path):
    _path, registry = _write_registry(tmp_path)
    normalized_registry = validate_contrast_registry(registry)
    rows = _cells(registry)
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValidationAblationError, match="duplicate method/seed/episode"):
        validate_cells(rows, normalized_registry)

    rows = _cells(registry)
    rows[0]["transition_f1"] = float("nan")
    with pytest.raises(ValidationAblationError, match="transition_f1 must be finite"):
        validate_cells(rows, normalized_registry)


def test_schema2_preserves_zero_fire_and_excludes_true_empty_union(tmp_path):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    rows[0].update(
        {
            "transition_f1": 0.0,
            "transition_f1_defined": True,
            "transition_f1_excluded": False,
            "transition_tp": 0,
            "transition_fp": 0,
            "transition_fn": 1,
            "transition_zero_fire_collapse": True,
        }
    )
    rows[1].update(
        {
            "transition_f1": None,
            "transition_f1_defined": False,
            "transition_f1_excluded": True,
            "transition_tp": 0,
            "transition_fp": 0,
            "transition_fn": 0,
            "transition_zero_fire_collapse": False,
        }
    )
    normalized_registry = validate_contrast_registry(registry)
    validated = validate_cells(rows, normalized_registry)
    assert validated["rows"][0]["transition_f1"] == 0.0
    assert validated["rows"][0]["transition_zero_fire_collapse"] is True
    assert validated["rows"][1]["transition_f1"] is None
    assert validated["rows"][1]["transition_f1_excluded"] is True

    report, normalized_rows = analyze_validation_ablations(rows, registry)
    evidence = report["transition_f1_evidence"]["overall"]
    assert evidence["defined_cell_count"] == 80
    assert evidence["excluded_empty_event_union_cell_count"] == 1
    assert evidence["zero_fire_collapse_cell_count"] == 1
    assert normalized_rows[1]["transition_f1"] is None
    transition = report["contrasts"][0]["guardrails"]["transition_f1"]
    assert transition["analysis_status"] == "partial_empty_event_union_exclusion"
    assert transition["paired_cell_count"] == 8
    assert transition["excluded_empty_union_cell_count"] == 1


def test_schema2_all_empty_transition_guardrail_is_explicitly_excluded(tmp_path):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    for row in rows:
        row.update(
            {
                "transition_f1": None,
                "transition_f1_defined": False,
                "transition_f1_excluded": True,
                "transition_tp": 0,
                "transition_fp": 0,
                "transition_fn": 0,
                "transition_zero_fire_collapse": False,
            }
        )
    report, _normalized_rows = analyze_validation_ablations(rows, registry)
    assert report["transition_f1_evidence"]["overall"][
        "excluded_empty_event_union_cell_count"
    ] == 81
    for contrast in report["contrasts"]:
        transition = contrast["guardrails"]["transition_f1"]
        assert transition["analysis_status"] == "excluded_no_paired_support"
        assert transition["paired_cell_count"] == 0
        assert transition["pass"] is False
        assert transition["noninferiority_rule"] == "no_paired_support_cannot_pass"
        assert contrast["decision"]["guardrails_pass"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            {
                "transition_f1": None,
                "transition_f1_defined": False,
                "transition_f1_excluded": False,
                "transition_tp": 0,
                "transition_fp": 0,
                "transition_fn": 0,
            },
            "empty transition event union",
        ),
        (
            {
                "transition_f1": None,
                "transition_f1_defined": False,
                "transition_f1_excluded": True,
                "transition_tp": 0,
                "transition_fp": 0,
                "transition_fn": 1,
                "transition_zero_fire_collapse": True,
            },
            "non-empty transition event union",
        ),
        (
            {"transition_f1": 0.5},
            "count-based tp/fp/fn definition",
        ),
        (
            {"transition_zero_fire_collapse": True},
            "zero_fire_collapse disagrees",
        ),
    ],
)
def test_schema2_transition_evidence_invariants_fail_closed(
    tmp_path, mutation, message
):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    rows[0].update(mutation)
    with pytest.raises(ValidationAblationError, match=message):
        validate_cells(rows, validate_contrast_registry(registry))


def test_selected_value_must_equal_three_episode_macro(tmp_path):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    for row in rows[:3]:
        row["selected_value"] += 0.01
    with pytest.raises(ValidationAblationError, match="selected_value disagrees"):
        validate_cells(rows, validate_contrast_registry(registry))


def test_centered_two_way_bootstrap_is_deterministic_and_not_frame_resampling():
    delta = np.asarray(
        [[-0.10, -0.08, -0.12], [-0.09, -0.07, -0.11], [-0.13, -0.06, -0.10]]
    )
    counts_a = _bootstrap_counts(iterations=10_000, analysis_seed=17)
    counts_b = _bootstrap_counts(iterations=10_000, analysis_seed=17)
    samples_a = _bootstrap_distribution(delta, *counts_a)
    samples_b = _bootstrap_distribution(delta, *counts_b)
    assert np.array_equal(samples_a, samples_b)
    assert np.quantile(samples_a, [0.025, 0.975]) == pytest.approx(
        [-0.11555555555555555, -0.07325], abs=1e-12
    )


def test_holm_known_vector_is_monotone_and_step_down():
    values = [0.001, 0.01, 0.02, 0.2, 0.04, 0.03, 0.5, 0.9]
    result = holm_bonferroni(values)
    ordered = sorted(zip(values, result), key=lambda item: item[0])
    adjusted = [item[1]["adjusted_p"] for item in ordered]
    assert adjusted == sorted(adjusted)
    assert result[0]["adjusted_p"] == pytest.approx(0.008)
    assert result[1]["adjusted_p"] == pytest.approx(0.07)
    assert result[0]["reject"] is True
    assert result[1]["reject"] is False
    assert all(not item["reject"] for item in result[1:])


def test_guardrail_harm_and_leave_one_out_sign_flip_block_decision(tmp_path):
    _path, registry = _write_registry(tmp_path)
    rows = _cells(registry)
    # Harm H0 forward SmoothL1 beyond the +0.02 non-inferiority margin.
    for row in rows:
        if row["method_id"] == "H0":
            row["smooth_l1_forward"] = 0.20
    # Make one seed strongly prefer the first ablation, creating a LOSO sign flip.
    first_reference = registry["contrasts"][0]["reference_id"]
    for row in rows:
        if row["method_id"] == "H0" and row["seed"] == 0:
            row["bce_at1"] += 0.5
    for method in ("H0",):
        for seed in range(3):
            run_rows = [row for row in rows if row["method_id"] == method and row["seed"] == seed]
            selected = float(np.mean([row["bce_at1"] for row in run_rows]))
            for row in run_rows:
                row["selected_value"] = selected
                row["run_end_best_validation_bce"] = selected
    report, _ = analyze_validation_ablations(rows, registry)
    contrast = next(
        item for item in report["contrasts"] if item["reference_id"] == first_reference
    )
    assert contrast["guardrails"]["smooth_l1_forward"]["pass"] is False
    assert contrast["decision"]["guardrails_pass"] is False
    assert contrast["decision"]["leave_one_out_sign_consistent"] is False
    assert contrast["decision"]["supported_improvement"] is False


def test_registry_rejects_incomplete_family_and_low_resolution_bootstrap(tmp_path):
    payload = _registry_payload()
    payload["contrasts"].pop()
    path, registry = _write_registry(tmp_path, payload)
    assert path.is_file()
    with pytest.raises(ValidationAblationError, match="exactly 8 contrasts"):
        validate_contrast_registry(registry)

    payload = _registry_payload(iterations=10_000)
    _path, registry = _write_registry(tmp_path, payload)
    with pytest.raises(ValidationAblationError, match="must be >= 200000"):
        validate_contrast_registry(registry)
