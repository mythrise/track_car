from pathlib import Path

import pytest

from scripts.diagnose_v6_f1_v7_dryrun import (
    DEV_DIAGNOSTICS_RELATIVE,
    DiagnosticError,
    PROJECT_ROOT,
    _assert_no_v7_input,
    build_diagnostic,
    count_f1_from_counts,
    legacy_f1_from_counts,
    validate_output_path,
)


def test_count_f1_separates_supported_miss_from_empty_union():
    assert legacy_f1_from_counts(0, 0, 66) is None
    assert count_f1_from_counts(0, 0, 66) == 0.0
    assert legacy_f1_from_counts(0, 0, 0) is None
    assert count_f1_from_counts(0, 0, 0) is None


def test_v7_formal_input_components_are_rejected(tmp_path):
    for component in (
        "validation_checkpoint_inventory_v7.json",
        "validation_eval_v7",
        "validation_analysis_v7",
    ):
        with pytest.raises(DiagnosticError, match=r"validation_\*_v7 input"):
            _assert_no_v7_input(tmp_path / component / "artifact.json", "test input")


def test_output_is_restricted_to_dev_diagnostics(tmp_path):
    allowed = tmp_path / DEV_DIAGNOSTICS_RELATIVE / "diagnostic.json"
    assert validate_output_path(tmp_path, allowed) == allowed.resolve()
    with pytest.raises(DiagnosticError, match="external_reviews/dev_diagnostics"):
        validate_output_path(tmp_path, tmp_path / "diagnostic.json")
    with pytest.raises(DiagnosticError, match=".json suffix"):
        validate_output_path(
            tmp_path, tmp_path / DEV_DIAGNOSTICS_RELATIVE / "diagnostic.md"
        )


def test_real_sealed_v6_diagnostic_has_exact_27_cell_delta_partition():
    report = build_diagnostic(PROJECT_ROOT, attempt_builder_dry_run=False)
    assert report["status"] == "passed"
    assert report["internal_test_opened"] is False
    assert report["formal_artifact_reuse_allowed"] is False
    assert report["summary"] == {
        "cell_count": 27,
        "nondegenerate_f1_bit_and_canonical_delta_count": 3,
        "nondegenerate_f1_bit_exact_unchanged_count": 12,
        "null_to_zero_f1_count": 12,
        "unexpected_delta_count": 0,
        "all_invariant_payloads_bound": True,
        "all_bce_objects_equal_sealed_results": True,
    }
    converted = [
        row
        for row in report["legacy_vs_count_delta"]
        if row["classification"] == "null_to_zero"
    ]
    assert len(converted) == 12
    assert all(row["legacy_f1"] is None for row in converted)
    assert all(row["count_f1"] == 0.0 for row in converted)
    assert all(row["tp"] == 0 for row in converted)
    assert all(row["fp"] == 0 for row in converted)
    assert all(row["fn"] > 0 for row in converted)


def test_metric_v2_dependency_dry_run_is_finite_and_main_builder_fails_closed():
    report = build_diagnostic(PROJECT_ROOT, attempt_builder_dry_run=True)
    dependency = report["dependency_metric_v2_dry_run"]
    assert dependency["status"] == "passed"
    assert dependency["cell_count"] == 27
    assert dependency["guardrail_null_count"] == 0
    assert dependency["temporary_directory_removed_after_run"] is True
    assert not Path(dependency["cells_path"]).exists()

    main_probe = report["current_main_builder_analyzer_probe"]
    assert main_probe["status"] == "blocked_by_current_main_builder"
    assert main_probe["temporary_output_exists"] is False
    assert main_probe["temporary_directory_removed_after_run"] is True
    assert not Path(main_probe["temporary_output_path"]).exists()


def test_full_v7_like_main_builder_and_analyzer_pass_in_isolated_temp_tree():
    report = build_diagnostic(PROJECT_ROOT, attempt_full_main_v7_like_dry_run=True)
    dry_run = report["full_main_v7_like_builder_analyzer_dry_run"]
    assert dry_run["status"] == "passed"
    assert dry_run["model_inference_executed"] is False
    assert dry_run["preflight_executed"] is False
    assert dry_run["internal_test_opened"] is False
    assert dry_run["cell_count"] == 27
    assert dry_run["transition_f1_null_count"] == 0
    assert dry_run["transition_zero_fire_collapse_count"] == 12
    assert dry_run["transition_count_mismatch_count"] == 0
    assert dry_run["builder_phase"]["exit_code"] == 0
    assert dry_run["analyzer_phase"]["exit_code"] == 0
    assert dry_run["report_analysis_class"] == "validation_main_v1"
    assert dry_run["temporary_directory_removed_after_run"] is True
    assert dry_run["temporary_artifacts_exist_after_run"] is False
    assert not Path(dry_run["temporary_root"]).exists()
    assert not Path(dry_run["diagnostic_stage_root"]).exists()
    assert all(artifact["exists"] for artifact in dry_run["artifacts"].values())
