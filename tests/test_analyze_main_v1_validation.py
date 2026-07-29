import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.analyze_main_v1_validation as analyzer
from scripts.analyze_main_v1_validation import (
    ANALYSIS_CLASS,
    EPISODE_IDS,
    METHOD_IDS,
    SEED_IDS,
    MainValidationAnalysisError,
    _contrast_result,
    _exact_two_way_bootstrap,
    analyze,
    load_builder_receipt,
    rebuild_and_verify_cells,
    validate_distinct_paths,
    validate_analyzer_source_binding,
    validate_cells,
    verify_upstream_artifact_hashes,
)


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _phase_args(tmp_path):
    analysis_root = tmp_path / "experiments/collected_v1_main/validation_analysis_v12"
    return SimpleNamespace(
        cells=str(analysis_root / "main_v1_validation_cells.jsonl"),
        expected_cells_sha256=_sha("cells"),
        builder_receipt=str(analysis_root / "main_v1_validation_cells.receipt.json"),
        expected_builder_receipt_sha256=_sha("builder-receipt"),
        controller_run_manifest=str(
            tmp_path
            / "experiments/collected_v1_main/validation_eval_v12/audit/run_manifest.json"
        ),
        expected_controller_run_manifest_sha256=_sha("controller-manifest"),
        checkpoint_inventory=str(
            tmp_path
            / "experiments/collected_v1_main/validation_checkpoint_inventory_v12.json"
        ),
        expected_checkpoint_inventory_sha256=_sha("inventory"),
        cell_registry=str(
            tmp_path
            / "experiments/collected_v1_main/main_v1_validation_cell_registry_v12.json"
        ),
        expected_cell_registry_sha256=_sha("registry"),
        expected_analyzer_sha256=_sha("analyzer"),
        stage_manifest=str(tmp_path / "stage/main_v1_stage_manifest.json"),
        expected_stage_manifest_sha256=_sha("stage-manifest"),
        json_output=str(analysis_root / "main_v1_validation_report.json"),
        markdown_output=str(analysis_root / "main_v1_validation_report.md"),
        command_receipt_output=str(
            tmp_path / analyzer.ANALYZER_PHASE_RECEIPTS["success"]
        ),
        failure_output=str(tmp_path / analyzer.ANALYZER_PHASE_RECEIPTS["failure"]),
    )


def _phase_argv(args):
    values = ["analyze_main_v1_validation.py"]
    for field in (
        "cells",
        "expected_cells_sha256",
        "builder_receipt",
        "expected_builder_receipt_sha256",
        "controller_run_manifest",
        "expected_controller_run_manifest_sha256",
        "checkpoint_inventory",
        "expected_checkpoint_inventory_sha256",
        "cell_registry",
        "expected_cell_registry_sha256",
        "expected_analyzer_sha256",
        "stage_manifest",
        "expected_stage_manifest_sha256",
        "json_output",
        "markdown_output",
        "command_receipt_output",
        "failure_output",
    ):
        values.extend([f"--{field}", str(getattr(args, field))])
    return values


def _write_predecessor_receipt(tmp_path, monkeypatch):
    receipt_path = tmp_path / "v11-full-failure-receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    source = analyzer.PROJECT_ROOT / analyzer.PREDECESSOR_FAILURE_RELATIVE
    receipt_path.write_bytes(source.read_bytes())
    receipt_sha256 = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    monkeypatch.setattr(analyzer, "PREDECESSOR_FAILURE_RELATIVE", receipt_path)
    monkeypatch.setattr(analyzer, "PREDECESSOR_FAILURE_SHA256", receipt_sha256)
    return receipt_path


def _registry():
    builder_dependency_sha = _sha("builder-dependency")
    builder_dependency_analyzer_sha = _sha("builder-dependency-analyzer")
    evaluator_file_sha = _sha("evaluator-file")
    experiment_binding_file_sha = _sha("experiment-binding-file")
    dependency_inventory = {
        "schema_version": 1,
        "files": [
            {
                "path": "scripts/build_validation_ablation_cells.py",
                "sha256": builder_dependency_sha,
            },
            {
                "path": "scripts/analyze_validation_ablations.py",
                "sha256": builder_dependency_analyzer_sha,
            },
            {
                "path": "scripts/eval_offline.py",
                "sha256": evaluator_file_sha,
            },
            {
                "path": "third_party/OpenTrackVLA/experiment_binding.py",
                "sha256": experiment_binding_file_sha,
            },
        ],
    }
    dependency_inventory_sha = hashlib.sha256(
        json.dumps(
            dependency_inventory,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "_registry_sha256": _sha("registry"),
        "schema_version": 1,
        "analysis_class": ANALYSIS_CLASS,
        "status": "frozen_before_validation_v12_evaluation",
        "formal_version": analyzer.FORMAL_VERSION,
        "predecessor_failure_receipt_sha256": (analyzer.PREDECESSOR_FAILURE_SHA256),
        "controller_sha256": analyzer.EXPECTED_CONTROLLER_SHA256,
        "method_ids": list(METHOD_IDS),
        "seed_ids": list(SEED_IDS),
        "episode_ids": list(EPISODE_IDS),
        "analyzer_source_sha256": _sha("analyzer-source"),
        "builder_source_sha256": _sha("builder-source"),
        "builder_dependency_sha256": builder_dependency_sha,
        "builder_dependency_analyzer_sha256": (builder_dependency_analyzer_sha),
        "builder_runtime_dependency_inventory_sha256": (dependency_inventory_sha),
        "evaluator_file_sha256": evaluator_file_sha,
        "experiment_binding_file_sha256": experiment_binding_file_sha,
        "runtime_isolation_contract_sha256": _sha("runtime-isolation"),
        "project_import_surface_sha256": _sha("project-import-surface"),
        "opentrackvla_import_surface_sha256": _sha("opentrack-import-surface"),
        "stage_source_contract_sha256": _sha("stage-source-contract"),
        "metric_contract_schema_version": 2,
        "metric_contract_sha256": (
            "c75d0c08fd209e120e59eac10bb9cdd0fac6fdd65582beaf3c63a1b18a6c5097"
        ),
        "validation_dataset": {"data_sha256": _sha("val-data")},
        "method_contracts": {
            "B0": {
                "state_mode": "stateless",
                "model_family": "opentrackvla_baseline",
                "treatment_config_sha256": _sha("B0"),
            },
            "B1": {
                "state_mode": "rolling",
                "model_family": "trackvla_pp_lite",
                "treatment_config_sha256": _sha("B1"),
            },
            "H0": {
                "state_mode": "rolling",
                "model_family": "pfem_harness",
                "treatment_config_sha256": _sha("H0"),
            },
        },
        "deterministic_baseline": {"model_state_sha256": _sha("B0-state")},
        "statistics": {
            "independence_unit": "training_seed",
            "exact_two_way_ordered_resamples": 729,
            "seed_cluster_t_critical_df2": analyzer.T_CRITICAL_DF2_95,
        },
        "contrasts": [
            {
                "contrast_id": "H0_vs_B1",
                "candidate": "H0",
                "reference": "B1",
                "status": "primary",
            },
            {
                "contrast_id": "B1_vs_B0",
                "candidate": "B1",
                "reference": "B0",
                "status": "secondary",
            },
            {
                "contrast_id": "H0_vs_B0",
                "candidate": "H0",
                "reference": "B0",
                "status": "secondary",
            },
        ],
    }


def _cells():
    values = {
        "B0": [0.30, 0.30, 0.30],
        "B1": [0.36, 0.37, 0.38],
        "H0": [0.37, 0.36, 0.35],
    }
    rows = []
    registry = _registry()
    shared = {
        field: _sha(field)
        for field in (
            "parent_main_registry_sha256",
            "source_tree_sha256",
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
            "evaluator_source_sha256",
            "metric_contract_sha256",
            "selection_replay_manifest_sha256",
            "predecessor_failure_receipt_sha256",
        )
    }
    shared["predecessor_failure_receipt_sha256"] = analyzer.PREDECESSOR_FAILURE_SHA256
    for method in METHOD_IDS:
        for seed in SEED_IDS:
            formal_by_episode = {
                episode: values[method][seed] + episode_index * 0.001
                for episode_index, episode in enumerate(EPISODE_IDS)
            }
            selection_by_episode = {
                episode: (
                    formal_by_episode[episode] - 0.0001
                    if method == "B0"
                    else formal_by_episode[episode]
                )
                for episode in EPISODE_IDS
            }
            formal_macro = float(
                np.mean(
                    np.asarray(
                        [formal_by_episode[episode] for episode in EPISODE_IDS],
                        dtype=np.float64,
                    )
                )
            )
            selection_macro = float(
                np.mean(
                    np.asarray(
                        [selection_by_episode[episode] for episode in EPISODE_IDS],
                        dtype=np.float64,
                    )
                )
            )
            run_hashes = {
                field: (
                    _sha("B0-state")
                    if field == "model_state_sha256" and method == "B0"
                    else _sha(f"{field}:B0-shared")
                    if method == "B0"
                    and field
                    in {
                        "selection_replay_receipt_sha256",
                        "selection_replay_predictions_sha256",
                        "selection_replay_evaluation_result_sha256",
                        "selection_replay_log_sha256",
                        "selection_replay_execution_contract_sha256",
                        "selection_replay_evaluator_command_sha256",
                    }
                    else _sha(f"{field}:{method}:{seed}")
                )
                for field in (
                    "checkpoint_sha256",
                    "model_state_sha256",
                    "training_log_sha256",
                    "checkpoint_event_sha256",
                    "run_end_event_sha256",
                    "selection_detail_sha256",
                    "training_selection_contract_sha256",
                    "evaluation_predictions_sha256",
                    "evaluation_execution_contract_sha256",
                    "evaluation_result_sha256",
                    "formal_evaluator_command_sha256",
                    "formal_log_sha256",
                    "selection_replay_receipt_sha256",
                    "selection_replay_predictions_sha256",
                    "selection_replay_evaluation_result_sha256",
                    "selection_replay_log_sha256",
                    "selection_replay_execution_contract_sha256",
                    "selection_replay_evaluator_command_sha256",
                )
            }
            if method in {"B1", "H0"}:
                run_hashes.update(
                    {
                        "selection_replay_predictions_sha256": run_hashes[
                            "evaluation_predictions_sha256"
                        ],
                        "selection_replay_evaluation_result_sha256": run_hashes[
                            "evaluation_result_sha256"
                        ],
                        "selection_replay_log_sha256": run_hashes["formal_log_sha256"],
                        "selection_replay_execution_contract_sha256": run_hashes[
                            "evaluation_execution_contract_sha256"
                        ],
                        "selection_replay_evaluator_command_sha256": run_hashes[
                            "formal_evaluator_command_sha256"
                        ],
                    }
                )
            for episode_index, episode in enumerate(EPISODE_IDS):
                bce = formal_by_episode[episode]
                selection_bce = selection_by_episode[episode]
                rows.append(
                    {
                        "schema_version": 2,
                        "analysis_class": ANALYSIS_CLASS,
                        "formal_version": analyzer.FORMAL_VERSION,
                        "dual_contract_id": analyzer.DUAL_CONTRACT_ID,
                        "split": "val",
                        "validation_only": True,
                        "paper_eligible": False,
                        "formal_primary": True,
                        "formal_batch_size": 1,
                        "method_id": method,
                        "seed": seed,
                        "episode": episode,
                        "state_mode": registry["method_contracts"][method][
                            "state_mode"
                        ],
                        "treatment_config_sha256": registry["method_contracts"][method][
                            "treatment_config_sha256"
                        ],
                        "checkpoint_role": "best_validation",
                        "selection_verified": True,
                        "selected_epoch": 0,
                        "checkpoint_epoch": 0,
                        "selected_value": selection_macro,
                        "run_end_status": "completed",
                        "run_end_error_count": 0,
                        "run_end_best_validation_bce": selection_macro,
                        "run_end_optimizer_updates": 6873,
                        "run_end_processed_samples": 13746,
                        "run_id": f"run-{method}-{seed}",
                        "bce_at1": bce,
                        "formal_bce_at1": bce,
                        "selection_bce_at1": selection_bce,
                        "formal_minus_selection_bce_at1": bce - selection_bce,
                        "formal_macro_bce_at1": formal_macro,
                        "selection_macro_bce_at1": selection_macro,
                        "formal_minus_selection_macro_bce_at1": formal_macro
                        - selection_macro,
                        "selection_replay_receipt_id": (
                            "B0_shared_batch2_selection_replay"
                            if method == "B0"
                            else f"{method}_seed{seed}_formal_same_contract_selection_replay"
                        ),
                        "selection_replay_kind": (
                            "executed_shared_model_state"
                            if method == "B0"
                            else "formal_artifact_same_contract"
                        ),
                        "selection_replay_batch_size": 2 if method == "B0" else 1,
                        "smooth_l1_forward": bce / 10,
                        "smooth_l1_yaw": bce / 5,
                        "turn_sign_accuracy": 1 - bce / 2,
                        "transition_f1": 1.0,
                        "transition_f1_defined": True,
                        "transition_f1_excluded": False,
                        "transition_tp": 1,
                        "transition_fp": 0,
                        "transition_fn": 0,
                        "transition_zero_fire_collapse": False,
                        "saturation_rate": bce / 4,
                        "cell_registry_sha256": registry["_registry_sha256"],
                        **shared,
                        **run_hashes,
                    }
                )
    return rows, registry


def test_analyzer_json_and_cells_hash_the_parsed_payload_once(tmp_path, monkeypatch):
    document_path = tmp_path / "receipt.json"
    document_path.write_text('{"trusted": true}\n', encoding="utf-8")
    cells_path = tmp_path / "cells.jsonl"
    cells_path.write_text('{"cell": 1}\n', encoding="utf-8")
    document_sha = hashlib.sha256(document_path.read_bytes()).hexdigest()
    cells_sha = hashlib.sha256(cells_path.read_bytes()).hexdigest()
    original_open = Path.open
    calls = {document_path.resolve(): 0, cells_path.resolve(): 0}
    replacements = {
        document_path.resolve(): b'{"trusted": false}\n',
        cells_path.resolve(): b'{"cell": 999}\n',
    }

    def racing_open(path, *args, **kwargs):
        resolved = path.resolve()
        mode = args[0] if args else kwargs.get("mode", "r")
        if resolved in calls and "r" in mode:
            calls[resolved] += 1
            if calls[resolved] == 2:
                with original_open(path, "wb") as handle:
                    handle.write(replacements[resolved])
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    document, actual_document_sha = analyzer._load_json(
        document_path, document_sha, "test document"
    )
    cells, actual_cells_sha = analyzer.load_cells(cells_path, cells_sha)
    assert document == {"trusted": True}
    assert cells == [{"cell": 1}]
    assert actual_document_sha == document_sha
    assert actual_cells_sha == cells_sha
    assert calls == {document_path.resolve(): 1, cells_path.resolve(): 1}


def test_v12_analyzer_parser_requires_both_one_shot_receipts(tmp_path):
    args = _phase_args(tmp_path)
    argv = _phase_argv(args)[1:]
    without_receipts = argv[:-4]
    with pytest.raises(SystemExit):
        analyzer.build_parser().parse_args(without_receipts)
    with pytest.raises(SystemExit):
        analyzer.build_parser().parse_args(
            without_receipts + ["--command_receipt_output", args.command_receipt_output]
        )
    parsed = analyzer.build_parser().parse_args(argv)
    assert parsed.command_receipt_output == args.command_receipt_output
    assert parsed.failure_output == args.failure_output


def test_v12_analyzer_receipt_paths_are_exact_absent_and_checked_early(tmp_path):
    args = _phase_args(tmp_path)
    success, failure = analyzer.validate_analyzer_phase_receipt_cli(
        args, project_root=tmp_path
    )
    assert success == (tmp_path / analyzer.ANALYZER_PHASE_RECEIPTS["success"]).resolve()
    assert failure == (tmp_path / analyzer.ANALYZER_PHASE_RECEIPTS["failure"]).resolve()

    args.command_receipt_output = str(tmp_path / "wrong-success.json")
    with pytest.raises(MainValidationAnalysisError, match="success receipt path"):
        analyzer.validate_analyzer_phase_receipt_cli(args, project_root=tmp_path)

    args = _phase_args(tmp_path)
    success = Path(args.command_receipt_output)
    success.parent.mkdir(parents=True)
    success.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MainValidationAnalysisError, match="already exists"):
        analyzer.validate_analyzer_phase_receipt_cli(args, project_root=tmp_path)


def test_v12_analyzer_success_receipt_binds_inputs_outputs_and_stage(
    tmp_path, monkeypatch
):
    args = _phase_args(tmp_path)
    for field in (
        "cells",
        "builder_receipt",
        "controller_run_manifest",
        "checkpoint_inventory",
        "cell_registry",
        "stage_manifest",
        "json_output",
        "markdown_output",
    ):
        path = Path(getattr(args, field))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{field}\n", encoding="utf-8")
    stage_root = Path(args.stage_manifest).parent
    source_file = stage_root / "scripts/analyze_main_v1_validation.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("trusted analyzer\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer,
        "_FORMAL_STAGE_BINDING",
        {
            "source_project_root": tmp_path,
            "stage_root": stage_root,
            "manifest_path": Path(args.stage_manifest),
            "manifest_sha256": args.expected_stage_manifest_sha256,
            "source_contract_sha256": _sha("stage-source"),
            "stage_tree_sha256": _sha("stage-tree"),
        },
    )
    monkeypatch.setattr(sys, "argv", _phase_argv(args))

    analyzer.write_analyzer_success_receipt(
        args,
        {"analysis_class": ANALYSIS_CLASS, "validation_only": True},
    )
    document = json.loads(Path(args.command_receipt_output).read_text(encoding="utf-8"))
    assert document["status"] == "completed"
    assert document["internal_test_opened"] is False
    assert document["stage"]["manifest_sha256"] == (args.expected_stage_manifest_sha256)
    assert document["input_artifacts"]["cells"]["sha256"] == analyzer.sha256_file(
        args.cells
    )
    assert document["input_artifacts"]["builder_receipt"][
        "sha256"
    ] == analyzer.sha256_file(args.builder_receipt)
    assert document["output_artifacts"]["json_output"][
        "sha256"
    ] == analyzer.sha256_file(args.json_output)
    exact_set = document["output_artifacts"]["output_parent_exact_sets"][0]["exact_set"]
    assert {entry["path"] for entry in exact_set} == {
        "main_v1_validation_cells.jsonl",
        "main_v1_validation_cells.receipt.json",
        "main_v1_validation_report.json",
        "main_v1_validation_report.md",
    }


def test_v12_analyzer_failure_receipt_records_partial_exact_sets_and_no_overwrite(
    tmp_path, monkeypatch
):
    args = _phase_args(tmp_path)
    cells = Path(args.cells)
    cells.parent.mkdir(parents=True)
    cells.write_text("partial cells\n", encoding="utf-8")
    json_output = Path(args.json_output)
    json_output.write_text("partial report\n", encoding="utf-8")
    stage_root = Path(args.stage_manifest).parent
    stage_manifest = Path(args.stage_manifest)
    stage_manifest.parent.mkdir(parents=True, exist_ok=True)
    stage_manifest.write_text("{}\n", encoding="utf-8")
    staged_source = stage_root / "scripts/analyze_main_v1_validation.py"
    staged_source.parent.mkdir(parents=True)
    staged_source.write_text("trusted analyzer\n", encoding="utf-8")
    monkeypatch.setattr(
        analyzer,
        "_FORMAL_STAGE_BINDING",
        {
            "source_project_root": tmp_path,
            "stage_root": stage_root,
            "manifest_path": stage_manifest,
            "manifest_sha256": args.expected_stage_manifest_sha256,
            "source_contract_sha256": _sha("stage-source"),
            "stage_tree_sha256": _sha("stage-tree"),
        },
    )
    argv = _phase_argv(args)
    analyzer.write_analyzer_failure_receipt(RuntimeError("boom"), argv)
    document = json.loads(Path(args.failure_output).read_text(encoding="utf-8"))
    assert document["status"] == "failed_closed"
    assert document["error_type"] == "RuntimeError"
    assert document["error"] == "boom"
    assert document["internal_test_opened"] is False
    output_exact_set = document["partial_artifacts"]["output_parent_exact_sets"][0][
        "exact_set"
    ]
    assert {entry["path"] for entry in output_exact_set} == {
        "main_v1_validation_cells.jsonl",
        "main_v1_validation_report.json",
    }
    stage_exact_set = document["partial_artifacts"]["stage_root"]["exact_set"]
    assert {entry["path"] for entry in stage_exact_set} == {
        "main_v1_stage_manifest.json",
        "scripts/analyze_main_v1_validation.py",
    }
    with pytest.raises(MainValidationAnalysisError, match="already exists"):
        analyzer.write_analyzer_failure_receipt(RuntimeError("again"), argv)


@pytest.mark.parametrize(
    "component",
    sorted(analyzer.SEALED_FORMAL_PATH_COMPONENTS),
)
def test_v12_analyzer_rejects_sealed_v3_through_v11_path_components(
    tmp_path, component
):
    path = tmp_path / component / "artifact.json"
    with pytest.raises(
        MainValidationAnalysisError,
        match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 path",
    ):
        analyzer._require_v6_artifact_path(path, "artifact")


@pytest.mark.parametrize(
    ("sealed_values", "label"),
    (
        (analyzer.SEALED_CONTROLLER_SHA256S, "controller"),
        (analyzer.SEALED_BUILDER_SHA256S, "builder"),
        (analyzer.SEALED_ANALYZER_SHA256S, "analyzer"),
        (analyzer.SEALED_CELL_REGISTRY_SHA256S, "cell registry"),
        (analyzer.SEALED_CHECKPOINT_INVENTORY_SHA256S, "inventory"),
        (analyzer.SEALED_RUN_MANIFEST_SHA256S, "run manifest"),
        (analyzer.SEALED_STAGE_MANIFEST_SHA256S, "stage manifest"),
        (analyzer.SEALED_STAGE_TREE_SHA256S, "stage tree"),
        (
            analyzer.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
    ),
)
def test_v12_analyzer_rejects_every_sealed_v3_through_v11_sha(sealed_values, label):
    for digest in sealed_values:
        with pytest.raises(
            MainValidationAnalysisError,
            match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 SHA-256",
        ):
            analyzer._require_fresh_v6_sha256(digest, sealed_values, label)
    fresh = _sha(f"fresh:{label}")
    assert analyzer._require_fresh_v6_sha256(fresh, sealed_values, label) == fresh


def test_v12_lifecycle_constants_bind_v11_full_failure_and_fresh_controller():
    assert analyzer.FORMAL_VERSION == "v12"
    assert analyzer.EXPECTED_CONTROLLER_SHA256 not in analyzer.SEALED_CONTROLLER_SHA256S
    assert analyzer.PREDECESSOR_FAILURE_SHA256 == (
        "b2ad831aa3372c827b8067f1a25f3dbd60af2697e9fcf834a14874191eb527a5"
    )
    assert analyzer.PREDECESSOR_FAILURE_RELATIVE.as_posix().endswith(
        "20260718_validation_v11_full_failure_receipt.json"
    )
    real_receipt = analyzer.PROJECT_ROOT / analyzer.PREDECESSOR_FAILURE_RELATIVE
    assert (
        hashlib.sha256(real_receipt.read_bytes()).hexdigest()
        == analyzer.PREDECESSOR_FAILURE_SHA256
    )
    assert analyzer.SEALED_V8_PREDECESSOR_FAILURE_SHA256 == (
        "49702ef4e9da3ece4f8b1368eea4df6d8b02deeb72bb0f2edc5bbd9b9485ec22"
    )
    assert analyzer.SEALED_V9_PREDECESSOR_FAILURE_SHA256 == (
        "fd124efe6e267d8c6f605c1beabd708b0c2fb5b14405944b41cac03b3a762755"
    )
    assert analyzer.SEALED_V10_PREDECESSOR_FAILURE_SHA256 == (
        "618175b58f2fd300e07b03ed0276c5d3b8b163736e82fadf19c82d74b05a9946"
    )
    assert analyzer.SEALED_V9_CONTROLLER_SHA256 == (
        "3a196bf8f0d71128339fc9ad3991b2945fb6792155d46f771c3a453768526d39"
    )
    assert analyzer.SEALED_V10_CONTROLLER_SHA256 == (
        "3d6a3865fcae526d4df48b8e8ae76d0e43ead6d711635faca257c129ca8b9e7d"
    )
    assert analyzer.SEALED_V11_CONTROLLER_SHA256 == (
        "0f4e782538087038178186bb6b7621d1b510152223b84ba11c34243b535b7d67"
    )
    assert {
        "validation_checkpoint_inventory_v9.json",
        "validation_eval_v9",
        "validation_analysis_v9",
        "validation_checkpoint_inventory_v10.json",
        "validation_eval_v10",
        "validation_analysis_v10",
        "validation_checkpoint_inventory_v11.json",
        "validation_eval_v11",
        "validation_analysis_v11",
    }.issubset(analyzer.SEALED_FORMAL_PATH_COMPONENTS)
    assert analyzer.SEALED_V8_CONTROLLER_SHA256 in analyzer.SEALED_CONTROLLER_SHA256S
    assert analyzer.SEALED_V9_CONTROLLER_SHA256 in analyzer.SEALED_CONTROLLER_SHA256S
    assert analyzer.SEALED_V10_CONTROLLER_SHA256 in analyzer.SEALED_CONTROLLER_SHA256S
    assert analyzer.SEALED_V11_CONTROLLER_SHA256 in analyzer.SEALED_CONTROLLER_SHA256S
    assert analyzer.SEALED_V8_BUILDER_SHA256 in analyzer.SEALED_BUILDER_SHA256S
    assert analyzer.SEALED_V9_BUILDER_SHA256 in analyzer.SEALED_BUILDER_SHA256S
    assert analyzer.SEALED_V10_BUILDER_SHA256 in analyzer.SEALED_BUILDER_SHA256S
    assert analyzer.SEALED_V11_BUILDER_SHA256 in analyzer.SEALED_BUILDER_SHA256S
    assert analyzer.SEALED_V8_ANALYZER_SHA256 in analyzer.SEALED_ANALYZER_SHA256S
    assert analyzer.SEALED_V9_ANALYZER_SHA256 in analyzer.SEALED_ANALYZER_SHA256S
    assert analyzer.SEALED_V10_ANALYZER_SHA256 in analyzer.SEALED_ANALYZER_SHA256S
    assert analyzer.SEALED_V11_ANALYZER_SHA256 in analyzer.SEALED_ANALYZER_SHA256S
    assert (
        analyzer.SEALED_V8_CELL_REGISTRY_SHA256
        in analyzer.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        analyzer.SEALED_V9_CELL_REGISTRY_SHA256
        in analyzer.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        analyzer.SEALED_V10_CELL_REGISTRY_SHA256
        in analyzer.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        analyzer.SEALED_V11_CELL_REGISTRY_SHA256
        in analyzer.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        analyzer.SEALED_V8_INVENTORY_SHA256
        in analyzer.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        analyzer.SEALED_V9_INVENTORY_SHA256
        in analyzer.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        analyzer.SEALED_V10_INVENTORY_SHA256
        in analyzer.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        analyzer.SEALED_V11_INVENTORY_SHA256
        in analyzer.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        analyzer.SEALED_V8_STAGE_MANIFEST_SHA256
        in analyzer.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        analyzer.SEALED_V9_STAGE_MANIFEST_SHA256
        in analyzer.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        analyzer.SEALED_V10_STAGE_MANIFEST_SHA256
        in analyzer.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        analyzer.SEALED_V11_STAGE_MANIFEST_SHA256
        in analyzer.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert analyzer.SEALED_V8_STAGE_TREE_SHA256 in analyzer.SEALED_STAGE_TREE_SHA256S
    assert analyzer.SEALED_V9_STAGE_TREE_SHA256 in analyzer.SEALED_STAGE_TREE_SHA256S
    assert analyzer.SEALED_V10_STAGE_TREE_SHA256 in analyzer.SEALED_STAGE_TREE_SHA256S
    assert analyzer.SEALED_V11_STAGE_TREE_SHA256 in analyzer.SEALED_STAGE_TREE_SHA256S
    assert (
        analyzer.SEALED_V8_STAGE_SOURCE_CONTRACT_SHA256
        in analyzer.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        analyzer.SEALED_V9_STAGE_SOURCE_CONTRACT_SHA256
        in analyzer.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        analyzer.SEALED_V10_STAGE_SOURCE_CONTRACT_SHA256
        in analyzer.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        analyzer.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256
        in analyzer.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert analyzer.ANALYZER_PHASE_RECEIPTS["success"].as_posix().endswith(
        "20260718_validation_v12_analyzer_command_receipt.json"
    )
    assert analyzer.ANALYZER_PHASE_RECEIPTS["failure"].as_posix().endswith(
        "20260718_validation_v12_analyzer_failure_receipt.json"
    )
    controller_path = analyzer.PROJECT_ROOT / "scripts/run_main_v1_validation_eval.py"
    assert analyzer.EXPECTED_CONTROLLER_SHA256 == (
        hashlib.sha256(controller_path.read_bytes()).hexdigest()
    )


def test_v12_registry_requires_exact_controller_identity(tmp_path):
    registry = _registry()
    registry.pop("_registry_sha256")
    path = tmp_path / "main_v1_validation_cell_registry_v12.json"

    def write_registry() -> str:
        path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    registry["controller_sha256"] = _sha("wrong-v12-controller")
    with pytest.raises(
        MainValidationAnalysisError,
        match="registry.controller_sha256 vs v12 controller",
    ):
        analyzer.load_registry(path, write_registry())

    registry["controller_sha256"] = analyzer.EXPECTED_CONTROLLER_SHA256
    loaded = analyzer.load_registry(path, write_registry())
    assert loaded["controller_sha256"] == analyzer.EXPECTED_CONTROLLER_SHA256


def test_validate_cells_accepts_exact_27_and_audits_states():
    rows, registry = _cells()
    validated = validate_cells(rows, registry)
    assert len(validated) == 27
    assert validated[0]["method_id"] == "B0"
    assert validated[-1]["method_id"] == "H0"
    assert validated[0]["formal_primary"] is True
    assert validated[0]["formal_minus_selection_bce_at1"] != 0.0
    assert (
        next(row for row in validated if row["method_id"] == "B1")[
            "formal_minus_selection_bce_at1"
        ]
        == 0.0
    )


def _inject_b1_selection_drift(rows):
    for row in rows:
        if row["method_id"] == "B1" and row["seed"] == 0:
            row["selection_bce_at1"] -= 1e-6
            row["formal_minus_selection_bce_at1"] = (
                row["formal_bce_at1"] - row["selection_bce_at1"]
            )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda rows: rows.pop(), "exactly 27"),
        (
            lambda rows: rows[0].update({"test_manifest_sha256": _sha("test")}),
            "forbidden",
        ),
        (
            lambda rows: rows[0].update({"treatment_config_sha256": _sha("wrong")}),
            "treatment_config",
        ),
        (
            lambda rows: [
                row.update({"model_state_sha256": _sha("reused-H0")})
                for row in rows
                if row["method_id"] == "H0"
            ],
            "H0 model-state SHA reuse",
        ),
        (
            lambda rows: [
                row.update({"selected_epoch": 1})
                for row in rows
                if row["method_id"] == "H0" and row["seed"] == 0
            ],
            "selected/checkpoint epoch",
        ),
        (
            lambda rows: [
                row.update({"selected_value": 9.0})
                for row in rows
                if row["method_id"] == "H0" and row["seed"] == 0
            ],
            "selected/run_end BCE mismatch",
        ),
        (
            lambda rows: rows[0].update({"bce_at1": rows[0]["selection_bce_at1"]}),
            "formal primary alias",
        ),
        (
            lambda rows: rows[0].update(
                {
                    "formal_minus_selection_bce_at1": rows[0][
                        "formal_minus_selection_bce_at1"
                    ]
                    + 1e-6
                }
            ),
            "episode drift identity",
        ),
        (
            _inject_b1_selection_drift,
            "selection episode-macro identity|same-contract",
        ),
    ),
)
def test_validate_cells_fails_closed(mutation, match):
    rows, registry = _cells()
    mutation(rows)
    with pytest.raises(MainValidationAnalysisError, match=match):
        validate_cells(rows, registry)


def test_exact_bootstrap_enumerates_729_ordered_draws():
    import numpy as np

    delta = np.arange(9, dtype=float).reshape(3, 3) / 100
    result = _exact_two_way_bootstrap(delta)
    assert result["ordered_resample_count"] == 729
    assert 0 <= result["favorable_probability"] <= 1
    assert result["unique_draw_values"] <= 729


def test_analyzer_source_must_match_cli_and_registry(tmp_path):
    analyzer = tmp_path / "analyzer.py"
    analyzer.write_text("trusted", encoding="utf-8")
    actual = hashlib.sha256(analyzer.read_bytes()).hexdigest()
    registry = {"analyzer_source_sha256": actual}
    assert validate_analyzer_source_binding(registry, actual, analyzer) == actual
    with pytest.raises(MainValidationAnalysisError, match="vs registry"):
        validate_analyzer_source_binding(
            {"analyzer_source_sha256": _sha("other")}, actual, analyzer
        )


def test_seed_cluster_contrast_and_report_are_validation_only():
    rows, registry = _cells()
    rows = validate_cells(rows, registry)
    primary = _contrast_result(rows, registry["contrasts"][0])
    assert primary["delta_by_seed"] == pytest.approx([0.01, -0.01, -0.03])
    assert primary["mean_delta"] == pytest.approx(-0.01)
    assert primary["seed_cluster_df"] == 2
    assert set(primary["leave_one_seed_out"]) == {"0", "1", "2"}
    assert set(primary["leave_one_episode_out"]) == set(EPISODE_IDS)
    assert primary["exact_two_way_bootstrap"]["ordered_resample_count"] == 729
    report = analyze(
        rows,
        registry,
        cells_sha256=_sha("cells"),
        analyzer_sha256=_sha("analyzer"),
        builder_receipt={
            "builder_source_sha256": _sha("builder-source"),
            "builder_dependency_sha256": _sha("builder-dependency"),
            "builder_dependency_analyzer_sha256": registry[
                "builder_dependency_analyzer_sha256"
            ],
            "builder_runtime_dependency_inventory_sha256": registry[
                "builder_runtime_dependency_inventory_sha256"
            ],
            "evaluator_file_sha256": registry["evaluator_file_sha256"],
            "experiment_binding_file_sha256": registry[
                "experiment_binding_file_sha256"
            ],
            "controller_run_manifest_sha256": _sha("controller-manifest"),
            "checkpoint_inventory_sha256": _sha("inventory"),
            "runtime_isolation_contract_sha256": registry[
                "runtime_isolation_contract_sha256"
            ],
            "selection_replay_manifest_sha256": _sha("selection-replay-manifest"),
            "predecessor_failure_receipt_sha256": (analyzer.PREDECESSOR_FAILURE_SHA256),
        },
        builder_receipt_sha256=_sha("receipt"),
    )
    assert report["validation_only"] is True
    assert report["paper_eligible"] is False
    assert report["methods"]["B0"]["unique_model_state_count"] == 1
    assert report["methods"]["B0"]["seed_sd_ddof1"] == pytest.approx(0.0)
    assert report["metric"] == "formal_batch1_episode_macro_BCE@1"
    assert report["methods"]["B0"]["selection_replay"]["selection_batch_size"] == 2
    assert report["methods"]["B1"]["selection_replay"][
        "formal_minus_selection_macro_by_seed"
    ] == [0.0, 0.0, 0.0]
    assert report["provenance"]["builder_receipt_sha256"] == _sha("receipt")


def test_builder_receipt_closes_upstream_chain(tmp_path, monkeypatch):
    predecessor_path = _write_predecessor_receipt(tmp_path, monkeypatch)
    registry = _registry()
    cells_sha = _sha("cells")
    receipt = {
        "schema_version": 2,
        "analysis_class": ANALYSIS_CLASS,
        "formal_version": analyzer.FORMAL_VERSION,
        "dual_contract_id": analyzer.DUAL_CONTRACT_ID,
        "status": "completed",
        "validation_only": True,
        "paper_eligible": False,
        "builder_source_sha256": registry["builder_source_sha256"],
        "builder_dependency_sha256": registry["builder_dependency_sha256"],
        "builder_dependency_analyzer_sha256": registry[
            "builder_dependency_analyzer_sha256"
        ],
        "builder_runtime_dependency_inventory_sha256": registry[
            "builder_runtime_dependency_inventory_sha256"
        ],
        "evaluator_file_sha256": registry["evaluator_file_sha256"],
        "experiment_binding_file_sha256": registry["experiment_binding_file_sha256"],
        "cell_registry_sha256": registry["_registry_sha256"],
        "validation_data_sha256": registry["validation_dataset"]["data_sha256"],
        "cells_sha256": cells_sha,
        "cell_count": 27,
        "controller_run_manifest_sha256": _sha("controller-manifest"),
        "checkpoint_inventory_sha256": _sha("inventory"),
        "runtime_isolation_contract_sha256": registry[
            "runtime_isolation_contract_sha256"
        ],
        "selection_replay_manifest_sha256": _sha("selection-replay-manifest"),
        "selection_replay_receipt_count": 7,
        "predecessor_failure_receipt": str(predecessor_path),
        "predecessor_failure_receipt_sha256": (analyzer.PREDECESSOR_FAILURE_SHA256),
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    loaded, actual = load_builder_receipt(
        path,
        expected,
        registry=registry,
        cells_sha256=cells_sha,
        controller_run_manifest_sha256=_sha("controller-manifest"),
        checkpoint_inventory_sha256=_sha("inventory"),
    )
    assert loaded == receipt
    assert actual == expected
    with pytest.raises(MainValidationAnalysisError, match="receipt.cells_sha256"):
        load_builder_receipt(
            path,
            expected,
            registry=registry,
            cells_sha256=_sha("other-cells"),
            controller_run_manifest_sha256=_sha("controller-manifest"),
            checkpoint_inventory_sha256=_sha("inventory"),
        )


def test_v12_predecessor_receipt_requires_exact_full_v11_failure_binding(
    tmp_path, monkeypatch
):
    receipt_path = _write_predecessor_receipt(tmp_path, monkeypatch)
    pristine = receipt_path.read_text(encoding="utf-8")
    binding = analyzer._load_v11_full_failure_receipt(receipt_path)
    document = binding
    assert document["path"] == str(receipt_path.resolve())
    assert document["sha256"] == analyzer.PREDECESSOR_FAILURE_SHA256
    assert document["schema_version"] == 1
    assert document["analysis_class"] == (
        "main_v1_validation_controller_phase_receipt"
    )
    assert document["formal_version"] == "v11"
    assert document["phase"] == "full"
    assert document["status"] == "failed_closed"
    assert document["exit_code"] == 1
    assert document["error_type"] == "ValidationEvalError"
    assert document["error"] == (
        "failed to parse argv for guarded process launcher"
    )
    assert document["internal_test_opened"] is False
    assert document["preflight_output"] is None
    assert document["required_python_options"] == ["-I", "-S", "-B", "-u"]
    assert document["controller_expected_sha256"] == (
        analyzer.SEALED_V11_CONTROLLER_SHA256
    )
    assert document["checkpoint_inventory_expected_sha256"] == (
        analyzer.SEALED_V11_INVENTORY_SHA256
    )
    assert document["stage_manifest_expected_sha256"] == (
        analyzer.SEALED_V11_STAGE_MANIFEST_SHA256
    )
    assert document["stage_source_contract_expected_sha256"] == (
        analyzer.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256
    )
    assert document["runtime_isolation_contract_expected_sha256"] == (
        analyzer.SEALED_V11_RUNTIME_ISOLATION_CONTRACT_SHA256
    )
    assert document["predecessor_failure_receipt_expected_sha256"] == (
        analyzer.SEALED_V10_PREDECESSOR_FAILURE_SHA256
    )
    assert Path(document["checkpoint_inventory"]).resolve() == (
        analyzer.PROJECT_ROOT
        / "experiments/collected_v1_main/validation_checkpoint_inventory_v11.json"
    ).resolve()
    assert Path(document["output_dir"]).resolve() == (
        analyzer.PROJECT_ROOT / "experiments/collected_v1_main/validation_eval_v11"
    ).resolve()
    assert Path(document["stage_manifest"]).resolve() == Path(
        "/private/tmp/track_car_main_v1_stage-79z0uynl/main_v1_stage_manifest.json"
    )
    assert Path(document["predecessor_failure_receipt"]).resolve() == (
        analyzer.PROJECT_ROOT
        / "experiments/collected_v1_main/external_reviews/"
        "20260718_validation_v10_full_failure_receipt.json"
    ).resolve()
    for field, expected in (
        ("argv", analyzer.SEALED_V11_FAILURE_ARGV_SHA256),
        ("partial_artifacts", analyzer.SEALED_V11_FAILURE_PARTIAL_ARTIFACTS_SHA256),
        ("error", analyzer.SEALED_V11_FAILURE_ERROR_SHA256),
    ):
        assert analyzer._canonical_json_sha256(document[field]) == expected
    assert document["partial_artifacts"]["output_dir"]["exists"] is False
    assert document["partial_artifacts"]["checkpoint_inventory"]["sha256"] == (
        analyzer.SEALED_V11_INVENTORY_SHA256
    )
    assert document["partial_artifacts"]["stage_manifest"]["sha256"] == (
        analyzer.SEALED_V11_STAGE_MANIFEST_SHA256
    )

    def rewrite(mutated_document):
        receipt_path.write_text(
            json.dumps(
                mutated_document, ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            analyzer,
            "PREDECESSOR_FAILURE_SHA256",
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        )

    mutated = json.loads(pristine)
    mutated["phase"] = "preflight"
    rewrite(mutated)
    with pytest.raises(
        MainValidationAnalysisError, match="predecessor.phase"
    ):
        analyzer._load_v11_full_failure_receipt(receipt_path)

    mutated = json.loads(pristine)
    mutated["controller_expected_sha256"] = _sha("wrong-v11-sealed-controller")
    rewrite(mutated)
    with pytest.raises(
        MainValidationAnalysisError,
        match="predecessor.controller_expected_sha256",
    ):
        analyzer._load_v11_full_failure_receipt(receipt_path)

    mutated = json.loads(pristine)
    mutated["output_dir"] = str(tmp_path / "evil_output_dir")
    rewrite(mutated)
    with pytest.raises(
        MainValidationAnalysisError, match="predecessor output path"
    ):
        analyzer._load_v11_full_failure_receipt(receipt_path)

    mutated = json.loads(pristine)
    mutated["argv"].append("--internal_test")
    rewrite(mutated)
    with pytest.raises(
        MainValidationAnalysisError, match="predecessor argv binding"
    ):
        analyzer._load_v11_full_failure_receipt(receipt_path)


def test_upstream_artifacts_are_physically_rehashed(tmp_path):
    controller = tmp_path / "controller.json"
    inventory = tmp_path / "inventory.json"
    controller.write_text("controller", encoding="utf-8")
    inventory.write_text("inventory", encoding="utf-8")
    controller_sha = hashlib.sha256(controller.read_bytes()).hexdigest()
    inventory_sha = hashlib.sha256(inventory.read_bytes()).hexdigest()
    resolved = verify_upstream_artifact_hashes(
        controller_run_manifest=controller,
        expected_controller_run_manifest_sha256=controller_sha,
        checkpoint_inventory=inventory,
        expected_checkpoint_inventory_sha256=inventory_sha,
    )
    assert resolved == (controller.resolve(), inventory.resolve())
    inventory.write_text("tampered", encoding="utf-8")
    with pytest.raises(MainValidationAnalysisError, match="inventory SHA-256 mismatch"):
        verify_upstream_artifact_hashes(
            controller_run_manifest=controller,
            expected_controller_run_manifest_sha256=controller_sha,
            checkpoint_inventory=inventory,
            expected_checkpoint_inventory_sha256=inventory_sha,
        )


def test_analyzer_reexecutes_trusted_builder(monkeypatch, tmp_path):
    rows, registry = _cells()
    rows = validate_cells(rows, registry)

    class FakeBuilder:
        @staticmethod
        def validate_builder_registry(value):
            assert value is registry

        @staticmethod
        def load_controller_manifest(*_):
            return {"runs": {}}

        @staticmethod
        def load_checkpoint_inventory(*_):
            return {"runs": {}}

        @staticmethod
        def build_main_cells(*_):
            return rows

    original_sha = analyzer.sha256_file

    def fake_sha(path):
        resolved = Path(path).resolve()
        if resolved == analyzer.BUILDER_PATH:
            return registry["builder_source_sha256"]
        if resolved == analyzer.BUILDER_DEPENDENCY_PATH:
            return registry["builder_dependency_sha256"]
        if resolved == analyzer.BUILDER_DEPENDENCY_ANALYZER_PATH:
            return registry["builder_dependency_analyzer_sha256"]
        if resolved == analyzer.BUILDER_DEPENDENCY_EVALUATOR_PATH:
            return registry["evaluator_file_sha256"]
        if resolved == analyzer.BUILDER_DEPENDENCY_EXPERIMENT_BINDING_PATH:
            return registry["experiment_binding_file_sha256"]
        return original_sha(path)

    monkeypatch.setattr(analyzer, "sha256_file", fake_sha)
    monkeypatch.setattr(analyzer, "load_verified_builder_module", lambda: FakeBuilder)
    rebuild_and_verify_cells(
        rows,
        registry,
        controller_run_manifest=tmp_path / "controller.json",
        controller_run_manifest_sha256=_sha("controller"),
        checkpoint_inventory=tmp_path / "inventory.json",
        checkpoint_inventory_sha256=_sha("inventory"),
    )
    with pytest.raises(MainValidationAnalysisError, match="rebuilt cells"):
        rebuild_and_verify_cells(
            rows[:-1],
            registry,
            controller_run_manifest=tmp_path / "controller.json",
            controller_run_manifest_sha256=_sha("controller"),
            checkpoint_inventory=tmp_path / "inventory.json",
            checkpoint_inventory_sha256=_sha("inventory"),
        )


def test_output_paths_cannot_overlap_each_other_or_inputs(tmp_path):
    source = tmp_path / "cells.jsonl"
    with pytest.raises(MainValidationAnalysisError, match="must be distinct"):
        validate_distinct_paths(
            input_paths=[source], output_paths=[tmp_path / "same", tmp_path / "same"]
        )
    with pytest.raises(MainValidationAnalysisError, match="overlaps"):
        validate_distinct_paths(
            input_paths=[source], output_paths=[source, tmp_path / "report.md"]
        )
