import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import scripts.build_main_v1_validation_cells as builder
from scripts.analyze_main_v1_validation import (
    ANALYSIS_CLASS,
    EPISODE_IDS,
    METHOD_IDS,
    SEED_IDS,
)
from scripts.build_main_v1_validation_cells import (
    MainValidationBuildError,
    build_main_cells,
    load_checkpoint_inventory,
    load_controller_manifest,
    validate_distinct_paths,
    validate_builder_registry,
)


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def _phase_args(tmp_path):
    analysis_root = tmp_path / "experiments/collected_v1_main/validation_analysis_v12"
    return SimpleNamespace(
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
        expected_builder_sha256=_sha("builder"),
        stage_manifest=str(tmp_path / "stage/main_v1_stage_manifest.json"),
        expected_stage_manifest_sha256=_sha("stage-manifest"),
        output=str(analysis_root / "main_v1_validation_cells.jsonl"),
        receipt_output=str(analysis_root / "main_v1_validation_cells.receipt.json"),
        command_receipt_output=str(
            tmp_path / builder.BUILDER_PHASE_RECEIPTS["success"]
        ),
        failure_output=str(tmp_path / builder.BUILDER_PHASE_RECEIPTS["failure"]),
    )


def _phase_argv(args):
    values = ["build_main_v1_validation_cells.py"]
    for field in (
        "controller_run_manifest",
        "expected_controller_run_manifest_sha256",
        "checkpoint_inventory",
        "expected_checkpoint_inventory_sha256",
        "cell_registry",
        "expected_cell_registry_sha256",
        "expected_builder_sha256",
        "stage_manifest",
        "expected_stage_manifest_sha256",
        "output",
        "receipt_output",
        "command_receipt_output",
        "failure_output",
    ):
        values.extend([f"--{field}", str(getattr(args, field))])
    return values


def _runtime_contract():
    return {
        "controller_cli": {
            "required_sys_flags": {
                "isolated": True,
                "no_site": True,
                "dont_write_bytecode": True,
                "ignore_environment": True,
                "safe_path": True,
                "unbuffered_stdout": True,
            }
        }
    }


def _predecessor_binding():
    path = builder.PROJECT_ROOT / builder.PREDECESSOR_FAILURE_RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": str(path.resolve()),
        "sha256": builder.PREDECESSOR_FAILURE_SHA256,
        **document,
    }


def _registry():
    runtime_contract_sha = builder.canonical_json_sha256(_runtime_contract())
    contracts = {
        "registry_sha256": _sha("main-registry"),
        "validation_data_sha256": _sha("val-data"),
        "validation_manifest_sha256": _sha("val-manifest"),
        "cache_manifest_sha256": _sha("cache-manifest"),
        "source_tree_sha256": _sha("source"),
        "evaluator_source_sha256": _sha("evaluator"),
        "evaluator_file_sha256": builder.EXPECTED_DEPENDENCY_EVALUATOR_SHA256,
        "metric_contract_sha256": (
            "c75d0c08fd209e120e59eac10bb9cdd0fac6fdd65582beaf3c63a1b18a6c5097"
        ),
        "runtime_isolation_contract_sha256": runtime_contract_sha,
        "stage_source_contract_sha256": _sha("stage-source-contract"),
        "provenance_sha256": _sha("cache-provenance"),
        "token_payload_sha256": _sha("cache-payload"),
        "dino_model_sha256": _sha("dino"),
        "siglip_model_sha256": _sha("siglip"),
    }
    return {
        "_registry_sha256": _sha("cell-registry"),
        "controller_sha256": builder.EXPECTED_CONTROLLER_SHA256,
        "builder_source_sha256": _sha("builder"),
        "builder_dependency_sha256": builder.EXPECTED_DEPENDENCY_SHA256,
        "builder_dependency_analyzer_sha256": (
            builder.EXPECTED_DEPENDENCY_ANALYZER_SHA256
        ),
        "builder_runtime_dependency_inventory_sha256": (
            builder.EXPECTED_RUNTIME_DEPENDENCY_INVENTORY_SHA256
        ),
        "analyzer_source_sha256": builder.EXPECTED_ANALYZER_SHA256,
        "parent_main_registry_sha256": contracts["registry_sha256"],
        "source_tree_sha256": contracts["source_tree_sha256"],
        "evaluator_source_sha256": contracts["evaluator_source_sha256"],
        "evaluator_file_sha256": contracts["evaluator_file_sha256"],
        "experiment_binding_file_sha256": (
            builder.EXPECTED_DEPENDENCY_EXPERIMENT_BINDING_SHA256
        ),
        "metric_contract_sha256": contracts["metric_contract_sha256"],
        "metric_contract_schema_version": 2,
        "runtime_isolation_contract_sha256": runtime_contract_sha,
        "stage_source_contract_sha256": contracts["stage_source_contract_sha256"],
        "fairness_contract_sha256": _sha("fairness"),
        "validation_dataset": {
            "path": "/tmp/val.jsonl",
            "data_sha256": contracts["validation_data_sha256"],
            "manifest_sha256": contracts["validation_manifest_sha256"],
            "sample_count": 2848,
        },
        "expected_controller_contracts": contracts,
        "method_contracts": {
            "B0": {
                "state_mode": "stateless",
                "model_family": "B0",
                "treatment_config_sha256": _sha("B0"),
            },
            "B1": {
                "state_mode": "rolling",
                "model_family": "B1",
                "treatment_config_sha256": _sha("B1"),
            },
            "H0": {
                "state_mode": "rolling",
                "model_family": "H0",
                "treatment_config_sha256": _sha("H0"),
            },
        },
        "deterministic_baseline": {"model_state_sha256": _sha("B0-state")},
    }


def _stage_binding(tmp_path, registry):
    return {
        "source_project_root": builder.PROJECT_ROOT,
        "stage_root": tmp_path / "stage-root",
        "manifest_path": tmp_path / "stage-root/main_v1_stage_manifest.json",
        "manifest_sha256": _sha("stage-manifest"),
        "source_contract_sha256": registry["stage_source_contract_sha256"],
        "stage_tree_sha256": _sha("stage-tree"),
    }


def test_builder_frozen_json_hashes_and_parses_one_payload(tmp_path, monkeypatch):
    path = tmp_path / "frozen.json"
    path.write_text('{"trusted": true}\n', encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    original_open = Path.open
    calls = 0

    def racing_open(source, *args, **kwargs):
        nonlocal calls
        mode = args[0] if args else kwargs.get("mode", "r")
        if source.resolve() == path.resolve() and "r" in mode:
            calls += 1
            if calls == 2:
                with original_open(source, "wb") as handle:
                    handle.write(b'{"trusted": false}\n')
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(Path, "open", racing_open)
    source, value, actual = builder._read_frozen_json(
        path, expected, "test frozen document"
    )
    assert source == path.resolve()
    assert value == {"trusted": True}
    assert actual == expected
    assert calls == 1


def test_v12_builder_parser_requires_both_one_shot_receipts(tmp_path):
    args = _phase_args(tmp_path)
    argv = _phase_argv(args)[1:]
    without_receipts = argv[:-4]
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args(without_receipts)
    with pytest.raises(SystemExit):
        builder.build_parser().parse_args(
            without_receipts + ["--command_receipt_output", args.command_receipt_output]
        )
    parsed = builder.build_parser().parse_args(argv)
    assert parsed.command_receipt_output == args.command_receipt_output
    assert parsed.failure_output == args.failure_output


def test_v12_builder_receipt_paths_are_exact_absent_and_checked_early(tmp_path):
    args = _phase_args(tmp_path)
    success, failure = builder.validate_builder_phase_receipt_cli(
        args, project_root=tmp_path
    )
    assert success == (tmp_path / builder.BUILDER_PHASE_RECEIPTS["success"]).resolve()
    assert failure == (tmp_path / builder.BUILDER_PHASE_RECEIPTS["failure"]).resolve()

    args.command_receipt_output = str(tmp_path / "wrong-success.json")
    with pytest.raises(MainValidationBuildError, match="success receipt path"):
        builder.validate_builder_phase_receipt_cli(args, project_root=tmp_path)

    args = _phase_args(tmp_path)
    success = Path(args.command_receipt_output)
    success.parent.mkdir(parents=True)
    success.write_text("{}\n", encoding="utf-8")
    with pytest.raises(MainValidationBuildError, match="already exists"):
        builder.validate_builder_phase_receipt_cli(args, project_root=tmp_path)


def test_v12_builder_success_receipt_binds_inputs_outputs_and_stage(
    tmp_path, monkeypatch
):
    args = _phase_args(tmp_path)
    for field in (
        "controller_run_manifest",
        "checkpoint_inventory",
        "cell_registry",
        "stage_manifest",
        "output",
        "receipt_output",
    ):
        path = Path(getattr(args, field))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{field}\n", encoding="utf-8")
    stage_root = Path(args.stage_manifest).parent
    staged_source = stage_root / "scripts/build_main_v1_validation_cells.py"
    staged_source.parent.mkdir(parents=True)
    staged_source.write_text("trusted builder\n", encoding="utf-8")
    monkeypatch.setattr(
        builder,
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
    builder._write_builder_phase_receipt(
        kind="success",
        status="completed",
        exit_code=0,
        result={"status": "completed"},
    )
    document = json.loads(Path(args.command_receipt_output).read_text(encoding="utf-8"))
    assert document["status"] == "completed"
    assert document["internal_test_opened"] is False
    assert document["stage"]["manifest_sha256"] == (args.expected_stage_manifest_sha256)
    assert document["input_artifacts"]["controller_run_manifest"][
        "sha256"
    ] == builder.sha256_file(args.controller_run_manifest)
    assert document["output_artifacts"]["cells"]["sha256"] == builder.sha256_file(
        args.output
    )
    assert document["output_artifacts"]["builder_receipt"][
        "sha256"
    ] == builder.sha256_file(args.receipt_output)
    exact_set = document["output_artifacts"]["output_parent_exact_sets"][0]["exact_set"]
    assert {entry["path"] for entry in exact_set} == {
        "main_v1_validation_cells.jsonl",
        "main_v1_validation_cells.receipt.json",
    }


def test_v12_builder_failure_receipt_records_partial_exact_sets_and_no_overwrite(
    tmp_path, monkeypatch
):
    args = _phase_args(tmp_path)
    output = Path(args.output)
    output.parent.mkdir(parents=True)
    output.write_text("partial cells\n", encoding="utf-8")
    stage_root = Path(args.stage_manifest).parent
    stage_manifest = Path(args.stage_manifest)
    stage_manifest.parent.mkdir(parents=True, exist_ok=True)
    stage_manifest.write_text("{}\n", encoding="utf-8")
    staged_source = stage_root / "scripts/build_main_v1_validation_cells.py"
    staged_source.parent.mkdir(parents=True)
    staged_source.write_text("trusted builder\n", encoding="utf-8")
    monkeypatch.setattr(
        builder,
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
    builder._write_builder_phase_receipt(
        kind="failure",
        status="failed_closed",
        exit_code=1,
        error=RuntimeError("boom"),
        argv=argv,
    )
    document = json.loads(Path(args.failure_output).read_text(encoding="utf-8"))
    assert document["status"] == "failed_closed"
    assert document["error_type"] == "RuntimeError"
    assert document["error"] == "boom"
    assert document["internal_test_opened"] is False
    output_exact_set = document["partial_artifacts"]["output_parent_exact_sets"][0][
        "exact_set"
    ]
    assert {entry["path"] for entry in output_exact_set} == {
        "main_v1_validation_cells.jsonl"
    }
    stage_exact_set = document["partial_artifacts"]["stage_root"]["exact_set"]
    assert {entry["path"] for entry in stage_exact_set} == {
        "main_v1_stage_manifest.json",
        "scripts/build_main_v1_validation_cells.py",
    }
    with pytest.raises(MainValidationBuildError, match="already exists"):
        builder._write_builder_phase_receipt(
            kind="failure",
            status="failed_closed",
            exit_code=1,
            error=RuntimeError("again"),
            argv=argv,
        )


@pytest.mark.parametrize(
    "component",
    sorted(builder.SEALED_FORMAL_PATH_COMPONENTS),
)
def test_v12_builder_rejects_sealed_v3_through_v11_path_components(
    tmp_path, component
):
    path = tmp_path / component / "artifact.json"
    with pytest.raises(
        MainValidationBuildError, match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 path"
    ):
        builder._require_v6_artifact_path(path, "artifact")


@pytest.mark.parametrize(
    ("sealed_values", "label"),
    (
        (builder.SEALED_CONTROLLER_SHA256S, "controller"),
        (builder.SEALED_BUILDER_SHA256S, "builder"),
        (builder.SEALED_ANALYZER_SHA256S, "analyzer"),
        (builder.SEALED_CELL_REGISTRY_SHA256S, "cell registry"),
        (builder.SEALED_CHECKPOINT_INVENTORY_SHA256S, "inventory"),
        (builder.SEALED_RUN_MANIFEST_SHA256S, "run manifest"),
        (builder.SEALED_STAGE_MANIFEST_SHA256S, "stage manifest"),
        (builder.SEALED_STAGE_TREE_SHA256S, "stage tree"),
        (builder.SEALED_STAGE_SOURCE_CONTRACT_SHA256S, "stage source contract"),
    ),
)
def test_v12_builder_rejects_every_sealed_v3_through_v11_sha(sealed_values, label):
    for digest in sealed_values:
        with pytest.raises(
            MainValidationBuildError,
            match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 SHA-256",
        ):
            builder._require_fresh_v6_sha256(digest, sealed_values, label)
    fresh = _sha(f"fresh:{label}")
    assert builder._require_fresh_v6_sha256(fresh, sealed_values, label) == fresh


def test_v12_predecessor_receipt_hash_and_lifecycle_are_frozen():
    path = builder.PROJECT_ROOT / builder.PREDECESSOR_FAILURE_RELATIVE
    payload = path.read_bytes()
    assert builder.FORMAL_VERSION == "v12"
    assert path.name == "20260718_validation_v11_full_failure_receipt.json"
    assert hashlib.sha256(payload).hexdigest() == builder.PREDECESSOR_FAILURE_SHA256
    document = json.loads(payload.decode("utf-8"))
    assert set(document) == builder.PREDECESSOR_FAILURE_DOCUMENT_FIELDS
    assert document["schema_version"] == 1
    assert document["analysis_class"] == (
        "main_v1_validation_controller_phase_receipt"
    )
    assert document["formal_version"] == "v11"
    assert document["phase"] == "full"
    assert document["status"] == "failed_closed"
    assert document["internal_test_opened"] is False
    assert document["error_type"] == "ValidationEvalError"
    assert document["error"] == "failed to parse argv for guarded process launcher"
    assert document["exit_code"] == 1
    assert document["preflight_output"] is None
    assert document["required_python_options"] == ["-I", "-S", "-B", "-u"]
    assert builder.SEALED_V11_CONTROLLER_SHA256 == (
        "0f4e782538087038178186bb6b7621d1b510152223b84ba11c34243b535b7d67"
    )
    assert document["controller_expected_sha256"] == (
        builder.SEALED_V11_CONTROLLER_SHA256
    )
    assert document["checkpoint_inventory_expected_sha256"] == (
        builder.SEALED_V11_CHECKPOINT_INVENTORY_SHA256
    )
    assert document["stage_manifest_expected_sha256"] == (
        builder.SEALED_V11_STAGE_MANIFEST_SHA256
    )
    assert document["stage_source_contract_expected_sha256"] == (
        builder.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256
    )
    assert document["runtime_isolation_contract_expected_sha256"] == (
        builder.SEALED_V11_RUNTIME_ISOLATION_CONTRACT_SHA256
    )
    assert builder.SEALED_V10_PREDECESSOR_FAILURE_SHA256 == (
        "618175b58f2fd300e07b03ed0276c5d3b8b163736e82fadf19c82d74b05a9946"
    )
    assert document["predecessor_failure_receipt_expected_sha256"] == (
        builder.SEALED_V10_PREDECESSOR_FAILURE_SHA256
    )
    assert document["predecessor_failure_receipt"].endswith(
        "experiments/collected_v1_main/external_reviews/"
        "20260718_validation_v10_full_failure_receipt.json"
    )
    assert document["partial_artifacts"]["output_dir"]["exists"] is False
    assert document["partial_artifacts"]["checkpoint_inventory"]["sha256"] == (
        builder.SEALED_V11_CHECKPOINT_INVENTORY_SHA256
    )
    assert document["partial_artifacts"]["stage_manifest"]["sha256"] == (
        builder.SEALED_V11_STAGE_MANIFEST_SHA256
    )
    assert builder.canonical_json_sha256(document["argv"]) == (
        builder.SEALED_V11_FAILURE_ARGV_SHA256
    )
    assert builder.canonical_json_sha256(document["partial_artifacts"]) == (
        builder.SEALED_V11_FAILURE_PARTIAL_ARTIFACTS_SHA256
    )
    assert builder.canonical_json_sha256(document["error"]) == (
        builder.SEALED_V11_FAILURE_ERROR_SHA256
    )


def test_v12_controller_builder_and_analyzer_source_identities_are_bound_and_fresh():
    controller_sha = builder.sha256_file(
        builder.PROJECT_ROOT / "scripts/run_main_v1_validation_eval.py"
    )
    builder_sha = builder.sha256_file(builder.BUILDER_PATH)
    analyzer_sha = builder.sha256_file(builder.ANALYZER_PATH)
    assert builder.EXPECTED_CONTROLLER_SHA256 == controller_sha
    assert builder.EXPECTED_CONTROLLER_SHA256 not in builder.SEALED_CONTROLLER_SHA256S
    assert controller_sha not in builder.SEALED_CONTROLLER_SHA256S
    assert builder_sha not in builder.SEALED_BUILDER_SHA256S
    assert analyzer_sha == builder.EXPECTED_ANALYZER_SHA256
    assert analyzer_sha not in builder.SEALED_ANALYZER_SHA256S
    assert builder.SEALED_V8_CONTROLLER_SHA256 in builder.SEALED_CONTROLLER_SHA256S
    assert builder.SEALED_V9_CONTROLLER_SHA256 in builder.SEALED_CONTROLLER_SHA256S
    assert builder.SEALED_V10_CONTROLLER_SHA256 in builder.SEALED_CONTROLLER_SHA256S
    assert builder.SEALED_V11_CONTROLLER_SHA256 in builder.SEALED_CONTROLLER_SHA256S
    assert builder.SEALED_V8_BUILDER_SHA256 in builder.SEALED_BUILDER_SHA256S
    assert builder.SEALED_V9_BUILDER_SHA256 in builder.SEALED_BUILDER_SHA256S
    assert builder.SEALED_V10_BUILDER_SHA256 in builder.SEALED_BUILDER_SHA256S
    assert builder.SEALED_V11_BUILDER_SHA256 in builder.SEALED_BUILDER_SHA256S
    assert builder.SEALED_V8_ANALYZER_SHA256 in builder.SEALED_ANALYZER_SHA256S
    assert builder.SEALED_V9_ANALYZER_SHA256 in builder.SEALED_ANALYZER_SHA256S
    assert builder.SEALED_V10_ANALYZER_SHA256 in builder.SEALED_ANALYZER_SHA256S
    assert builder.SEALED_V11_ANALYZER_SHA256 in builder.SEALED_ANALYZER_SHA256S
    assert (
        builder.SEALED_V8_CELL_REGISTRY_SHA256
        in builder.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        builder.SEALED_V9_CELL_REGISTRY_SHA256
        in builder.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        builder.SEALED_V10_CELL_REGISTRY_SHA256
        in builder.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        builder.SEALED_V11_CELL_REGISTRY_SHA256
        in builder.SEALED_CELL_REGISTRY_SHA256S
    )
    assert (
        builder.SEALED_V8_CHECKPOINT_INVENTORY_SHA256
        in builder.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        builder.SEALED_V9_CHECKPOINT_INVENTORY_SHA256
        in builder.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        builder.SEALED_V10_CHECKPOINT_INVENTORY_SHA256
        in builder.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        builder.SEALED_V11_CHECKPOINT_INVENTORY_SHA256
        in builder.SEALED_CHECKPOINT_INVENTORY_SHA256S
    )
    assert (
        builder.SEALED_V8_STAGE_MANIFEST_SHA256
        in builder.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        builder.SEALED_V9_STAGE_MANIFEST_SHA256
        in builder.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        builder.SEALED_V10_STAGE_MANIFEST_SHA256
        in builder.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert (
        builder.SEALED_V11_STAGE_MANIFEST_SHA256
        in builder.SEALED_STAGE_MANIFEST_SHA256S
    )
    assert builder.SEALED_V8_STAGE_TREE_SHA256 in builder.SEALED_STAGE_TREE_SHA256S
    assert builder.SEALED_V9_STAGE_TREE_SHA256 in builder.SEALED_STAGE_TREE_SHA256S
    assert builder.SEALED_V10_STAGE_TREE_SHA256 in builder.SEALED_STAGE_TREE_SHA256S
    assert builder.SEALED_V11_STAGE_TREE_SHA256 in builder.SEALED_STAGE_TREE_SHA256S
    assert (
        builder.SEALED_V8_STAGE_SOURCE_CONTRACT_SHA256
        in builder.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        builder.SEALED_V9_STAGE_SOURCE_CONTRACT_SHA256
        in builder.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        builder.SEALED_V10_STAGE_SOURCE_CONTRACT_SHA256
        in builder.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )
    assert (
        builder.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256
        in builder.SEALED_STAGE_SOURCE_CONTRACT_SHA256S
    )


def test_v12_registry_rejects_a_fresh_but_wrong_controller_identity():
    registry = _registry()
    registry["analyzer_source_sha256"] = builder.sha256_file(builder.ANALYZER_PATH)
    registry["controller_sha256"] = _sha("wrong-v12-controller")
    with pytest.raises(MainValidationBuildError, match="controller source SHA mismatch"):
        validate_builder_registry(registry)


def _write_documents(
    tmp_path,
    registry,
    *,
    stage_binding,
    controller_mutation=None,
    inventory_mutation=None,
):
    inventory_runs = []
    controller_runs = []
    for method in METHOD_IDS:
        for seed in SEED_IDS:
            run_id = f"{method}_seed{seed}"
            checkpoint = tmp_path / f"{run_id}.pt"
            training = tmp_path / f"{run_id}.metrics.jsonl"
            result = tmp_path / f"{run_id}.val.json"
            predictions = tmp_path / f"{run_id}.pred.jsonl"
            log = tmp_path / f"{run_id}.log"
            for path in (checkpoint, training, result, predictions, log):
                path.write_text(run_id, encoding="utf-8")
            selection_value = 0.29 if method == "B0" else 0.3
            selection_by_episode = {episode: selection_value for episode in EPISODE_IDS}
            support = {episode: {"forward": 1} for episode in EPISODE_IDS}
            selection_contract = {
                "schema_version": 1,
                "split": "val",
                "data_sha256": registry["validation_dataset"]["data_sha256"],
                "manifest_sha256": registry["validation_dataset"]["manifest_sha256"],
                "sample_count": 2848,
                "batch_size": 2 if method == "B0" else 1,
                "shuffle": False,
                "num_workers": 0,
                "ordered_record_validation": True,
                "metric": "validation_episode_macro_BCE@1",
                "mode": "min",
                "rule": "strict_improvement_earliest_epoch",
            }
            training_selection = {
                "batch_size": 2 if method == "B0" else 1,
                "selected_epoch": 0,
                "selected_value": selection_value,
                "by_episode": selection_by_episode,
                "support": support,
                "selection_detail_sha256": builder.canonical_json_sha256(
                    {
                        "value": selection_value,
                        "by_episode": selection_by_episode,
                        "support": support,
                    }
                ),
                "training_selection_contract": selection_contract,
                "training_selection_contract_sha256": builder.canonical_json_sha256(
                    selection_contract
                ),
            }
            inventory_runs.append(
                {
                    "run_id": run_id,
                    "experiment_id": method,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": _sha(f"ckpt:{run_id}"),
                    "model_state_sha256": _sha("B0-state")
                    if method == "B0"
                    else _sha(f"state:{run_id}"),
                    "training_metrics": str(training),
                    "training_metrics_sha256": _sha(f"train:{run_id}"),
                    "selected_epoch": 0,
                    "selected_value": selection_value,
                    "training_selection": training_selection,
                }
            )
            controller_runs.append(
                {
                    "run_id": run_id,
                    "experiment_id": method,
                    "model_family": method,
                    "seed": seed,
                    "checkpoint_sha256": _sha(f"ckpt:{run_id}"),
                    "evaluation_result": str(result),
                    "evaluation_result_sha256": _sha(f"eval:{run_id}"),
                    "predictions": str(predictions),
                    "predictions_sha256": _sha(f"pred:{run_id}"),
                    "prediction_count": 2848,
                    "bce_at1": 0.3,
                    "by_episode_bce_at1": {episode: 0.3 for episode in EPISODE_IDS},
                    "log": str(log),
                    "log_sha256": _sha(f"log:{run_id}"),
                    "formal_primary": True,
                    "comparison_batch_size": 1,
                    "evaluator_command_sha256": _sha(f"command:{run_id}"),
                    "training_selection": {
                        **training_selection,
                        "training_metrics": str(training),
                        "training_metrics_sha256": _sha(f"train:{run_id}"),
                    },
                    "formal_minus_selection": {
                        "value": 0.3 - selection_value,
                        "by_episode": {
                            episode: 0.3 - selection_value for episode in EPISODE_IDS
                        },
                    },
                    "selection_replay_receipt_id": (
                        "B0_shared_batch2_selection_replay"
                        if method == "B0"
                        else f"{run_id}_formal_same_contract_selection_replay"
                    ),
                }
            )
    inventory = {
        "schema_version": 2,
        "analysis_class": "main_v1_validation_checkpoint_inventory",
        "formal_version": builder.FORMAL_VERSION,
        "status": "preflight_passed",
        "internal_test_opened": False,
        "controller_sha256": registry["controller_sha256"],
        "dual_contract": builder._dual_contract(),
        "predecessor_failure_receipt": _predecessor_binding(),
        "hermetic_stage": {
            "source_project_root": str(stage_binding["source_project_root"]),
            "stage_root": str(stage_binding["stage_root"]),
            "manifest_path": str(stage_binding["manifest_path"]),
            "manifest_sha256": stage_binding["manifest_sha256"],
            "source_contract_sha256": stage_binding["source_contract_sha256"],
            "stage_tree_sha256": stage_binding["stage_tree_sha256"],
        },
        "runtime_isolation": _runtime_contract(),
        "static_contracts": registry["expected_controller_contracts"],
        "deterministic_baseline_audit": {
            "expected_model_state_sha256": registry["deterministic_baseline"][
                "model_state_sha256"
            ]
        },
        "runs": inventory_runs,
    }
    if inventory_mutation:
        inventory_mutation(inventory)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory, sort_keys=True), encoding="utf-8")
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    controller = {
        "schema_version": 2,
        "analysis_class": "main_v1_validation_evaluation",
        "formal_version": builder.FORMAL_VERSION,
        "status": "completed",
        "internal_test_opened": False,
        "hermetic_stage": inventory["hermetic_stage"],
        "runtime_isolation": _runtime_contract(),
        "controller": {"sha256": registry["controller_sha256"]},
        "checkpoint_inventory": {"sha256": inventory_sha},
        "dual_contract": builder._dual_contract(),
        "predecessor_failure_receipt": _predecessor_binding(),
        "selection_replay": {
            "path": str(tmp_path / "selection_replay_manifest.json"),
            "sha256": _sha("selection-replay-manifest"),
            "receipt_count": 7,
            "executed_replay_count": 1,
            "same_contract_receipt_count": 6,
        },
        "contracts": registry["expected_controller_contracts"],
        "environment": {
            "formal_runtime": {
                "isolated": True,
                "no_site": True,
                "dont_write_bytecode": True,
                "ignore_environment": True,
                "safe_path": True,
                "unbuffered_stdout": True,
                "prefix": "/opt/conda/envs/pytorch",
            }
        },
        "validation_dataset": {
            "data_sha256": registry["validation_dataset"]["data_sha256"],
            "manifest_sha256": registry["validation_dataset"]["manifest_sha256"],
            "sample_count": 2848,
            "episodes": list(EPISODE_IDS),
        },
        "runs": controller_runs,
    }
    if controller_mutation:
        controller_mutation(controller)
    controller_path = tmp_path / "controller.json"
    controller_path.write_text(json.dumps(controller, sort_keys=True), encoding="utf-8")
    controller_sha = hashlib.sha256(controller_path.read_bytes()).hexdigest()
    return controller_path, controller_sha, inventory_path, inventory_sha


def test_registry_and_frozen_manifests_bind_exact_matrix(tmp_path, monkeypatch):
    registry = validate_builder_registry(_registry())
    stage_binding = _stage_binding(tmp_path, registry)
    monkeypatch.setattr(builder, "_FORMAL_STAGE_BINDING", stage_binding)
    controller_path, controller_sha, inventory_path, inventory_sha = _write_documents(
        tmp_path, registry, stage_binding=stage_binding
    )
    controller_manifest = load_controller_manifest(
        controller_path, controller_sha, registry
    )
    inventory = load_checkpoint_inventory(
        inventory_path, inventory_sha, registry, controller_manifest
    )
    assert len(controller_manifest["runs"]) == 9
    assert len(inventory["runs"]) == 9


def test_manifest_rejects_internal_test_and_inventory_drift(tmp_path, monkeypatch):
    registry = validate_builder_registry(_registry())
    stage_binding = _stage_binding(tmp_path, registry)
    monkeypatch.setattr(builder, "_FORMAL_STAGE_BINDING", stage_binding)
    controller_path, controller_sha, _, _ = _write_documents(
        tmp_path,
        registry,
        stage_binding=stage_binding,
        controller_mutation=lambda value: value.update({"internal_test_opened": True}),
    )
    with pytest.raises(MainValidationBuildError, match="internal_test_opened mismatch"):
        load_controller_manifest(controller_path, controller_sha, registry)


def test_v12_predecessor_requires_exact_full_v11_failure_binding(
    tmp_path, monkeypatch
):
    registry = validate_builder_registry(_registry())
    stage_binding = _stage_binding(tmp_path, registry)
    monkeypatch.setattr(builder, "_FORMAL_STAGE_BINDING", stage_binding)

    def mutate_phase(value):
        value["predecessor_failure_receipt"]["phase"] = "preflight"

    def mutate_argv(value):
        value["predecessor_failure_receipt"]["argv"] = value[
            "predecessor_failure_receipt"
        ]["argv"][:-1]

    def mutate_controller_identity(value):
        value["predecessor_failure_receipt"]["controller_expected_sha256"] = _sha(
            "tampered-v11-controller"
        )

    for mutation in (mutate_phase, mutate_argv, mutate_controller_identity):
        controller_path, controller_sha, _, _ = _write_documents(
            tmp_path,
            registry,
            stage_binding=stage_binding,
            controller_mutation=mutation,
        )
        with pytest.raises(MainValidationBuildError, match="predecessor full binding"):
            load_controller_manifest(controller_path, controller_sha, registry)


def test_build_main_cells_uses_formal_primary_without_overwriting_b0(monkeypatch):
    registry = _registry()
    controller_manifest = {"runs": {}}
    inventory = {"runs": {}}
    evidence = {}
    replay_receipts = {}
    for method in METHOD_IDS:
        for seed in SEED_IDS:
            run_id = f"{method}_seed{seed}"
            selection = 0.30 + seed * 0.01
            formal = selection + (0.001 if method == "B0" else 0.0)
            receipt_id = (
                "B0_shared_batch2_selection_replay"
                if method == "B0"
                else f"{run_id}_formal_same_contract_selection_replay"
            )
            controller_manifest["runs"][run_id] = {
                "evaluation_results": {},
                "predictions": {},
                "log": {
                    "path": Path(f"/{run_id}.log"),
                    "sha256": _sha(f"log:{run_id}"),
                },
                "checkpoint_sha256": _sha(f"ckpt:{run_id}"),
                "bce_at1": formal,
                "by_episode_bce_at1": {episode: formal for episode in EPISODE_IDS},
                "evaluator_command_sha256": _sha(f"formal-command:{run_id}"),
                "selection_replay_receipt_id": receipt_id,
                "training_selection": {
                    "selected_value": selection,
                    "by_episode": {episode: selection for episode in EPISODE_IDS},
                    "training_selection_contract_sha256": _sha(
                        f"training-selection-contract:{run_id}"
                    ),
                },
            }
            inventory["runs"][run_id] = {
                "checkpoint_artifact": {},
                "training_artifact": {},
            }
            provenance = {
                field: _sha(field)
                for field in (
                    "source_tree_sha256",
                    "train_manifest_sha256",
                    "train_data_sha256",
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
                    "evaluator_source_sha256",
                    "metric_contract_sha256",
                )
            }
            provenance["evaluation_execution_contract_sha256"] = _sha(
                f"formal-exec:{method}"
            )
            evidence[run_id] = {
                "checkpoint_info": {
                    "sha256": _sha(f"ckpt:{run_id}"),
                    "model_state_sha256": (
                        _sha("B0-state") if method == "B0" else _sha(f"state:{run_id}")
                    ),
                    "checkpoint": {"epoch": 0},
                },
                "meta": {
                    "state_mode": registry["method_contracts"][method]["state_mode"],
                    "checkpoint_role": "best_validation",
                    "selection_verified": True,
                    "selected_epoch": 0,
                },
                "training": {
                    "sha256": _sha(f"training:{run_id}"),
                    "checkpoint_event_sha256": _sha(f"checkpoint-event:{run_id}"),
                    "run_end_event_sha256": _sha(f"run-end:{run_id}"),
                    "selection_detail_sha256": _sha(f"selection:{run_id}"),
                    "selected_value": selection,
                    "run_id": f"train-{run_id}",
                    "run_end": {
                        "status": "completed",
                        "alert_counts": {"error": 0},
                        "summary": {
                            "best_validation_BCE_at_1": selection,
                            "optimizer_updates": 6873,
                            "processed_samples": 13746,
                        },
                    },
                },
                "predictions": {"sha256": _sha(f"formal-pred:{run_id}")},
                "evaluation": {
                    "sha256": _sha(f"formal-result:{run_id}"),
                    "provenance": provenance,
                },
                "episode_metrics": {
                    episode: {
                        "bce_at1": formal,
                        "smooth_l1_forward": 0.1,
                        "smooth_l1_yaw": 0.1,
                        "turn_sign_accuracy": 0.9,
                        "transition_f1": 0.8,
                        "saturation_rate": 0.0,
                    }
                    for episode in EPISODE_IDS
                },
                "support_sha256": _sha("support"),
                "formal_value": formal,
                "formal_by_episode": {episode: formal for episode in EPISODE_IDS},
                "selection_value": selection,
                "selection_by_episode": {episode: selection for episode in EPISODE_IDS},
                "formal_minus_selection": {
                    "value": formal - selection,
                    "by_episode": {
                        episode: formal - selection for episode in EPISODE_IDS
                    },
                },
            }
            if receipt_id not in replay_receipts:
                replay_receipts[receipt_id] = {
                    "receipt_id": receipt_id,
                    "receipt_sha256": _sha(f"receipt:{receipt_id}"),
                    "receipt_kind": (
                        "executed_shared_model_state"
                        if method == "B0"
                        else "formal_artifact_same_contract"
                    ),
                    "selection_batch_size": 2 if method == "B0" else 1,
                    "artifacts": {
                        "predictions": {"sha256": _sha(f"replay-pred:{receipt_id}")},
                        "evaluation_result": {
                            "sha256": _sha(f"replay-result:{receipt_id}")
                        },
                        "log": {"sha256": _sha(f"replay-log:{receipt_id}")},
                    },
                    "evaluation_execution_contract_sha256": _sha(
                        f"replay-exec:{receipt_id}"
                    ),
                    "evaluator_command_sha256": _sha(f"replay-command:{receipt_id}"),
                }
    monkeypatch.setattr(
        builder, "_load_validation_dataset", lambda _: {"raw_records": []}
    )
    monkeypatch.setattr(
        builder,
        "load_selection_replay_manifest",
        lambda *_args, **_kwargs: {
            "sha256": _sha("selection-replay-manifest"),
            "receipts": replay_receipts,
        },
    )
    monkeypatch.setattr(
        builder,
        "_load_formal_run_evidence",
        lambda run, **_: evidence[run["run_name"]],
    )
    replay_calls = []
    monkeypatch.setattr(
        builder,
        "_verify_b0_shared_replay",
        lambda *args, **kwargs: replay_calls.append((args, kwargs)),
    )
    controller_manifest["selection_replay"] = {}
    controller_manifest["predecessor_failure_receipt"] = {
        "sha256": builder.PREDECESSOR_FAILURE_SHA256
    }
    monkeypatch.setattr(builder, "validate_cells", lambda rows, _: rows)
    rows = build_main_cells(registry, controller_manifest, inventory)
    assert len(rows) == 27
    assert all(row["analysis_class"] == ANALYSIS_CLASS for row in rows)
    assert len(replay_calls) == 1
    b0 = next(row for row in rows if row["method_id"] == "B0")
    assert b0["bce_at1"] == b0["formal_bce_at1"] == pytest.approx(0.301)
    assert b0["selection_bce_at1"] == pytest.approx(0.30)
    assert b0["formal_minus_selection_bce_at1"] == pytest.approx(0.001)
    assert b0["selection_replay_kind"] == "executed_shared_model_state"
    same_contract = next(row for row in rows if row["method_id"] == "B1")
    assert same_contract["formal_minus_selection_bce_at1"] == 0.0


def test_builder_outputs_cannot_overlap_each_other_or_inputs(tmp_path):
    source = tmp_path / "controller.json"
    with pytest.raises(MainValidationBuildError, match="must be distinct"):
        validate_distinct_paths(
            input_paths=[source], output_paths=[tmp_path / "same", tmp_path / "same"]
        )
    with pytest.raises(MainValidationBuildError, match="overlaps"):
        validate_distinct_paths(
            input_paths=[source], output_paths=[source, tmp_path / "receipt.json"]
        )


def test_selection_replay_artifact_byte_tamper_fails_closed(tmp_path):
    result = tmp_path / "result.json"
    predictions = tmp_path / "predictions.jsonl"
    log = tmp_path / "run.log"
    result.write_text("{}\n", encoding="utf-8")
    predictions.write_text("{}\n", encoding="utf-8")
    log.write_text("ok\n", encoding="utf-8")
    artifacts = {
        "evaluation_result": {
            "path": str(result),
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        },
        "predictions": {
            "path": str(predictions),
            "sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
            "count": 2848,
        },
        "log": {
            "path": str(log),
            "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        },
    }
    normalized = builder._normalize_receipt_artifacts(artifacts, receipt_id="receipt")
    assert normalized["predictions"]["count"] == 2848
    result.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(MainValidationBuildError, match="physical SHA-256"):
        builder._normalize_receipt_artifacts(artifacts, receipt_id="receipt")
