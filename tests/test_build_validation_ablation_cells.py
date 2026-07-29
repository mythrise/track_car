import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.analyze_validation_ablations import (
    ANALYSIS_CLASS,
    VALIDATION_EPISODE_IDS,
)
from scripts.build_validation_ablation_cells import (
    RUN_MANIFEST_ANALYSIS_CLASS,
    ValidationCellBuildError,
    deterministic_model_state_sha256,
    main,
)
from scripts.eval_offline import (
    PROJECT_ROOT,
    build_evaluator_source,
    build_fairness_contract,
    build_metric_contract,
    build_method_contract,
    canonical_json_sha256,
    evaluate_predictions,
)


METHODS = (
    "H0",
    "H0-noTIM",
    "H0-noFuture",
    "H0-noVerifier",
    "H0-noEventBank",
    "H0-noOrchestrator",
    "H0-S",
    "H0-noPolar",
    "H0-noTurnBalance",
)


def _sha(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prediction_records(error):
    rows = []
    for episode_index, episode in enumerate(VALIDATION_EPISODE_IDS):
        for frame, (command, target) in enumerate(
            (
                ("forward", [1.0, 0.0, 0.0]),
                ("turn_left", [0.5, 0.0, 1.0]),
            )
        ):
            prediction = [target[0] + error + 0.001 * episode_index, 0.0, target[2]]
            rows.append(
                {
                    "step_actions": [target],
                    "pred_step_actions": [prediction],
                    "prev_action": [0.0, 0.0, 0.0] if frame == 0 else [1.0, 0.0, 0.0],
                    "valid_mask": [True],
                    "transition_type": "steady_forward" if frame == 0 else "turn_onset",
                    "episode": episode,
                    "source_raw_dir": episode,
                    "sequence_id": f"{episode}-sequence",
                    "chunk_id": f"{episode}-sequence",
                    "clip_id": f"{episode}-clip",
                    "frame_idx": frame,
                    "mirrored": False,
                    "command": command,
                }
            )
    return rows


def _checkpoint_meta(
    method,
    seed,
    selected_value,
    selection_detail,
    *,
    validation_data_sha=None,
):
    state_mode = "stateless" if method == "H0-S" else "rolling"
    disabled = [] if method == "H0" else [method.removeprefix("H0-")]
    return {
        "schema_version": 1,
        "model_family": "pfem_harness",
        "experiment_id": method,
        "seed": seed,
        "history": 1,
        "n_waypoints": 1,
        "dt": 0.1,
        "label_mode": "absolute",
        "action_semantics": "arc_turn_v2",
        "data_manifest_hash": _sha("train-manifest"),
        "data_jsonl_sha256": _sha("train-data"),
        "sample_count": 6,
        "base_model_sha256": _sha("base"),
        "base_model_artifact": {
            "schema_version": 1,
            "format": "huggingface_pretrained",
            "weight_layout": "safetensors_single",
            "files": [
                {
                    "path": "config.json",
                    "role": "config",
                    "size": 1,
                    "sha256": _sha("config"),
                }
            ],
            "artifact_sha256": _sha("base"),
        },
        "qwen_model_sha256": _sha("qwen"),
        "vision_cache_manifest_sha256": _sha("cache-manifest"),
        "vision_cache_provenance_sha256": _sha("cache-provenance"),
        "vision_cache_token_payload_sha256": _sha("cache-payload"),
        "dino_model_sha256": _sha("dino"),
        "siglip_model_sha256": _sha("siglip"),
        "training_source_raw_dirs": ["train0"],
        "state_mode": state_mode,
        "checkpoint_selection": {
            "metric": "validation_episode_macro_BCE@1",
            "mode": "min",
            "rule": "strict_improvement_earliest_epoch",
        },
        "checkpoint_role": "best_validation",
        "selection_verified": True,
        "selected_epoch": 0,
        "selected_value": selected_value,
        "optimizer_updates": 2,
        "processed_samples": 6,
        "sampling_policy": "ordered_jsonl",
        "batch_size": 1,
        "grad_accum_steps": 1,
        "effective_batch_size": 1,
        "base_lr": 2e-5,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "validation": {
            "data_manifest_hash": _sha("val-manifest"),
            "data_jsonl_sha256": validation_data_sha or _sha("val-data"),
            "sample_count": 6,
        },
        "best_validation": {
            "selection_bce_at1": selected_value,
            "family_loss": 0.1,
            "selection_detail": selection_detail,
        },
        "disabled_components": disabled,
    }


def _contracts(meta, checkpoint_sha, label_mode="absolute"):
    evaluator_source = build_evaluator_source(PROJECT_ROOT)
    metric_contract = build_metric_contract(0.2)
    execution = {
        "schema_version": 1,
        "evaluation_data": {
            "split": "val",
            "manifest_sha256": meta["validation"]["data_manifest_hash"],
            "data_sha256": meta["validation"]["data_jsonl_sha256"],
            "sample_count": 6,
        },
        "observation": {"history": 1, "n_waypoints": 1, "dt": 0.1},
        "state": {
            "declared_mode": meta["state_mode"],
            "effective_mode": meta["state_mode"],
            "override": False,
            "sequence_id_required": meta["state_mode"] == "rolling",
            "reset_policy": (
                "clean_sequence_boundary_or_frame_gap"
                if meta["state_mode"] == "rolling"
                else "functionally_stateless"
            ),
        },
        "label": {
            "declared_mode": "absolute",
            "effective_mode": label_mode,
            "override": False,
        },
        "loader": {
            "batch_size": 1,
            "shuffle": False,
            "num_workers": 0,
            "ordered_record_validation": True,
        },
        "runtime": {
            "device": "cpu",
            "device_type": "cpu",
            "parameter_dtypes": ["torch.float32"],
            "buffer_dtypes": ["torch.float32"],
            "torch_default_dtype": "torch.float32",
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
        "evaluation_identity": {"tier": "locked_final", "class": "validation"},
    }
    return evaluator_source, metric_contract, execution


def _build_fixture(
    tmp_path,
    *,
    eval_mutator=None,
    validation_records_mutator=None,
    prediction_records_mutator=None,
    legacy_paths=False,
):
    validation_records = _prediction_records(0.0)
    if validation_records_mutator is not None:
        validation_records_mutator(validation_records)
    for record in validation_records:
        record.pop("pred_step_actions")
        record.pop("valid_mask")
    validation_path = tmp_path / "validation.jsonl"
    validation_path.write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in validation_records
        ),
        encoding="utf-8",
    )
    validation_data_sha = _file_sha(validation_path)
    registry_payload = {
        "schema_version": 1,
        "analysis_class": ANALYSIS_CLASS,
        "family_id": "validation-builder-test",
        "family_size": 8,
        "seed_ids": [0, 1, 2],
        "episode_ids": list(VALIDATION_EPISODE_IDS),
        "primary_metric": "bce_at1",
        "alpha": 0.05,
        "bootstrap_iterations": 200_000,
        "analysis_seed": 7,
        "expected_optimizer_updates": 2,
        "expected_processed_samples": 6,
        "full_method_id": "H0",
        "contrasts": [
            {
                "contrast_id": f"H0_vs_{method}",
                "candidate_id": "H0",
                "reference_id": method,
            }
            for method in METHODS[1:]
        ],
        "method_contracts": {},
        "guardrails": {
            "smooth_l1_forward": {"direction": "lower", "harm_margin": 0.1},
            "smooth_l1_yaw": {"direction": "lower", "harm_margin": 0.1},
            "turn_sign_accuracy": {"direction": "higher", "harm_margin": 0.1},
            "transition_f1": {"direction": "higher", "harm_margin": 0.1},
            "saturation_rate": {"direction": "lower", "harm_margin": 0.1},
        },
    }
    run_specs = []
    pending = []
    for method_index, method in enumerate(METHODS):
        example_meta = _checkpoint_meta(
            method,
            0,
            0.0,
            {"value": 0.0, "by_episode": {}, "support": {}},
            validation_data_sha=validation_data_sha,
        )
        treatment_sha = canonical_json_sha256(build_method_contract(example_meta))
        registry_payload["method_contracts"][method] = {
            "treatment_name": "full" if method == "H0" else method,
            "treatment_config_sha256": treatment_sha,
            "state_mode": example_meta["state_mode"],
        }
        for seed in range(3):
            run_name = f"{method}_seed{seed}"
            run_dir = tmp_path / run_name
            run_dir.mkdir()
            records = _prediction_records(0.01 * method_index + 0.001 * seed)
            if prediction_records_mutator is not None:
                prediction_records_mutator(records, method, seed)
            predictions = np.asarray([row["pred_step_actions"] for row in records])
            metrics = evaluate_predictions(predictions, records)
            selected_value = metrics["balanced_control_error_at1"]["value"]
            selection_detail = {
                "value": selected_value,
                "by_episode": metrics["balanced_control_error_at1"]["by_episode"],
                "support": metrics["balanced_control_error_at1"]["support"],
            }
            meta = _checkpoint_meta(
                method,
                seed,
                selected_value,
                selection_detail,
                validation_data_sha=validation_data_sha,
            )
            checkpoint_path = run_dir / "best.pt"
            torch.save(
                {
                    "epoch": 0,
                    "loss": selected_value,
                    "model_state": {
                        "weight": torch.tensor([float(method_index), float(seed)])
                    },
                    "meta": meta,
                },
                checkpoint_path,
            )
            checkpoint_sha = _file_sha(checkpoint_path)
            predictions_path = run_dir / "predictions.jsonl"
            predictions_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
                encoding="utf-8",
            )
            logged_checkpoint_path = (
                f"/old-machine/archive/{run_name}/best.pt"
                if legacy_paths
                else str(checkpoint_path.resolve())
            )
            logged_predictions_path = (
                f"/old-machine/archive/{run_name}/predictions.jsonl"
                if legacy_paths
                else str(predictions_path.resolve())
            )
            evaluator_source, metric_contract, execution = _contracts(
                meta, checkpoint_sha
            )
            fairness = build_fairness_contract(meta)
            method_contract = build_method_contract(meta)
            provenance = {
                "state_mode": meta["state_mode"],
                "state_mode_override": False,
                "checkpoint_experiment_id": method,
                "effective_experiment_id": method,
                "evaluation_tier": "locked_final",
                "evaluation_class": "validation",
                "headline_eligible": False,
                "test_manifest_sha256": None,
                "train_manifest_sha256": meta["data_manifest_hash"],
                "train_data_sha256": meta["data_jsonl_sha256"],
                "validation_manifest_sha256": meta["validation"]["data_manifest_hash"],
                "validation_data_sha256": meta["validation"]["data_jsonl_sha256"],
                "base_model_sha256": meta["base_model_sha256"],
                "qwen_model_sha256": meta["qwen_model_sha256"],
                "vision_cache_manifest_sha256": meta["vision_cache_manifest_sha256"],
                "vision_cache_provenance_sha256": meta["vision_cache_provenance_sha256"],
                "vision_cache_token_payload_sha256": meta["vision_cache_token_payload_sha256"],
                "dino_model_sha256": meta["dino_model_sha256"],
                "siglip_model_sha256": meta["siglip_model_sha256"],
                "source_tree_sha256": _sha("source"),
                "experiment_registry_sha256": _sha("parent-registry"),
                "fairness_contract": fairness,
                "fairness_contract_sha256": canonical_json_sha256(fairness),
                "method_contract": method_contract,
                "method_contract_sha256": canonical_json_sha256(method_contract),
                "validation_selection_detail_sha256": canonical_json_sha256(selection_detail),
                "checkpoint_role": "best_validation",
                "selection_verified": True,
                "selected_epoch": 0,
                "selected_value": selected_value,
                "checkpoint_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
                "evaluation_predictions_sha256": _file_sha(predictions_path),
                "evaluator_source": evaluator_source,
                "evaluator_source_sha256": canonical_json_sha256(evaluator_source),
                "metric_contract": metric_contract,
                "metric_contract_sha256": canonical_json_sha256(metric_contract),
                "evaluation_execution_contract": execution,
                "evaluation_execution_contract_sha256": canonical_json_sha256(execution),
            }
            metrics.update(
                {
                    "model_family": "pfem_harness",
                    "state_mode": meta["state_mode"],
                    "experiment_id": method,
                    "checkpoint_experiment_id": method,
                    "evaluation_class": "validation",
                    "headline_eligible": False,
                    "seed": seed,
                }
            )
            eval_payload = {
                run_name: {
                    "checkpoint": logged_checkpoint_path,
                    "predictions": logged_predictions_path,
                    "label_mode": "absolute",
                    "metrics": metrics,
                    "provenance": provenance,
                }
            }
            if eval_mutator is not None:
                eval_mutator(eval_payload, method, seed)
            eval_path = run_dir / "eval.json"
            _write_json(eval_path, eval_payload)
            config = {"method": method, "seed": seed}
            run_id = f"run-{method}-{seed}"
            best_event = {
                "phase": "checkpoint",
                "role": "best_validation",
                "path": logged_checkpoint_path,
                "sha256": checkpoint_sha,
                "epoch": 0,
                "optimizer_updates": 2,
                "selected_value": selected_value,
                "run_id": run_id,
                "sequence": 2,
            }
            run_end = {
                "phase": "run_end",
                "status": "completed",
                "summary": {
                    "final_epoch": 0,
                    "optimizer_updates": 2,
                    "processed_samples": 6,
                    "best_validation_BCE_at_1": selected_value,
                },
                "alert_counts": {"error": 0, "warning": 0, "info": 0},
                "checkpoints": [
                    {
                        "role": "best_validation",
                        "path": logged_checkpoint_path,
                        "sha256": checkpoint_sha,
                        "epoch": 0,
                        "optimizer_updates": 2,
                        "selected_value": selected_value,
                    }
                ],
                "run_id": run_id,
                "sequence": 3,
            }
            log_records = [
                {
                    "phase": "run_start",
                    "checkpoint_meta": copy.deepcopy(meta),
                    "config": config,
                    "config_sha256": canonical_json_sha256(config),
                    "provenance": {
                        field: copy.deepcopy(meta[field])
                        for field in (
                            "data_manifest_hash",
                            "data_jsonl_sha256",
                            "base_model_sha256",
                            "qwen_model_sha256",
                            "vision_cache_manifest_sha256",
                            "vision_cache_provenance_sha256",
                            "vision_cache_token_payload_sha256",
                            "dino_model_sha256",
                            "siglip_model_sha256",
                            "validation",
                        )
                    },
                    "run_id": run_id,
                    "sequence": 0,
                },
                {
                    "phase": "validation",
                    "epoch": 0,
                    "BCE_at_1": selected_value,
                    "family_loss": 0.1,
                    "selection_detail": selection_detail,
                    "run_id": run_id,
                    "sequence": 1,
                },
                best_event,
                run_end,
            ]
            metrics_path = run_dir / "metrics.jsonl"
            metrics_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in log_records),
                encoding="utf-8",
            )
            pending.append(
                {
                    "run_name": run_name,
                    "method_id": method,
                    "seed": seed,
                    "checkpoint_experiment_id": method,
                    "checkpoint": checkpoint_path,
                    "training_metrics": metrics_path,
                    "evaluation_results": eval_path,
                    "predictions": predictions_path,
                }
            )
    registry_path = tmp_path / "contrast_registry.json"
    _write_json(registry_path, registry_payload)
    for item in pending:
        run_specs.append(
            {
                "run_name": item["run_name"],
                "method_id": item["method_id"],
                "seed": item["seed"],
                "checkpoint_experiment_id": item["checkpoint_experiment_id"],
                **{
                    field: {
                        "path": str(item[field].relative_to(tmp_path)),
                        "sha256": _file_sha(item[field]),
                    }
                    for field in (
                        "checkpoint",
                        "training_metrics",
                        "evaluation_results",
                        "predictions",
                    )
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "analysis_class": RUN_MANIFEST_ANALYSIS_CLASS,
        "status": "frozen_after_validation_evaluation",
        "parent_main_registry_sha256": _sha("parent-registry"),
        "expected_shared_provenance": {
            "source_tree_sha256": _sha("source"),
            "evaluation_registry_sha256": _sha("parent-registry"),
            "fairness_contract_sha256": canonical_json_sha256(
                build_fairness_contract(
                    _checkpoint_meta(
                        "H0",
                        0,
                        0.0,
                        {"value": 0.0, "by_episode": {}, "support": {}},
                        validation_data_sha=validation_data_sha,
                    )
                )
            ),
            "evaluator_source_sha256": canonical_json_sha256(
                build_evaluator_source(PROJECT_ROOT)
            ),
            "metric_contract_sha256": canonical_json_sha256(
                build_metric_contract(0.2)
            ),
        },
        "validation_dataset": {
            "path": str(validation_path.relative_to(tmp_path)),
            "sha256": validation_data_sha,
        },
        "runs": run_specs,
    }
    manifest_path = tmp_path / "run_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, registry_path, pending


def _args(tmp_path, manifest_path, registry_path):
    return [
        "--run_manifest",
        str(manifest_path),
        "--expected_run_manifest_sha256",
        _file_sha(manifest_path),
        "--contrast_registry",
        str(registry_path),
        "--expected_contrast_registry_sha256",
        _file_sha(registry_path),
        "--output",
        str(tmp_path / "cells.jsonl"),
    ]


def _sync_run_artifact_hashes(manifest_path, run_name, *artifact_names):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    run = next(item for item in manifest["runs"] if item["run_name"] == run_name)
    for artifact_name in artifact_names:
        artifact_path = Path(manifest_path).parent / run[artifact_name]["path"]
        run[artifact_name]["sha256"] = _file_sha(artifact_path)
    _write_json(manifest_path, manifest)


def test_end_to_end_builds_strict_81_cells(tmp_path):
    manifest_path, registry_path, _runs = _build_fixture(tmp_path)
    result = main(_args(tmp_path, manifest_path, registry_path))
    assert result["cell_count"] == 81
    rows = [
        json.loads(line)
        for line in (tmp_path / "cells.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 81
    assert all(row["split"] == "val" for row in rows)
    assert all(row["validation_only"] is True for row in rows)
    assert all("evaluator_source_sha256" in row for row in rows)
    assert all("evaluation_result_sha256" in row for row in rows)
    assert all(row["transition_f1"] == pytest.approx(1.0) for row in rows)
    assert all(row["transition_f1_defined"] is True for row in rows)
    assert all(row["transition_f1_excluded"] is False for row in rows)
    assert all(row["transition_tp"] == 1 for row in rows)
    assert all(row["transition_fp"] == 0 for row in rows)
    assert all(row["transition_fn"] == 0 for row in rows)
    assert all(row["transition_zero_fire_collapse"] is False for row in rows)


def test_end_to_end_preserves_supported_zero_f1_cell(tmp_path):
    target_episode = VALIDATION_EPISODE_IDS[1]

    def miss_supported_event(records, method, seed):
        if method == "H0" and seed == 0:
            for record in records:
                if record["episode"] == target_episode:
                    record["pred_step_actions"][0][2] = 0.0

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path,
        prediction_records_mutator=miss_supported_event,
    )
    main(_args(tmp_path, manifest_path, registry_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "cells.jsonl").read_text().splitlines()
    ]
    row = next(
        item
        for item in rows
        if item["method_id"] == "H0"
        and item["seed"] == 0
        and item["episode"] == target_episode
    )
    assert row["transition_f1"] == 0.0
    assert row["transition_f1_defined"] is True
    assert row["transition_f1_excluded"] is False
    assert row["transition_tp"] == 0
    assert row["transition_fp"] == 0
    assert row["transition_fn"] == 1
    assert row["transition_zero_fire_collapse"] is True


def test_end_to_end_preserves_empty_event_union_as_excluded_null(tmp_path):
    target_episode = VALIDATION_EPISODE_IDS[1]

    def make_truth_transition_free(records):
        for record in records:
            if record["episode"] == target_episode:
                record["step_actions"][0][2] = 1.0
                record["prev_action"][2] = 1.0

    def make_predictions_transition_free(records, _method, _seed):
        make_truth_transition_free(records)
        for record in records:
            if record["episode"] == target_episode:
                record["pred_step_actions"][0][2] = 1.0

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path,
        validation_records_mutator=make_truth_transition_free,
        prediction_records_mutator=make_predictions_transition_free,
    )
    main(_args(tmp_path, manifest_path, registry_path))
    rows = [
        json.loads(line)
        for line in (tmp_path / "cells.jsonl").read_text().splitlines()
    ]
    excluded = [row for row in rows if row["episode"] == target_episode]
    assert len(excluded) == len(METHODS) * 3
    assert all(row["transition_f1"] is None for row in excluded)
    assert all(row["transition_f1_defined"] is False for row in excluded)
    assert all(row["transition_f1_excluded"] is True for row in excluded)
    assert all(row["transition_tp"] == 0 for row in excluded)
    assert all(row["transition_fp"] == 0 for row in excluded)
    assert all(row["transition_fn"] == 0 for row in excluded)
    assert all(row["transition_zero_fire_collapse"] is False for row in excluded)


def test_tampered_artifact_is_rejected_before_cell_generation(tmp_path):
    manifest_path, registry_path, runs = _build_fixture(tmp_path)
    runs[0]["predictions"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationCellBuildError, match="predictions SHA-256 mismatch"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_validation_builder_rejects_test_provenance(tmp_path):
    def mutate(payload, method, seed):
        if method == "H0" and seed == 0:
            payload[f"{method}_seed{seed}"]["provenance"]["test_manifest_sha256"] = _sha("test")

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path, eval_mutator=mutate
    )
    with pytest.raises(ValidationCellBuildError, match="validation-only payloads"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_execution_contract_cannot_self_hash_an_invalid_split(tmp_path):
    def mutate(payload, method, seed):
        if method == "H0" and seed == 0:
            provenance = payload[f"{method}_seed{seed}"]["provenance"]
            provenance["evaluation_execution_contract"]["evaluation_data"][
                "split"
            ] = "test"
            provenance["evaluation_execution_contract_sha256"] = canonical_json_sha256(
                provenance["evaluation_execution_contract"]
            )

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path, eval_mutator=mutate
    )
    with pytest.raises(ValidationCellBuildError, match="execution evaluation split"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_guardrail_metrics_are_recomputed_from_predictions(tmp_path):
    def mutate(payload, method, seed):
        if method == "H0" and seed == 0:
            payload[f"{method}_seed{seed}"]["metrics"]["by_episode"][
                VALIDATION_EPISODE_IDS[0]
            ]["smooth_l1"]["forward"] += 0.5

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path, eval_mutator=mutate
    )
    with pytest.raises(ValidationCellBuildError, match="smooth_l1.forward"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_predictions_ground_truth_cannot_replace_frozen_validation_truth(tmp_path):
    manifest_path, registry_path, runs = _build_fixture(tmp_path)
    run = runs[0]
    records = [
        json.loads(line)
        for line in run["predictions"].read_text(encoding="utf-8").splitlines()
    ]
    records[0]["step_actions"][0][0] += 0.25
    run["predictions"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    evaluation = json.loads(run["evaluation_results"].read_text(encoding="utf-8"))
    evaluation[run["run_name"]]["provenance"][
        "evaluation_predictions_sha256"
    ] = _file_sha(run["predictions"])
    _write_json(run["evaluation_results"], evaluation)
    _sync_run_artifact_hashes(
        manifest_path,
        run["run_name"],
        "predictions",
        "evaluation_results",
    )

    with pytest.raises(
        ValidationCellBuildError,
        match="step_actions mismatch against frozen validation dataset",
    ):
        main(_args(tmp_path, manifest_path, registry_path))


def test_prediction_identity_must_align_with_validation_dataset_in_order(tmp_path):
    manifest_path, registry_path, runs = _build_fixture(tmp_path)
    run = runs[0]
    records = [
        json.loads(line)
        for line in run["predictions"].read_text(encoding="utf-8").splitlines()
    ]
    records[0]["command"] = "forged-command"
    run["predictions"].write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    evaluation = json.loads(run["evaluation_results"].read_text(encoding="utf-8"))
    evaluation[run["run_name"]]["provenance"][
        "evaluation_predictions_sha256"
    ] = _file_sha(run["predictions"])
    _write_json(run["evaluation_results"], evaluation)
    _sync_run_artifact_hashes(
        manifest_path,
        run["run_name"],
        "predictions",
        "evaluation_results",
    )

    with pytest.raises(ValidationCellBuildError, match="identity.command"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_evaluation_predictions_sha_must_match_manifest_artifact(tmp_path):
    def mutate(payload, method, seed):
        if method == "H0" and seed == 0:
            payload[f"{method}_seed{seed}"]["provenance"][
                "evaluation_predictions_sha256"
            ] = _sha("wrong-predictions")

    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path, eval_mutator=mutate
    )
    with pytest.raises(ValidationCellBuildError, match="evaluation predictions SHA"):
        main(_args(tmp_path, manifest_path, registry_path))


def test_builder_accepts_moved_artifacts_with_historical_absolute_paths(tmp_path):
    manifest_path, registry_path, _runs = _build_fixture(
        tmp_path, legacy_paths=True
    )
    result = main(_args(tmp_path, manifest_path, registry_path))
    assert result["cell_count"] == 81


def test_model_state_hash_is_order_independent_and_value_sensitive():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    changed = {"a": torch.tensor([1.0]), "b": torch.tensor([3.0])}
    assert deterministic_model_state_sha256(first) == deterministic_model_state_sha256(second)
    assert deterministic_model_state_sha256(first) != deterministic_model_state_sha256(changed)
