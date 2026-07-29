import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.build_headline_cells import (
    BUILDER_ANALYSIS_CLASS,
    HeadlineBuildError,
    RUN_MANIFEST_ANALYSIS_CLASS,
    TRUST_ROOTS_ANALYSIS_CLASS,
    main,
)
from scripts.eval_offline import (
    PROJECT_ROOT,
    build_evaluator_source,
    build_fairness_contract,
    build_method_contract,
    build_metric_contract,
    canonical_json_sha256,
    evaluate_predictions,
    source_tree_sha256,
)


EPISODES = ("test004", "test0006", "test010", "test017", "test020")


def _sha(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows):
    Path(path).write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _test_records():
    rows = []
    for episode_index, episode in enumerate(EPISODES):
        for frame_idx, (command, target) in enumerate(
            (
                ("forward", [0.5, 0.0, 0.0]),
                ("turn_left", [0.5, 0.0, 0.4]),
            )
        ):
            row = {
                    "episode": f"clean_{episode}",
                    "source_raw_dir": episode,
                    "sequence_id": f"{episode}_sequence",
                    "chunk_id": f"{episode}_sequence",
                    "clip_id": f"{episode}_clip",
                    "frame_idx": frame_idx,
                    "mirrored": False,
                    "command": command,
                    "transition_type": "steady_forward" if frame_idx == 0 else "turn_onset",
                    "step_actions": [target],
                    "prev_action": [0.0, 0.0, 0.0] if frame_idx == 0 else [0.5, 0.0, 0.0],
                    "fixture_marker": episode_index,
                }
            # Exercise both dataset fallbacks used by real collected JSONL:
            # valid_idx-derived masks and the implicit all-valid default.
            if episode_index % 2 == 0:
                row["valid_idx"] = [0]
            rows.append(row)
    return rows


def _checkpoint_meta(selection_detail):
    selected_value = float(selection_detail["value"])
    return {
        "schema_version": 1,
        "model_family": "pfem_harness",
        "experiment_id": "H0",
        "seed": 0,
        "history": 1,
        "n_waypoints": 1,
        "dt": 0.1,
        "label_mode": "absolute",
        "action_semantics": "arc_turn_v2",
        "data_manifest_hash": _sha("train-manifest"),
        "data_jsonl_sha256": _sha("train-jsonl"),
        "sample_count": 4,
        "base_model_sha256": _sha("base"),
        "base_model_artifact": {
            "artifact_sha256": _sha("base"),
            "files": [{"path": "weights", "sha256": _sha("weights")}],
        },
        "qwen_model_sha256": _sha("qwen"),
        "vision_cache_manifest_sha256": _sha("cache-manifest"),
        "vision_cache_provenance_sha256": _sha("cache-provenance"),
        "vision_cache_token_payload_sha256": _sha("cache-payload"),
        "dino_model_sha256": _sha("dino"),
        "siglip_model_sha256": _sha("siglip"),
        "training_source_raw_dirs": ["train0"],
        "state_mode": "rolling",
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
        "processed_samples": 4,
        "sampling_policy": "ordered_jsonl",
        "batch_size": 1,
        "grad_accum_steps": 2,
        "effective_batch_size": 2,
        "base_lr": 2e-5,
        "head_lr": 3e-4,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "validation": {
            "data_manifest_hash": _sha("val-manifest"),
            "data_jsonl_sha256": _sha("val-jsonl"),
            "sample_count": 3,
        },
        "best_validation": {
            "selection_bce_at1": selected_value,
            "selection_detail": selection_detail,
        },
    }


def _refresh_manifest(fixture):
    manifest = json.loads(fixture["manifest"].read_text(encoding="utf-8"))
    manifest["test_dataset"]["jsonl"]["sha256"] = _file_sha(fixture["test_jsonl"])
    manifest["test_dataset"]["manifest"]["sha256"] = _file_sha(
        fixture["test_manifest"]
    )
    manifest["registry"]["sha256"] = _file_sha(fixture["registry"])
    manifest["trust_roots"]["sha256"] = _file_sha(fixture["trust_roots"])
    for field in (
        "checkpoint",
        "training_metrics",
        "evaluation_results",
        "predictions",
    ):
        manifest["runs"][0][field]["sha256"] = _file_sha(fixture[field])
    _write_json(fixture["manifest"], manifest)
    return _file_sha(fixture["manifest"])


def _args(fixture, expected_manifest_sha=None, output=None):
    return [
        "--run_manifest",
        str(fixture["manifest"]),
        "--expected_run_manifest_sha256",
        expected_manifest_sha or _file_sha(fixture["manifest"]),
        "--expected_trust_roots_sha256",
        _file_sha(fixture["trust_roots"]),
        "--output",
        str(output or fixture["output"]),
    ]


def _build_fixture(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    records = _test_records()
    test_jsonl = root / "test.jsonl"
    _write_jsonl(test_jsonl, records)
    test_manifest = root / "test.jsonl.manifest.json"
    _write_json(
        test_manifest,
        {
            "schema_version": 1,
            "split": "test",
            "data_jsonl_sha256": _file_sha(test_jsonl),
            "sample_count": len(records),
            "history": 1,
            "n_waypoints": 1,
            "dt": 0.1,
        },
    )
    predictions_rows = []
    for index, row in enumerate(records):
        prediction = copy.deepcopy(row["step_actions"])
        prediction[0][0] += 0.02 + index * 0.0001
        predictions_rows.append(
            {
                **copy.deepcopy(row),
                "valid_mask": [True],
                "pred_step_actions": prediction,
            }
        )
    predictions = root / "predictions.jsonl"
    _write_jsonl(predictions, predictions_rows)
    prediction_array = np.asarray(
        [row["pred_step_actions"] for row in predictions_rows], dtype=np.float64
    )
    metrics = evaluate_predictions(prediction_array, records, threshold=0.2)
    metrics.update(
        {
            "model_family": "pfem_harness",
            "state_mode": "rolling",
            "experiment_id": "H0",
            "checkpoint_experiment_id": "H0",
            "evaluation_class": "headline",
            "headline_eligible": True,
            "seed": 0,
        }
    )
    selection_detail = {
        "value": 0.25,
        "by_episode": {episode: 0.25 for episode in ("val0",)},
        "support": {"val0": {"forward": 1}},
    }
    meta = _checkpoint_meta(selection_detail)
    registry = root / "experiment_registry.json"
    _write_json(
        registry,
        {
            "schema_version": 1,
            "status": "frozen_synthetic_headline_fixture",
            "history": 1,
            "prediction_horizon": 1,
            "dt": 0.1,
            "sampling_policy": "ordered_jsonl",
            "max_optimizer_updates": 2,
            "processed_samples_per_run": 4,
            "weight_decay": 1e-4,
            "grad_clip": 1.0,
            "checkpoint_selection": copy.deepcopy(meta["checkpoint_selection"]),
            "data": {
                "train_manifest_sha256": meta["data_manifest_hash"],
                "train_data_sha256": meta["data_jsonl_sha256"],
                "train_count": meta["sample_count"],
                "val_manifest_sha256": meta["validation"]["data_manifest_hash"],
                "val_data_sha256": meta["validation"]["data_jsonl_sha256"],
                "val_count": meta["validation"]["sample_count"],
            },
            "artifacts": {
                "base_model_sha256": meta["base_model_sha256"],
                "qwen_model_sha256": meta["qwen_model_sha256"],
                "vision_cache_manifest_sha256": meta[
                    "vision_cache_manifest_sha256"
                ],
                "vision_cache_provenance_sha256": meta[
                    "vision_cache_provenance_sha256"
                ],
                "vision_cache_token_payload_sha256": meta[
                    "vision_cache_token_payload_sha256"
                ],
                "dinov3_sha256": meta["dino_model_sha256"],
                "siglip_sha256": meta["siglip_model_sha256"],
            },
            "models": {
                "H0": {
                    field: copy.deepcopy(meta[field])
                    for field in (
                        "model_family",
                        "state_mode",
                        "batch_size",
                        "grad_accum_steps",
                        "effective_batch_size",
                        "base_lr",
                        "head_lr",
                    )
                }
            },
        },
    )
    checkpoint = root / "best.pt"
    torch.save(
        {
            "epoch": 0,
            "loss": 0.25,
            "model_state": {"weight": torch.tensor([1.0])},
            "meta": meta,
        },
        checkpoint,
    )
    checkpoint_sha = _file_sha(checkpoint)

    training_metrics = root / "metrics.jsonl"
    run_id = "synthetic-run"
    config = {"method": "H0", "seed": 0}
    training_rows = [
        {
            "phase": "run_start",
            "sequence": 0,
            "run_id": run_id,
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
        },
        {
            "phase": "validation",
            "sequence": 1,
            "run_id": run_id,
            "epoch": 0,
            "BCE_at_1": 0.25,
            "selection_detail": copy.deepcopy(selection_detail),
        },
        {
            "phase": "checkpoint",
            "sequence": 2,
            "run_id": run_id,
            "role": "best_validation",
            "path": "/historical/machine/H0_s0/best.pt",
            "sha256": checkpoint_sha,
            "epoch": 0,
            "optimizer_updates": 2,
            "selected_value": 0.25,
        },
        {
            "phase": "run_end",
            "sequence": 3,
            "run_id": run_id,
            "status": "completed",
            "error": None,
            "alert_counts": {"error": 0, "warning": 0, "info": 0},
            "summary": {
                "best_validation_BCE_at_1": 0.25,
                "optimizer_updates": 2,
                "processed_samples": 4,
            },
            "checkpoints": [
                {
                    "role": "best_validation",
                    "path": "/historical/machine/H0_s0/best.pt",
                    "sha256": checkpoint_sha,
                    "epoch": 0,
                    "optimizer_updates": 2,
                    "selected_value": 0.25,
                }
            ],
        },
    ]
    _write_jsonl(training_metrics, training_rows)

    evaluator_source = build_evaluator_source(PROJECT_ROOT)
    metric_contract = build_metric_contract(0.2)
    execution_contract = {
        "schema_version": 1,
        "evaluation_data": {
            "split": "test",
            "manifest_sha256": _file_sha(test_manifest),
            "data_sha256": _file_sha(test_jsonl),
            "sample_count": len(records),
        },
        "observation": {"history": 1, "n_waypoints": 1, "dt": 0.1},
        "state": {
            "declared_mode": "rolling",
            "effective_mode": "rolling",
            "override": False,
            "sequence_id_required": True,
            "reset_policy": "clean_sequence_boundary_or_frame_gap",
        },
        "label": {
            "declared_mode": "absolute",
            "effective_mode": "absolute",
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
            "torch_default_dtype": "torch.float32",
            "parameter_dtypes": ["torch.float32"],
            "buffer_dtypes": ["torch.float32"],
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
        "evaluation_identity": {"tier": "locked_final", "class": "headline"},
    }
    fairness_contract = build_fairness_contract(meta)
    method_contract = build_method_contract(meta)
    provenance = {
        "state_mode": "rolling",
        "test_manifest_sha256": _file_sha(test_manifest),
        "train_manifest_sha256": meta["data_manifest_hash"],
        "train_data_sha256": meta["data_jsonl_sha256"],
        "validation_manifest_sha256": meta["validation"]["data_manifest_hash"],
        "validation_data_sha256": meta["validation"]["data_jsonl_sha256"],
        "base_model_sha256": meta["base_model_sha256"],
        "qwen_model_sha256": meta["qwen_model_sha256"],
        "vision_cache_manifest_sha256": meta["vision_cache_manifest_sha256"],
        "vision_cache_provenance_sha256": meta[
            "vision_cache_provenance_sha256"
        ],
        "vision_cache_token_payload_sha256": meta[
            "vision_cache_token_payload_sha256"
        ],
        "dino_model_sha256": meta["dino_model_sha256"],
        "siglip_model_sha256": meta["siglip_model_sha256"],
        "source_tree_sha256": source_tree_sha256(
            PROJECT_ROOT / "third_party" / "OpenTrackVLA"
        ),
        "evaluator_source": evaluator_source,
        "evaluator_source_sha256": canonical_json_sha256(evaluator_source),
        "metric_contract": metric_contract,
        "metric_contract_sha256": canonical_json_sha256(metric_contract),
        "evaluation_execution_contract": execution_contract,
        "evaluation_execution_contract_sha256": canonical_json_sha256(
            execution_contract
        ),
        "evaluation_predictions_sha256": _file_sha(predictions),
        "experiment_registry_sha256": _file_sha(registry),
        "fairness_contract": fairness_contract,
        "fairness_contract_sha256": canonical_json_sha256(fairness_contract),
        "method_contract": method_contract,
        "method_contract_sha256": canonical_json_sha256(method_contract),
        "validation_selection_detail_sha256": canonical_json_sha256(
            selection_detail
        ),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "best_validation",
        "selection_verified": True,
        "selected_epoch": 0,
        "selected_value": 0.25,
        "checkpoint_seed": 0,
        "evaluation_tier": "locked_final",
        "evaluation_class": "headline",
        "headline_eligible": True,
        "state_mode_override": False,
        "checkpoint_experiment_id": "H0",
        "effective_experiment_id": "H0",
    }
    evaluation_results = root / "evaluation.json"
    _write_json(
        evaluation_results,
        {
            "H0_s0": {
                "checkpoint": "/historical/machine/H0_s0/best.pt",
                "predictions": "/historical/machine/H0_s0/predictions.jsonl",
                "label_mode": "absolute",
                "metrics": metrics,
                "provenance": provenance,
            }
        },
    )
    trust_roots = root / "headline_trust_roots.json"
    bootstrap_analysis_contract = {
        "schema_version": 1,
        "baseline_experiment_id": "B1",
        "candidate_experiment_id": "H0",
        "iterations": 10_000,
        "analysis_seed": 20_260_715,
        "metric": "episode_macro_BCE@1",
        "seed_ids": [0, 1, 2],
        "episode_ids": list(EPISODES),
    }
    shared_evaluation_contract = {
        "schema_version": 1,
        "loader": {
            "shuffle": False,
            "num_workers": 0,
            "ordered_record_validation": True,
        },
        "runtime": {
            "device_type": "cpu",
            "torch_default_dtype": "torch.float32",
            "parameter_dtypes": ["torch.float32"],
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
    }
    _write_json(
        trust_roots,
        {
            "schema_version": 1,
            "analysis_class": TRUST_ROOTS_ANALYSIS_CLASS,
            "status": "frozen_before_internal_test_synthetic_fixture",
            "experiment_registry_sha256": _file_sha(registry),
            "source_tree_sha256": source_tree_sha256(
                PROJECT_ROOT / "third_party" / "OpenTrackVLA"
            ),
            "evaluator_source_sha256": canonical_json_sha256(evaluator_source),
            "metric_contract_sha256": canonical_json_sha256(metric_contract),
            "builder_source_sha256": _file_sha(
                PROJECT_ROOT / "scripts" / "build_headline_cells.py"
            ),
            "bootstrap_source_sha256": _file_sha(
                PROJECT_ROOT / "scripts" / "bootstrap_experiments.py"
            ),
            "bootstrap_analysis_contract": bootstrap_analysis_contract,
            "bootstrap_analysis_contract_sha256": canonical_json_sha256(
                bootstrap_analysis_contract
            ),
            "shared_evaluation_contract": shared_evaluation_contract,
            "shared_evaluation_contract_sha256": canonical_json_sha256(
                shared_evaluation_contract
            ),
            "test_manifest_sha256": _file_sha(test_manifest),
            "test_data_sha256": _file_sha(test_jsonl),
        },
    )
    manifest = root / "headline_run_manifest.json"
    _write_json(
        manifest,
        {
            "schema_version": 1,
            "analysis_class": RUN_MANIFEST_ANALYSIS_CLASS,
            "status": "frozen_after_locked_test_evaluation",
            "trust_roots": {
                "path": "headline_trust_roots.json",
                "sha256": _file_sha(trust_roots),
            },
            "test_dataset": {
                "jsonl": {"path": "test.jsonl", "sha256": _file_sha(test_jsonl)},
                "manifest": {
                    "path": "test.jsonl.manifest.json",
                    "sha256": _file_sha(test_manifest),
                },
            },
            "registry": {
                "path": "experiment_registry.json",
                "sha256": _file_sha(registry),
            },
            "runs": [
                {
                    "run_name": "H0_s0",
                    "experiment_id": "H0",
                    "seed": 0,
                    "checkpoint": {"path": "best.pt", "sha256": checkpoint_sha},
                    "training_metrics": {
                        "path": "metrics.jsonl",
                        "sha256": _file_sha(training_metrics),
                    },
                    "evaluation_results": {
                        "path": "evaluation.json",
                        "sha256": _file_sha(evaluation_results),
                    },
                    "predictions": {
                        "path": "predictions.jsonl",
                        "sha256": _file_sha(predictions),
                    },
                }
            ],
        },
    )
    return {
        "root": root,
        "test_jsonl": test_jsonl,
        "test_manifest": test_manifest,
        "trust_roots": trust_roots,
        "registry": registry,
        "checkpoint": checkpoint,
        "training_metrics": training_metrics,
        "evaluation_results": evaluation_results,
        "predictions": predictions,
        "manifest": manifest,
        "output": root / "verified_headline.json",
    }


def test_builder_verifies_bytes_and_recomputes_all_core_metrics(tmp_path):
    fixture = _build_fixture(tmp_path)
    summary = main(_args(fixture))
    assert summary["metric_recomputation_passed"] is True
    document = json.loads(fixture["output"].read_text(encoding="utf-8"))
    assert document["analysis_class"] == BUILDER_ANALYSIS_CLASS
    assert document["receipt"]["verification_status"] == "verified"
    assert document["receipt"]["metric_recomputation"]["passed"] is True
    assert "/historical/" not in fixture["output"].read_text(encoding="utf-8")
    assert set(document["receipt"]["metric_recomputation"]["metrics"]) == {
        "balanced_control_error_at1",
        "smooth_l1",
        "turn_sign_accuracy",
        "chronological_transition",
        "saturation_rate",
    }


@pytest.mark.parametrize(
    ("metric_path", "delta", "match"),
    (
        (("balanced_control_error_at1", "value"), 0.2, "balanced_control_error_at1"),
        (("smooth_l1", "forward"), 0.2, "smooth_l1"),
        (("saturation_rate", "overall"), 0.2, "saturation_rate"),
    ),
)
def test_metrics_only_tamper_is_rejected(tmp_path, metric_path, delta, match):
    fixture = _build_fixture(tmp_path)
    document = json.loads(fixture["evaluation_results"].read_text(encoding="utf-8"))
    target = document["H0_s0"]["metrics"]
    for key in metric_path[:-1]:
        target = target[key]
    target[metric_path[-1]] += delta
    _write_json(fixture["evaluation_results"], document)
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match=match):
        main(_args(fixture, expected_manifest))


def test_prediction_bytes_tamper_is_rejected_by_frozen_sha(tmp_path):
    fixture = _build_fixture(tmp_path)
    fixture["predictions"].write_bytes(fixture["predictions"].read_bytes() + b" \n")
    with pytest.raises(HeadlineBuildError, match="predictions SHA-256 mismatch"):
        main(_args(fixture))


def test_prediction_and_metrics_tamper_cannot_replace_external_manifest_root(tmp_path):
    fixture = _build_fixture(tmp_path)
    expected_manifest = _file_sha(fixture["manifest"])
    rows = [json.loads(line) for line in fixture["predictions"].read_text().splitlines()]
    rows[0]["pred_step_actions"][0][0] += 0.4
    _write_jsonl(fixture["predictions"], rows)
    evaluation = json.loads(fixture["evaluation_results"].read_text())
    trusted = _test_records()
    changed = np.asarray([row["pred_step_actions"] for row in rows])
    forged_metrics = evaluate_predictions(changed, trusted, threshold=0.2)
    forged_metrics.update(
        {
            key: value
            for key, value in evaluation["H0_s0"]["metrics"].items()
            if key
            in {
                "model_family",
                "state_mode",
                "experiment_id",
                "checkpoint_experiment_id",
                "evaluation_class",
                "headline_eligible",
                "seed",
            }
        }
    )
    evaluation["H0_s0"]["metrics"] = forged_metrics
    evaluation["H0_s0"]["provenance"]["evaluation_predictions_sha256"] = _file_sha(
        fixture["predictions"]
    )
    _write_json(fixture["evaluation_results"], evaluation)
    _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="run manifest SHA-256 mismatch"):
        main(_args(fixture, expected_manifest))


def test_checkpoint_semantic_tamper_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    checkpoint = torch.load(fixture["checkpoint"], map_location="cpu", weights_only=False)
    checkpoint["meta"]["seed"] = 1
    torch.save(checkpoint, fixture["checkpoint"])
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="checkpoint seed"):
        main(_args(fixture, expected_manifest))


def test_training_selection_tamper_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["training_metrics"].read_text().splitlines()]
    rows[1]["BCE_at_1"] = 0.9
    _write_jsonl(fixture["training_metrics"], rows)
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="validation selected BCE"):
        main(_args(fixture, expected_manifest))


@pytest.mark.parametrize("mode", ("missing", "duplicate", "extra"))
def test_prediction_sample_key_closure_is_enforced(tmp_path, mode):
    fixture = _build_fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["predictions"].read_text().splitlines()]
    if mode == "missing":
        rows.pop()
    elif mode == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    else:
        extra = copy.deepcopy(rows[-1])
        extra["frame_idx"] = 999
        rows.append(extra)
    _write_jsonl(fixture["predictions"], rows)
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="sample key"):
        main(_args(fixture, expected_manifest))


def test_prediction_embedded_ground_truth_forgery_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["predictions"].read_text().splitlines()]
    rows[0]["step_actions"][0][0] += 0.5
    _write_jsonl(fixture["predictions"], rows)
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="embedded ground truth disagrees"):
        main(_args(fixture, expected_manifest))


def test_frozen_test_frame_idx_is_strict_before_dataset_style_coercion(tmp_path):
    fixture = _build_fixture(tmp_path)
    records = [
        json.loads(line)
        for line in fixture["test_jsonl"].read_text(encoding="utf-8").splitlines()
    ]
    records[0]["frame_idx"] = 0.5
    _write_jsonl(fixture["test_jsonl"], records)
    test_manifest = json.loads(
        fixture["test_manifest"].read_text(encoding="utf-8")
    )
    test_manifest["data_jsonl_sha256"] = _file_sha(fixture["test_jsonl"])
    _write_json(fixture["test_manifest"], test_manifest)
    trust_roots = json.loads(
        fixture["trust_roots"].read_text(encoding="utf-8")
    )
    trust_roots["test_data_sha256"] = _file_sha(fixture["test_jsonl"])
    trust_roots["test_manifest_sha256"] = _file_sha(fixture["test_manifest"])
    _write_json(fixture["trust_roots"], trust_roots)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(HeadlineBuildError, match="frame_idx must be an integer"):
        main(_args(fixture, expected_manifest))


def test_prediction_command_identity_poisoning_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    rows = [json.loads(line) for line in fixture["predictions"].read_text().splitlines()]
    rows[0]["command"] = "turn_right"
    _write_jsonl(fixture["predictions"], rows)
    expected_manifest = _refresh_manifest(fixture)
    with pytest.raises(HeadlineBuildError, match="sample keys mismatch"):
        main(_args(fixture, expected_manifest))


def test_stale_absolute_paths_and_bundle_relocation_are_canonical(tmp_path):
    fixture = _build_fixture(tmp_path / "original")
    original_output = fixture["root"] / "first.json"
    main(_args(fixture, output=original_output))
    moved_root = tmp_path / "moved" / "bundle"
    moved_root.parent.mkdir()
    shutil.copytree(fixture["root"], moved_root)
    moved_fixture = {
        key: (moved_root / value.name if isinstance(value, Path) and value.parent == fixture["root"] else value)
        for key, value in fixture.items()
    }
    moved_fixture["root"] = moved_root
    moved_fixture["output"] = moved_root / "second.json"
    main(_args(moved_fixture, output=moved_fixture["output"]))
    assert original_output.read_bytes() == moved_fixture["output"].read_bytes()


def test_run_manifest_input_sha_mismatch_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    with pytest.raises(HeadlineBuildError, match="run manifest SHA-256 mismatch"):
        main(_args(fixture, expected_manifest_sha=_sha("wrong")))


def test_registry_content_drift_is_rejected_even_when_manifest_sha_is_refrozen(
    tmp_path,
):
    fixture = _build_fixture(tmp_path)
    registry = json.loads(fixture["registry"].read_text(encoding="utf-8"))
    registry["review_marker"] = "drifted-after-evaluation"
    _write_json(fixture["registry"], registry)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(HeadlineBuildError, match="registry"):
        main(_args(fixture, expected_manifest_sha=expected_manifest))


def test_evaluation_source_tree_sha_drift_is_rejected(tmp_path):
    fixture = _build_fixture(tmp_path)
    evaluation = json.loads(
        fixture["evaluation_results"].read_text(encoding="utf-8")
    )
    evaluation["H0_s0"]["provenance"]["source_tree_sha256"] = _sha(
        "drifted-source-tree"
    )
    _write_json(fixture["evaluation_results"], evaluation)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(
        HeadlineBuildError,
        match="source_tree_sha256|source binding",
    ):
        main(_args(fixture, expected_manifest_sha=expected_manifest))


@pytest.mark.parametrize(
    ("section", "field", "value", "match"),
    (
        ("observation", "history", 99, "execution observation.history"),
        ("state", "effective_mode", "stateless", "execution effective state"),
        ("label", "override", True, "execution label override"),
        ("loader", "shuffle", True, "execution loader shuffle"),
        (
            "runtime",
            "device_type",
            "mps",
            "pre-registered runtime.device_type",
        ),
        ("runtime", "autocast", True, "execution autocast"),
        (
            "evaluation_identity",
            "class",
            "sensitivity",
            "execution evaluation class",
        ),
    ),
)
def test_execution_contract_semantic_tamper_is_rejected(
    tmp_path, section, field, value, match
):
    fixture = _build_fixture(tmp_path)
    evaluation = json.loads(
        fixture["evaluation_results"].read_text(encoding="utf-8")
    )
    provenance = evaluation["H0_s0"]["provenance"]
    provenance["evaluation_execution_contract"][section][field] = value
    provenance["evaluation_execution_contract_sha256"] = canonical_json_sha256(
        provenance["evaluation_execution_contract"]
    )
    _write_json(fixture["evaluation_results"], evaluation)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(HeadlineBuildError, match=match):
        main(_args(fixture, expected_manifest_sha=expected_manifest))


def test_metric_contract_cannot_replace_preregistered_trust_root(tmp_path):
    fixture = _build_fixture(tmp_path)
    evaluation = json.loads(
        fixture["evaluation_results"].read_text(encoding="utf-8")
    )
    provenance = evaluation["H0_s0"]["provenance"]
    changed_contract = build_metric_contract(0.7)
    provenance["metric_contract"] = changed_contract
    provenance["metric_contract_sha256"] = canonical_json_sha256(changed_contract)
    _write_json(fixture["evaluation_results"], evaluation)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(HeadlineBuildError, match="pre-registered metric-contract SHA"):
        main(_args(fixture, expected_manifest_sha=expected_manifest))


def test_bootstrap_source_cannot_replace_preregistered_trust_root(tmp_path):
    fixture = _build_fixture(tmp_path)
    trust_roots = json.loads(
        fixture["trust_roots"].read_text(encoding="utf-8")
    )
    trust_roots["bootstrap_source_sha256"] = _sha("different-bootstrap")
    _write_json(fixture["trust_roots"], trust_roots)
    expected_manifest = _refresh_manifest(fixture)

    with pytest.raises(HeadlineBuildError, match="headline-bootstrap SHA"):
        main(_args(fixture, expected_manifest_sha=expected_manifest))


def test_builder_refuses_to_overwrite_frozen_output(tmp_path):
    fixture = _build_fixture(tmp_path)
    main(_args(fixture))
    with pytest.raises(HeadlineBuildError, match="output already exists"):
        main(_args(fixture))
