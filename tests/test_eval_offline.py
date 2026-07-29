import numpy as np
import pytest
import hashlib
import json
from pathlib import Path

from data_pipeline.kinematics import integrate_actions
from scripts.eval_offline import (
    _f1_from_counts,
    _match_events,
    balanced_control_error_at1,
    build_evaluation_execution_contract,
    build_evaluator_source,
    build_fairness_contract,
    build_metric_contract,
    build_method_contract,
    canonical_json_sha256,
    checkpoint_model_family,
    chronological_transition_metrics,
    compute_metrics,
    continues_evaluation_sequence,
    evaluation_sequence_key,
    evaluate_predictions,
    horizon_metrics,
    load_checkpoint_with_sha256,
    load_experiment_registry,
    main as eval_main,
    render_comparison_table,
    resolve_label_mode_for_evaluation,
    resolve_evaluation_identity,
    transition_event_mask,
    validate_checkpoint_metadata,
    validate_comparison_contracts,
    validate_evaluation_dataset,
    validate_expected_contract_sha256,
    validate_label_mode_override,
    validate_mode_override_names,
    validate_ordered_evaluation_records,
    validate_registry_checkpoint_binding,
    verify_checkpoint_file_unchanged,
    waypoints_to_step_actions,
)


def complete_checkpoint_meta(**overrides):
    meta = {
        "schema_version": 1,
        "model_family": "opentrackvla_baseline",
        "experiment_id": "B0",
        "seed": 0,
        "history": 31,
        "n_waypoints": 8,
        "dt": 0.1,
        "label_mode": "absolute",
        "action_semantics": "arc_turn_v2",
        "data_manifest_hash": "train-manifest",
        "data_jsonl_sha256": "train-jsonl",
        "sample_count": 200,
        "base_model_sha256": "base",
        "base_model_artifact": {
            "schema_version": 1,
            "format": "huggingface_pretrained",
            "weight_layout": "safetensors_single",
            "files": [
                {
                    "path": "config.json",
                    "role": "config",
                    "size": 1,
                    "sha256": "config",
                },
                {
                    "path": "model.safetensors",
                    "role": "weights",
                    "size": 1,
                    "sha256": "weights",
                },
            ],
            "artifact_sha256": "base",
        },
        "qwen_model_sha256": "qwen",
        "vision_cache_manifest_sha256": "cache-manifest",
        "vision_cache_provenance_sha256": "cache-provenance",
        "vision_cache_token_payload_sha256": "cache-payload",
        "dino_model_sha256": "dino",
        "siglip_model_sha256": "siglip",
        "training_source_raw_dirs": ["test001"],
        "state_mode": "stateless",
        "checkpoint_selection": {
            "metric": "validation_episode_macro_BCE@1",
            "mode": "min",
            "rule": "strict_improvement_earliest_epoch",
        },
        "checkpoint_role": "best_validation",
        "selection_verified": True,
        "selected_epoch": 2,
        "selected_value": 0.25,
        "optimizer_updates": 100,
        "processed_samples": 200,
        "sampling_policy": "ordered_jsonl",
        "batch_size": 2,
        "grad_accum_steps": 1,
        "effective_batch_size": 2,
        "base_lr": 2e-5,
        "head_lr": None,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "validation": {
            "data_manifest_hash": "val-manifest",
            "data_jsonl_sha256": "val-jsonl",
            "sample_count": 10,
        },
        "best_validation": {
            "selection_bce_at1": 0.25,
            "selection_detail": {
                "value": 0.25,
                "by_episode": {"val0": 0.25},
                "support": {"val0": {"forward": 10}},
            },
        },
    }
    meta.update(overrides)
    return meta


def matching_registry(meta=None):
    meta = complete_checkpoint_meta() if meta is None else meta
    return {
        "schema_version": 1,
        "status": "frozen_before_validation_ablation",
        "source_tree_sha256": "a" * 64,
        "history": meta["history"],
        "prediction_horizon": meta["n_waypoints"],
        "dt": meta["dt"],
        "sampling_policy": meta["sampling_policy"],
        "max_optimizer_updates": meta["optimizer_updates"],
        "processed_samples_per_run": meta["processed_samples"],
        "weight_decay": meta["weight_decay"],
        "grad_clip": meta["grad_clip"],
        "checkpoint_selection": meta["checkpoint_selection"],
        "data": {
            "train_count": meta["sample_count"],
            "train_data_sha256": meta["data_jsonl_sha256"],
            "train_manifest_sha256": meta["data_manifest_hash"],
            "val_count": meta["validation"]["sample_count"],
            "val_data_sha256": meta["validation"]["data_jsonl_sha256"],
            "val_manifest_sha256": meta["validation"]["data_manifest_hash"],
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
            meta["experiment_id"]: {
                "model_family": meta["model_family"],
                "state_mode": meta["state_mode"],
                "batch_size": meta["batch_size"],
                "grad_accum_steps": meta["grad_accum_steps"],
                "effective_batch_size": meta["effective_batch_size"],
                "base_lr": meta["base_lr"],
                "head_lr": meta["head_lr"],
            }
        },
    }


def test_absolute_waypoints_are_inverted_to_step_actions():
    actions = np.asarray([[1.0, 0.0, 1.0], [0.5, 0.2, -0.5]], dtype=np.float32)
    waypoints = integrate_actions(actions, 0.1)
    recovered = waypoints_to_step_actions(waypoints, 0.1)
    np.testing.assert_allclose(recovered, actions, atol=1e-5)


def test_offline_metrics_cover_axis_sign_transition_and_saturation():
    gt = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    pred = gt.copy()
    metrics = compute_metrics(pred, gt, np.zeros((1, 3)), np.ones((1, 3), dtype=bool))
    assert metrics["smooth_l1"] == {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}
    assert metrics["turn_sign_accuracy"] == 1.0
    assert metrics["transition"]["f1"] == 1.0
    assert metrics["saturation_rate"]["forward"] == pytest.approx(2.0 / 3.0)


def test_combined_saturation_counts_timesteps_not_axis_elements():
    target = np.zeros((1, 2, 3), dtype=np.float64)
    prediction = np.asarray([[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    metrics = compute_metrics(
        prediction,
        target,
        np.zeros((1, 3)),
        np.ones((1, 2), dtype=bool),
    )
    assert metrics["saturation_rate"]["overall"] == 0.5


def test_transition_f1_is_zero_when_ground_truth_events_are_fully_missed():
    target = np.asarray([[[0.0, 0.0, 1.0]]])
    prediction = np.zeros_like(target)
    metrics = compute_metrics(
        prediction,
        target,
        np.zeros((1, 3)),
        np.ones((1, 1), dtype=bool),
    )
    assert metrics["transition"] == {
        "precision": None,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 0,
        "fn": 1,
    }


def test_transition_events_follow_turn_activity_and_sign_not_yaw_difference():
    actions = np.asarray([[[0.0, 0.0, 0.25], [0.0, 0.0, 0.7], [0.0, 0.0, -0.4], [0.0, 0.0, 0.0]]])
    previous = np.asarray([[0.0, 0.0, 0.1]])
    np.testing.assert_array_equal(
        transition_event_mask(actions, previous, threshold=0.2),
        [[True, False, True, True]],
    )


def test_offline_evaluation_groups_transition_types_and_renders_multiple_runs():
    records = [
        {
            "step_actions": [[1.0, 0.0, 0.0]],
            "prev_action": [1.0, 0.0, 0.0],
            "transition_type": "steady_forward",
        },
        {
            "step_actions": [[0.8, 0.0, -1.0]],
            "prev_action": [1.0, 0.0, 0.0],
            "transition_type": "turn_onset",
        },
    ]
    pred = np.asarray([record["step_actions"] for record in records])
    metrics = evaluate_predictions(pred, records)
    assert set(metrics["by_transition_type"]) == {"steady_forward", "turn_onset"}
    results = {
        "absolute": {"label_mode": "absolute", "metrics": metrics},
        "step": {"label_mode": "step_action", "metrics": metrics},
    }
    table = render_comparison_table(results)
    assert "absolute" in table and "step_action" in table


def test_checkpoint_family_fails_closed_instead_of_guessing_pfem():
    with pytest.raises(ValueError, match="missing explicit model_family"):
        checkpoint_model_family({"meta": {}})


def test_locked_final_requires_predictions_dir_before_reading_artifacts():
    with pytest.raises(ValueError, match="requires --predictions_dir"):
        eval_main(
            [
                "--val_json",
                "does-not-exist.jsonl",
                "--ckpt",
                "run=does-not-exist.pt",
                "--cache_root",
                "does-not-exist-cache",
                "--experiment_registry",
                "does-not-exist-registry.json",
                "--expected_registry_sha256",
                "a" * 64,
            ]
        )


def test_checkpoint_metadata_requires_reproducibility_contract():
    meta = validate_checkpoint_metadata({"meta": complete_checkpoint_meta()})
    assert meta["state_mode"] == "stateless"
    broken = complete_checkpoint_meta()
    del broken["qwen_model_sha256"]
    with pytest.raises(ValueError, match="qwen_model_sha256"):
        validate_checkpoint_metadata({"meta": broken})


def test_registry_binding_emits_complete_cross_invocation_provenance():
    meta = complete_checkpoint_meta()
    registry = matching_registry(meta)
    provenance = validate_registry_checkpoint_binding(
        meta,
        registry,
        registry_sha256="b" * 64,
        actual_source_tree_sha256="a" * 64,
    )
    assert provenance["train_manifest_sha256"] == "train-manifest"
    assert provenance["validation_data_sha256"] == "val-jsonl"
    assert provenance["base_model_sha256"] == "base"
    assert provenance["vision_cache_provenance_sha256"] == "cache-provenance"
    assert provenance["source_tree_sha256"] == "a" * 64
    assert provenance["experiment_registry_sha256"] == "b" * 64
    assert provenance["checkpoint_role"] == "best_validation"
    assert provenance["selection_verified"] is True
    assert provenance["checkpoint_seed"] == 0
    assert provenance["fairness_contract_sha256"] == canonical_json_sha256(
        build_fairness_contract(meta)
    )
    assert provenance["method_contract_sha256"] == canonical_json_sha256(
        build_method_contract(meta)
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ({"data_jsonl_sha256": "other-train"}, "train_data_sha256"),
        ({"base_model_sha256": "other-base"}, "base_model_sha256"),
        ({"qwen_model_sha256": "other-qwen"}, "qwen_model_sha256"),
        (
            {"vision_cache_provenance_sha256": "other-cache"},
            "vision_cache_provenance_sha256",
        ),
        ({"dino_model_sha256": "other-dino"}, "dinov3_sha256"),
        ({"sampling_policy": "weighted_random"}, "sampling_policy"),
        ({"processed_samples": 199}, "processed_samples_per_run"),
    ),
)
def test_registry_binding_rejects_checkpoint_or_fairness_drift(mutation, match):
    registry = matching_registry()
    meta = complete_checkpoint_meta(**mutation)
    with pytest.raises(ValueError, match=match):
        validate_registry_checkpoint_binding(
            meta,
            registry,
            registry_sha256="b" * 64,
            actual_source_tree_sha256="a" * 64,
        )


def test_registry_binding_rejects_validation_and_source_drift():
    meta = complete_checkpoint_meta()
    registry = matching_registry(meta)
    changed_validation = dict(meta["validation"])
    changed_validation["data_jsonl_sha256"] = "other-val"
    changed_meta = {**meta, "validation": changed_validation}
    with pytest.raises(ValueError, match="val_data_sha256"):
        validate_registry_checkpoint_binding(
            changed_meta,
            registry,
            registry_sha256="b" * 64,
            actual_source_tree_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="source-tree"):
        validate_registry_checkpoint_binding(
            meta,
            registry,
            registry_sha256="b" * 64,
            actual_source_tree_sha256="c" * 64,
        )


def test_registry_file_hash_is_explicit_and_fail_closed(tmp_path):
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(matching_registry()), encoding="utf-8")
    expected = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    registry, actual = load_experiment_registry(registry_path, expected)
    assert registry["status"].startswith("frozen")
    assert actual == expected
    with pytest.raises(ValueError, match="registry SHA-256 mismatch"):
        load_experiment_registry(registry_path, "f" * 64)


def test_evaluator_source_and_metric_contracts_are_deterministic_and_bound():
    evaluator_source = build_evaluator_source()
    paths = [item["path"] for item in evaluator_source["files"]]
    assert "scripts/eval_offline.py" in paths
    assert "data_pipeline/kinematics.py" in paths
    assert any(path.startswith("inference_pipeline/") for path in paths)
    source_hash = canonical_json_sha256(evaluator_source)
    assert build_evaluator_source() == evaluator_source
    assert validate_expected_contract_sha256(
        source_hash,
        {"evaluator_source_sha256": source_hash},
        "evaluator_source_sha256",
    ) == source_hash
    with pytest.raises(ValueError, match="evaluator_source_sha256 mismatch"):
        validate_expected_contract_sha256(
            source_hash,
            {"evaluator_source_sha256": "f" * 64},
            "evaluator_source_sha256",
        )

    default_metric = build_metric_contract(0.2)
    changed_metric = build_metric_contract(0.3)
    assert default_metric["schema_version"] == 2
    assert default_metric["transition"] == {
        "yaw_active_threshold": 0.2,
        "event_types": ["onset", "exit", "sign_flip"],
        "chronological_horizon": 1,
        "one_to_one_tolerance_frames": 2,
        "f1_definition": "2*tp/(2*tp+fp+fn)",
        "count_aggregation": "micro_sum_tp_fp_fn_before_f1",
        "precision_recall_null_semantics": {
            "precision": "null iff tp+fp=0",
            "recall": "null iff tp+fn=0",
        },
        "matcher_semantics": (
            "maximum_cardinality_one_to_one_per_sequence_and_event_type_"
            "within_tolerance_frames"
        ),
        "empty_event_union_f1": None,
        "supported_zero_true_positive_f1": 0.0,
    }
    assert default_metric["transition"]["one_to_one_tolerance_frames"] == 2
    assert default_metric["horizon_mae"]["horizons"] == [1, 2, 4, 8]
    assert default_metric["saturation"]["absolute_threshold"] == 0.95
    assert canonical_json_sha256(default_metric) != canonical_json_sha256(
        changed_metric
    )


def test_evaluation_execution_contract_binds_data_state_loader_and_dtype():
    binding = {
        "split": "val",
        "_verified_manifest_sha256": "a" * 64,
        "_verified_data_jsonl_sha256": "b" * 64,
    }
    identity = {
        "state_mode": "rolling",
        "state_mode_override": False,
        "evaluation_tier": "locked_final",
        "evaluation_class": "validation",
    }
    contract = build_evaluation_execution_contract(
        evaluation_binding=binding,
        identity=identity,
        declared_state_mode="rolling",
        declared_label_mode="absolute",
        effective_label_mode="absolute",
        label_mode_override=False,
        batch_size=1,
        history=31,
        n_waypoints=8,
        dt=0.1,
        device="cpu",
        default_dtype="torch.float32",
        parameter_dtypes={"torch.float32"},
        buffer_dtypes={"torch.float32"},
        dataset_sample_count=10,
    )
    assert contract["evaluation_data"]["split"] == "val"
    assert contract["state"]["sequence_id_required"] is True
    assert contract["loader"] == {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": 0,
        "ordered_record_validation": True,
    }
    changed = build_evaluation_execution_contract(
        evaluation_binding=binding,
        identity={**identity, "state_mode": "stateless"},
        declared_state_mode="stateless",
        declared_label_mode="absolute",
        effective_label_mode="absolute",
        label_mode_override=False,
        batch_size=2,
        history=31,
        n_waypoints=8,
        dt=0.1,
        device="cpu",
        default_dtype="torch.float32",
        parameter_dtypes={"torch.float32"},
        buffer_dtypes={"torch.float32"},
        dataset_sample_count=10,
    )
    assert canonical_json_sha256(contract) != canonical_json_sha256(changed)


def test_checkpoint_hash_binds_loaded_bytes_and_detects_replacement(tmp_path):
    import torch

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": 0, "meta": {"marker": "original"}}, checkpoint_path)
    expected = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()

    checkpoint, actual = load_checkpoint_with_sha256(checkpoint_path)
    assert checkpoint["meta"]["marker"] == "original"
    assert actual == expected
    assert verify_checkpoint_file_unchanged(checkpoint_path, actual) == actual

    checkpoint_path.write_bytes(b"replacement")
    with pytest.raises(RuntimeError, match="checkpoint changed during evaluation"):
        verify_checkpoint_file_unchanged(checkpoint_path, actual)


def test_comparison_contract_rejects_different_training_data():
    first = complete_checkpoint_meta()
    second = complete_checkpoint_meta(data_jsonl_sha256="different")
    with pytest.raises(ValueError, match="data_jsonl_sha256"):
        validate_comparison_contracts([("B0", first), ("B1", second)])


def test_comparison_contract_enforces_matched_budget_and_optimizer_fields():
    b1 = complete_checkpoint_meta(
        model_family="trackvla_pp_lite",
        experiment_id="B1",
        state_mode="rolling",
        batch_size=1,
        grad_accum_steps=2,
        head_lr=3e-4,
    )
    h0 = complete_checkpoint_meta(
        model_family="pfem_harness",
        experiment_id="H0",
        state_mode="rolling",
        batch_size=1,
        grad_accum_steps=2,
        head_lr=3e-4,
    )
    assert validate_comparison_contracts([("B1", b1), ("H0", h0)])

    unequal_updates = {**h0, "optimizer_updates": 99}
    with pytest.raises(ValueError, match="optimizer_updates"):
        validate_comparison_contracts([("B1", b1), ("H0", unequal_updates)])

    same_effective_different_accum = {
        **h0,
        "batch_size": 2,
        "grad_accum_steps": 1,
    }
    with pytest.raises(ValueError, match="batch_size|grad_accum_steps"):
        validate_comparison_contracts(
            [("B1", b1), ("H0", same_effective_different_accum)]
        )

    unequal_head_lr = {**h0, "head_lr": 1e-4}
    with pytest.raises(ValueError, match="head_lr"):
        validate_comparison_contracts([("B1", b1), ("H0", unequal_head_lr)])

    unequal_sampling = {**h0, "sampling_policy": "weighted_random"}
    with pytest.raises(ValueError, match="sampling_policy"):
        validate_comparison_contracts([("B1", b1), ("H0", unequal_sampling)])


@pytest.mark.parametrize(
    ("experiment_id", "family", "state_mode"),
    (
        ("B0", "opentrackvla_baseline", "stateless"),
        ("B1", "trackvla_pp_lite", "rolling"),
        ("B1-P", "trackvla_pp_lite", "rolling"),
        ("H0", "pfem_harness", "rolling"),
        ("H0-ablation:tim", "pfem_harness", "rolling"),
        ("H0-S", "pfem_harness", "stateless"),
    ),
)
def test_experiment_id_binds_model_family_and_state(
    experiment_id, family, state_mode
):
    head_lr = None if family == "opentrackvla_baseline" else 3e-4
    meta = complete_checkpoint_meta(
        experiment_id=experiment_id,
        model_family=family,
        state_mode=state_mode,
        head_lr=head_lr,
    )
    assert validate_checkpoint_metadata({"meta": meta})["experiment_id"] == experiment_id


def test_experiment_id_rejects_wrong_family_or_state():
    wrong_state = complete_checkpoint_meta(
        model_family="trackvla_pp_lite",
        experiment_id="B1",
        state_mode="stateless",
        head_lr=3e-4,
    )
    with pytest.raises(ValueError, match="requires state_mode=rolling"):
        validate_checkpoint_metadata({"meta": wrong_state})
    wrong_family = complete_checkpoint_meta(
        model_family="pfem_harness",
        experiment_id="B1",
        state_mode="rolling",
        head_lr=3e-4,
    )
    with pytest.raises(ValueError, match="requires model_family=trackvla_pp_lite"):
        validate_checkpoint_metadata({"meta": wrong_family})


def test_test_evaluation_requires_frozen_hash_and_no_train_overlap(tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "source_raw_dir": "test004",
                "sequence_id": "seq",
                "frame_idx": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    sidecar = Path(str(dataset) + ".manifest.json")
    manifest = {
        "split": "test",
        "history": 31,
        "n_waypoints": 8,
        "dt": 0.1,
        "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
    }
    sidecar.write_text(json.dumps(manifest), encoding="utf-8")
    meta = complete_checkpoint_meta()
    with pytest.raises(ValueError, match="expected_eval_manifest"):
        validate_evaluation_dataset(dataset, meta)
    frozen_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    checkpoint = {"epoch": 2, "loss": 0.25, "meta": meta}
    validate_evaluation_dataset(
        dataset,
        meta,
        checkpoint=checkpoint,
        expected_manifest_sha256=frozen_hash,
    )
    overlap = complete_checkpoint_meta(training_source_raw_dirs=["test004"])
    with pytest.raises(ValueError, match="overlap"):
        validate_evaluation_dataset(
            dataset,
            overlap,
            checkpoint={"epoch": 2, "loss": 0.25, "meta": overlap},
            expected_manifest_sha256=frozen_hash,
        )


def test_locked_test_rejects_epoch_checkpoint_but_exploratory_is_explicit(tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text(
        json.dumps({"source_raw_dir": "test004"}) + "\n", encoding="utf-8"
    )
    sidecar = Path(str(dataset) + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "split": "test",
                "history": 31,
                "n_waypoints": 8,
                "dt": 0.1,
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    frozen_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    epoch_meta = complete_checkpoint_meta(
        checkpoint_role="epoch",
        selection_verified=False,
        selected_epoch=None,
        selected_value=None,
        best_validation=None,
    )
    epoch_checkpoint = {"epoch": 2, "loss": 1.2, "meta": epoch_meta}
    with pytest.raises(ValueError, match="best_validation"):
        validate_evaluation_dataset(
            dataset,
            epoch_meta,
            checkpoint=epoch_checkpoint,
            expected_manifest_sha256=frozen_hash,
        )
    binding = validate_evaluation_dataset(
        dataset,
        epoch_meta,
        checkpoint=epoch_checkpoint,
        evaluation_tier="exploratory",
        expected_manifest_sha256=frozen_hash,
    )
    assert binding["_verified_manifest_sha256"] == frozen_hash


def test_locked_test_rejects_unverified_or_inconsistent_best_checkpoint(tmp_path):
    dataset = tmp_path / "test.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    sidecar = Path(str(dataset) + ".manifest.json")
    sidecar.write_text(
        json.dumps(
            {
                "split": "test",
                "data_jsonl_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    frozen_hash = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    unverified = complete_checkpoint_meta(selection_verified=False)
    with pytest.raises(ValueError, match="selection_verified"):
        validate_evaluation_dataset(
            dataset,
            unverified,
            checkpoint={"epoch": 2, "loss": 0.25, "meta": unverified},
            expected_manifest_sha256=frozen_hash,
        )
    inconsistent = complete_checkpoint_meta(selected_value=0.3)
    with pytest.raises(ValueError, match="best_validation"):
        validate_evaluation_dataset(
            dataset,
            inconsistent,
            checkpoint={"epoch": 2, "loss": 0.3, "meta": inconsistent},
            expected_manifest_sha256=frozen_hash,
        )


def test_sequence_identity_crosses_clip_blocks_but_resets_on_gap():
    first = {
        "episode": "chunk0",
        "sequence_id": "chunk0",
        "clip_id": "clip0",
        "frame_idx": 90,
        "mirrored": False,
    }
    second = {**first, "clip_id": "clip1", "frame_idx": 91}
    assert continues_evaluation_sequence(
        evaluation_sequence_key(first), evaluation_sequence_key(second)
    )
    gap = {**second, "frame_idx": 93}
    assert not continues_evaluation_sequence(
        evaluation_sequence_key(second), evaluation_sequence_key(gap)
    )
    assert validate_ordered_evaluation_records(
        [first, second, gap], require_sequence_id=True
    )


def test_balanced_control_error_is_episode_and_command_macro():
    records = [
        {
            "episode": "ep1",
            "command": "forward",
            "step_actions": [[1.0, 0.0, 0.0]],
            "valid_mask": [True],
        },
        {
            "episode": "ep1",
            "command": "turn_right",
            "step_actions": [[0.5, 0.0, 1.0]],
            "valid_mask": [True],
        },
        {
            "episode": "ep2",
            "command": "forward",
            "step_actions": [[1.0, 0.0, 0.0]],
            "valid_mask": [True],
        },
    ]
    predictions = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    metric = balanced_control_error_at1(predictions, records)
    # ep1=(0 + 2/3)/2=1/3; ep2=1/3; macro=1/3.
    assert metric["value"] == pytest.approx(1.0 / 3.0)
    assert metric["support"]["ep1"] == {"forward": 1, "turn_right": 1}


def test_horizon_metrics_report_selected_forecast_steps():
    records = [
        {
            "step_actions": [[0.0, 0.0, 0.0]] * 8,
            "valid_mask": [True] * 8,
        }
    ]
    predictions = np.ones((1, 8, 3), dtype=np.float64)
    metrics = horizon_metrics(predictions, records)
    assert set(metrics) == {"1", "2", "4", "8"}
    assert metrics["8"]["forward_mae"] == 1.0
    assert metrics["8"]["yaw_mae"] == 1.0


def test_chronological_transition_matches_one_frame_shift_with_tolerance():
    records = []
    gt_yaws = [0.0, 1.0, 1.0, 0.0]
    pred_yaws = [0.0, 0.0, 1.0, 1.0]
    for frame, yaw in enumerate(gt_yaws):
        records.append(
            {
                "sequence_id": "seq",
                "chunk_id": "seq",
                "frame_idx": frame,
                "step_actions": [[0.0, 0.0, yaw]],
                "prev_action": [0.0, 0.0, 0.0 if frame == 0 else gt_yaws[frame - 1]],
                "valid_mask": [True],
            }
        )
    predictions = np.asarray([[[0.0, 0.0, yaw]] for yaw in pred_yaws])
    metric = chronological_transition_metrics(
        predictions, records, threshold=0.2, tolerance=2
    )
    assert metric["by_type"]["onset"]["f1"] == 1.0
    assert metric["by_type"]["exit"]["f1"] == 0.0
    assert metric["f1"] == pytest.approx(2.0 / 3.0)


def test_event_matching_f1_is_zero_for_disjoint_predictions_and_targets():
    metric = _match_events([0], [10], tolerance=2)
    assert metric == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }


def test_event_matching_f1_is_none_when_both_sides_have_no_events():
    metric = _match_events([], [], tolerance=2)
    assert metric == {
        "precision": None,
        "recall": None,
        "f1": None,
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def test_event_matching_f1_is_zero_for_prediction_only_events():
    metric = _match_events([0], [], tolerance=2)
    assert metric == {
        "precision": 0.0,
        "recall": None,
        "f1": 0.0,
        "tp": 0,
        "fp": 1,
        "fn": 0,
    }


@pytest.mark.parametrize(
    "counts",
    [(-1, 0, 0), (0, -1, 0), (0, 0, -1), (0.0, 0, 0), (True, 0, 0)],
)
def test_count_f1_rejects_non_integer_or_negative_counts(counts):
    with pytest.raises(ValueError, match="non-negative integers"):
        _f1_from_counts(*counts)


def test_event_matching_is_maximum_cardinality_not_nearest_first_greedy():
    metric = _match_events([0, 3], [2, 3], tolerance=2)
    assert metric == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 2,
        "fp": 0,
        "fn": 0,
    }


def test_state_override_is_renamed_and_cannot_enter_locked_headline():
    meta = complete_checkpoint_meta(
        model_family="pfem_harness",
        experiment_id="H0",
        state_mode="rolling",
        head_lr=3e-4,
    )
    with pytest.raises(ValueError, match="cannot use evaluation_tier=locked_final"):
        resolve_evaluation_identity(
            meta,
            evaluation_split="test",
            evaluation_tier="locked_final",
            requested_state_mode="stateless",
            allow_state_mode_override=True,
        )
    sensitivity = resolve_evaluation_identity(
        meta,
        evaluation_split="test",
        evaluation_tier="exploratory",
        requested_state_mode="stateless",
        allow_state_mode_override=True,
    )
    assert sensitivity["evaluation_class"] == "sensitivity"
    assert sensitivity["headline_eligible"] is False
    assert sensitivity["state_mode_override"] is True
    assert sensitivity["effective_experiment_id"] == (
        "H0-sensitivity:state_mode=stateless"
    )


def test_only_locked_best_test_without_override_is_headline_eligible():
    meta = complete_checkpoint_meta()
    headline = resolve_evaluation_identity(
        meta,
        evaluation_split="test",
        evaluation_tier="locked_final",
    )
    assert headline["evaluation_class"] == "headline"
    assert headline["headline_eligible"] is True
    validation = resolve_evaluation_identity(
        meta,
        evaluation_split="val",
        evaluation_tier="locked_final",
    )
    assert validation["evaluation_class"] == "validation"
    assert validation["headline_eligible"] is False


def test_locked_final_rejects_label_mode_override():
    assert (
        validate_label_mode_override(
            "absolute", "absolute", evaluation_tier="locked_final"
        )
        is False
    )
    with pytest.raises(ValueError, match="label-mode override"):
        validate_label_mode_override(
            "absolute", "step_action", evaluation_tier="locked_final"
        )
    assert (
        validate_label_mode_override(
            "absolute", "step_action", evaluation_tier="exploratory"
        )
        is True
    )


def test_family_label_mode_is_validated_before_normalization():
    with pytest.raises(ValueError, match="label-mode override"):
        resolve_label_mode_for_evaluation(
            "absolute",
            "step_action",
            model_family="opentrackvla_baseline",
            evaluation_tier="locked_final",
        )
    with pytest.raises(ValueError, match="supports label_mode=absolute only"):
        resolve_label_mode_for_evaluation(
            "absolute",
            "step_action",
            model_family="trackvla_pp_lite",
            evaluation_tier="exploratory",
        )
    assert resolve_label_mode_for_evaluation(
        "step_action",
        None,
        model_family="pfem_harness",
        evaluation_tier="locked_final",
    ) == ("step_action", False)


def test_unknown_label_mode_override_name_is_rejected():
    assert validate_mode_override_names({"H0": "absolute"}, ["H0"]) is True
    with pytest.raises(ValueError, match="do not match any --ckpt run"):
        validate_mode_override_names({"H0_typo": "absolute"}, ["H0"])
