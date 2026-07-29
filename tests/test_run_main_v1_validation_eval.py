import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import shutil
import subprocess
import sys
import types

import pytest

import scripts.eval_offline as eval_offline
import scripts.run_main_v1_validation_eval as controller

from scripts.run_main_v1_validation_eval import (
    CACHE_MANIFEST_SHA256,
    CACHE_PAYLOAD_SHA256,
    CACHE_PROVENANCE_SHA256,
    DINO_SHA256,
    EVALUATOR_FILE_SHA256,
    EVALUATOR_ISOLATED_BOOTSTRAP,
    EVALUATOR_SOURCE_SHA256,
    EXPECTED_B0_MODEL_STATE_SHA256,
    METRIC_CONTRACT_SHA256,
    REGISTRY_SHA256,
    RUN_SPECS,
    SIGLIP_SHA256,
    SOURCE_TREE_SHA256,
    VAL_DATA_SHA256,
    VAL_EPISODES,
    VAL_MANIFEST_SHA256,
    VAL_SAMPLE_COUNT,
    ValidationEvalError,
    _prediction_line_count,
    append_hermetic_stage_import_roots,
    assert_no_competing_processes,
    assert_controller_sha256,
    build_eval_command,
    build_evaluator_environment,
    build_preflight_document,
    build_stage_source_contract,
    create_hermetic_stage,
    deterministic_model_state_sha256,
    global_execution_lock,
    independent_build_runtime_isolation_contract,
    load_frozen_checkpoint_inventory,
    preflight_checkpoints,
    reject_preloaded_reserved_local_modules,
    validate_static_contracts,
    validate_run_artifacts,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _predecessor_receipt_payload(project_root: Path | None = None):
    path = controller.PROJECT_ROOT / controller.PREDECESSOR_FAILURE_RECEIPT_RELATIVE
    document = json.loads(path.read_text(encoding="utf-8"))
    if project_root is None:
        return document

    sealed_root = str(controller.PROJECT_ROOT.resolve())
    replacement_root = str(project_root.resolve())

    def rebind(value):
        if isinstance(value, str):
            return value.replace(sealed_root, replacement_root)
        if isinstance(value, list):
            return [rebind(item) for item in value]
        if isinstance(value, dict):
            return {key: rebind(item) for key, item in value.items()}
        return value

    return rebind(document)


def _bind_predecessor_canonical_hashes(monkeypatch, receipt) -> None:
    for constant, field in (
        ("SEALED_V11_FAILURE_ARGV_SHA256", "argv"),
        ("SEALED_V11_FAILURE_PARTIAL_ARTIFACTS_SHA256", "partial_artifacts"),
        ("SEALED_V11_FAILURE_ERROR_SHA256", "error"),
    ):
        monkeypatch.setattr(
            controller,
            constant,
            controller.canonical_json_sha256(receipt[field]),
        )


def _cleanup_stage(binding) -> None:
    stage_root = binding["stage_root"]
    os.chmod(stage_root, 0o755)
    for path in stage_root.rglob("*"):
        if path.is_dir():
            os.chmod(path, 0o755)
        else:
            os.chmod(path, 0o644)
    shutil.rmtree(stage_root)


@pytest.fixture(scope="module")
def hermetic_stage():
    binding = create_hermetic_stage(controller.PROJECT_ROOT)
    yield binding
    _cleanup_stage(binding)


def _copy_stage_sources(destination: Path) -> Path:
    for source in controller._stage_source_paths(controller.PROJECT_ROOT):
        relative = source.relative_to(controller.PROJECT_ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    return destination


def _write_artifacts(tmp_path: Path, *, mutation=None, batch_size=1):
    spec = RUN_SPECS[0]
    output = tmp_path / "out"
    for directory in ("json", "predictions", "logs"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha(checkpoint)
    predictions = output / "predictions" / f"{spec.run_id}.jsonl"
    log = output / "logs" / f"{spec.run_id}.log"
    log.write_text("ok\n", encoding="utf-8")
    records = []
    predictions_for_metrics = []
    episode_size = VAL_SAMPLE_COUNT // len(VAL_EPISODES)
    for index in range(VAL_SAMPLE_COUNT):
        episode_index = min(index // episode_size, len(VAL_EPISODES) - 1)
        episode = VAL_EPISODES[episode_index]
        frame_idx = index - episode_index * episode_size
        records.append(
            {
                "step_actions": [[0.6, 0.0, 0.0] for _ in range(8)],
                "prev_action": [0.0, 0.0, 0.0],
                "valid_mask": [True] * 8,
                "transition_type": "steady_forward",
                "episode": f"collected_{episode}__chunk{episode_index:03d}",
                "source_raw_dir": episode,
                "sequence_id": episode,
                "chunk_id": f"{episode}:chunk0",
                "clip_id": f"{episode}:clip0",
                "frame_idx": frame_idx,
                "mirrored": False,
                "command": "forward",
            }
        )
        predictions_for_metrics.append([[0.0, 0.0, 0.0] for _ in range(8)])
    predictions.write_text(
        "".join(
            json.dumps(
                {**record, "pred_step_actions": prediction},
                sort_keys=True,
            )
            + "\n"
            for record, prediction in zip(records, predictions_for_metrics)
        ),
        encoding="utf-8",
    )
    prediction_sha = _sha(predictions)
    metrics = eval_offline.evaluate_predictions(predictions_for_metrics, records)
    metrics.update(
        {
            "checkpoint_experiment_id": spec.experiment_id,
            "experiment_id": spec.experiment_id,
            "seed": spec.seed,
            "model_family": spec.model_family,
            "state_mode": "stateless",
            "evaluation_class": "validation",
            "headline_eligible": False,
        }
    )
    execution_contract = {
        "schema_version": 1,
        "evaluation_data": {
            "split": "val",
            "manifest_sha256": VAL_MANIFEST_SHA256,
            "data_sha256": VAL_DATA_SHA256,
            "sample_count": VAL_SAMPLE_COUNT,
        },
        "observation": {
            "history": 31,
            "n_waypoints": 8,
            "dt": 0.1,
        },
        "state": {
            "declared_mode": "stateless",
            "effective_mode": "stateless",
            "override": False,
            "sequence_id_required": False,
            "reset_policy": "functionally_stateless",
        },
        "label": {
            "declared_mode": "absolute",
            "effective_mode": "absolute",
            "override": False,
        },
        "loader": {
            "batch_size": batch_size,
            "shuffle": False,
            "num_workers": 0,
            "ordered_record_validation": True,
        },
        "runtime": {
            "device": "mps",
            "device_type": "mps",
            "torch_default_dtype": "torch.float32",
            "parameter_dtypes": ["torch.bfloat16", "torch.float32"],
            "buffer_dtypes": ["torch.float32"],
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
        "evaluation_identity": {
            "tier": "locked_final",
            "class": "validation",
        },
    }
    payload = {
        spec.run_id: {
            "checkpoint": str(checkpoint),
            "label_mode": "absolute",
            "metrics": metrics,
            "provenance": {
                "evaluation_tier": "locked_final",
                "evaluation_class": "validation",
                "headline_eligible": False,
                "state_mode_override": False,
                "checkpoint_role": "best_validation",
                "selection_verified": True,
                "checkpoint_seed": spec.seed,
                "checkpoint_sha256": checkpoint_sha,
                "experiment_registry_sha256": REGISTRY_SHA256,
                "source_tree_sha256": SOURCE_TREE_SHA256,
                "evaluator_source_sha256": EVALUATOR_SOURCE_SHA256,
                "metric_contract_sha256": METRIC_CONTRACT_SHA256,
                "validation_manifest_sha256": VAL_MANIFEST_SHA256,
                "validation_data_sha256": VAL_DATA_SHA256,
                "vision_cache_manifest_sha256": CACHE_MANIFEST_SHA256,
                "vision_cache_provenance_sha256": CACHE_PROVENANCE_SHA256,
                "vision_cache_token_payload_sha256": CACHE_PAYLOAD_SHA256,
                "dino_model_sha256": DINO_SHA256,
                "siglip_model_sha256": SIGLIP_SHA256,
                "test_manifest_sha256": None,
                "evaluation_predictions_sha256": prediction_sha,
                "evaluation_execution_contract_sha256": (
                    controller.canonical_json_sha256(execution_contract)
                ),
                "evaluation_execution_contract": execution_contract,
            },
        }
    }
    if mutation is not None:
        mutation(payload, predictions)
    result = output / "json" / f"{spec.run_id}.val.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return spec, output, checkpoint_sha


def _mock_static_tree(tmp_path, monkeypatch):
    registry = tmp_path / controller.REGISTRY_RELATIVE
    validation = tmp_path / controller.VAL_RELATIVE
    validation_manifest = tmp_path / controller.VAL_MANIFEST_RELATIVE
    cache_manifest = tmp_path / controller.CACHE_RELATIVE / "cache_manifest.json"
    for path in (registry, validation, validation_manifest, cache_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text("{}", encoding="utf-8")
    validation.write_text("{}\n", encoding="utf-8")
    validation_manifest.write_text("{}", encoding="utf-8")
    cache_manifest.write_text(
        json.dumps(
            {
                "provenance_sha256": CACHE_PROVENANCE_SHA256,
                "token_payload_sha256": CACHE_PAYLOAD_SHA256,
                "dino_model_sha256": DINO_SHA256,
                "siglip_model_sha256": SIGLIP_SHA256,
            }
        ),
        encoding="utf-8",
    )
    cache_manifest_sha = _sha(cache_manifest)
    monkeypatch.setattr(controller, "CACHE_MANIFEST_SHA256", cache_manifest_sha)
    artifact_dirs = [tmp_path / name for name in ("base", "qwen", "siglip")]
    for path in artifact_dirs:
        path.mkdir()
    dino_dir = tmp_path / "weights/modelscope/dinov3-vits16-pretrain-lvd1689m"
    dino_dir.mkdir(parents=True)
    monkeypatch.setattr(controller, "BASE_MODEL_DIR", artifact_dirs[0])
    monkeypatch.setattr(controller, "QWEN_MODEL_DIR", artifact_dirs[1])
    monkeypatch.setattr(controller, "SIGLIP_MODEL_DIR", artifact_dirs[2])
    expected_hashes = {
        registry.resolve(): REGISTRY_SHA256,
        validation.resolve(): VAL_DATA_SHA256,
        validation_manifest.resolve(): VAL_MANIFEST_SHA256,
        cache_manifest.resolve(): cache_manifest_sha,
        (tmp_path / "scripts/eval_offline.py").resolve(): EVALUATOR_FILE_SHA256,
    }
    monkeypatch.setattr(
        controller,
        "sha256_file",
        lambda path: expected_hashes[Path(path).resolve()],
    )
    monkeypatch.setattr(
        controller,
        "independent_source_tree_sha256",
        lambda _: SOURCE_TREE_SHA256,
    )
    monkeypatch.setattr(
        controller,
        "independent_build_evaluator_source",
        lambda _: {"contract": "evaluator"},
    )
    monkeypatch.setattr(
        controller,
        "independent_build_metric_contract",
        lambda: {"contract": "metric"},
    )
    monkeypatch.setattr(
        controller,
        "independent_import_surface",
        lambda path: {
            "contract": (
                "opentrack-surface"
                if Path(path).resolve()
                == (tmp_path / "third_party/OpenTrackVLA").resolve()
                else "project-surface"
            )
        },
    )
    monkeypatch.setattr(
        controller,
        "build_stage_source_contract",
        lambda _: {"contract": "stage-source"},
    )
    monkeypatch.setattr(
        controller,
        "canonical_json_sha256",
        lambda value: {
            "evaluator": EVALUATOR_SOURCE_SHA256,
            "metric": METRIC_CONTRACT_SHA256,
            "stage-source": hashlib.sha256(b"stage-source").hexdigest(),
        }.get(
            value.get("contract") if isinstance(value, dict) else None,
            hashlib.sha256(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
        ),
    )
    return cache_manifest


def _mock_checkpoint_preflight(tmp_path, monkeypatch, *, duplicate_non_b0=False):
    specs_by_path = {}
    support = {
        "test0012": {"forward": 855, "turn_left": 34, "turn_right": 33},
        "test015": {
            "backward": 35,
            "forward": 784,
            "turn_left": 125,
            "turn_right": 18,
        },
        "test0189": {"forward": 865, "turn_left": 66, "turn_right": 31},
    }
    for spec in RUN_SPECS:
        path = tmp_path / spec.checkpoint_relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(spec.run_id.encode("utf-8"))
        selected_value = 0.2 if spec.experiment_id == "B0" else 0.2 + spec.seed / 100
        batch_size = 2 if spec.experiment_id == "B0" else 1
        (path.parent / "metrics.jsonl").write_text(
            "\n".join(
                (
                    json.dumps(
                        {
                            "phase": "run_start",
                            "checkpoint_meta": {
                                "batch_size": batch_size,
                                "checkpoint_selection": {
                                    "metric": "validation_episode_macro_BCE@1",
                                    "mode": "min",
                                    "rule": "strict_improvement_earliest_epoch",
                                },
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "phase": "validation",
                            "epoch": 0,
                            "selection_detail": {
                                "value": selected_value,
                                "by_episode": {
                                    episode: selected_value for episode in VAL_EPISODES
                                },
                                "support": support,
                            },
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        specs_by_path[path.resolve()] = spec

    monkeypatch.setattr(
        eval_offline,
        "load_experiment_registry",
        lambda *_: ({"status": "frozen"}, REGISTRY_SHA256),
    )
    monkeypatch.setattr(
        eval_offline,
        "validate_checkpoint_metadata",
        lambda checkpoint: checkpoint["meta"],
    )
    monkeypatch.setattr(
        eval_offline, "validate_frozen_test_checkpoint", lambda *_, **__: True
    )
    monkeypatch.setattr(
        eval_offline, "validate_registry_checkpoint_binding", lambda *_, **__: {}
    )

    def load_checkpoint(path):
        spec = specs_by_path[Path(path).resolve()]
        selected_value = 0.2 if spec.experiment_id == "B0" else 0.2 + spec.seed / 100
        meta = {
            "experiment_id": spec.experiment_id,
            "model_family": spec.model_family,
            "seed": spec.seed,
            "checkpoint_role": "best_validation",
            "selection_verified": True,
            "selected_epoch": 0,
            "selected_value": selected_value,
            "validation": {
                "data_jsonl_sha256": VAL_DATA_SHA256,
                "data_manifest_hash": VAL_MANIFEST_SHA256,
                "sample_count": VAL_SAMPLE_COUNT,
            },
        }
        return {
            "meta": meta,
            "epoch": 0,
            "model_state": {"run_id": spec.run_id},
        }, hashlib.sha256(spec.run_id.encode()).hexdigest()

    def model_state_sha(model_state):
        run_id = model_state["run_id"]
        if run_id.startswith("B0_seed"):
            return EXPECTED_B0_MODEL_STATE_SHA256
        if duplicate_non_b0 and run_id.startswith("B1_seed"):
            return "d" * 64
        return hashlib.sha256(f"state:{run_id}".encode()).hexdigest()

    monkeypatch.setattr(eval_offline, "load_checkpoint_with_sha256", load_checkpoint)
    monkeypatch.setattr(
        controller, "load_verified_evaluator_module", lambda _: eval_offline
    )
    monkeypatch.setattr(
        controller,
        "independent_source_tree_sha256",
        lambda _: SOURCE_TREE_SHA256,
    )
    monkeypatch.setattr(controller, "deterministic_model_state_sha256", model_state_sha)


def test_eval_command_is_validation_only_and_has_no_override(tmp_path, hermetic_stage):
    command = build_eval_command(
        RUN_SPECS[0],
        project_root=tmp_path,
        output_root=tmp_path / "out",
        stage_binding=hermetic_stage,
    )
    joined = " ".join(command)
    assert command[0] == sys.executable
    assert Path(command[0]).is_absolute()
    assert command[1:8:1] == [
        "-I",
        "-S",
        "-B",
        "-u",
        "-X",
        f"pycache_prefix={hermetic_stage['stage_root'] / '.pycache-disabled'}",
        "-c",
    ]
    assert command[8] == EVALUATOR_ISOLATED_BOOTSTRAP
    assert command[9] == str(hermetic_stage["manifest_path"])
    assert command[10] == hermetic_stage["manifest_sha256"]
    assert command[11] == str(hermetic_stage["stage_root"] / "scripts/eval_offline.py")
    assert "--val_json" in command
    assert "data/collected_v1/datasets/val.jsonl" in joined
    assert "--evaluation_tier locked_final" in joined
    assert "--batch_size 1" in joined
    assert "--mode" not in command
    assert "--state_mode" not in command
    assert "test.jsonl" not in joined
    assert "test_manifest" not in joined


def test_selection_replay_command_is_batch2_and_uses_dedicated_tree(
    tmp_path, hermetic_stage
):
    replay_root = tmp_path / "selection_replay"
    command = build_eval_command(
        RUN_SPECS[0],
        project_root=tmp_path,
        output_root=replay_root,
        stage_binding=hermetic_stage,
        batch_size=2,
    )
    joined = " ".join(command)
    assert "--batch_size 2" in joined
    assert str(replay_root / "json/B0_seed0.val.json") in command
    assert str(replay_root / "predictions") in command
    assert "test.jsonl" not in joined
    with pytest.raises(ValidationEvalError, match="batch_size must be 1 or 2"):
        build_eval_command(
            RUN_SPECS[0],
            project_root=tmp_path,
            output_root=replay_root,
            stage_binding=hermetic_stage,
            batch_size=3,
        )


def test_controller_sha_must_be_explicit_and_exact():
    import scripts.run_main_v1_validation_eval as module

    actual = _sha(Path(module.__file__))
    assert assert_controller_sha256(actual) == actual
    with pytest.raises(ValidationEvalError, match="explicit SHA-256"):
        assert_controller_sha256("not-a-sha")
    with pytest.raises(ValidationEvalError, match="controller SHA-256 mismatch"):
        assert_controller_sha256("0" * 64)


def test_predecessor_failure_receipt_requires_exact_path_and_literal_sha(
    tmp_path, monkeypatch
):
    receipt_path = tmp_path / controller.PREDECESSOR_FAILURE_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True)
    receipt = _predecessor_receipt_payload(tmp_path)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = _sha(receipt_path)
    monkeypatch.setattr(controller, "PREDECESSOR_FAILURE_RECEIPT_SHA256", receipt_sha)
    _bind_predecessor_canonical_hashes(monkeypatch, receipt)
    binding = controller.validate_v11_full_failure_receipt(
        tmp_path, receipt_path, receipt_sha
    )
    assert binding["sha256"] == receipt_sha
    assert binding["phase"] == "full"
    assert binding["status"] == "failed_closed"
    assert binding["internal_test_opened"] is False
    assert set(binding) == {"path", "sha256", *receipt}
    with pytest.raises(ValidationEvalError, match="literal SHA-256"):
        controller.validate_v11_full_failure_receipt(
            tmp_path, receipt_path, "0" * 64
        )
    alias = tmp_path / "alias.json"
    alias.write_bytes(receipt_path.read_bytes())
    with pytest.raises(ValidationEvalError, match="receipt path mismatch"):
        controller.validate_v11_full_failure_receipt(tmp_path, alias, receipt_sha)


def test_real_v11_full_failure_receipt_is_the_only_v12_predecessor():
    binding = controller.validate_v11_full_failure_receipt(
        controller.PROJECT_ROOT,
        controller.PROJECT_ROOT / controller.PREDECESSOR_FAILURE_RECEIPT_RELATIVE,
        controller.PREDECESSOR_FAILURE_RECEIPT_SHA256,
    )
    assert binding["formal_version"] == "v11"
    assert binding["phase"] == "full"
    assert binding["exit_code"] == 1
    assert binding["error_type"] == "ValidationEvalError"
    assert binding["error"] == "failed to parse argv for guarded process launcher"
    assert binding["controller_expected_sha256"] == (
        controller.SEALED_V11_CONTROLLER_SHA256
    )
    assert binding["checkpoint_inventory_expected_sha256"] == (
        controller.SEALED_V11_INVENTORY_SHA256
    )
    assert binding["stage_manifest_expected_sha256"] == (
        controller.SEALED_V11_STAGE_MANIFEST_SHA256
    )
    assert binding["stage_source_contract_expected_sha256"] == (
        controller.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256
    )
    assert binding["runtime_isolation_contract_expected_sha256"] == (
        controller.SEALED_V11_RUNTIME_ISOLATION_CONTRACT_SHA256
    )
    assert binding["predecessor_failure_receipt_expected_sha256"] == (
        controller.SEALED_V10_FAILURE_RECEIPT_SHA256
    )
    assert binding["preflight_output"] is None
    assert binding["required_python_options"] == ["-I", "-S", "-B", "-u"]
    assert Path(binding["predecessor_failure_receipt"]).name == (
        "20260718_validation_v10_full_failure_receipt.json"
    )
    assert binding["partial_artifacts"]["output_dir"]["exists"] is False
    assert binding["partial_artifacts"]["checkpoint_inventory"]["sha256"] == (
        controller.SEALED_V11_INVENTORY_SHA256
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda receipt: receipt.update({"schema_version": 2}),
            r"predecessor_failure_receipt\.schema_version mismatch",
        ),
        (
            lambda receipt: receipt.update({"analysis_class": "other_receipt"}),
            r"predecessor_failure_receipt\.analysis_class mismatch",
        ),
        (
            lambda receipt: receipt.update({"status": "completed"}),
            r"predecessor_failure_receipt\.status mismatch",
        ),
        (
            lambda receipt: receipt.update({"formal_version": "v10"}),
            r"predecessor_failure_receipt\.formal_version mismatch",
        ),
        (
            lambda receipt: receipt.update({"phase": "preflight"}),
            "phase mismatch",
        ),
        (
            lambda receipt: receipt.update({"exit_code": 0}),
            r"predecessor_failure_receipt\.exit_code mismatch",
        ),
        (
            lambda receipt: receipt.update({"error_type": "RuntimeError"}),
            r"predecessor_failure_receipt\.error_type mismatch",
        ),
        (
            lambda receipt: receipt.update({"controller_expected_sha256": "0" * 64}),
            r"predecessor_failure_receipt\.controller_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"checkpoint_inventory_expected_sha256": "0" * 64}
            ),
            "checkpoint_inventory_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"stage_manifest_expected_sha256": "0" * 64}
            ),
            "stage_manifest_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"stage_source_contract_expected_sha256": "0" * 64}
            ),
            "stage_source_contract_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"runtime_isolation_contract_expected_sha256": "0" * 64}
            ),
            "runtime_isolation_contract_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"predecessor_failure_receipt_expected_sha256": "0" * 64}
            ),
            "predecessor_failure_receipt_expected_sha256 mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"preflight_output": "unexpected_preflight.json"}
            ),
            r"predecessor_failure_receipt\.preflight_output mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {"required_python_options": ["-I", "-S", "-B"]}
            ),
            r"predecessor_failure_receipt\.required_python_options mismatch",
        ),
        (
            lambda receipt: receipt.update({"internal_test_opened": True}),
            "internal_test_opened mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {
                    "checkpoint_inventory": receipt["checkpoint_inventory"].replace(
                        "validation_checkpoint_inventory_v11",
                        "validation_checkpoint_inventory_v10",
                    )
                }
            ),
            "v11 checkpoint inventory path mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {
                    "output_dir": receipt["output_dir"].replace(
                        "validation_eval_v11", "validation_eval_v10"
                    )
                }
            ),
            "v11 output path mismatch",
        ),
        (
            lambda receipt: receipt.update(
                {
                    "stage_manifest": (
                        "/private/tmp/track_car_main_v1_stage-other/"
                        "main_v1_stage_manifest.json"
                    )
                }
            ),
            "v11 stage manifest path mismatch",
        ),
        (
            lambda receipt: receipt["argv"].append("--internal_test"),
            "v11 failure argv binding mismatch",
        ),
        (
            lambda receipt: receipt["partial_artifacts"]["output_dir"].update(
                {"exists": True}
            ),
            "v11 partial artifact binding mismatch",
        ),
        (
            lambda receipt: receipt.update({"error": "different failure"}),
            "v11 guard parse-failure error binding mismatch",
        ),
        (
            lambda receipt: receipt.update({"unexpected": True}),
            r"predecessor_failure_receipt\.keys mismatch",
        ),
    ),
)
def test_v12_predecessor_receipt_mutations_fail_closed(
    tmp_path, monkeypatch, mutation, match
):
    receipt_path = tmp_path / controller.PREDECESSOR_FAILURE_RECEIPT_RELATIVE
    receipt_path.parent.mkdir(parents=True)
    receipt = _predecessor_receipt_payload(tmp_path)
    _bind_predecessor_canonical_hashes(monkeypatch, receipt)
    mutation(receipt)
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = _sha(receipt_path)
    monkeypatch.setattr(controller, "PREDECESSOR_FAILURE_RECEIPT_SHA256", receipt_sha)
    with pytest.raises(ValidationEvalError, match=match):
        controller.validate_v11_full_failure_receipt(
            tmp_path, receipt_path, receipt_sha
        )


def test_v12_paths_are_exact_and_v3_through_v11_are_rejected(tmp_path):
    output = tmp_path / controller.DEFAULT_OUTPUT_RELATIVE
    inventory = tmp_path / controller.DEFAULT_INVENTORY_RELATIVE
    controller.validate_fresh_v12_paths(
        project_root=tmp_path,
        output_root=output,
        inventory_path=inventory,
        preflight_output=inventory,
    )
    for version in ("v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11"):
        with pytest.raises(
            ValidationEvalError, match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 path"
        ):
            controller.validate_fresh_v12_paths(
                project_root=tmp_path,
                output_root=(
                    tmp_path
                    / f"experiments/collected_v1_main/validation_eval_{version}"
                ),
            )
        with pytest.raises(
            ValidationEvalError, match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 path"
        ):
            controller.validate_fresh_v12_paths(
                project_root=tmp_path,
                output_root=output,
                inventory_path=(
                    tmp_path / "experiments/collected_v1_main/"
                    f"validation_checkpoint_inventory_{version}.json"
                ),
            )


def test_v12_own_stage_and_lifecycle_sha_are_not_self_denied(tmp_path):
    fresh_stage = tmp_path / "track_car_main_v1_stage-v12fresh"
    controller._reject_sealed_stage_root(fresh_stage, "fresh v12 stage")
    fresh_sha = "f" * 64
    for sealed_set in (
        controller.SEALED_CONTROLLER_SHA256S,
        controller.SEALED_INVENTORY_SHA256S,
        controller.SEALED_RUN_MANIFEST_SHA256S,
        controller.SEALED_REPLAY_MANIFEST_SHA256S,
        controller.SEALED_EVAL_SUMS_SHA256S,
        controller.SEALED_STAGE_MANIFEST_SHA256S,
        controller.SEALED_STAGE_TREE_SHA256S,
        controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
    ):
        controller._reject_sealed_sha256(fresh_sha, sealed_set, "fresh v12 identity")


def test_v12_phase_receipt_paths_are_exact_and_pair_required(tmp_path):
    success = tmp_path / controller.PHASE_RECEIPT_RELATIVES["preflight"]["success"]
    failure = tmp_path / controller.PHASE_RECEIPT_RELATIVES["preflight"]["failure"]
    args = types.SimpleNamespace(
        preflight_only=True,
        command_receipt_output=str(success),
        failure_output=str(failure),
    )
    controller.validate_phase_receipt_cli(args, tmp_path)
    args.failure_output = None
    with pytest.raises(ValidationEvalError, match="must be supplied together"):
        controller.validate_phase_receipt_cli(args, tmp_path)


def test_v12_one_shot_failure_receipt_records_error_and_refuses_overwrite(tmp_path):
    failure = tmp_path / controller.PHASE_RECEIPT_RELATIVES["preflight"]["failure"]
    argv = [
        "run_main_v1_validation_eval.py",
        "--project_root",
        str(tmp_path),
        "--preflight_only",
        "--failure_output",
        str(failure),
        "--expected_controller_sha256",
        "a" * 64,
        "--predecessor_failure_receipt",
        "receipt.json",
        "--expected_predecessor_failure_receipt_sha256",
        controller.PREDECESSOR_FAILURE_RECEIPT_SHA256,
    ]
    controller.write_phase_failure_receipt(RuntimeError("boom"), argv)
    document = json.loads(failure.read_text(encoding="utf-8"))
    assert document["formal_version"] == "v12"
    assert document["phase"] == "preflight"
    assert document["status"] == "failed_closed"
    assert document["error_type"] == "RuntimeError"
    assert document["error"] == "boom"
    assert document["internal_test_opened"] is False
    with pytest.raises(ValidationEvalError, match="already exists"):
        controller.write_phase_failure_receipt(RuntimeError("again"), argv)


def test_v12_one_shot_success_receipt_binds_preflight_inventory(tmp_path, monkeypatch):
    success = tmp_path / controller.PHASE_RECEIPT_RELATIVES["preflight"]["success"]
    inventory = tmp_path / "validation_checkpoint_inventory_v12.json"
    inventory.write_text("{}\n", encoding="utf-8")
    argv = [
        "run_main_v1_validation_eval.py",
        "--project_root",
        str(tmp_path),
        "--preflight_only",
        "--preflight_output",
        str(inventory),
        "--command_receipt_output",
        str(success),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = types.SimpleNamespace(
        project_root=str(tmp_path),
        preflight_only=True,
        preflight_output=str(inventory),
        output_dir=str(tmp_path / controller.DEFAULT_OUTPUT_RELATIVE),
        command_receipt_output=str(success),
    )
    controller.write_phase_success_receipt(args, {"status": "preflight_passed"})
    document = json.loads(success.read_text(encoding="utf-8"))
    assert document["status"] == "completed"
    assert document["primary_artifact"]["sha256"] == _sha(inventory)
    assert document["internal_test_opened"] is False


@pytest.mark.parametrize(
    ("digest", "sealed_set", "field"),
    (
        (
            controller.SEALED_V3_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.PREDECESSOR_ROOT_FIX_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V5_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V6_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V3_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.PREDECESSOR_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V6_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V6_RUN_MANIFEST_SHA256,
            controller.SEALED_RUN_MANIFEST_SHA256S,
            "run manifest",
        ),
        (
            controller.SEALED_V6_REPLAY_MANIFEST_SHA256,
            controller.SEALED_REPLAY_MANIFEST_SHA256S,
            "selection replay manifest",
        ),
        (
            controller.SEALED_V6_EVAL_SUMS_SHA256,
            controller.SEALED_EVAL_SUMS_SHA256S,
            "eval sums",
        ),
        (
            controller.SEALED_V3_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.PREDECESSOR_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V5_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V5_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V5_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V6_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V6_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V6_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V7_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V7_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V7_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V7_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V7_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V8_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V8_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V8_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V8_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V8_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V9_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V9_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V9_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V9_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V9_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V10_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V10_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V10_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V10_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V10_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
        (
            controller.SEALED_V11_CONTROLLER_SHA256,
            controller.SEALED_CONTROLLER_SHA256S,
            "controller",
        ),
        (
            controller.SEALED_V11_INVENTORY_SHA256,
            controller.SEALED_INVENTORY_SHA256S,
            "checkpoint inventory",
        ),
        (
            controller.SEALED_V11_STAGE_MANIFEST_SHA256,
            controller.SEALED_STAGE_MANIFEST_SHA256S,
            "stage manifest",
        ),
        (
            controller.SEALED_V11_STAGE_TREE_SHA256,
            controller.SEALED_STAGE_TREE_SHA256S,
            "stage tree",
        ),
        (
            controller.SEALED_V11_STAGE_SOURCE_CONTRACT_SHA256,
            controller.SEALED_STAGE_SOURCE_CONTRACT_SHA256S,
            "stage source contract",
        ),
    ),
)
def test_v12_rejects_all_sealed_v3_through_v11_lifecycle_shas(
    digest, sealed_set, field
):
    with pytest.raises(
        ValidationEvalError, match="sealed v3/v4/v5/v6/v7/v8/v9/v10/v11 SHA-256"
    ):
        controller._reject_sealed_sha256(digest, sealed_set, field)


def test_v12_runtime_command_and_environment_literals_are_frozen():
    assert controller.PROCESS_LISTING_COMMAND == (
        "/bin/ps",
        "-axo",
        "pid=,ucomm=,args=",
    )
    assert controller.FORMAL_PATH == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert controller.FORMAL_ENVIRONMENT == {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/Users/mythrise",
        "TMPDIR": "/private/tmp",
        "XDG_CACHE_HOME": "/Users/mythrise/.cache",
    }


@pytest.mark.parametrize("sealed_root", sorted(controller.SEALED_STAGE_ROOTS))
def test_stage_verifier_rejects_exact_sealed_v5_through_v11_stage_roots(sealed_root):
    with pytest.raises(ValidationEvalError, match="verified stage reuses"):
        controller.verify_hermetic_stage(
            Path(sealed_root) / controller.STAGE_MANIFEST_FILENAME, "a" * 64
        )


def test_v12_runtime_guard_supersedes_v8_substring_audit():
    audit_path = (
        controller.PROJECT_ROOT / "experiments/collected_v1_main/external_reviews/"
        "20260717_validation_v8_runtime_guard_audit.json"
    )
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit["formal_version"] == "v8"
    assert audit["predecessor_failure_receipt_sha256"] == (
        "5ae212d6af80c666b7572ae68a0d058dd963be7dfa4375c0e3c76244c31ebde6"
    )
    assert set(audit["forbidden_process_scripts"]) == set(
        controller.FORBIDDEN_FORMAL_PROCESS_SCRIPTS
    )
    assert "cli_introspection_including_help" in audit["forbidden_modes"]
    assert audit["guard_checkpoints"] == [
        "before_preflight",
        "before_each_formal_cell",
        "before_selection_replay",
    ]
    assert audit["weakening_allowed"] is False
    assert audit["internal_test_opened"] is False

    guard = """
emulate -LR zsh
setopt ERR_EXIT NO_UNSET PIPE_FAIL
EXPECTED_PATH='/usr/bin:/bin:/usr/sbin:/sbin'
export PATH="$EXPECTED_PATH"
readonly EXPECTED_PATH PATH path
candidate_path=''
for candidate_path in /private/tmp/a /private/tmp/b; do
  [[ "$PATH" == "$EXPECTED_PATH" ]]
  [[ "${(j/:/)path}" == "$EXPECTED_PATH" ]]
done
print -r -- "$PATH"
"""
    completed = subprocess.run(
        ["/bin/zsh", "-f", "-c", guard],
        check=False,
        capture_output=True,
        text=True,
        env=controller.FORMAL_ENVIRONMENT,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == controller.FORMAL_PATH

    mutation = """
emulate -LR zsh
EXPECTED_PATH='/usr/bin:/bin:/usr/sbin:/sbin'
export PATH="$EXPECTED_PATH"
readonly EXPECTED_PATH PATH path
path=(/attacker)
"""
    rejected = subprocess.run(
        ["/bin/zsh", "-f", "-c", mutation],
        check=False,
        capture_output=True,
        text=True,
        env=controller.FORMAL_ENVIRONMENT,
    )
    assert rejected.returncode != 0
    assert "read-only variable: path" in rejected.stderr


@pytest.mark.parametrize(
    "command",
    (
        "99991 python scripts/train_baseline.py --seed 0\n",
        "99992 python scripts/train_trackvla_lite.py --seed 0\n",
        "99993 python scripts/train_pfem.py --seed 0\n",
        "99994 python scripts/eval_offline.py --val_json val.jsonl\n",
        "99995 python third_party/OpenTrackVLA/scripts/train_pfem.py --help\n",
        "99996 /bin/zsh -c 'python scripts/eval_offline.py --help | sed -n 1,20p'\n",
    ),
)
def test_process_gate_catches_short_script_paths(monkeypatch, command):
    completed = type("Completed", (), {"stdout": command})()
    monkeypatch.setattr(controller.subprocess, "run", lambda *_, **__: completed)
    with pytest.raises(ValidationEvalError, match="already active"):
        assert_no_competing_processes()


def test_v12_process_gate_allows_real_v8_sky_payload_false_positive(monkeypatch):
    receipt_path = (
        controller.PROJECT_ROOT
        / "experiments/collected_v1_main/external_reviews/"
        "20260717_validation_v8_full_failure_receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    process_listing = receipt["error"].split("already active:\n", 1)[1] + "\n"
    sky_arguments = process_listing.strip().partition(" ")[2]
    assert not controller._process_invocation_invokes_forbidden_script(
        "SkyComputerUseClient", sky_arguments
    )
    completed = type("Completed", (), {"stdout": process_listing})()
    monkeypatch.setattr(controller.subprocess, "run", lambda *_, **__: completed)
    assert_no_competing_processes()


@pytest.mark.parametrize(
    ("executable", "arguments"),
    (
        ("python", 'python -c "unterminated'),
        ("zsh", '/bin/zsh -c "python \'unterminated"'),
    ),
)
def test_v12_process_parser_fails_closed_on_guarded_malformed_quoting(
    executable, arguments
):
    with pytest.raises(ValidationEvalError, match="failed to parse"):
        controller._process_invocation_invokes_forbidden_script(
            executable, arguments
        )


def test_v12_process_parser_leaves_unknown_malformed_payload_opaque():
    assert not controller._process_invocation_invokes_forbidden_script(
        "SkyComputerUseClient",
        '{"payload":"python scripts/train_pfem.py}',
    )


@pytest.mark.parametrize(
    "command",
    (
        "99997 rg -n 'python scripts/train_pfem.py --help' docs/notes.md",
        "99998 echo python scripts/eval_offline.py --help",
        "99999 python -c \"print('scripts/train_pfem.py --help')\"",
        (
            "/Applications/SkyComputerUseClient turn-ended "
            "{\"payload\":\"python scripts/train_pfem.py --help\"}"
        ),
    ),
)
def test_v12_process_parser_ignores_payload_and_search_text(command):
    assert not controller._process_command_invokes_forbidden_script(command)


@pytest.mark.parametrize(
    "command",
    (
        "python scripts/train_pfem.py --seed 0",
        "/opt/conda/bin/python -I -u third_party/OpenTrackVLA/scripts/train_pfem.py --help",
        "python -m scripts.eval_offline --help",
        "/bin/zsh -c 'cd /tmp && python scripts/train_baseline.py --seed 0'",
        "/usr/bin/env HF_HUB_OFFLINE=1 python scripts/train_trackvla_lite.py",
        "conda run --no-capture-output -n pytorch python scripts/train_pfem.py",
        "gtimeout --signal=INT 5m python scripts/eval_offline.py --help",
        "./third_party/OpenTrackVLA/scripts/train_pfem.py --help",
    ),
)
def test_v12_process_parser_detects_actual_invocation_chains(command):
    assert controller._process_command_invokes_forbidden_script(command)


@pytest.mark.parametrize(
    "command",
    (
        (
            "python -c \"import runpy; "
            "runpy.run_path('third_party/OpenTrackVLA/scripts/train_pfem.py')\""
        ),
        "python -c \"import scripts.eval_offline\"",
        (
            "python -c \"import importlib; "
            "importlib.import_module('scripts.eval_offline')\""
        ),
        (
            "python -c \"exec(open("
            "'third_party/OpenTrackVLA/scripts/train_pfem.py').read())\""
        ),
    ),
)
def test_v12_process_parser_detects_python_dash_c_execution(command):
    assert controller._process_command_invokes_forbidden_script(command)


def test_v12_process_gate_detects_real_macos_python_ucomm_line(monkeypatch):
    completed = type(
        "Completed",
        (),
        {
            "stdout": (
                "99990 Python /opt/miniconda3/envs/pytorch/bin/python3.11 "
                "-I -S -B -u scripts/train_pfem.py --seed 0\n"
            )
        },
    )()
    monkeypatch.setattr(controller.subprocess, "run", lambda *_, **__: completed)
    with pytest.raises(ValidationEvalError, match="already active"):
        assert_no_competing_processes()


def test_v12_python_ucomm_family_match_does_not_prepend_duplicate_argv0(
    monkeypatch,
):
    captured = {}

    def inspect(argv):
        captured["argv"] = argv
        return False

    monkeypatch.setattr(controller, "_argv_invokes_forbidden_script", inspect)
    assert not controller._process_invocation_invokes_forbidden_script(
        "Python",
        "/opt/miniconda3/envs/pytorch/bin/python3.11 -I -c \"print('ok')\"",
    )
    assert captured["argv"] == (
        "/opt/miniconda3/envs/pytorch/bin/python3.11",
        "-I",
        "-c",
        "print('ok')",
    )


def test_process_gate_uses_absolute_ps_under_clobbered_path(monkeypatch):
    original_run = controller.subprocess.run
    captured = {}

    def run_bound_command(command, **kwargs):
        captured["command"] = tuple(command)
        captured["environment"] = kwargs["env"]
        poisoned_kwargs = dict(kwargs)
        poisoned_kwargs["env"] = {"PATH": "/definitely-not-a-real-bin-directory"}
        original_run(command, **poisoned_kwargs)
        return type("Completed", (), {"stdout": ""})()

    monkeypatch.setenv("PATH", "/definitely-not-a-real-bin-directory")
    monkeypatch.setattr(controller.subprocess, "run", run_bound_command)
    assert_no_competing_processes()
    assert captured["command"] == controller.PROCESS_LISTING_COMMAND
    assert Path(captured["command"][0]).is_absolute()
    assert captured["environment"] == controller.FORMAL_ENVIRONMENT


def test_process_gate_wraps_bound_ps_execution_failure(monkeypatch):
    def fail(*_args, **_kwargs):
        raise FileNotFoundError("simulated bound tool failure")

    monkeypatch.setattr(controller.subprocess, "run", fail)
    with pytest.raises(ValidationEvalError, match="bound process-listing command"):
        assert_no_competing_processes()


def test_v12_process_gate_retries_transient_parse_failure_then_passes(monkeypatch):
    scans = {"count": 0}
    sleeps = []

    def flaky_scan():
        scans["count"] += 1
        if scans["count"] == 1:
            raise ValidationEvalError(
                "failed to parse argv for guarded process launcher: python"
            )

    monkeypatch.setattr(
        controller, "_assert_no_competing_processes_once", flaky_scan
    )
    monkeypatch.setattr(
        controller.time, "sleep", lambda seconds: sleeps.append(seconds)
    )
    assert_no_competing_processes()
    assert scans["count"] == 2
    assert sleeps == [2.0]


def test_v12_process_gate_fails_closed_after_three_parse_failure_scans(monkeypatch):
    scans = {"count": 0}
    sleeps = []

    def parse_failure_scan():
        scans["count"] += 1
        raise ValidationEvalError(
            "failed to parse argv for guarded process launcher: python"
        )

    monkeypatch.setattr(
        controller, "_assert_no_competing_processes_once", parse_failure_scan
    )
    monkeypatch.setattr(
        controller.time, "sleep", lambda seconds: sleeps.append(seconds)
    )
    with pytest.raises(
        ValidationEvalError,
        match=(
            "guarded process launcher argv remained unparseable across "
            "3 consecutive scans 2.0 s apart; failing closed"
        ),
    ):
        assert_no_competing_processes()
    assert scans["count"] == 3
    assert sleeps == [2.0, 2.0]


def test_v12_process_gate_does_not_retry_forbidden_invocation(monkeypatch):
    scans = {"count": 0}

    def forbidden_scan():
        scans["count"] += 1
        raise ValidationEvalError(
            "trainer/evaluator process is already active:\n"
            "pid=1 executable=python match=forbidden-script-invocation"
        )

    monkeypatch.setattr(
        controller, "_assert_no_competing_processes_once", forbidden_scan
    )
    monkeypatch.setattr(
        controller.time,
        "sleep",
        lambda seconds: pytest.fail("forbidden invocation must not be retried"),
    )
    with pytest.raises(ValidationEvalError, match="already active"):
        assert_no_competing_processes()
    assert scans["count"] == 1


def test_global_lock_rejects_second_controller(tmp_path):
    lock_parent = tmp_path / "experiments/collected_v1_main"
    lock_parent.mkdir(parents=True)
    with global_execution_lock(tmp_path):
        with pytest.raises(ValidationEvalError, match="global lock"):
            with global_execution_lock(tmp_path):
                pass


def test_static_contracts_are_independently_bound(tmp_path, monkeypatch):
    cache_manifest = _mock_static_tree(tmp_path, monkeypatch)
    expected_stage_source = hashlib.sha256(b"stage-source").hexdigest()
    expected_runtime_isolation = controller.canonical_json_sha256(
        independent_build_runtime_isolation_contract()
    )
    contracts = validate_static_contracts(
        tmp_path,
        expected_stage_source_contract_sha256=expected_stage_source,
        expected_runtime_isolation_contract_sha256=expected_runtime_isolation,
    )
    assert contracts["registry_sha256"] == REGISTRY_SHA256
    assert contracts["source_tree_sha256"] == SOURCE_TREE_SHA256
    assert contracts["evaluator_source_sha256"] == EVALUATOR_SOURCE_SHA256
    assert contracts["evaluator_file_sha256"] == EVALUATOR_FILE_SHA256
    assert contracts["stage_source_contract_sha256"] == expected_stage_source
    cache_manifest.write_text(
        json.dumps(
            {
                "provenance_sha256": "0" * 64,
                "token_payload_sha256": CACHE_PAYLOAD_SHA256,
                "dino_model_sha256": DINO_SHA256,
                "siglip_model_sha256": SIGLIP_SHA256,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(controller, "CACHE_MANIFEST_SHA256", _sha(cache_manifest))
    with pytest.raises(ValidationEvalError, match="cache.provenance_sha256 mismatch"):
        validate_static_contracts(
            tmp_path,
            expected_stage_source_contract_sha256=expected_stage_source,
            expected_runtime_isolation_contract_sha256=expected_runtime_isolation,
        )


@pytest.mark.parametrize(
    "missing_option",
    (
        "--expected_stage_source_contract_sha256",
        "--expected_runtime_isolation_contract_sha256",
    ),
)
def test_parser_requires_external_contract_sha256s(missing_option):
    required = {
        "--expected_controller_sha256": "a" * 64,
        "--predecessor_failure_receipt": "predecessor.json",
        "--expected_predecessor_failure_receipt_sha256": "b" * 64,
        "--expected_stage_source_contract_sha256": "c" * 64,
        "--expected_runtime_isolation_contract_sha256": "d" * 64,
    }
    argv = [
        item
        for option, value in required.items()
        if option != missing_option
        for item in (option, value)
    ]
    with pytest.raises(SystemExit) as exc_info:
        controller.build_parser().parse_args(argv)
    assert exc_info.value.code == 2


@pytest.mark.parametrize(
    ("stage_sha256", "runtime_sha256", "match"),
    (
        ("invalid", "d" * 64, "expected_stage_source_contract_sha256"),
        ("c" * 64, "invalid", "expected_runtime_isolation_contract_sha256"),
    ),
)
def test_static_contract_expected_sha256s_must_be_explicit(
    tmp_path, stage_sha256, runtime_sha256, match
):
    with pytest.raises(ValidationEvalError, match=match):
        validate_static_contracts(
            tmp_path,
            expected_stage_source_contract_sha256=stage_sha256,
            expected_runtime_isolation_contract_sha256=runtime_sha256,
        )


def test_static_contract_rejects_sealed_v5_stage_source_contract(tmp_path):
    with pytest.raises(
        ValidationEvalError, match="expected stage source contract reuses"
    ):
        validate_static_contracts(
            tmp_path,
            expected_stage_source_contract_sha256=(
                controller.SEALED_V5_STAGE_SOURCE_CONTRACT_SHA256
            ),
            expected_runtime_isolation_contract_sha256="d" * 64,
        )


def test_static_contract_external_sha256_mismatches_fail_closed(tmp_path, monkeypatch):
    _mock_static_tree(tmp_path, monkeypatch)
    actual_stage = hashlib.sha256(b"stage-source").hexdigest()
    actual_runtime = controller.canonical_json_sha256(
        independent_build_runtime_isolation_contract()
    )
    with pytest.raises(
        ValidationEvalError, match="stage_source_contract_sha256 mismatch"
    ):
        validate_static_contracts(
            tmp_path,
            expected_stage_source_contract_sha256="a" * 64,
            expected_runtime_isolation_contract_sha256=actual_runtime,
        )
    with pytest.raises(
        ValidationEvalError, match="runtime_isolation_contract_sha256 mismatch"
    ):
        validate_static_contracts(
            tmp_path,
            expected_stage_source_contract_sha256=actual_stage,
            expected_runtime_isolation_contract_sha256="b" * 64,
        )


def test_real_static_contracts_are_computed_without_importing_evaluator():
    assert (
        controller.independent_source_tree_sha256(
            controller.PROJECT_ROOT / "third_party/OpenTrackVLA"
        )
        == SOURCE_TREE_SHA256
    )
    assert (
        controller.canonical_json_sha256(
            controller.independent_build_evaluator_source(controller.PROJECT_ROOT)
        )
        == EVALUATOR_SOURCE_SHA256
    )
    assert (
        controller.canonical_json_sha256(controller.independent_build_metric_contract())
        == METRIC_CONTRACT_SHA256
    )
    assert (
        controller.sha256_file(controller.PROJECT_ROOT / "scripts/eval_offline.py")
        == EVALUATOR_FILE_SHA256
    )
    assert len(build_stage_source_contract(controller.PROJECT_ROOT)["files"]) >= 60


def test_hermetic_stage_evaluator_does_not_copy_malicious_pythonpath(
    tmp_path, monkeypatch, hermetic_stage
):
    malicious = tmp_path / "malicious" / "scripts"
    malicious.mkdir(parents=True)
    (malicious / "__init__.py").write_text("", encoding="utf-8")
    (malicious / "eval_offline.py").write_text(
        "raise RuntimeError('PYTHONPATH hijack')\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(malicious.parent))
    staged_evaluator = hermetic_stage["stage_root"] / "scripts/eval_offline.py"
    assert controller.sha256_file(staged_evaluator) == EVALUATOR_FILE_SHA256
    assert malicious.parent.resolve() != hermetic_stage["stage_root"]


def test_checkpoint_preflight_binds_all_nine_and_b0_determinism(tmp_path, monkeypatch):
    _mock_checkpoint_preflight(tmp_path, monkeypatch)
    evaluator_loads = []
    monkeypatch.setattr(
        controller,
        "load_verified_evaluator_module",
        lambda binding: evaluator_loads.append(binding) or eval_offline,
    )
    records, evaluator = preflight_checkpoints(tmp_path, {"stage_root": tmp_path})
    assert evaluator is eval_offline
    assert evaluator_loads == [{"stage_root": tmp_path}]
    assert len(records) == 9
    assert [row["run_id"] for row in records] == [spec.run_id for spec in RUN_SPECS]
    b0_hashes = {
        row["model_state_sha256"] for row in records if row["experiment_id"] == "B0"
    }
    assert b0_hashes == {EXPECTED_B0_MODEL_STATE_SHA256}
    assert len({row["checkpoint_sha256"] for row in records}) == 9
    assert all(len(row["training_metrics_sha256"]) == 64 for row in records)
    assert [row["training_selection"]["batch_size"] for row in records] == [
        2,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    assert all(
        len(row["training_selection"]["selection_detail_sha256"]) == 64
        for row in records
    )


def test_checkpoint_preflight_rejects_non_b0_state_reuse(tmp_path, monkeypatch):
    _mock_checkpoint_preflight(tmp_path, monkeypatch, duplicate_non_b0=True)
    with pytest.raises(ValidationEvalError, match="non-B0 model-state"):
        preflight_checkpoints(tmp_path, {"stage_root": tmp_path})


def test_checkpoint_preflight_stops_unreplayed_multi_epoch_ordering(
    tmp_path, monkeypatch
):
    _mock_checkpoint_preflight(tmp_path, monkeypatch)
    metrics = tmp_path / "experiments/collected_v1_main/B0_seed0/metrics.jsonl"
    with metrics.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "phase": "validation",
                    "epoch": 1,
                    "selection_detail": {
                        "value": 0.1,
                        "by_episode": {episode: 0.1 for episode in VAL_EPISODES},
                        "support": {
                            "test0012": {
                                "forward": 855,
                                "turn_left": 34,
                                "turn_right": 33,
                            },
                            "test015": {
                                "backward": 35,
                                "forward": 784,
                                "turn_left": 125,
                                "turn_right": 18,
                            },
                            "test0189": {
                                "forward": 865,
                                "turn_left": 66,
                                "turn_right": 31,
                            },
                        },
                    },
                }
            )
            + "\n"
        )
    with pytest.raises(ValidationEvalError, match="selection_order_divergence"):
        preflight_checkpoints(tmp_path, {"stage_root": tmp_path})


def test_frozen_inventory_is_external_sha_gate(tmp_path, hermetic_stage):
    records = [
        {"run_id": spec.run_id, "checkpoint_sha256": "a" * 64} for spec in RUN_SPECS
    ]
    contracts = {
        "source_tree_sha256": SOURCE_TREE_SHA256,
        "runtime_isolation_contract_sha256": controller.canonical_json_sha256(
            independent_build_runtime_isolation_contract()
        ),
        "stage_source_contract_sha256": hermetic_stage["source_contract_sha256"],
    }
    predecessor = {
        "path": "/artifact/predecessor.json",
        "sha256": "d" * 64,
        "status": "failed_closed",
    }
    document = build_preflight_document(
        controller_sha256="c" * 64,
        free_bytes=3 * 1024**3,
        static_contracts=contracts,
        checkpoint_records=records,
        stage_binding=hermetic_stage,
        predecessor_failure_receipt=predecessor,
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    expected_sha = _sha(inventory)
    loaded, actual_sha = load_frozen_checkpoint_inventory(
        inventory,
        expected_sha,
        controller_sha256="c" * 64,
        static_contracts=contracts,
        checkpoint_records=records,
        stage_binding=hermetic_stage,
        predecessor_failure_receipt=predecessor,
    )
    assert loaded["runs"] == records
    assert actual_sha == expected_sha
    with pytest.raises(ValidationEvalError, match="inventory.runs mismatch"):
        load_frozen_checkpoint_inventory(
            inventory,
            expected_sha,
            controller_sha256="c" * 64,
            static_contracts=contracts,
            checkpoint_records=records[:-1],
            stage_binding=hermetic_stage,
            predecessor_failure_receipt=predecessor,
        )


def test_inventory_bootstrap_rejects_sealed_v5_stage_source_contract(
    tmp_path, monkeypatch
):
    inventory = tmp_path / "validation_checkpoint_inventory_v12.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "formal_version": controller.FORMAL_VERSION,
                "predecessor_failure_receipt": {
                    "sha256": controller.PREDECESSOR_FAILURE_RECEIPT_SHA256
                },
                "hermetic_stage": {},
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "source_project_root": tmp_path,
        "stage_root": tmp_path / "stage",
        "manifest_path": tmp_path / "stage/manifest.json",
        "manifest_sha256": "a" * 64,
        "source_contract_sha256": (controller.SEALED_V5_STAGE_SOURCE_CONTRACT_SHA256),
        "stage_tree_sha256": "b" * 64,
        "excluded_source_import_artifact_count": 0,
    }
    monkeypatch.setattr(controller, "verify_hermetic_stage", lambda *_: binding)
    with pytest.raises(
        ValidationEvalError, match="inventory stage source contract reuses"
    ):
        controller.load_stage_binding_from_inventory(inventory, _sha(inventory))


def test_inventory_bootstrap_rejects_exact_sealed_v5_stage_root(tmp_path, monkeypatch):
    inventory = tmp_path / "validation_checkpoint_inventory_v12.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "formal_version": controller.FORMAL_VERSION,
                "predecessor_failure_receipt": {
                    "sha256": controller.PREDECESSOR_FAILURE_RECEIPT_SHA256
                },
                "hermetic_stage": {},
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "source_project_root": tmp_path,
        "stage_root": Path(next(iter(controller.SEALED_STAGE_ROOTS))),
        "manifest_path": tmp_path / "stage/manifest.json",
        "manifest_sha256": "a" * 64,
        "source_contract_sha256": "b" * 64,
        "stage_tree_sha256": "c" * 64,
        "excluded_source_import_artifact_count": 0,
    }
    monkeypatch.setattr(controller, "verify_hermetic_stage", lambda *_: binding)
    with pytest.raises(ValidationEvalError, match="inventory stage reuses"):
        controller.load_stage_binding_from_inventory(inventory, _sha(inventory))


def test_model_state_hash_is_order_independent_and_content_sensitive():
    import torch

    first = {
        "b": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "a": torch.tensor([3], dtype=torch.int64),
    }
    reordered = {"a": first["a"].clone(), "b": first["b"].clone()}
    changed = {"a": first["a"].clone(), "b": first["b"].clone() + 1.0}
    assert deterministic_model_state_sha256(first) == deterministic_model_state_sha256(
        reordered
    )
    assert deterministic_model_state_sha256(first) != deterministic_model_state_sha256(
        changed
    )


def test_prediction_jsonl_rejects_blank_or_invalid_rows(tmp_path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    assert _prediction_line_count(path) == 1
    path.write_text('{"ok": true}\n\n', encoding="utf-8")
    with pytest.raises(ValidationEvalError, match="blank prediction"):
        _prediction_line_count(path)
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValidationEvalError, match="invalid prediction JSON"):
        _prediction_line_count(path)


def test_controller_json_and_jsonl_hash_the_parsed_bytes_once(tmp_path, monkeypatch):
    document_path = tmp_path / "document.json"
    document_path.write_text('{"trusted": true}\n', encoding="utf-8")
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text('{"frame": 1}\n', encoding="utf-8")
    expected_document_sha = _sha(document_path)
    expected_predictions_sha = _sha(predictions_path)
    original_open = Path.open
    calls = {document_path.resolve(): 0, predictions_path.resolve(): 0}
    replacements = {
        document_path.resolve(): b'{"trusted": false}\n',
        predictions_path.resolve(): b'{"frame": 999}\n',
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
    document, document_sha = controller._read_json(
        document_path,
        "test document",
        expected_sha256=expected_document_sha,
    )
    records, predictions_sha = controller._read_prediction_jsonl(predictions_path)
    assert document == {"trusted": True}
    assert document_sha == expected_document_sha
    assert records == [{"frame": 1}]
    assert predictions_sha == expected_predictions_sha
    assert calls == {
        document_path.resolve(): 1,
        predictions_path.resolve(): 1,
    }


def test_valid_run_artifacts_close_prediction_and_checkpoint_bytes(tmp_path):
    spec, output, checkpoint_sha = _write_artifacts(tmp_path)
    document = json.loads(
        (output / "json" / f"{spec.run_id}.val.json").read_text(encoding="utf-8")
    )
    assert all(
        "balanced_control_error_at1" not in episode_metrics
        for episode_metrics in document[spec.run_id]["metrics"]["by_episode"].values()
    )
    record = validate_run_artifacts(
        spec,
        output_root=output,
        expected_checkpoint_sha256=checkpoint_sha,
        evaluator_module=eval_offline,
    )
    assert record["run_id"] == spec.run_id
    assert record["prediction_count"] == VAL_SAMPLE_COUNT
    assert record["bce_at1"] == pytest.approx(0.2)
    assert set(record["by_episode_bce_at1"]) == set(VAL_EPISODES)
    assert set(record["prediction_support"]) == set(VAL_EPISODES)
    assert (
        sum(
            sum(command_counts.values())
            for command_counts in record["prediction_support"].values()
        )
        == VAL_SAMPLE_COUNT
    )


def test_batch2_replay_artifact_contract_is_exact(tmp_path):
    spec, output, checkpoint_sha = _write_artifacts(tmp_path, batch_size=2)
    record = validate_run_artifacts(
        spec,
        output_root=output,
        expected_checkpoint_sha256=checkpoint_sha,
        evaluator_module=eval_offline,
        expected_batch_size=2,
    )
    assert record["evaluation_batch_size"] == 2
    with pytest.raises(ValidationEvalError, match="loader.batch_size mismatch"):
        validate_run_artifacts(
            spec,
            output_root=output,
            expected_checkpoint_sha256=checkpoint_sha,
            evaluator_module=eval_offline,
            expected_batch_size=1,
        )


def _mutate_first_prediction(path: Path, mutation) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    mutation(first)
    lines[0] = json.dumps(first, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mutate_first_prediction_and_rebind(payload, path: Path, mutation) -> None:
    _mutate_first_prediction(path, mutation)
    payload[RUN_SPECS[0].run_id]["provenance"]["evaluation_predictions_sha256"] = _sha(
        path
    )


def _mutate_execution_and_rebind(payload, mutation) -> None:
    provenance = payload[RUN_SPECS[0].run_id]["provenance"]
    mutation(provenance["evaluation_execution_contract"])
    provenance["evaluation_execution_contract_sha256"] = (
        controller.canonical_json_sha256(provenance["evaluation_execution_contract"])
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"].update(
                {"evaluation_class": "headline"}
            ),
            "evaluation_class mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["provenance"].update(
                {"test_manifest_sha256": "f" * 64}
            ),
            "test_manifest_sha256 mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "by_episode"
            ].pop(VAL_EPISODES[0]),
            "episodes mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "balanced_control_error_at1"
            ]["by_episode"].pop(VAL_EPISODES[0]),
            "BCE episodes mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "balanced_control_error_at1"
            ].update({"value": "0.2"}),
            "must be a JSON number",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "balanced_control_error_at1"
            ]["by_episode"].update({VAL_EPISODES[0]: True}),
            "must be a JSON number",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "balanced_control_error_at1"
            ].update({"value": 0.3}),
            "prediction-recomputed metrics SHA-256 mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"][
                "balanced_control_error_at1"
            ]["support"][VAL_EPISODES[0]].update({"forward": 1}),
            "prediction-recomputed metrics SHA-256 mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["metrics"]["by_episode"][
                VAL_EPISODES[0]
            ].update({"samples": 1}),
            "prediction-recomputed metrics SHA-256 mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["provenance"].update(
                {"evaluation_execution_contract_sha256": "0" * 64}
            ),
            "evaluation_execution_contract SHA-256 mismatch",
        ),
        (
            lambda payload, _: payload[RUN_SPECS[0].run_id]["provenance"][
                "evaluation_execution_contract"
            ]["evaluation_data"].update({"sample_count": str(VAL_SAMPLE_COUNT)}),
            "execution.sample_count must be an integer",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["loader"].update({"batch_size": True}),
            ),
            "loader.batch_size must be an integer",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["loader"].update({"shuffle": 0}),
            ),
            "loader.shuffle must be a boolean",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["loader"].update(
                    {"ordered_record_validation": 1}
                ),
            ),
            "loader.ordered must be a boolean",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["runtime"].update({"inference_mode": 1}),
            ),
            "runtime.inference_mode must be a boolean",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["runtime"].update(
                    {"device": "cpu", "device_type": "cpu"}
                ),
            ),
            "runtime.device mismatch",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["runtime"].update(
                    {"parameter_dtypes": ["torch.float32"]}
                ),
            ),
            "runtime.parameter_dtypes mismatch",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["observation"].update({"history": 32}),
            ),
            "observation.history mismatch",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["state"].update({"override": 0}),
            ),
            "state.override must be a boolean",
        ),
        (
            lambda payload, _: _mutate_execution_and_rebind(
                payload,
                lambda execution: execution["evaluation_data"].update(
                    {"actual_split": "test"}
                ),
            ),
            "execution.evaluation_data.keys mismatch",
        ),
        (
            lambda payload, predictions: _mutate_first_prediction_and_rebind(
                payload,
                predictions,
                lambda row: row.update({"unexpected": True}),
            ),
            r"prediction\[0\].keys mismatch",
        ),
        (
            lambda payload, predictions: _mutate_first_prediction_and_rebind(
                payload,
                predictions,
                lambda row: row.update(
                    {"episode": "test0012", "source_raw_dir": "not-validation"}
                ),
            ),
            r"prediction\[0\].source_raw_dir is outside validation split",
        ),
        (
            lambda payload, predictions: predictions.write_text(
                predictions.read_text(encoding="utf-8").splitlines()[0] + "\n",
                encoding="utf-8",
            ),
            "prediction_count mismatch",
        ),
    ),
)
def test_run_artifacts_fail_closed_on_protocol_drift(tmp_path, mutation, match):
    spec, output, checkpoint_sha = _write_artifacts(tmp_path, mutation=mutation)
    with pytest.raises(ValidationEvalError, match=match):
        validate_run_artifacts(
            spec,
            output_root=output,
            expected_checkpoint_sha256=checkpoint_sha,
            evaluator_module=eval_offline,
        )


def test_prediction_command_support_groups_by_source_raw_dir():
    prediction_values = [
        {
            "episode": "collected_test002__chunk000",
            "source_raw_dir": "test0012",
            "command": "forward",
        },
        {
            "episode": "collected_test002__chunk001",
            "source_raw_dir": "test0012",
            "command": "turn_left",
        },
        {
            "episode": "collected_test015__chunk000",
            "source_raw_dir": "test015",
            "command": "forward",
        },
        {
            "episode": "collected_test019__chunk000",
            "source_raw_dir": "test0189",
            "command": "turn_right",
        },
    ]

    support = controller._prediction_command_support(prediction_values)

    assert support == {
        "test0012": {"forward": 1, "turn_left": 1},
        "test015": {"forward": 1},
        "test0189": {"turn_right": 1},
    }


def _synthetic_dual_contract_inputs(tmp_path):
    support = {
        "test0012": {"forward": 855, "turn_left": 34, "turn_right": 33},
        "test015": {
            "backward": 35,
            "forward": 784,
            "turn_left": 125,
            "turn_right": 18,
        },
        "test0189": {"forward": 865, "turn_left": 66, "turn_right": 31},
    }
    prediction_support = json.loads(json.dumps(support))
    prediction_support["test0012"]["strafe_right"] = 2
    checkpoint_by_run = {}
    formal_runs = []
    for index, spec in enumerate(RUN_SPECS):
        selected_value = 0.3 if spec.experiment_id == "B0" else 0.4 + index / 100
        selected_by_episode = {episode: selected_value for episode in VAL_EPISODES}
        batch_size = 2 if spec.experiment_id == "B0" else 1
        selection_contract = {
            "schema_version": 1,
            "split": "val",
            "batch_size": batch_size,
        }
        checkpoint_by_run[spec.run_id] = {
            "run_id": spec.run_id,
            "experiment_id": spec.experiment_id,
            "checkpoint_sha256": hashlib.sha256(
                f"checkpoint:{spec.run_id}".encode()
            ).hexdigest(),
            "model_state_sha256": (
                EXPECTED_B0_MODEL_STATE_SHA256
                if spec.experiment_id == "B0"
                else hashlib.sha256(f"state:{spec.run_id}".encode()).hexdigest()
            ),
            "training_metrics": str(tmp_path / f"{spec.run_id}.metrics.jsonl"),
            "training_metrics_sha256": hashlib.sha256(
                f"metrics:{spec.run_id}".encode()
            ).hexdigest(),
            "training_selection": {
                "batch_size": batch_size,
                "selected_epoch": 0,
                "selected_value": selected_value,
                "by_episode": selected_by_episode,
                "support": support,
                "selection_detail_sha256": hashlib.sha256(
                    f"selection:{spec.run_id}".encode()
                ).hexdigest(),
                "training_selection_contract": selection_contract,
                "training_selection_contract_sha256": (
                    controller.canonical_json_sha256(selection_contract)
                ),
            },
        }
        formal_value = (
            selected_value + 0.001 if spec.experiment_id == "B0" else selected_value
        )
        formal_by_episode = {
            episode: (value + 0.001 if spec.experiment_id == "B0" else value)
            for episode, value in selected_by_episode.items()
        }
        artifact_sha = hashlib.sha256(f"artifact:{spec.run_id}".encode()).hexdigest()
        formal_runs.append(
            {
                "run_id": spec.run_id,
                "bce_at1": formal_value,
                "by_episode_bce_at1": formal_by_episode,
                "evaluation_result": str(tmp_path / "json" / f"{spec.run_id}.json"),
                "evaluation_result_sha256": artifact_sha,
                "predictions": str(tmp_path / "predictions" / f"{spec.run_id}.jsonl"),
                "predictions_sha256": artifact_sha,
                "prediction_count": VAL_SAMPLE_COUNT,
                "log": str(tmp_path / "logs" / f"{spec.run_id}.log"),
                "log_sha256": artifact_sha,
                "evaluation_execution_contract_sha256": artifact_sha,
                "evaluator_command_sha256": artifact_sha,
                "evaluation_batch_size": 1,
                "prediction_support": prediction_support,
            }
        )
    replay_sha = hashlib.sha256(b"B0 replay").hexdigest()
    b0_replay = {
        "run_id": "B0_seed0",
        "bce_at1": 0.3,
        "by_episode_bce_at1": {episode: 0.3 for episode in VAL_EPISODES},
        "evaluation_result": str(tmp_path / "selection_replay/json/B0_seed0.val.json"),
        "evaluation_result_sha256": replay_sha,
        "predictions": str(tmp_path / "selection_replay/predictions/B0_seed0.jsonl"),
        "predictions_sha256": replay_sha,
        "prediction_count": VAL_SAMPLE_COUNT,
        "log": str(tmp_path / "selection_replay/logs/B0_seed0.log"),
        "log_sha256": replay_sha,
        "evaluation_execution_contract_sha256": replay_sha,
        "evaluation_batch_size": 2,
        "prediction_support": prediction_support,
    }
    return checkpoint_by_run, formal_runs, b0_replay


def test_formal_primary_is_preserved_and_drift_is_recorded(tmp_path):
    checkpoint_by_run, formal_runs, _ = _synthetic_dual_contract_inputs(tmp_path)
    formal = formal_runs[0]
    primary_before = formal["bce_at1"]
    by_episode_before = dict(formal["by_episode_bce_at1"])
    controller.enrich_formal_run_record(
        formal, checkpoint_by_run["B0_seed0"], ["formal", "batch1"]
    )
    assert formal["bce_at1"] == primary_before
    assert formal["by_episode_bce_at1"] == by_episode_before
    assert formal["formal_primary"] is True
    assert formal["formal_minus_selection"]["value"] == pytest.approx(0.001)
    assert formal["training_selection"]["selected_value"] == 0.3


def test_selection_replay_manifest_binds_one_b0_run_and_six_formal_receipts(
    tmp_path,
):
    checkpoint_by_run, formal_runs, b0_replay = _synthetic_dual_contract_inputs(
        tmp_path
    )
    for formal in formal_runs:
        formal.setdefault("formal_primary", True)
    document = controller.build_selection_replay_document(
        project_root=tmp_path,
        controller_sha256="c" * 64,
        inventory_path=tmp_path / controller.DEFAULT_INVENTORY_RELATIVE,
        inventory_sha256="i" * 64,
        predecessor_failure_receipt={"sha256": "p" * 64},
        static_contracts={"source_tree_sha256": SOURCE_TREE_SHA256},
        checkpoint_by_run=checkpoint_by_run,
        formal_runs=formal_runs,
        b0_replay_run=b0_replay,
        b0_replay_command=["selection", "batch2"],
    )
    assert len(document["receipts"]) == 7
    b0 = document["receipts"][0]
    assert b0["receipt_kind"] == "executed_shared_model_state"
    assert b0["selection_batch_size"] == 2
    assert len(b0["checkpoint_bindings"]) == 3
    assert (
        len({binding["checkpoint_sha256"] for binding in b0["checkpoint_bindings"]})
        == 3
    )
    assert "strafe_right" not in b0["replayed_support"]["test0012"]
    same_contract = document["receipts"][1]
    assert same_contract["receipt_kind"] == "formal_artifact_same_contract"
    assert (
        same_contract["artifacts"]["evaluation_result"]["path"]
        == formal_runs[3]["evaluation_result"]
    )


@pytest.mark.parametrize("target", ("b0_replay", "same_contract"))
def test_selection_replay_exact_match_mutations_fail_closed(tmp_path, target):
    checkpoint_by_run, formal_runs, b0_replay = _synthetic_dual_contract_inputs(
        tmp_path
    )
    if target == "b0_replay":
        b0_replay["bce_at1"] += 1e-12
        match = "B0 shared replay B0_seed0.value mismatch"
    else:
        formal_runs[3]["bce_at1"] += 1e-12
        match = "B1_seed0 formal same-contract replay.value mismatch"
    with pytest.raises(ValidationEvalError, match=match):
        controller.build_selection_replay_document(
            project_root=tmp_path,
            controller_sha256="c" * 64,
            inventory_path=tmp_path / controller.DEFAULT_INVENTORY_RELATIVE,
            inventory_sha256="i" * 64,
            predecessor_failure_receipt={"sha256": "p" * 64},
            static_contracts={"source_tree_sha256": SOURCE_TREE_SHA256},
            checkpoint_by_run=checkpoint_by_run,
            formal_runs=formal_runs,
            b0_replay_run=b0_replay,
            b0_replay_command=["selection", "batch2"],
        )


def test_sha256sums_covers_selection_replay_and_both_manifests(tmp_path):
    output = tmp_path / "validation_eval_v6"
    for relative in (
        "json/B0_seed0.val.json",
        "selection_replay/json/B0_seed0.val.json",
        "selection_replay/predictions/B0_seed0.jsonl",
        "selection_replay/logs/B0_seed0.log",
        "audit/selection_replay_manifest.json",
        "audit/run_manifest.json",
    ):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    sums = controller.write_sha256sums(output)
    indexed = {
        line.split("  ", 1)[1] for line in sums.read_text(encoding="utf-8").splitlines()
    }
    assert indexed == {
        "json/B0_seed0.val.json",
        "selection_replay/json/B0_seed0.val.json",
        "selection_replay/predictions/B0_seed0.jsonl",
        "selection_replay/logs/B0_seed0.log",
        "audit/selection_replay_manifest.json",
        "audit/run_manifest.json",
    }


def test_execution_rejects_exact_sealed_v5_stage_root(tmp_path):
    args = types.SimpleNamespace(
        hermetic_stage_binding={
            "stage_root": Path(next(iter(controller.SEALED_STAGE_ROOTS)))
        }
    )
    with pytest.raises(ValidationEvalError, match="execution stage reuses"):
        controller._execute_locked(
            args,
            project_root=tmp_path,
            output_root=tmp_path / controller.DEFAULT_OUTPUT_RELATIVE,
        )


def test_execution_rejects_sealed_v5_stage_source_contract(tmp_path, monkeypatch):
    sealed_source = controller.SEALED_V5_STAGE_SOURCE_CONTRACT_SHA256
    stage_binding = {
        "source_project_root": tmp_path,
        "stage_root": tmp_path / "stage",
        "manifest_path": tmp_path / "stage/manifest.json",
        "manifest_sha256": "m" * 64,
        "source_contract_sha256": sealed_source,
        "stage_tree_sha256": "t" * 64,
        "excluded_source_import_artifact_count": 0,
    }
    monkeypatch.setattr(controller, "assert_pytorch_environment", lambda: None)
    monkeypatch.setattr(controller, "assert_controller_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(
        controller,
        "validate_v11_full_failure_receipt",
        lambda *_: {"sha256": "p" * 64},
    )
    monkeypatch.setattr(controller, "assert_no_competing_processes", lambda: None)
    monkeypatch.setattr(controller, "assert_free_space", lambda _: 3 * 1024**3)
    monkeypatch.setattr(
        controller,
        "validate_static_contracts",
        lambda *_args, **_kwargs: {"stage_source_contract_sha256": sealed_source},
    )
    args = types.SimpleNamespace(
        hermetic_stage_binding=stage_binding,
        expected_controller_sha256="c" * 64,
        predecessor_failure_receipt=str(tmp_path / "predecessor.json"),
        expected_predecessor_failure_receipt_sha256="p" * 64,
        expected_stage_source_contract_sha256=sealed_source,
        expected_runtime_isolation_contract_sha256="r" * 64,
        preflight_only=True,
        preflight_output=None,
    )
    with pytest.raises(
        ValidationEvalError, match="execution stage source contract reuses"
    ):
        controller._execute_locked(
            args,
            project_root=tmp_path,
            output_root=tmp_path / controller.DEFAULT_OUTPUT_RELATIVE,
        )


def test_execute_locked_runs_nine_formal_then_one_b0_replay(tmp_path, monkeypatch):
    checkpoint_by_run, formal_runs, b0_replay = _synthetic_dual_contract_inputs(
        tmp_path
    )
    formal_by_run = {run["run_id"]: run for run in formal_runs}
    output_root = tmp_path / controller.DEFAULT_OUTPUT_RELATIVE
    inventory_path = tmp_path / controller.DEFAULT_INVENTORY_RELATIVE
    stage_binding = {
        "source_project_root": tmp_path,
        "stage_root": tmp_path / "stage",
        "manifest_path": tmp_path / "stage/manifest.json",
        "manifest_sha256": "m" * 64,
        "source_contract_sha256": "s" * 64,
        "stage_tree_sha256": "t" * 64,
        "excluded_source_import_artifact_count": 0,
    }
    predecessor = {
        "path": str(tmp_path / "predecessor.json"),
        "sha256": "p" * 64,
        **_predecessor_receipt_payload(),
    }
    monkeypatch.setattr(controller, "assert_pytorch_environment", lambda: None)
    monkeypatch.setattr(controller, "assert_controller_sha256", lambda _: "c" * 64)
    monkeypatch.setattr(controller, "assert_no_competing_processes", lambda: None)
    monkeypatch.setattr(controller, "assert_free_space", lambda _: 3 * 1024**3)
    monkeypatch.setattr(
        controller,
        "validate_v11_full_failure_receipt",
        lambda *_: predecessor,
    )
    monkeypatch.setattr(
        controller,
        "validate_static_contracts",
        lambda *_args, **_kwargs: {"stage_source_contract_sha256": "s" * 64},
    )
    monkeypatch.setattr(
        controller,
        "preflight_checkpoints",
        lambda *_: (list(checkpoint_by_run.values()), types.ModuleType("evaluator")),
    )
    monkeypatch.setattr(
        controller,
        "load_frozen_checkpoint_inventory",
        lambda *_args, **_kwargs: (
            {"runs": list(checkpoint_by_run.values())},
            "i" * 64,
        ),
    )
    evaluator_commands = []

    def fake_run_evaluator(command, **_):
        evaluator_commands.append(command)

    monkeypatch.setattr(controller, "run_evaluator", fake_run_evaluator)

    def fake_validate(spec, *, output_root, expected_batch_size, **_):
        source = b0_replay if expected_batch_size == 2 else formal_by_run[spec.run_id]
        return json.loads(json.dumps(source))

    monkeypatch.setattr(controller, "validate_run_artifacts", fake_validate)
    args = types.SimpleNamespace(
        hermetic_stage_binding=stage_binding,
        expected_controller_sha256="c" * 64,
        predecessor_failure_receipt=str(tmp_path / "predecessor.json"),
        expected_predecessor_failure_receipt_sha256="p" * 64,
        expected_stage_source_contract_sha256="s" * 64,
        expected_runtime_isolation_contract_sha256="r" * 64,
        preflight_only=False,
        expected_checkpoint_inventory_sha256="i" * 64,
        checkpoint_inventory=str(inventory_path),
        formal_runtime_snapshot={"isolated": True, "no_site": True},
    )
    manifest = controller._execute_locked(
        args, project_root=tmp_path, output_root=output_root
    )
    assert len(evaluator_commands) == 10
    assert all(
        command[command.index("--batch_size") + 1] == "1"
        for command in evaluator_commands[:9]
    )
    assert evaluator_commands[9][evaluator_commands[9].index("--batch_size") + 1] == "2"
    assert manifest["schema_version"] == 2
    assert manifest["selection_replay"]["receipt_count"] == 7
    replay_document = json.loads(
        (output_root / "audit/selection_replay_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(replay_document["receipts"]) == 7
    assert (output_root / "audit/SHA256SUMS.txt").is_file()


def _write_python_startup_attack(tmp_path: Path) -> tuple[dict[str, str], list[Path]]:
    attack = tmp_path / "pythonpath_attack"
    attack.mkdir()
    marker_dir = tmp_path / "markers"
    marker_dir.mkdir()
    markers = []

    def write_attack(path: Path, label: str) -> None:
        marker = marker_dir / f"{label}.marker"
        markers.append(marker)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n"
            f"raise RuntimeError({label!r} + ' hijack executed')\n",
            encoding="utf-8",
        )

    for module_name in (
        "json",
        "hashlib",
        "numpy",
        "torch",
        "transformers",
        "experiment_binding",
        "model",
        "local_weights",
    ):
        write_attack(attack / f"{module_name}.py", module_name)
    write_attack(attack / "scripts/__init__.py", "scripts_package")
    write_attack(attack / "sitecustomize.py", "sitecustomize")

    user_site = (
        tmp_path
        / "userbase/lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    user_site.mkdir(parents=True)
    pth_marker = marker_dir / "pth.marker"
    markers.append(pth_marker)
    (user_site / "attack.pth").write_text(
        "import pathlib; "
        f"pathlib.Path({str(pth_marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    write_attack(user_site / "sitecustomize.py", "user_sitecustomize")

    environment = os.environ.copy()
    environment.update(
        {
            "PATH": controller.FORMAL_PATH,
            "PYTHONPATH": str(attack),
            "PYTHONUSERBASE": str(tmp_path / "userbase"),
            "PYTHONHOME": str(tmp_path / "fake_python_home"),
            "PYTHONINSPECT": "1",
        }
    )
    return environment, markers


def test_formal_clis_block_python_startup_and_dependency_hijacks(
    tmp_path, hermetic_stage
):
    environment, markers = _write_python_startup_attack(tmp_path)
    commands = (
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            str(controller.PROJECT_ROOT / "scripts/run_main_v1_validation_eval.py"),
            "--help",
        ],
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            str(controller.PROJECT_ROOT / "scripts/build_main_v1_validation_cells.py"),
            "--stage_manifest",
            str(hermetic_stage["manifest_path"]),
            "--expected_stage_manifest_sha256",
            hermetic_stage["manifest_sha256"],
            "--expected_builder_sha256",
            controller.sha256_file(
                controller.PROJECT_ROOT / "scripts/build_main_v1_validation_cells.py"
            ),
            "--help",
        ],
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            str(controller.PROJECT_ROOT / "scripts/analyze_main_v1_validation.py"),
            "--stage_manifest",
            str(hermetic_stage["manifest_path"]),
            "--expected_stage_manifest_sha256",
            hermetic_stage["manifest_sha256"],
            "--expected_analyzer_sha256",
            controller.sha256_file(
                controller.PROJECT_ROOT / "scripts/analyze_main_v1_validation.py"
            ),
            "--help",
        ],
    )
    for command in commands:
        completed = subprocess.run(
            command,
            cwd=controller.PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
    assert not [path for path in markers if path.exists()]


@pytest.mark.parametrize(
    "script_name",
    (
        "run_main_v1_validation_eval.py",
        "build_main_v1_validation_cells.py",
        "analyze_main_v1_validation.py",
    ),
)
def test_nonisolated_cli_refuses_before_evidence_work(script_name):
    completed = subprocess.run(
        [
            sys.executable,
            "-u",
            str(controller.PROJECT_ROOT / "scripts" / script_name),
            "--help",
        ],
        cwd=controller.PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert completed.returncode != 0
    assert "python -I -S -B -u" in completed.stderr


def test_isolated_controller_runtime_rejects_path_drift():
    probe = (
        "import runpy,sys; "
        "namespace=runpy.run_path(sys.argv[1],run_name='_path_probe'); "
        "namespace['assert_formal_runtime_isolation'](require_ml=False)"
    )
    environment = os.environ.copy()
    environment["PATH"] = "/definitely-not-the-formal-path"
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            "-c",
            probe,
            str(controller.CONTROLLER_PATH),
        ],
        cwd=controller.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode != 0
    assert "formal runtime PATH mismatch" in completed.stderr


def test_isolated_inprocess_evaluator_path_order_and_origins(tmp_path, hermetic_stage):
    environment, markers = _write_python_startup_attack(tmp_path)
    probe = """
import json
import pathlib
import runpy
import sys

namespace = runpy.run_path(sys.argv[1], run_name="_main_v1_controller_probe")
namespace["assert_formal_runtime_isolation"](require_ml=True)
binding = namespace["verify_hermetic_stage"](sys.argv[2], sys.argv[3])
module = namespace["load_verified_evaluator_module"](binding)
project = str(binding["stage_root"])
car_runtime = str(binding["stage_root"] / "car_runtime")
opentrack = str(binding["stage_root"] / "third_party/OpenTrackVLA")
site_paths = [str(path) for path in namespace["_trusted_site_package_paths"]()]
print(json.dumps({
    "project_index": sys.path.index(project),
    "car_runtime_index": sys.path.index(car_runtime),
    "opentrack_index": sys.path.index(opentrack),
    "site_indices": [sys.path.index(path) for path in site_paths],
    "project_count": sys.path.count(project),
    "car_runtime_count": sys.path.count(car_runtime),
    "opentrack_count": sys.path.count(opentrack),
    "evaluator": str(pathlib.Path(module.__file__).resolve()),
    "experiment_binding": str(pathlib.Path(sys.modules["experiment_binding"].__file__).resolve()),
    "numpy": str(pathlib.Path(sys.modules["numpy"].__file__).resolve()),
    "torch": str(pathlib.Path(sys.modules["torch"].__file__).resolve()),
    "transformers": str(pathlib.Path(sys.modules["transformers"].__file__).resolve()),
}, sort_keys=True))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            "-X",
            f"pycache_prefix={hermetic_stage['stage_root'] / '.pycache-disabled'}",
            "-c",
            probe,
            str(controller.CONTROLLER_PATH),
            str(hermetic_stage["manifest_path"]),
            hermetic_stage["manifest_sha256"],
        ],
        cwd=controller.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert max(result["site_indices"]) < result["project_index"]
    assert (
        result["project_index"]
        < result["car_runtime_index"]
        < result["opentrack_index"]
    )
    assert (
        result["project_count"]
        == result["car_runtime_count"]
        == result["opentrack_count"]
        == 1
    )
    assert result["evaluator"] == str(
        (hermetic_stage["stage_root"] / "scripts/eval_offline.py").resolve()
    )
    assert result["experiment_binding"].startswith(
        str((hermetic_stage["stage_root"] / "third_party/OpenTrackVLA").resolve())
    )
    trusted_site = str(Path(sys.prefix).resolve())
    assert result["numpy"].startswith(trusted_site)
    assert result["torch"].startswith(trusted_site)
    assert result["transformers"].startswith(trusted_site)
    assert not [path for path in markers if path.exists()]


def test_isolated_evaluator_child_bootstrap_blocks_pythonpath_attack(
    tmp_path, hermetic_stage
):
    environment, markers = _write_python_startup_attack(tmp_path)
    command = build_eval_command(
        RUN_SPECS[0],
        project_root=controller.PROJECT_ROOT,
        output_root=tmp_path / "unused-output",
        stage_binding=hermetic_stage,
    )
    completed = subprocess.run(
        [*command, "--help"],
        cwd=controller.PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage: eval_offline.py" in completed.stdout
    assert not [path for path in markers if path.exists()]


def test_environment_strip_command_tamper_and_preloaded_modules_fail(
    tmp_path, monkeypatch, hermetic_stage
):
    injected = {
        "HOME": "/trusted/home",
        "TMPDIR": "/trusted/tmp",
        "XDG_CACHE_HOME": "/trusted/cache",
        "PATH": "/attacker/bin",
        "PYTHONPATH": "/attacker/python",
        "pythonhome": "/attacker/home",
        "DYLD_INSERT_LIBRARIES": "/attacker/inject.dylib",
        "DYLD_LIBRARY_PATH": "/attacker/lib",
        "LD_PRELOAD": "/attacker/inject.so",
        "BASH_ENV": "/attacker/bash_env",
        "ZDOTDIR": "/attacker/zsh",
        "UNRELATED_SECRET": "must-not-propagate",
        "PYTORCH_ENABLE_MPS_FALLBACK": "0",
    }
    formal = controller.build_formal_environment(injected)
    assert formal == controller.FORMAL_ENVIRONMENT
    sanitized = build_evaluator_environment(injected)
    assert sanitized == controller.EVALUATOR_ENVIRONMENT

    command = build_eval_command(
        RUN_SPECS[0],
        project_root=controller.PROJECT_ROOT,
        output_root=tmp_path,
        stage_binding=hermetic_stage,
    )
    tampered = list(command)
    tampered[2] = "-E"
    with pytest.raises(ValidationEvalError, match="isolation contract"):
        controller.run_evaluator(
            tampered,
            log_path=tmp_path / "tampered.log",
            cwd=controller.PROJECT_ROOT,
        )

    reserved = [
        name
        for name in tuple(controller.sys.modules)
        if name.partition(".")[0] in controller.LOCAL_MODULE_ROOTS
    ]
    for name in reserved:
        monkeypatch.delitem(controller.sys.modules, name, raising=False)
    fake_model = types.ModuleType("model")
    fake_model.__file__ = str(
        controller.PROJECT_ROOT / "third_party/OpenTrackVLA/model.py"
    )
    monkeypatch.setitem(controller.sys.modules, "model", fake_model)
    with pytest.raises(ValidationEvalError, match="preloaded reserved"):
        reject_preloaded_reserved_local_modules()
    monkeypatch.delitem(controller.sys.modules, "model", raising=False)

    fake_torch = types.ModuleType("torch")
    fake_torch.__file__ = str(tmp_path / "torch.py")
    monkeypatch.setitem(controller.sys.modules, "torch", fake_torch)
    with pytest.raises(ValidationEvalError, match="outside trusted roots"):
        controller.validate_preloaded_runtime_modules()


def test_evaluator_popen_receives_only_allowlisted_environment(
    tmp_path, monkeypatch, hermetic_stage
):
    for name in (
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_PRELOAD",
        "BASH_ENV",
        "ZDOTDIR",
        "UNRELATED_SECRET",
    ):
        monkeypatch.setenv(name, "/attacker")
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return types.SimpleNamespace(pid=12345, returncode=0, poll=lambda: 0)

    monkeypatch.setattr(controller.subprocess, "Popen", fake_popen)
    command = build_eval_command(
        RUN_SPECS[0],
        project_root=controller.PROJECT_ROOT,
        output_root=tmp_path / "unused-output",
        stage_binding=hermetic_stage,
    )
    controller.run_evaluator(
        command,
        log_path=tmp_path / "evaluator.log",
        cwd=controller.PROJECT_ROOT,
    )
    assert captured["command"][0] == sys.executable
    assert captured["environment"] == controller.EVALUATOR_ENVIRONMENT
    assert not {
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_PRELOAD",
        "BASH_ENV",
        "ZDOTDIR",
        "UNRELATED_SECRET",
    }.intersection(captured["environment"])


def test_runtime_contract_binds_same_byte_evaluator_bootstrap():
    contract = independent_build_runtime_isolation_contract()
    assert (
        contract["controller_cli"]["fixed_environment"] == controller.FORMAL_ENVIRONMENT
    )
    process_listing = contract["controller_cli"]["process_listing"]
    assert process_listing == {
        "command": list(controller.PROCESS_LISTING_COMMAND),
        "resolution": "absolute_path_no_PATH_lookup",
        "outer_executable_field": "ucomm",
        "arguments_field": "args",
        "classification": (
            "inspect argv only when the outer executable is an exact python, "
            "shell, conda, env, timeout, or direct-script launcher"
        ),
        "payload_policy": (
            "never match forbidden script names in arguments belonging to an "
            "unrecognized outer executable"
        ),
        "guarded_parse_failure_policy": (
            "fail closed on malformed outer argv or nested shell-command "
            "quoting for every recognized launcher"
        ),
        "python_dash_c_policy": (
            "parse Python -c code as AST; reject direct/importlib/runpy "
            "module execution and literal exec/open execution of a forbidden "
            "script; syntax-invalid code fails closed"
        ),
        "argv0_reconciliation": {
            "equivalent_launcher_families": [
                "exact_normalized_name",
                "python_family",
                "shell_family",
                "timeout_family",
            ],
            "prepend_ucomm_only_when": (
                "args[0] is absent or is not equivalent to the guarded ucomm "
                "launcher"
            ),
            "residual": (
                "unrecognized outer executables remain opaque even when their "
                "payload text names a forbidden script"
            ),
        },
    }
    evaluator_contract = contract["evaluator_subprocess"]
    assert evaluator_contract["fixed_environment"] == controller.EVALUATOR_ENVIRONMENT
    assert evaluator_contract["target_file_sha256"] == EVALUATOR_FILE_SHA256
    assert evaluator_contract["target_loading"] == (
        "absolute_path_hash_then_compile_exec_same_bytes"
    )
    assert (
        evaluator_contract["bootstrap_sha256"]
        == hashlib.sha256(EVALUATOR_ISOLATED_BOOTSTRAP.encode("utf-8")).hexdigest()
    )
    assert "target.read_bytes()" in EVALUATOR_ISOLATED_BOOTSTRAP
    assert "compile(target_bytes" in EVALUATOR_ISOLATED_BOOTSTRAP
    assert evaluator_contract["sys_path_order"][-3:] == [
        "read-only hermetic stage root",
        "read-only staged car_runtime legacy flat-import root",
        "read-only staged OpenTrackVLA root",
    ]
    assert set(
        evaluator_contract["legacy_import_compatibility"]["flat_module_origins"]
    ) == {
        "car_hardware",
        "car_protocol",
        "process_cleanup",
        "uart_transport",
        "wheel_trim",
    }
    assert (
        contract["artifact_trust_boundary"]["concurrent_writer_policy"]
        == "no-concurrent-artifact-writer"
    )
    source_scope = contract["hermetic_stage"]["source_contract_scope"]
    assert "target_detector.py" in source_scope["legacy_evaluator_source"]
    assert (
        "complete staged car_runtime closure" in source_scope["stage_source_contract"]
    )


def test_evaluator_bootstrap_binds_legacy_car_runtime_flat_imports(tmp_path):
    source_root = _copy_stage_sources(tmp_path / "source")
    evaluator = source_root / "scripts/eval_offline.py"
    evaluator.write_text(
        """
import atexit
import json
import pathlib
import sys

import car_runtime
import inference_pipeline.mac_server

def emit_audit():
    stage_root = pathlib.Path(__file__).resolve().parents[1]
    car_runtime_root = stage_root / "car_runtime"
    opentrackvla_root = stage_root / "third_party/OpenTrackVLA"
    names = (
        "car_runtime",
        "car_hardware",
        "car_protocol",
        "process_cleanup",
        "uart_transport",
        "wheel_trim",
    )
    print(json.dumps({
        "root_tail": sys.path[-3:],
        "root_counts": {
            str(root): sys.path.count(str(root))
            for root in (stage_root, car_runtime_root, opentrackvla_root)
        },
        "origins": {
            name: str(pathlib.Path(sys.modules[name].__file__).resolve())
            for name in names
        },
    }, sort_keys=True))

atexit.register(emit_audit)
""".lstrip(),
        encoding="utf-8",
    )
    binding = create_hermetic_stage(source_root)
    try:
        stage_root = binding["stage_root"]
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-u",
            "-X",
            f"pycache_prefix={stage_root / '.pycache-disabled'}",
            "-c",
            EVALUATOR_ISOLATED_BOOTSTRAP,
            str(binding["manifest_path"]),
            binding["manifest_sha256"],
            str(stage_root / "scripts/eval_offline.py"),
        ]
        completed = subprocess.run(
            command,
            cwd=controller.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        audit = json.loads(completed.stdout.strip().splitlines()[-1])
        car_runtime_root = stage_root / "car_runtime"
        opentrackvla_root = stage_root / "third_party/OpenTrackVLA"
        assert audit["root_tail"] == [
            str(stage_root),
            str(car_runtime_root),
            str(opentrackvla_root),
        ]
        assert set(audit["root_counts"].values()) == {1}
        assert all(
            pathlib.startswith(str(car_runtime_root))
            for pathlib in audit["origins"].values()
        )
    finally:
        _cleanup_stage(binding)


def test_reserved_site_package_collision_fails_before_local_root_append(
    monkeypatch, hermetic_stage
):
    original_find_spec = controller.importlib.machinery.PathFinder.find_spec

    def fake_find_spec(name, path=None, target=None):
        if name == "model":
            return object()
        return original_find_spec(name, path, target)

    monkeypatch.setattr(
        controller.importlib.machinery.PathFinder,
        "find_spec",
        staticmethod(fake_find_spec),
    )
    with pytest.raises(ValidationEvalError, match="shadows reserved"):
        append_hermetic_stage_import_roots(hermetic_stage["stage_root"])


def test_stage_excludes_matching_header_and_sourceless_pycache(tmp_path):
    source_root = _copy_stage_sources(tmp_path / "source")
    marker = tmp_path / "pyc-executed.marker"
    trusted_source = source_root / "scripts/eval_offline.py"
    code = compile(
        f"from pathlib import Path; Path({str(marker)!r}).write_text('executed')",
        str(trusted_source),
        "exec",
    )
    pyc_path = Path(importlib.util.cache_from_source(str(trusted_source)))
    pyc_path.parent.mkdir(parents=True)
    header = (
        importlib.util.MAGIC_NUMBER
        + (0).to_bytes(4, "little")
        + int(trusted_source.stat().st_mtime).to_bytes(4, "little")
        + int(trusted_source.stat().st_size).to_bytes(4, "little")
    )
    pyc_path.write_bytes(header + marshal.dumps(code))
    orphan = pyc_path.parent / "orphan.cpython-test.pyc"
    orphan.write_bytes(header + marshal.dumps(code))

    binding = create_hermetic_stage(source_root)
    try:
        assert binding["excluded_source_import_artifact_count"] >= 2
        assert not list(binding["stage_root"].rglob("*.pyc"))
        command = build_eval_command(
            RUN_SPECS[0],
            project_root=controller.PROJECT_ROOT,
            output_root=tmp_path / "unused",
            stage_binding=binding,
        )
        completed = subprocess.run(
            [*command, "--help"],
            cwd=controller.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, completed.stderr
        assert not marker.exists()
    finally:
        _cleanup_stage(binding)


@pytest.mark.parametrize("artifact_kind", ("sourceless_pyc", "extension"))
def test_stage_rejects_adjacent_import_artifacts(tmp_path, artifact_kind):
    source_root = _copy_stage_sources(tmp_path / artifact_kind)
    if artifact_kind == "sourceless_pyc":
        artifact = source_root / "data_pipeline/orphan.pyc"
    else:
        suffix = controller.importlib.machinery.EXTENSION_SUFFIXES[0]
        artifact = source_root / "third_party/OpenTrackVLA/orphan"
        artifact = artifact.with_name(artifact.name + suffix)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"not executable")
    with pytest.raises(ValidationEvalError, match="non-__pycache__"):
        create_hermetic_stage(source_root)


def test_stage_verifier_rejects_post_freeze_import_artifact(tmp_path):
    source_root = _copy_stage_sources(tmp_path / "tamper-source")
    binding = create_hermetic_stage(source_root)
    stage_root = binding["stage_root"]
    scripts_dir = stage_root / "scripts"
    try:
        os.chmod(stage_root, 0o755)
        os.chmod(scripts_dir, 0o755)
        injected = scripts_dir / "orphan.pyc"
        injected.write_bytes(b"not executable")
        os.chmod(injected, 0o444)
        os.chmod(scripts_dir, 0o555)
        os.chmod(stage_root, 0o555)
        with pytest.raises(ValidationEvalError, match="unexpected file"):
            controller.verify_hermetic_stage(
                binding["manifest_path"], binding["manifest_sha256"]
            )
    finally:
        _cleanup_stage(binding)


def test_stage_root_normalization_handles_verified_duplicate_insert(
    monkeypatch, hermetic_stage
):
    project = hermetic_stage["stage_root"]
    car_runtime = project / "car_runtime"
    opentrack = project / "third_party/OpenTrackVLA"
    synthetic_path = [
        str(car_runtime),
        str(opentrack),
        *controller.sys.path,
        str(project),
        str(car_runtime),
        str(opentrack),
    ]
    monkeypatch.setattr(controller.sys, "path", synthetic_path)
    controller.normalize_stage_import_roots(project, car_runtime, opentrack)
    assert controller.sys.path.count(str(project)) == 1
    assert controller.sys.path.count(str(car_runtime)) == 1
    assert controller.sys.path.count(str(opentrack)) == 1
    assert controller.sys.path[-3:] == [
        str(project),
        str(car_runtime),
        str(opentrack),
    ]


def test_car_runtime_flat_imports_are_stage_bound_and_normalized(
    tmp_path, monkeypatch, hermetic_stage
):
    monkeypatch.setattr(controller.sys, "path", list(controller.sys.path))
    for name in tuple(controller.sys.modules):
        if name.partition(".")[0] in controller.LOCAL_MODULE_ROOTS:
            monkeypatch.delitem(controller.sys.modules, name, raising=False)

    roots = append_hermetic_stage_import_roots(hermetic_stage["stage_root"])
    project, car_runtime, opentrack = roots
    controller.sys.path.insert(0, str(car_runtime))
    module_files = {
        "car_runtime": car_runtime / "__init__.py",
        "car_hardware": car_runtime / "car_hardware.py",
        "car_protocol": car_runtime / "car_protocol.py",
        "process_cleanup": car_runtime / "process_cleanup.py",
        "uart_transport": car_runtime / "uart_transport.py",
        "wheel_trim": car_runtime / "wheel_trim.py",
    }
    for name, path in module_files.items():
        module = types.ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(controller.sys.modules, name, module)

    controller.validate_preloaded_runtime_modules(project)
    controller.validate_runtime_path_order(*roots)
    assert controller.sys.path[-3:] == [
        str(project),
        str(car_runtime),
        str(opentrack),
    ]
    assert all(controller.sys.path.count(str(root)) == 1 for root in roots)

    controller.sys.modules["wheel_trim"].__file__ = str(tmp_path / "wheel_trim.py")
    with pytest.raises(ValidationEvalError, match="outside the trusted root"):
        controller.validate_preloaded_runtime_modules(project)


def test_controller_reexec_injects_artifact_root(monkeypatch, hermetic_stage):
    captured = {}
    for name in (
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_PRELOAD",
        "BASH_ENV",
        "ZDOTDIR",
        "UNRELATED_SECRET",
    ):
        monkeypatch.setenv(name, "/attacker")

    def fake_execve(executable, command, environment):
        captured["executable"] = executable
        captured["command"] = command
        captured["environment"] = environment
        raise RuntimeError("captured")

    monkeypatch.setattr(controller.os, "execve", fake_execve)
    monkeypatch.setattr(
        controller.sys,
        "argv",
        [
            str(controller.CONTROLLER_PATH),
            "--expected_controller_sha256",
            "a" * 64,
            "--expected_stage_source_contract_sha256",
            "b" * 64,
            "--expected_runtime_isolation_contract_sha256",
            "c" * 64,
        ],
    )
    args = types.SimpleNamespace(
        staged_execution=False,
        expected_controller_sha256="a" * 64,
        expected_stage_source_contract_sha256="b" * 64,
        expected_runtime_isolation_contract_sha256="c" * 64,
    )
    with pytest.raises(RuntimeError, match="captured"):
        controller._reexec_staged_controller(args, hermetic_stage)
    project_indexes = [
        index
        for index, value in enumerate(captured["command"])
        if value == "--project_root"
    ]
    assert project_indexes
    assert captured["command"][project_indexes[-1] + 1] == str(
        hermetic_stage["source_project_root"]
    )
    assert captured["environment"] == controller.FORMAL_ENVIRONMENT
    assert not {
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "LD_PRELOAD",
        "BASH_ENV",
        "ZDOTDIR",
        "UNRELATED_SECRET",
    }.intersection(captured["environment"])
    for option, expected in (
        ("--expected_stage_source_contract_sha256", "b" * 64),
        ("--expected_runtime_isolation_contract_sha256", "c" * 64),
    ):
        index = len(captured["command"]) - 1 - captured["command"][::-1].index(option)
        assert captured["command"][index + 1] == expected
