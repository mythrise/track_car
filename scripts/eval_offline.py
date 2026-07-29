#!/usr/bin/env python3
"""Offline action-space evaluation for absolute and step-action checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPENTRACKVLA_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
for path in (PROJECT_ROOT, OPENTRACKVLA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiment_binding import (
    bind_hf_model_artifact,
    sha256_artifact,
    verify_vision_cache,
)


AXIS_NAMES = ("forward", "strafe", "yaw")
MODEL_FAMILIES = {
    "opentrackvla_baseline",
    "trackvla_pp_lite",
    "pfem_harness",
}
EVALUATION_TIERS = {"locked_final", "exploratory", "smoke"}
EXPECTED_CHECKPOINT_SELECTION = {
    "metric": "validation_episode_macro_BCE@1",
    "mode": "min",
    "rule": "strict_improvement_earliest_epoch",
}
FAIRNESS_META_FIELDS = (
    "optimizer_updates",
    "processed_samples",
    "sampling_policy",
    "batch_size",
    "grad_accum_steps",
    "effective_batch_size",
    "base_lr",
    "head_lr",
    "weight_decay",
    "grad_clip",
)
REGISTRY_MODEL_FIELDS = (
    "model_family",
    "state_mode",
    "batch_size",
    "grad_accum_steps",
    "effective_batch_size",
    "base_lr",
    "head_lr",
)
PRIMARY_COMMANDS = (
    "forward",
    "turn_left",
    "turn_right",
    "backward",
    "stop",
)
METRIC_CONTRACT_SCHEMA_VERSION = 2
EVALUATION_EXECUTION_CONTRACT_SCHEMA_VERSION = 1
SMOOTH_L1_BETA = 1.0
SATURATION_THRESHOLD = 0.95
EVENT_TOLERANCE_FRAMES = 2
HORIZON_IDS = (1, 2, 4, 8)
CONTROL_ERROR_WEIGHTS = {"forward": 1.0, "yaw": 2.0}
CHECKPOINT_META_FIELDS = (
    "schema_version",
    "model_family",
    "experiment_id",
    "seed",
    "history",
    "n_waypoints",
    "dt",
    "label_mode",
    "action_semantics",
    "data_manifest_hash",
    "data_jsonl_sha256",
    "sample_count",
    "base_model_sha256",
    "base_model_artifact",
    "qwen_model_sha256",
    "vision_cache_manifest_sha256",
    "vision_cache_provenance_sha256",
    "vision_cache_token_payload_sha256",
    "dino_model_sha256",
    "siglip_model_sha256",
    "training_source_raw_dirs",
    "state_mode",
    "checkpoint_selection",
    "checkpoint_role",
    "selection_verified",
    "selected_epoch",
    "selected_value",
    *FAIRNESS_META_FIELDS,
)


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint_with_sha256(path):
    """Load the exact checkpoint bytes identified by the returned SHA-256."""

    import torch

    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    payload = checkpoint_path.read_bytes()
    checkpoint_sha256 = hashlib.sha256(payload).hexdigest()
    checkpoint = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=False
    )
    del payload
    return checkpoint, checkpoint_sha256


def verify_checkpoint_file_unchanged(path, expected_sha256):
    """Fail closed if a checkpoint is replaced during offline evaluation."""

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "checkpoint changed during evaluation: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    return actual_sha256


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def source_tree_sha256(opentrackvla_root) -> str:
    """Hash the Python source tree using the formal-training path schema."""

    root = Path(opentrackvla_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OpenTrackVLA source root does not exist: {root}")
    files = sorted(root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError(f"OpenTrackVLA source root has no Python files: {root}")
    digest = hashlib.sha256()
    prefix = Path("third_party") / "OpenTrackVLA"
    for path in files:
        relative = (prefix / path.relative_to(root)).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_evaluator_source(project_root=PROJECT_ROOT) -> dict:
    """Describe every local source file that defines evaluation behavior."""

    root = Path(project_root).expanduser().resolve()
    inference_root = root / "inference_pipeline"
    files = [root / "scripts" / "eval_offline.py"]
    if not inference_root.is_dir():
        raise FileNotFoundError(f"evaluation source directory is missing: {inference_root}")
    files.extend(sorted(inference_root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()))
    files.append(root / "data_pipeline" / "kinematics.py")
    unique_files = sorted(set(files), key=lambda path: path.relative_to(root).as_posix())
    missing = [str(path) for path in unique_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("evaluation source files are missing: " + ", ".join(missing))
    return {
        "schema_version": 1,
        "hash_scope": (
            "scripts/eval_offline.py + inference_pipeline/**/*.py + "
            "data_pipeline/kinematics.py"
        ),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            for path in unique_files
        ],
    }


def build_metric_contract(transition_threshold=0.2) -> dict:
    """Return the canonical definition of every metric emitted by this evaluator."""

    threshold = float(transition_threshold)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("transition_threshold must be finite and >= 0")
    return {
        "schema_version": METRIC_CONTRACT_SCHEMA_VERSION,
        "primary": {
            "name": "episode_macro_BCE@1",
            "direction": "lower",
            "commands": list(PRIMARY_COMMANDS),
            "command_aggregation": "equal_macro_over_commands_present_per_episode",
            "episode_aggregation": "equal_macro_over_episodes",
            "horizon": 1,
            "absolute_error_weights": dict(CONTROL_ERROR_WEIGHTS),
            "weight_divisor": float(sum(CONTROL_ERROR_WEIGHTS.values())),
            "strafe_included": False,
        },
        "smooth_l1": {"beta": SMOOTH_L1_BETA, "axes": list(AXIS_NAMES)},
        "turn_sign_accuracy": {"yaw_active_threshold": threshold},
        "transition": {
            "yaw_active_threshold": threshold,
            "event_types": ["onset", "exit", "sign_flip"],
            "chronological_horizon": 1,
            "one_to_one_tolerance_frames": EVENT_TOLERANCE_FRAMES,
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
        },
        "horizon_mae": {
            "horizons": list(HORIZON_IDS),
            "axes": ["forward", "yaw"],
        },
        "saturation": {
            "absolute_threshold": SATURATION_THRESHOLD,
            "control_axes": ["forward", "yaw"],
            "all_axes_diagnostic": list(AXIS_NAMES),
        },
        "validity": {"invalid_forecast_steps_excluded": True},
    }


def validate_expected_contract_sha256(
    actual_sha256, registry, field, *, explicit_expected=None
):
    """Require a predeclared source/metric hash and compare it fail-closed."""

    expected = explicit_expected or registry.get(field)
    if not _is_sha256(expected):
        option = "--expected_" + field
        raise ValueError(
            f"{field} is unbound; provide {option} or freeze {field} in the registry"
        )
    if actual_sha256 != expected:
        raise ValueError(
            f"{field} mismatch: expected={expected}, actual={actual_sha256}"
        )
    return actual_sha256


def build_evaluation_execution_contract(
    *,
    evaluation_binding,
    identity,
    declared_state_mode,
    declared_label_mode,
    effective_label_mode,
    label_mode_override,
    batch_size,
    history,
    n_waypoints,
    dt,
    device,
    default_dtype,
    parameter_dtypes,
    buffer_dtypes,
    dataset_sample_count,
) -> dict:
    """Bind data and runtime choices that can change predictions or metrics."""

    state_mode = str(identity["state_mode"])
    return {
        "schema_version": EVALUATION_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "evaluation_data": {
            "split": str(evaluation_binding["split"]),
            "manifest_sha256": evaluation_binding["_verified_manifest_sha256"],
            "data_sha256": evaluation_binding["_verified_data_jsonl_sha256"],
            "sample_count": int(dataset_sample_count),
        },
        "observation": {
            "history": int(history),
            "n_waypoints": int(n_waypoints),
            "dt": float(dt),
        },
        "state": {
            "declared_mode": str(declared_state_mode),
            "effective_mode": state_mode,
            "override": bool(identity["state_mode_override"]),
            "sequence_id_required": state_mode == "rolling",
            "reset_policy": (
                "clean_sequence_boundary_or_frame_gap"
                if state_mode == "rolling"
                else "functionally_stateless"
            ),
        },
        "label": {
            "declared_mode": str(declared_label_mode),
            "effective_mode": str(effective_label_mode),
            "override": bool(label_mode_override),
        },
        "loader": {
            "batch_size": int(batch_size),
            "shuffle": False,
            "num_workers": 0,
            "ordered_record_validation": True,
        },
        "runtime": {
            "device": str(device),
            "device_type": str(getattr(device, "type", str(device))),
            "torch_default_dtype": str(default_dtype),
            "parameter_dtypes": sorted(str(value) for value in parameter_dtypes),
            "buffer_dtypes": sorted(str(value) for value in buffer_dtypes),
            "inference_mode": True,
            "autocast": False,
            "cache_payload_verified": True,
        },
        "evaluation_identity": {
            "tier": identity["evaluation_tier"],
            "class": identity["evaluation_class"],
        },
    }


def load_experiment_registry(path, expected_sha256):
    registry_path = Path(path).expanduser().resolve()
    if not registry_path.is_file():
        raise FileNotFoundError(f"experiment registry is missing: {registry_path}")
    if not _is_sha256(expected_sha256):
        raise ValueError("--expected_registry_sha256 must be an explicit SHA-256")
    actual_sha256 = sha256_file(registry_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "experiment registry SHA-256 mismatch: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or int(registry.get("schema_version", -1)) != 1:
        raise ValueError("experiment registry must be a schema-version 1 object")
    if not str(registry.get("status", "")).startswith("frozen"):
        raise ValueError("experiment registry must be frozen before evaluation")
    return registry, actual_sha256


def _require_equal(actual, expected, field):
    if actual != expected:
        raise ValueError(
            f"experiment registry mismatch for {field}: "
            f"checkpoint={actual!r}, registry={expected!r}"
        )


def build_fairness_contract(meta):
    validation = meta.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("checkpoint validation binding is required for provenance")
    return {
        "history": int(meta["history"]),
        "n_waypoints": int(meta["n_waypoints"]),
        "dt": float(meta["dt"]),
        "label_mode": str(meta["label_mode"]),
        "action_semantics": str(meta["action_semantics"]),
        "train": {
            "manifest_sha256": meta["data_manifest_hash"],
            "data_sha256": meta["data_jsonl_sha256"],
            "sample_count": int(meta["sample_count"]),
        },
        "validation": {
            "manifest_sha256": validation["data_manifest_hash"],
            "data_sha256": validation["data_jsonl_sha256"],
            "sample_count": int(validation["sample_count"]),
        },
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
        "checkpoint_selection": meta["checkpoint_selection"],
        "optimizer_updates": int(meta["optimizer_updates"]),
        "processed_samples": int(meta["processed_samples"]),
        "sampling_policy": str(meta["sampling_policy"]),
        "effective_batch_size": int(meta["effective_batch_size"]),
        "base_lr": float(meta["base_lr"]),
        "weight_decay": float(meta["weight_decay"]),
        "grad_clip": float(meta["grad_clip"]),
    }


def build_method_contract(meta):
    contract = {
        "experiment_id": str(meta["experiment_id"]),
        "model_family": str(meta["model_family"]),
        "state_mode": str(meta["state_mode"]),
        "batch_size": int(meta["batch_size"]),
        "grad_accum_steps": int(meta["grad_accum_steps"]),
        "effective_batch_size": int(meta["effective_batch_size"]),
        "base_lr": float(meta["base_lr"]),
        "head_lr": None if meta["head_lr"] is None else float(meta["head_lr"]),
    }
    for field in (
        "trackvla_lite_variant",
        "disabled_components",
        "state_reset_policy",
        "shuffle",
        "sampler",
    ):
        if field in meta:
            contract[field] = meta[field]
    return contract


def validate_registry_checkpoint_binding(
    meta,
    registry,
    *,
    registry_sha256,
    actual_source_tree_sha256,
    expected_source_tree_sha256=None,
):
    """Bind one checkpoint to the frozen suite registry and source snapshot."""

    expected_source = expected_source_tree_sha256 or registry.get(
        "source_tree_sha256"
    )
    if not _is_sha256(expected_source):
        raise ValueError(
            "source provenance is unbound; provide --expected_source_tree_sha256 "
            "or freeze source_tree_sha256 in the experiment registry"
        )
    if actual_source_tree_sha256 != expected_source:
        raise ValueError(
            "OpenTrackVLA source-tree SHA-256 mismatch: "
            f"expected={expected_source}, actual={actual_source_tree_sha256}"
        )

    data = registry.get("data")
    artifacts = registry.get("artifacts")
    models = registry.get("models")
    if not isinstance(data, dict) or not isinstance(artifacts, dict):
        raise ValueError("experiment registry must bind data and artifacts")
    if not isinstance(models, dict):
        raise ValueError("experiment registry must bind model contracts")
    experiment_id = str(meta["experiment_id"])
    model_contract = models.get(experiment_id)
    if not isinstance(model_contract, dict):
        raise ValueError(
            f"experiment registry has no exact model contract for {experiment_id!r}"
        )
    validation = meta.get("validation")
    if not isinstance(validation, dict):
        raise ValueError("checkpoint validation binding is required")

    bindings = (
        (meta.get("data_manifest_hash"), data.get("train_manifest_sha256"), "train_manifest_sha256"),
        (meta.get("data_jsonl_sha256"), data.get("train_data_sha256"), "train_data_sha256"),
        (meta.get("sample_count"), data.get("train_count"), "train_count"),
        (validation.get("data_manifest_hash"), data.get("val_manifest_sha256"), "val_manifest_sha256"),
        (validation.get("data_jsonl_sha256"), data.get("val_data_sha256"), "val_data_sha256"),
        (validation.get("sample_count"), data.get("val_count"), "val_count"),
        (meta.get("base_model_sha256"), artifacts.get("base_model_sha256"), "base_model_sha256"),
        (meta.get("qwen_model_sha256"), artifacts.get("qwen_model_sha256"), "qwen_model_sha256"),
        (meta.get("vision_cache_manifest_sha256"), artifacts.get("vision_cache_manifest_sha256"), "vision_cache_manifest_sha256"),
        (meta.get("vision_cache_provenance_sha256"), artifacts.get("vision_cache_provenance_sha256"), "vision_cache_provenance_sha256"),
        (meta.get("vision_cache_token_payload_sha256"), artifacts.get("vision_cache_token_payload_sha256"), "vision_cache_token_payload_sha256"),
        (meta.get("dino_model_sha256"), artifacts.get("dinov3_sha256"), "dinov3_sha256"),
        (meta.get("siglip_model_sha256"), artifacts.get("siglip_sha256"), "siglip_sha256"),
        (meta.get("history"), registry.get("history"), "history"),
        (meta.get("n_waypoints"), registry.get("prediction_horizon"), "prediction_horizon"),
        (meta.get("dt"), registry.get("dt"), "dt"),
        (meta.get("sampling_policy"), registry.get("sampling_policy"), "sampling_policy"),
        (meta.get("optimizer_updates"), registry.get("max_optimizer_updates"), "max_optimizer_updates"),
        (meta.get("processed_samples"), registry.get("processed_samples_per_run"), "processed_samples_per_run"),
        (meta.get("weight_decay"), registry.get("weight_decay"), "weight_decay"),
        (meta.get("grad_clip"), registry.get("grad_clip"), "grad_clip"),
        (meta.get("checkpoint_selection"), registry.get("checkpoint_selection"), "checkpoint_selection"),
    )
    for actual, expected, field in bindings:
        _require_equal(actual, expected, field)

    for field in REGISTRY_MODEL_FIELDS:
        _require_equal(meta.get(field), model_contract.get(field), f"models.{experiment_id}.{field}")
    optional_model_fields = {
        "variant": "trackvla_lite_variant",
        "disabled_components": "disabled_components",
        "state_reset_policy": "state_reset_policy",
        "shuffle": "shuffle",
        "sampler": "sampler",
    }
    for registry_field, meta_field in optional_model_fields.items():
        if registry_field in model_contract:
            _require_equal(
                meta.get(meta_field),
                model_contract[registry_field],
                f"models.{experiment_id}.{registry_field}",
            )

    best_validation = meta.get("best_validation")
    selection_detail = (
        best_validation.get("selection_detail")
        if isinstance(best_validation, dict)
        else None
    )
    if not isinstance(selection_detail, dict):
        raise ValueError("best-validation checkpoint must bind selection_detail")
    selected_value = _finite_number(
        meta.get("selected_value"), "selected_value", allow_zero=True
    )
    detail_value = _finite_number(
        selection_detail.get("value"),
        "best_validation.selection_detail.value",
        allow_zero=True,
    )
    if not math.isclose(selected_value, detail_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selected_value disagrees with validation selection_detail")

    fairness_contract = build_fairness_contract(meta)
    method_contract = build_method_contract(meta)
    return {
        "train_manifest_sha256": meta["data_manifest_hash"],
        "train_data_sha256": meta["data_jsonl_sha256"],
        "validation_manifest_sha256": validation["data_manifest_hash"],
        "validation_data_sha256": validation["data_jsonl_sha256"],
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
        "source_tree_sha256": actual_source_tree_sha256,
        "experiment_registry_sha256": registry_sha256,
        "fairness_contract": fairness_contract,
        "fairness_contract_sha256": canonical_json_sha256(fairness_contract),
        "method_contract": method_contract,
        "method_contract_sha256": canonical_json_sha256(method_contract),
        "validation_selection_detail_sha256": canonical_json_sha256(
            selection_detail
        ),
        "checkpoint_role": meta["checkpoint_role"],
        "selection_verified": meta["selection_verified"],
        "selected_epoch": meta["selected_epoch"],
        "selected_value": selected_value,
        "checkpoint_seed": int(meta["seed"]),
    }


def checkpoint_model_family(checkpoint):
    """Return an explicit supported family; never guess from an old checkpoint."""

    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("meta"), dict):
        raise ValueError("checkpoint must contain a metadata object")
    family = checkpoint["meta"].get("model_family")
    if not isinstance(family, str) or not family:
        raise ValueError("checkpoint metadata is missing explicit model_family")
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported checkpoint model_family={family!r}")
    return family


def validate_checkpoint_metadata(checkpoint):
    """Fail closed on metadata needed to identify a reproducible evaluation run."""

    family = checkpoint_model_family(checkpoint)
    meta = dict(checkpoint["meta"])
    missing = [field for field in CHECKPOINT_META_FIELDS if field not in meta]
    if missing:
        raise ValueError(
            "checkpoint metadata missing required fields: " + ", ".join(missing)
        )
    if meta.get("state_mode") not in {"stateless", "rolling"}:
        raise ValueError(
            f"{family} checkpoint must declare state_mode=stateless|rolling"
        )
    validate_experiment_identity(meta)
    if int(meta["history"]) <= 0 or int(meta["n_waypoints"]) <= 0:
        raise ValueError("checkpoint history and n_waypoints must be positive")
    if float(meta["dt"]) <= 0:
        raise ValueError("checkpoint dt must be positive")
    if not isinstance(meta["training_source_raw_dirs"], list) or not meta[
        "training_source_raw_dirs"
    ]:
        raise ValueError("checkpoint must bind non-empty training_source_raw_dirs")
    base_artifact = meta["base_model_artifact"]
    if (
        not isinstance(base_artifact, dict)
        or base_artifact.get("artifact_sha256") != meta["base_model_sha256"]
        or not isinstance(base_artifact.get("files"), list)
        or not base_artifact["files"]
    ):
        raise ValueError("checkpoint base model artifact binding is invalid")
    selection = meta["checkpoint_selection"]
    if selection != EXPECTED_CHECKPOINT_SELECTION:
        raise ValueError(
            "checkpoint selection must use the frozen validation BCE@1 rule: "
            f"{EXPECTED_CHECKPOINT_SELECTION}"
        )
    if meta["checkpoint_role"] not in {"epoch", "best_validation", "smoke"}:
        raise ValueError("checkpoint_role must be epoch|best_validation|smoke")
    if not isinstance(meta["selection_verified"], bool):
        raise ValueError("selection_verified must be a boolean")
    _validate_optimization_metadata(meta)
    return meta


def _experiment_contract(experiment_id):
    """Map a declared experiment ID to its immutable family/state contract."""

    value = str(experiment_id)
    if value == "B0":
        return "opentrackvla_baseline", "stateless"
    if value == "B1" or value.startswith(("B1-", "B1:")):
        return "trackvla_pp_lite", "rolling"
    if value == "H0-S" or value.startswith(("H0-S-", "H0-S:")):
        return "pfem_harness", "stateless"
    if value == "H0" or value.startswith(("H0-", "H0:")):
        return "pfem_harness", "rolling"
    raise ValueError(f"unsupported experiment_id={value!r}")


def validate_experiment_identity(meta):
    expected_family, expected_state = _experiment_contract(meta.get("experiment_id"))
    if meta.get("model_family") != expected_family:
        raise ValueError(
            f"experiment_id={meta.get('experiment_id')!r} requires "
            f"model_family={expected_family}"
        )
    if meta.get("state_mode") != expected_state:
        raise ValueError(
            f"experiment_id={meta.get('experiment_id')!r} requires "
            f"state_mode={expected_state}"
        )
    return True


def _finite_number(value, field, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    minimum_ok = numeric >= 0.0 if allow_zero else numeric > 0.0
    if not math.isfinite(numeric) or not minimum_ok:
        operator = ">=" if allow_zero else ">"
        raise ValueError(f"{field} must be finite and {operator} 0")
    return numeric


def _validate_optimization_metadata(meta):
    for field in ("optimizer_updates", "processed_samples", "batch_size", "grad_accum_steps"):
        value = meta[field]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
            raise ValueError(f"{field} must be a positive integer")
    effective = meta["effective_batch_size"]
    if (
        isinstance(effective, bool)
        or not isinstance(effective, (int, np.integer))
        or int(effective) <= 0
    ):
        raise ValueError("effective_batch_size must be a positive integer")
    expected_effective = int(meta["batch_size"]) * int(meta["grad_accum_steps"])
    if int(effective) != expected_effective:
        raise ValueError(
            "effective_batch_size must equal batch_size * grad_accum_steps"
        )
    _finite_number(meta["base_lr"], "base_lr")
    if meta["head_lr"] is None:
        if meta["model_family"] != "opentrackvla_baseline":
            raise ValueError("head_lr may be null only for the headless B0 baseline")
    else:
        _finite_number(meta["head_lr"], "head_lr")
    _finite_number(meta["weight_decay"], "weight_decay", allow_zero=True)
    _finite_number(meta["grad_clip"], "grad_clip")
    return True


def validate_frozen_test_checkpoint(checkpoint, meta, *, evaluation_tier):
    """Only validation-selected checkpoints may enter the locked test report."""

    if evaluation_tier not in EVALUATION_TIERS:
        raise ValueError(f"unsupported evaluation_tier={evaluation_tier!r}")
    if evaluation_tier != "locked_final":
        return False
    validation = meta.get("validation")
    if not isinstance(validation, dict) or not all(
        validation.get(field) not in (None, "")
        for field in ("data_manifest_hash", "data_jsonl_sha256", "sample_count")
    ):
        raise ValueError(
            "locked final test requires a checkpoint validation binding"
        )
    best = meta.get("best_validation")
    if not isinstance(best, dict) or "selection_bce_at1" not in best:
        raise ValueError("locked final test requires best_validation metadata")
    if meta.get("checkpoint_role") != "best_validation":
        raise ValueError(
            "locked final test requires checkpoint_role=best_validation"
        )
    if meta.get("selection_verified") is not True:
        raise ValueError("locked final test requires selection_verified=true")
    selected_epoch = meta.get("selected_epoch")
    checkpoint_epoch = checkpoint.get("epoch")
    if (
        isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, (int, np.integer))
        or int(selected_epoch) < 0
        or isinstance(checkpoint_epoch, bool)
        or not isinstance(checkpoint_epoch, (int, np.integer))
        or int(selected_epoch) != int(checkpoint_epoch)
    ):
        raise ValueError(
            "locked final test selected_epoch must equal the checkpoint epoch"
        )
    selected_value = _finite_number(
        meta.get("selected_value"), "selected_value", allow_zero=True
    )
    best_value = _finite_number(
        best.get("selection_bce_at1"),
        "best_validation.selection_bce_at1",
        allow_zero=True,
    )
    if not math.isclose(selected_value, best_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selected_value disagrees with best_validation")
    checkpoint_value = _finite_number(
        checkpoint.get("loss"), "checkpoint loss", allow_zero=True
    )
    if not math.isclose(selected_value, checkpoint_value, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("selected_value disagrees with checkpoint loss")
    return True


def validate_comparison_contracts(named_metas):
    """Require all compared runs to share the frozen matched-experiment contract."""

    invariant_fields = (
        "history",
        "n_waypoints",
        "dt",
        "action_semantics",
        "data_manifest_hash",
        "data_jsonl_sha256",
        "base_model_sha256",
        "base_model_artifact",
        "qwen_model_sha256",
        "vision_cache_manifest_sha256",
        "vision_cache_provenance_sha256",
        "vision_cache_token_payload_sha256",
        "dino_model_sha256",
        "siglip_model_sha256",
        "checkpoint_selection",
        "optimizer_updates",
        "processed_samples",
        "sampling_policy",
        "effective_batch_size",
        "base_lr",
        "weight_decay",
        "grad_clip",
    )
    reference_name = None
    reference = None
    state_references = {}
    head_reference = None
    for name, meta in named_metas:
        _validate_optimization_metadata(meta)
        if reference is None:
            reference_name, reference = name, meta
        else:
            mismatched = [
                field
                for field in invariant_fields
                if meta.get(field) != reference.get(field)
            ]
            if mismatched:
                raise ValueError(
                    f"comparison contract mismatch between {reference_name} and {name}: "
                    + ", ".join(mismatched)
                )

        # Stateful methods are constrained to micro-batch 1. Compare their raw
        # accumulation contract within each state regime while the effective
        # batch above remains invariant across all methods.
        state_mode = str(meta["state_mode"])
        state_reference = state_references.get(state_mode)
        if state_reference is None:
            state_references[state_mode] = (name, meta)
        else:
            state_name, state_meta = state_reference
            state_mismatched = [
                field
                for field in ("batch_size", "grad_accum_steps")
                if meta.get(field) != state_meta.get(field)
            ]
            if state_mismatched:
                raise ValueError(
                    f"comparison contract mismatch between {state_name} and {name}: "
                    + ", ".join(state_mismatched)
                )

        # B0 has no method-specific head. Every method that does have one must
        # nevertheless use the same head learning rate.
        if meta.get("head_lr") is not None:
            if head_reference is None:
                head_reference = (name, meta["head_lr"])
            elif meta["head_lr"] != head_reference[1]:
                raise ValueError(
                    f"comparison contract mismatch between {head_reference[0]} and {name}: "
                    "head_lr"
                )
    return True


def evaluation_sequence_key(record, *, require_sequence_id=True):
    """Return the clean-sequence identity used for rolling-state reset decisions.

    ``clip_id`` is intentionally absent from the identity. Clips are bounded
    loading/statistics blocks; rolling state may cross their boundary inside one
    clean chunk when frame indices remain consecutive.
    """

    sequence_id = record.get("sequence_id", record.get("clean_chunk_id"))
    if sequence_id in (None, ""):
        if require_sequence_id:
            raise ValueError(
                "rolling evaluation requires sequence_id=clean_chunk_id; "
                "episode/clip fallback is unsafe"
            )
        sequence_id = None
    if "frame_idx" not in record:
        raise ValueError("evaluation record is missing frame_idx")
    return (
        str(record.get("episode", "")),
        None if sequence_id is None else str(sequence_id),
        int(record["frame_idx"]),
        bool(record.get("mirrored", False)),
    )


def continues_evaluation_sequence(previous_key, current_key):
    """True for adjacent clean frames, including a boundary between clip blocks."""

    if previous_key is None or current_key is None:
        return False
    prev_episode, prev_sequence, prev_frame, prev_mirrored = previous_key
    episode, sequence, frame, mirrored = current_key
    return (
        not prev_mirrored
        and not mirrored
        and sequence is not None
        and episode == prev_episode
        and sequence == prev_sequence
        and frame == prev_frame + 1
    )


def validate_ordered_evaluation_records(records, *, require_sequence_id):
    """Reject ambiguous order while allowing frame gaps to reset rolling state."""

    seen = set()
    closed_sequences = set()
    previous_key = None
    for record in records:
        key = evaluation_sequence_key(
            record, require_sequence_id=require_sequence_id
        )
        episode, sequence, frame_idx, mirrored = key
        if mirrored:
            raise ValueError("validation/test evaluation cannot contain mirrored rows")
        identity = (episode, sequence, frame_idx)
        if identity in seen:
            raise ValueError(f"duplicate evaluation sample key: {identity}")
        seen.add(identity)
        if previous_key is not None:
            prev_episode, prev_sequence, prev_frame, _ = previous_key
            prev_identity = (prev_episode, prev_sequence)
            current_identity = (episode, sequence)
            if current_identity != prev_identity:
                closed_sequences.add(prev_identity)
                if current_identity in closed_sequences:
                    raise ValueError(
                        f"evaluation sequence is not contiguous in JSONL order: {current_identity}"
                    )
            elif frame_idx <= prev_frame:
                raise ValueError(
                    f"frame_idx must increase inside a sequence: {prev_frame} -> {frame_idx}"
                )
        previous_key = key
    return True


def validate_evaluation_dataset(
    dataset_path,
    checkpoint_meta,
    *,
    checkpoint=None,
    evaluation_tier="locked_final",
    expected_manifest_sha256=None,
):
    path = Path(dataset_path).expanduser().resolve()
    manifest_path = Path(str(path) + ".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"evaluation dataset/manifest missing: {path}")
    raw_manifest = manifest_path.read_bytes()
    manifest = json.loads(raw_manifest.decode("utf-8"))
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    actual_data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_data_hash != manifest.get("data_jsonl_sha256"):
        raise ValueError("evaluation JSONL sha256 does not match its manifest")
    if int(manifest.get("history", checkpoint_meta["history"])) != int(
        checkpoint_meta["history"]
    ):
        raise ValueError("evaluation history disagrees with checkpoint")
    if int(manifest.get("n_waypoints", checkpoint_meta["n_waypoints"])) != int(
        checkpoint_meta["n_waypoints"]
    ):
        raise ValueError("evaluation horizon disagrees with checkpoint")
    manifest_dt = manifest.get("dt", checkpoint_meta["dt"])
    if isinstance(manifest_dt, list) or not math.isclose(
        float(manifest_dt), float(checkpoint_meta["dt"]), rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError("evaluation dt disagrees with checkpoint")
    split = str(manifest.get("split", ""))
    if split not in {"val", "test"}:
        raise ValueError(f"evaluation dataset must declare split=val|test, got {split!r}")
    if split == "test":
        if not expected_manifest_sha256:
            raise ValueError(
                "test evaluation requires --expected_eval_manifest_sha256"
            )
        if manifest_hash != expected_manifest_sha256:
            raise ValueError("test manifest does not match the frozen expected hash")
        if not isinstance(checkpoint, dict):
            raise ValueError(
                "test evaluation requires the loaded checkpoint for selection verification"
            )
        validate_frozen_test_checkpoint(
            checkpoint, checkpoint_meta, evaluation_tier=evaluation_tier
        )
    validation = checkpoint_meta.get("validation")
    if split == "val":
        if not isinstance(validation, dict):
            raise ValueError("validation evaluation requires checkpoint validation binding")
        if manifest_hash != validation.get("data_manifest_hash"):
            raise ValueError("validation manifest hash disagrees with checkpoint")
        if actual_data_hash != validation.get("data_jsonl_sha256"):
            raise ValueError("validation data hash disagrees with checkpoint")
    evaluation_sources = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source = row.get("source_raw_dir")
            if source not in (None, ""):
                evaluation_sources.add(str(source))
    training_sources = {
        str(value) for value in checkpoint_meta.get("training_source_raw_dirs") or []
    }
    overlap = sorted(training_sources & evaluation_sources)
    if overlap:
        raise ValueError(
            "training/evaluation source_raw_dir overlap: " + ", ".join(overlap)
        )
    return {
        **manifest,
        "_verified_manifest_sha256": manifest_hash,
        "_verified_data_jsonl_sha256": actual_data_hash,
    }


def resolve_evaluation_identity(
    meta,
    *,
    evaluation_split,
    evaluation_tier,
    requested_state_mode=None,
    allow_state_mode_override=False,
):
    """Classify headline, validation, exploratory, smoke, and sensitivity runs."""

    if evaluation_split not in {"val", "test"}:
        raise ValueError(f"unsupported evaluation split={evaluation_split!r}")
    if evaluation_tier not in EVALUATION_TIERS:
        raise ValueError(f"unsupported evaluation_tier={evaluation_tier!r}")
    declared = str(meta["state_mode"])
    requested = requested_state_mode or declared
    override = requested != declared
    if override and not allow_state_mode_override:
        raise ValueError(
            "state-mode override is forbidden; use --allow_state_mode_override "
            "only for an explicitly non-headline sensitivity run"
        )
    if override and evaluation_tier == "locked_final":
        raise ValueError(
            "state-mode sensitivity runs cannot use evaluation_tier=locked_final"
        )
    checkpoint_experiment_id = str(meta["experiment_id"])
    if override:
        effective_experiment_id = (
            f"{checkpoint_experiment_id}-sensitivity:state_mode={requested}"
        )
        evaluation_class = "sensitivity"
    else:
        effective_experiment_id = checkpoint_experiment_id
        if evaluation_split == "test" and evaluation_tier == "locked_final":
            evaluation_class = "headline"
        elif evaluation_split == "val" and evaluation_tier == "locked_final":
            evaluation_class = "validation"
        else:
            evaluation_class = evaluation_tier
    return {
        "state_mode": requested,
        "state_mode_override": override,
        "checkpoint_experiment_id": checkpoint_experiment_id,
        "effective_experiment_id": effective_experiment_id,
        "evaluation_tier": evaluation_tier,
        "evaluation_class": evaluation_class,
        "headline_eligible": evaluation_class == "headline",
    }


def validate_label_mode_override(
    declared_label_mode,
    effective_label_mode,
    *,
    evaluation_tier,
):
    """Reject post-hoc label interpretation changes from locked-final results."""

    if evaluation_tier not in EVALUATION_TIERS:
        raise ValueError(f"unsupported evaluation_tier={evaluation_tier!r}")
    declared = str(declared_label_mode)
    effective = str(effective_label_mode)
    override = effective != declared
    if override and evaluation_tier == "locked_final":
        raise ValueError(
            "label-mode override cannot use evaluation_tier=locked_final; "
            "run it as an explicitly non-headline sensitivity analysis"
        )
    return override


def resolve_label_mode_for_evaluation(
    declared_label_mode,
    requested_label_mode,
    *,
    model_family,
    evaluation_tier,
):
    declared = str(declared_label_mode)
    effective = (
        declared if requested_label_mode in (None, "") else str(requested_label_mode)
    )
    override = validate_label_mode_override(
        declared, effective, evaluation_tier=evaluation_tier
    )
    if model_family in {"opentrackvla_baseline", "trackvla_pp_lite"}:
        if effective != "absolute":
            raise ValueError(f"{model_family} evaluation supports label_mode=absolute only")
        effective = "absolute"
    return effective, override


def smooth_l1_values(pred, target):
    difference = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    beta = float(SMOOTH_L1_BETA)
    return np.where(
        difference < beta,
        0.5 * difference**2 / beta,
        difference - 0.5 * beta,
    )


def waypoints_to_step_actions(waypoints, dt):
    """Invert the shared discrete pose composition for absolute checkpoints."""

    poses = np.asarray(waypoints, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None, ...]
        squeeze = True
    elif poses.ndim == 3:
        squeeze = False
    else:
        raise ValueError("waypoints must have shape (T, 3) or (B, T, 3)")
    if poses.shape[-1] != 3 or float(dt) <= 0:
        raise ValueError("waypoints must have 3 axes and dt must be > 0")

    previous = np.concatenate((np.zeros_like(poses[:, :1]), poses[:, :-1]), axis=1)
    world_delta = poses[..., :2] - previous[..., :2]
    yaw_before = previous[..., 2]
    cos_yaw = np.cos(yaw_before)
    sin_yaw = np.sin(yaw_before)
    forward = (cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]) / float(dt)
    strafe = (-sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]) / float(dt)
    yaw = (poses[..., 2] - previous[..., 2]) / float(dt)
    actions = np.stack((forward, strafe, yaw), axis=-1)
    return actions[0] if squeeze else actions


def transition_event_mask(actions, prev_actions, threshold=0.2):
    sequence = np.asarray(actions, dtype=np.float64)
    previous = np.asarray(prev_actions, dtype=np.float64)
    prior_yaw = np.concatenate((previous[:, None, 2], sequence[:, :-1, 2]), axis=1)
    yaw = sequence[..., 2]
    active = np.abs(yaw) > float(threshold)
    prior_active = np.abs(prior_yaw) > float(threshold)
    sign_flip = active & prior_active & (np.sign(yaw) != np.sign(prior_yaw))
    return (active != prior_active) | sign_flip


def _safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else None


def _f1_from_counts(true_positive, false_positive, false_negative):
    counts = (true_positive, false_positive, false_negative)
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 0
        for value in counts
    ):
        raise ValueError("F1 counts must be non-negative integers")
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return None
    return float(2 * true_positive) / float(denominator)


def compute_metrics(pred_actions, gt_actions, prev_actions, valid_mask=None, threshold=0.2):
    pred = np.asarray(pred_actions, dtype=np.float64)
    target = np.asarray(gt_actions, dtype=np.float64)
    previous = np.asarray(prev_actions, dtype=np.float64)
    if pred.shape != target.shape or pred.ndim != 3 or pred.shape[-1] != 3:
        raise ValueError("pred_actions and gt_actions must share shape (N, T, 3)")
    if previous.shape != (pred.shape[0], 3):
        raise ValueError("prev_actions must have shape (N, 3)")
    mask = (
        np.ones(pred.shape[:2], dtype=bool)
        if valid_mask is None
        else np.asarray(valid_mask, dtype=bool)
    )
    if mask.shape != pred.shape[:2]:
        raise ValueError("valid_mask must have shape (N, T)")

    errors = smooth_l1_values(pred, target)
    per_axis = {}
    saturation = {}
    for axis, name in enumerate(AXIS_NAMES):
        values = errors[..., axis][mask]
        per_axis[name] = float(values.mean()) if values.size else None
        saturated = np.abs(pred[..., axis])[mask] > SATURATION_THRESHOLD
        saturation[name] = float(saturated.mean()) if saturated.size else None

    turn_mask = mask & (np.abs(target[..., 2]) > float(threshold))
    sign_correct = np.sign(pred[..., 2]) == np.sign(target[..., 2])
    turn_sign_accuracy = _safe_ratio(np.count_nonzero(sign_correct & turn_mask), np.count_nonzero(turn_mask))

    gt_events = transition_event_mask(target, previous, threshold) & mask
    pred_events = transition_event_mask(pred, previous, threshold) & mask
    true_positive = int(np.count_nonzero(gt_events & pred_events))
    false_positive = int(np.count_nonzero(~gt_events & pred_events & mask))
    false_negative = int(np.count_nonzero(gt_events & ~pred_events & mask))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _f1_from_counts(true_positive, false_positive, false_negative)
    saturated_control = (
        np.abs(pred[..., (0, 2)]) > SATURATION_THRESHOLD
    ).any(axis=-1)[mask]
    saturated_all = (
        np.abs(pred)[np.repeat(mask[..., None], 3, axis=-1)]
        > SATURATION_THRESHOLD
    )
    return {
        "samples": int(pred.shape[0]),
        "valid_steps": int(np.count_nonzero(mask)),
        "smooth_l1": per_axis,
        "turn_sign_accuracy": turn_sign_accuracy,
        "transition": {
            "precision": precision,
            "recall": recall,
            "f1": None if f1 is None else float(f1),
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
        },
        "saturation_rate": {
            "overall": (
                float(saturated_control.mean()) if saturated_control.size else None
            ),
            "all_axes_diagnostic": (
                float(saturated_all.mean()) if saturated_all.size else None
            ),
            **saturation,
        },
    }


def _event_type(previous_yaw, yaw, threshold):
    previous_active = abs(float(previous_yaw)) > float(threshold)
    active = abs(float(yaw)) > float(threshold)
    if not previous_active and active:
        return "onset"
    if previous_active and not active:
        return "exit"
    if previous_active and active and np.sign(yaw) != np.sign(previous_yaw):
        return "sign_flip"
    return None


def _match_events(predicted, target, tolerance):
    """Maximum-cardinality one-to-one matching inside a frame tolerance.

    A nearest-first greedy match is not sufficient: predicted ``[0, 3]`` and
    target ``[2, 3]`` at tolerance 2 must produce two matches, not one.  On the
    sorted one-dimensional timelines an optimal matching can always be
    uncrossed, so the rolling dynamic program below returns the exact maximum.
    """

    predicted_frames = sorted(int(frame) for frame in predicted)
    target_frames = sorted(int(frame) for frame in target)
    tolerance = int(tolerance)
    if tolerance < 0:
        raise ValueError("event matching tolerance must be >= 0")
    previous = [0] * (len(target_frames) + 1)
    for predicted_frame in predicted_frames:
        current = [0]
        for target_index, target_frame in enumerate(target_frames, start=1):
            best = max(previous[target_index], current[target_index - 1])
            if abs(predicted_frame - target_frame) <= tolerance:
                best = max(best, previous[target_index - 1] + 1)
            current.append(best)
        previous = current
    true_positive = previous[-1]
    false_positive = len(predicted) - true_positive
    false_negative = len(target) - true_positive
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _f1_from_counts(true_positive, false_positive, false_negative)
    return {
        "precision": precision,
        "recall": recall,
        "f1": None if f1 is None else float(f1),
        "tp": int(true_positive),
        "fp": int(false_positive),
        "fn": int(false_negative),
    }


def chronological_transition_metrics(
    pred_actions, records, *, threshold=0.2, tolerance=2
):
    """Match h=1 onset/exit/sign-flip events on each clean sequence timeline."""

    prediction = np.asarray(pred_actions, dtype=np.float64)
    if prediction.ndim != 3 or prediction.shape[0] != len(records):
        raise ValueError("pred_actions must have shape (N,T,3) matching records")
    events = {
        "pred": defaultdict(list),
        "gt": defaultdict(list),
    }
    previous = {}
    for index, record in enumerate(records):
        sequence = str(
            record.get("sequence_id")
            or record.get("chunk_id")
            or record.get("episode")
            or "__legacy__"
        )
        frame = int(record.get("frame_idx", index))
        valid = record.get("valid_mask")
        if valid is not None and not bool(valid[0]):
            previous.pop(sequence, None)
            continue
        gt_yaw = float(np.asarray(record["step_actions"], dtype=np.float64)[0, 2])
        pred_yaw = float(prediction[index, 0, 2])
        prior = previous.get(sequence)
        if prior is None or frame != prior["frame"] + 1:
            initial = float(np.asarray(record["prev_action"], dtype=np.float64)[2])
            prior_gt = initial
            prior_pred = initial
        else:
            prior_gt = prior["gt"]
            prior_pred = prior["pred"]
        gt_type = _event_type(prior_gt, gt_yaw, threshold)
        pred_type = _event_type(prior_pred, pred_yaw, threshold)
        if gt_type is not None:
            events["gt"][sequence, gt_type].append(frame)
        if pred_type is not None:
            events["pred"][sequence, pred_type].append(frame)
        previous[sequence] = {"frame": frame, "gt": gt_yaw, "pred": pred_yaw}

    by_type = {}
    for event_type in ("onset", "exit", "sign_flip"):
        totals = {"tp": 0, "fp": 0, "fn": 0}
        sequence_keys = {
            key[0]
            for key in set(events["pred"]) | set(events["gt"])
            if key[1] == event_type
        }
        for sequence in sequence_keys:
            matched = _match_events(
                events["pred"].get((sequence, event_type), []),
                events["gt"].get((sequence, event_type), []),
                tolerance,
            )
            for field in totals:
                totals[field] += matched[field]
        precision = _safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
        recall = _safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
        f1 = _f1_from_counts(totals["tp"], totals["fp"], totals["fn"])
        by_type[event_type] = {
            "precision": precision,
            "recall": recall,
            "f1": None if f1 is None else float(f1),
            **totals,
        }
    total_tp = sum(value["tp"] for value in by_type.values())
    total_fp = sum(value["fp"] for value in by_type.values())
    total_fn = sum(value["fn"] for value in by_type.values())
    precision = _safe_ratio(total_tp, total_tp + total_fp)
    recall = _safe_ratio(total_tp, total_tp + total_fn)
    f1 = _f1_from_counts(total_tp, total_fp, total_fn)
    return {
        "tolerance_frames": int(tolerance),
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": None if f1 is None else float(f1),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "by_type": by_type,
    }


def _mae(values, targets, mask):
    selected = np.abs(values - targets)[mask]
    return float(selected.mean()) if selected.size else None


def balanced_control_error_at1(pred_actions, records):
    """Episode-macro, command-balanced first-action control error."""

    pred = np.asarray(pred_actions, dtype=np.float64)
    by_episode = defaultdict(lambda: defaultdict(list))
    support = defaultdict(lambda: defaultdict(int))
    for index, record in enumerate(records):
        valid = record.get("valid_mask")
        if valid is not None and not bool(valid[0]):
            continue
        command = str(record.get("command", ""))
        if command not in PRIMARY_COMMANDS:
            continue
        target = np.asarray(record["step_actions"], dtype=np.float64)[0]
        error = (
            CONTROL_ERROR_WEIGHTS["forward"]
            * abs(float(pred[index, 0, 0] - target[0]))
            + CONTROL_ERROR_WEIGHTS["yaw"]
            * abs(float(pred[index, 0, 2] - target[2]))
        ) / float(sum(CONTROL_ERROR_WEIGHTS.values()))
        episode = str(record.get("source_raw_dir") or record.get("episode") or "unknown")
        by_episode[episode][command].append(error)
        support[episode][command] += 1

    episode_values = {}
    command_values = defaultdict(list)
    for episode, groups in by_episode.items():
        per_command = {
            command: float(np.mean(values))
            for command, values in groups.items()
            if values
        }
        if not per_command:
            continue
        episode_values[episode] = float(np.mean(list(per_command.values())))
        for command, value in per_command.items():
            command_values[command].append(value)
    return {
        "value": (
            float(np.mean(list(episode_values.values())))
            if episode_values
            else None
        ),
        "by_episode": episode_values,
        "by_command": {
            command: float(np.mean(values))
            for command, values in sorted(command_values.items())
        },
        "support": {
            episode: dict(sorted(commands.items()))
            for episode, commands in sorted(support.items())
        },
    }


def horizon_metrics(pred_actions, records, horizons=HORIZON_IDS):
    pred = np.asarray(pred_actions, dtype=np.float64)
    target = np.asarray([record["step_actions"] for record in records], dtype=np.float64)
    valid = np.asarray(
        [record.get("valid_mask", [True] * target.shape[1]) for record in records],
        dtype=bool,
    )
    result = {}
    for horizon in horizons:
        index = int(horizon) - 1
        if index < 0 or index >= pred.shape[1]:
            continue
        mask = valid[:, index]
        result[str(horizon)] = {
            "forward_mae": _mae(pred[:, index, 0], target[:, index, 0], mask),
            "yaw_mae": _mae(pred[:, index, 2], target[:, index, 2], mask),
        }
    return result


def evaluate_predictions(pred_actions, records, threshold=0.2):
    if not records:
        raise ValueError("validation dataset is empty")
    gt = np.asarray([record["step_actions"] for record in records], dtype=np.float64)
    previous = np.asarray([record["prev_action"] for record in records], dtype=np.float64)
    valid = np.asarray(
        [record.get("valid_mask", [True] * gt.shape[1]) for record in records],
        dtype=bool,
    )
    transitions = [str(record.get("transition_type", "other")) for record in records]
    result = compute_metrics(pred_actions, gt, previous, valid, threshold)
    result["balanced_control_error_at1"] = balanced_control_error_at1(
        pred_actions, records
    )
    result["chronological_transition"] = chronological_transition_metrics(
        pred_actions,
        records,
        threshold=threshold,
        tolerance=EVENT_TOLERANCE_FRAMES,
    )
    result["horizons"] = horizon_metrics(
        pred_actions, records, horizons=HORIZON_IDS
    )
    result["by_transition_type"] = {}
    for transition_type in sorted(set(transitions)):
        indices = [index for index, value in enumerate(transitions) if value == transition_type]
        result["by_transition_type"][transition_type] = compute_metrics(
            np.asarray(pred_actions)[indices],
            gt[indices],
            previous[indices],
            valid[indices],
            threshold,
        )
    result["by_episode"] = {}
    episode_names = [
        str(record.get("source_raw_dir") or record.get("episode") or "unknown")
        for record in records
    ]
    for episode in sorted(set(episode_names)):
        indices = [i for i, value in enumerate(episode_names) if value == episode]
        result["by_episode"][episode] = compute_metrics(
            np.asarray(pred_actions)[indices],
            gt[indices],
            previous[indices],
            valid[indices],
            threshold,
        )
    commands = [str(record.get("command", "unknown")) for record in records]
    result["by_command"] = {}
    for command in sorted(set(commands)):
        indices = [i for i, value in enumerate(commands) if value == command]
        result["by_command"][command] = compute_metrics(
            np.asarray(pred_actions)[indices],
            gt[indices],
            previous[indices],
            valid[indices],
            threshold,
        )
    return result


def _parse_named_paths(values):
    runs = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
        else:
            path = value
            name = Path(path).stem
        if not name or not path:
            raise ValueError(f"invalid checkpoint spec: {value!r}")
        runs.append((name, Path(path).expanduser().resolve()))
    return runs


def _parse_mode_overrides(values):
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"mode override must be NAME=MODE: {value!r}")
        name, mode = value.split("=", 1)
        if mode not in {"absolute", "step_action"}:
            raise ValueError(f"invalid mode override: {value!r}")
        overrides[name] = mode
    return overrides


def validate_mode_override_names(overrides, run_names):
    unknown = sorted(set(overrides) - {str(name) for name in run_names})
    if unknown:
        raise ValueError(
            "label-mode override names do not match any --ckpt run: "
            + ", ".join(unknown)
        )
    return True


def _collect_checkpoint_predictions(
    checkpoint_path,
    val_json,
    args,
    mode_override=None,
    *,
    provenance_contract,
):
    import torch
    from torch.utils.data import DataLoader

    from inference_pipeline import mac_server
    from harness.baseline_adapter import OpenTrackVLABaselineAdapter
    from harness.sequence_state import continues_sequence, detach_state, sample_sequence_key
    from harness.trackvla_lite import TrackVLAPlusPlusLite
    from model import DataConfig, JsonTrackingDataset, collate_batch

    checkpoint, checkpoint_sha256 = load_checkpoint_with_sha256(checkpoint_path)
    if provenance_contract.get("checkpoint_sha256") != checkpoint_sha256:
        raise RuntimeError(
            "checkpoint bytes changed after provenance preflight: "
            f"expected={provenance_contract.get('checkpoint_sha256')}, "
            f"actual={checkpoint_sha256}"
        )
    meta = validate_checkpoint_metadata(checkpoint)
    best_validation = meta.get("best_validation")
    selection_detail = (
        best_validation.get("selection_detail")
        if isinstance(best_validation, dict)
        else None
    )
    preflight_checks = {
        "fairness_contract_sha256": canonical_json_sha256(
            build_fairness_contract(meta)
        ),
        "method_contract_sha256": canonical_json_sha256(
            build_method_contract(meta)
        ),
        "validation_selection_detail_sha256": (
            canonical_json_sha256(selection_detail)
            if isinstance(selection_detail, dict)
            else None
        ),
        "checkpoint_role": meta.get("checkpoint_role"),
        "selection_verified": meta.get("selection_verified"),
        "selected_epoch": meta.get("selected_epoch"),
        "selected_value": meta.get("selected_value"),
        "checkpoint_seed": int(meta["seed"]),
    }
    changed = [
        field
        for field, value in preflight_checks.items()
        if provenance_contract.get(field) != value
    ]
    if changed:
        raise RuntimeError(
            "checkpoint metadata changed after provenance preflight: "
            + ", ".join(changed)
        )
    model_family = meta["model_family"]
    declared_label_mode = str(meta.get("label_mode", "absolute"))
    label_mode, label_mode_override = resolve_label_mode_for_evaluation(
        declared_label_mode,
        mode_override,
        model_family=model_family,
        evaluation_tier=args.evaluation_tier,
    )
    history = int(meta.get("history", args.history))
    n_waypoints = int(meta.get("n_waypoints", args.n_waypoints))
    dt = float(meta.get("dt", args.dt))
    train_args = meta.get("train_args") if isinstance(meta.get("train_args"), dict) else {}
    aux_delta_vel = bool(meta.get("aux_delta_vel", train_args.get("aux_delta_vel", False)))
    evaluation_binding = validate_evaluation_dataset(
        val_json,
        meta,
        checkpoint=checkpoint,
        evaluation_tier=args.evaluation_tier,
        expected_manifest_sha256=args.expected_eval_manifest_sha256,
    )
    identity = resolve_evaluation_identity(
        meta,
        evaluation_split=str(evaluation_binding["split"]),
        evaluation_tier=args.evaluation_tier,
        requested_state_mode=args.state_mode,
        allow_state_mode_override=args.allow_state_mode_override,
    )

    root = mac_server.resolve_opentrackvla_root(args.opentrackvla_root)
    weight_args = SimpleNamespace(
        qwen_model_path=args.qwen_model_path,
        dinov3_model_path=args.dinov3_model_path,
        siglip_model_path=args.siglip_model_path,
        base_hf_model_dir=args.base_hf_model_dir or meta.get("base_hf_model_dir"),
    )
    mac_server.configure_default_weight_paths(weight_args, root)
    base_artifact = bind_hf_model_artifact(weight_args.base_hf_model_dir)
    if (
        base_artifact != meta["base_model_artifact"]
        or base_artifact["artifact_sha256"] != meta["base_model_sha256"]
    ):
        raise RuntimeError(
            "loaded base OpenTrackVLA config/weights disagree with checkpoint"
        )
    if sha256_artifact(weight_args.qwen_model_path) != meta["qwen_model_sha256"]:
        raise RuntimeError("loaded Qwen weights disagree with checkpoint")
    cache_info = verify_vision_cache(
        args.cache_root, [val_json], verify_payload=True
    )
    expected_cache = {
        "cache_manifest_sha256": meta["vision_cache_manifest_sha256"],
        "cache_provenance_sha256": meta[
            "vision_cache_provenance_sha256"
        ],
        "token_payload_sha256": meta["vision_cache_token_payload_sha256"],
        "dino_model_sha256": meta["dino_model_sha256"],
        "siglip_model_sha256": meta["siglip_model_sha256"],
    }
    if any(cache_info[field] != value for field, value in expected_cache.items()):
        raise RuntimeError("evaluation vision cache disagrees with checkpoint")
    device = torch.device(args.device or mac_server.default_device())
    if model_family == "opentrackvla_baseline":
        if not weight_args.base_hf_model_dir:
            raise FileNotFoundError("B0 evaluation requires --base_hf_model_dir")
        base = mac_server.load_official_base(weight_args.base_hf_model_dir).to(device)
        missing, unexpected = base.load_state_dict(
            checkpoint.get("model_state", {}), strict=False
        )
        missing = [key for key in missing if not key.startswith("llm.")]
        if missing or unexpected:
            raise RuntimeError(
                f"B0 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        print(
            f"[eval] loaded B0: {len(missing)} missing, {len(unexpected)} unexpected"
        )
        model = OpenTrackVLABaselineAdapter(base).to(device).eval()
    elif model_family == "trackvla_pp_lite":
        if not weight_args.base_hf_model_dir:
            raise FileNotFoundError("B1 evaluation requires --base_hf_model_dir")
        base = mac_server.load_official_base(weight_args.base_hf_model_dir)
        lite_variant = str(meta.get("trackvla_lite_variant", "polar_tim4"))
        model = TrackVLAPlusPlusLite(
            base,
            expected_history=history,
            use_tim=lite_variant != "polar_only",
            tim_tokens=16 if lite_variant == "polar_tim16" else 4,
        ).to(device)
        missing, unexpected = model.load_state_dict(
            checkpoint.get("model_state", {}), strict=False
        )
        missing = [key for key in missing if not key.startswith("base.llm.")]
        if missing or unexpected:
            raise RuntimeError(
                f"B1 checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        model.eval()
        print("[eval] loaded B1 TrackVLA++-Lite")
    elif model_family == "pfem_harness":
        model = mac_server.load_model(
            checkpoint,
            device,
            root,
            base_hf_model_dir=weight_args.base_hf_model_dir,
            n_waypoints=n_waypoints,
            label_mode=label_mode,
            control_dt=dt,
            aux_delta_vel=aux_delta_vel,
            strict_checkpoint=True,
        )
    else:
        raise ValueError(f"unsupported checkpoint model_family={model_family!r}")
    dataset = JsonTrackingDataset(
        DataConfig(
            train_json=str(val_json),
            n_waypoints=n_waypoints,
            history=history,
            cache_root=args.cache_root,
            default_dt=dt,
            require_cached_tokens=True,
        )
    )
    if int(evaluation_binding.get("sample_count", -1)) != len(dataset):
        raise ValueError(
            "evaluation dataset sample_count disagrees with its verified manifest"
        )
    state_mode = identity["state_mode"]
    if state_mode == "rolling" and args.batch_size != 1:
        raise ValueError("rolling offline evaluation requires --batch_size 1")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    evaluation_execution_contract = build_evaluation_execution_contract(
        evaluation_binding=evaluation_binding,
        identity=identity,
        declared_state_mode=meta["state_mode"],
        declared_label_mode=declared_label_mode,
        effective_label_mode=label_mode,
        label_mode_override=label_mode_override,
        batch_size=args.batch_size,
        history=history,
        n_waypoints=n_waypoints,
        dt=dt,
        device=device,
        default_dtype=torch.get_default_dtype(),
        parameter_dtypes={parameter.dtype for parameter in model.parameters()},
        buffer_dtypes={buffer.dtype for buffer in model.buffers()},
        dataset_sample_count=len(dataset),
    )
    predictions = []
    records = []
    raw_examples = [dataset.get_example(index) for index in range(len(dataset))]
    validate_ordered_evaluation_records(
        raw_examples, require_sequence_id=state_mode == "rolling"
    )
    model.eval()
    rolling_state = None
    previous_key = None
    with torch.inference_mode():
        for batch in loader:
            batch_size = batch["coarse_tokens"].size(0)
            record_offset = len(records)
            current_key = sample_sequence_key(batch) if state_mode == "rolling" else None
            carry_forward = (
                state_mode == "rolling" and current_key is not None and not current_key[2]
            )
            batch_state = rolling_state
            if (
                not carry_forward
                or not continues_sequence(previous_key, current_key)
            ):
                batch_state = model.init_state(batch_size, device)
            output = model.forward_step(
                coarse_tokens=batch["coarse_tokens"].to(device),
                coarse_tidx=batch["coarse_tidx"].to(device),
                fine_tokens=batch["fine_tokens"].to(device),
                fine_tidx=batch["fine_tidx"].to(device),
                instructions=batch["instruction"],
                prev_state=batch_state,
                yaw_hist=batch["yaw_hist"].to(device),
                yaw_curr=batch["yaw_curr"].to(device),
                prev_action=batch["prev_action"].to(device),
            )
            if label_mode == "step_action":
                batch_predictions = output["step_actions"].detach().cpu().numpy()
            else:
                batch_predictions = waypoints_to_step_actions(
                    output["waypoints"].detach().cpu().numpy(), dt
                )
            predictions.extend(batch_predictions.tolist())
            for index in range(batch_size):
                records.append(
                    {
                        "step_actions": batch["step_actions"][index].tolist(),
                        "prev_action": batch["prev_action"][index].tolist(),
                        "valid_mask": batch["valid_mask"][index].tolist(),
                        "transition_type": batch["transition_type"][index],
                        "episode": batch["episode"][index],
                        "sequence_id": batch["sequence_id"][index],
                        "chunk_id": batch["chunk_id"][index],
                        "clip_id": batch["clip_id"][index],
                        "frame_idx": int(batch["frame_idx"][index].item()),
                        "mirrored": bool(batch["mirrored"][index].item()),
                        "command": str(raw_examples[record_offset + index].get("command", "unknown")),
                        "source_raw_dir": raw_examples[record_offset + index].get("source_raw_dir"),
                    }
                )
            if carry_forward:
                rolling_state = detach_state(output.get("new_state", {}))
                previous_key = current_key
    metrics = evaluate_predictions(
        np.asarray(predictions), records, args.transition_threshold
    )
    metrics["model_family"] = model_family
    metrics["state_mode"] = state_mode
    metrics["experiment_id"] = identity["effective_experiment_id"]
    metrics["checkpoint_experiment_id"] = identity["checkpoint_experiment_id"]
    metrics["evaluation_class"] = identity["evaluation_class"]
    metrics["headline_eligible"] = identity["headline_eligible"]
    metrics["seed"] = int(meta["seed"])
    verify_checkpoint_file_unchanged(checkpoint_path, checkpoint_sha256)
    provenance = {
        **provenance_contract,
        **identity,
        "evaluation_execution_contract": evaluation_execution_contract,
        "evaluation_execution_contract_sha256": canonical_json_sha256(
            evaluation_execution_contract
        ),
        "test_manifest_sha256": (
            evaluation_binding["_verified_manifest_sha256"]
            if evaluation_binding["split"] == "test"
            else None
        ),
        "vision_cache_manifest_sha256": cache_info["cache_manifest_sha256"],
        "vision_cache_provenance_sha256": cache_info[
            "cache_provenance_sha256"
        ],
        "vision_cache_token_payload_sha256": cache_info["token_payload_sha256"],
        "dino_model_sha256": cache_info["dino_model_sha256"],
        "siglip_model_sha256": cache_info["siglip_model_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
    }
    return label_mode, metrics, predictions, records, provenance


def _fmt(value):
    return "n/a" if value is None or not math.isfinite(float(value)) else f"{float(value):.4f}"


def render_comparison_table(results):
    lines = [
        "| run | experiment | seed | state | mode | BCE@1 ↓ | fwd SmoothL1 | yaw SmoothL1 | turn-sign acc | timeline F1 | saturation(fwd+yaw) |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, payload in results.items():
        metrics = payload["metrics"]
        lines.append(
            "| " + " | ".join(
                (
                    name,
                    str(metrics.get("experiment_id", "n/a")),
                    str(metrics.get("seed", "n/a")),
                    str(metrics.get("state_mode", "n/a")),
                    payload["label_mode"],
                    _fmt(metrics["balanced_control_error_at1"]["value"]),
                    _fmt(metrics["smooth_l1"]["forward"]),
                    _fmt(metrics["smooth_l1"]["yaw"]),
                    _fmt(metrics["turn_sign_accuracy"]),
                    _fmt(metrics["chronological_transition"]["f1"]),
                    _fmt(metrics["saturation_rate"]["overall"]),
                )
            ) + " |"
        )
    return "\n".join(lines)


def render_group_tables(results):
    sections = []
    for name, payload in results.items():
        lines = [
            f"### {name} by transition_type",
            "",
            "| transition_type | samples | fwd SmoothL1 | strafe SmoothL1 | yaw SmoothL1 | turn-sign acc | transition F1 | saturation |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for group, metrics in payload["metrics"]["by_transition_type"].items():
            lines.append(
                f"| {group} | {metrics['samples']} | {_fmt(metrics['smooth_l1']['forward'])} | "
                f"{_fmt(metrics['smooth_l1']['strafe'])} | {_fmt(metrics['smooth_l1']['yaw'])} | "
                f"{_fmt(metrics['turn_sign_accuracy'])} | "
                f"{_fmt(metrics['transition']['f1'])} | {_fmt(metrics['saturation_rate']['overall'])} |"
            )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_json", required=True)
    parser.add_argument(
        "--ckpt",
        action="append",
        required=True,
        help="Checkpoint path or NAME=PATH; repeat for a comparison table.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=[],
        help="Optional NAME=absolute|step_action override; otherwise checkpoint meta is used.",
    )
    parser.add_argument("--json_output", default=None)
    parser.add_argument(
        "--predictions_dir",
        default=None,
        help="Optional directory for per-anchor prediction JSONL artifacts.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument("--n_waypoints", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--transition_threshold", type=float, default=0.2)
    parser.add_argument(
        "--state_mode", choices=("stateless", "rolling"), default=None
    )
    parser.add_argument("--allow_state_mode_override", action="store_true")
    parser.add_argument(
        "--evaluation_tier",
        choices=tuple(sorted(EVALUATION_TIERS)),
        default="locked_final",
        help=(
            "locked_final admits only validation-selected best checkpoints on test; "
            "exploratory/smoke outputs are never headline-eligible"
        ),
    )
    parser.add_argument("--expected_eval_manifest_sha256", default=None)
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--opentrackvla_root", default=None)
    parser.add_argument("--base_hf_model_dir", default=None)
    parser.add_argument("--qwen_model_path", default=None)
    parser.add_argument("--dinov3_model_path", default=None)
    parser.add_argument("--siglip_model_path", default=None)
    parser.add_argument(
        "--experiment_registry",
        required=True,
        help="Frozen experiment registry used to bind every compared checkpoint.",
    )
    parser.add_argument(
        "--expected_registry_sha256",
        required=True,
        help="Predeclared SHA-256 of --experiment_registry.",
    )
    parser.add_argument(
        "--expected_source_tree_sha256",
        default=None,
        help=(
            "Predeclared OpenTrackVLA Python-tree SHA-256. May be omitted only "
            "when the frozen registry contains source_tree_sha256."
        ),
    )
    parser.add_argument(
        "--expected_evaluator_source_sha256",
        default=None,
        help=(
            "Predeclared evaluator-source SHA-256. May be omitted only when "
            "the frozen registry contains evaluator_source_sha256."
        ),
    )
    parser.add_argument(
        "--expected_metric_contract_sha256",
        default=None,
        help=(
            "Predeclared canonical metric-contract SHA-256. May be omitted only "
            "when the frozen registry contains metric_contract_sha256."
        ),
    )
    return parser


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args = build_parser().parse_args(argv)
    if args.evaluation_tier == "locked_final" and not args.predictions_dir:
        raise ValueError(
            "locked_final evaluation requires --predictions_dir so predictions "
            "can be content-bound to the result provenance"
        )
    runs = _parse_named_paths(args.ckpt)
    overrides = _parse_mode_overrides(args.mode)
    validate_mode_override_names(overrides, [name for name, _ in runs])
    from inference_pipeline import mac_server

    registry, registry_sha256 = load_experiment_registry(
        args.experiment_registry, args.expected_registry_sha256
    )
    evaluator_source = build_evaluator_source(PROJECT_ROOT)
    evaluator_source_sha256 = canonical_json_sha256(evaluator_source)
    validate_expected_contract_sha256(
        evaluator_source_sha256,
        registry,
        "evaluator_source_sha256",
        explicit_expected=args.expected_evaluator_source_sha256,
    )
    metric_contract = build_metric_contract(args.transition_threshold)
    metric_contract_sha256 = canonical_json_sha256(metric_contract)
    validate_expected_contract_sha256(
        metric_contract_sha256,
        registry,
        "metric_contract_sha256",
        explicit_expected=args.expected_metric_contract_sha256,
    )
    source_root = mac_server.resolve_opentrackvla_root(args.opentrackvla_root)
    actual_source_sha256 = source_tree_sha256(source_root)

    comparison_metas = []
    provenance_contracts = {}
    for name, checkpoint_path in runs:
        checkpoint, checkpoint_sha256 = load_checkpoint_with_sha256(
            checkpoint_path
        )
        meta = validate_checkpoint_metadata(checkpoint)
        if args.evaluation_tier == "locked_final":
            validate_frozen_test_checkpoint(
                checkpoint, meta, evaluation_tier=args.evaluation_tier
            )
        comparison_metas.append((name, meta))
        provenance_contract = validate_registry_checkpoint_binding(
            meta,
            registry,
            registry_sha256=registry_sha256,
            actual_source_tree_sha256=actual_source_sha256,
            expected_source_tree_sha256=args.expected_source_tree_sha256,
        )
        provenance_contract["checkpoint_sha256"] = checkpoint_sha256
        provenance_contract.update(
            {
                "evaluator_source": evaluator_source,
                "evaluator_source_sha256": evaluator_source_sha256,
                "metric_contract": metric_contract,
                "metric_contract_sha256": metric_contract_sha256,
            }
        )
        provenance_contracts[name] = provenance_contract
        del checkpoint
    validate_comparison_contracts(comparison_metas)
    results = {}
    for name, checkpoint_path in runs:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        label_mode, metrics, predictions, records, provenance = _collect_checkpoint_predictions(
            checkpoint_path,
            Path(args.val_json).expanduser().resolve(),
            args,
            overrides.get(name),
            provenance_contract=provenance_contracts[name],
        )
        results[name] = {
            "checkpoint": str(checkpoint_path),
            "label_mode": label_mode,
            "metrics": metrics,
            "provenance": provenance,
        }
        if args.predictions_dir:
            prediction_path = Path(args.predictions_dir).expanduser() / f"{name}.jsonl"
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            with prediction_path.open("w", encoding="utf-8") as handle:
                for prediction, record in zip(predictions, records):
                    handle.write(
                        json.dumps(
                            {**record, "pred_step_actions": prediction},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            results[name]["predictions"] = str(prediction_path)
            results[name]["provenance"][
                "evaluation_predictions_sha256"
            ] = sha256_file(prediction_path)
    print(render_comparison_table(results))
    print()
    print(render_group_tables(results))
    if args.json_output:
        output = Path(args.json_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
