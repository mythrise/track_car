from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from f2_experiment.assembly import EVAL_MODE_CONTRACT
from f2_experiment.controller import DEFAULT_CONFIG
from f2_experiment.model import ARCHITECTURE_LOCK as F2_ARCHITECTURE_LOCK
from f2_experiment.runner import RunnerRow

import ibr1_experiment.eval_guard as eval_guard_module
from ibr1_experiment.assembly_model import (
    FAMILY_TO_ENGINE_ARM,
    IBR1_FROZEN_AUX_COEFFICIENTS,
)
from ibr1_experiment.authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    CAL_NUMERIC_EVIDENCE_CLASS,
    SUPPORT_BINDING_CLASS,
    canonical_json_bytes,
    canonical_json_sha256,
)
from ibr1_experiment.diagnostics import GeometryCollector
from ibr1_experiment.eval_guard import IBR1_EVAL_PHASES, IBR1EvalOrderGuard
from ibr1_experiment.artifacts import (
    DIAGNOSTICS_MANIFEST_FILENAME,
    EXPECTED_OPTIMIZER_RECORDS,
    write_diagnostics_bundle,
)
from ibr1_experiment.gates import (
    IBR1_CANDIDATE_LOCK_CLASS,
    IBR1_NEGATIVE_SEAL_CLASS,
    IBR1_PASS_SEAL_CLASS,
    IBR1GateContractError,
    _g6_updates_from_gradient_geometry,
    build_ibr1_candidate_lock_receipt,
    build_ibr1_combined_gate_receipt,
    build_ibr1_negative_result_seal,
    build_ibr1_pass_seal,
    evaluate_i1,
    evaluate_i2,
    evaluate_i3,
    evaluate_i4,
    evaluate_i5,
    evaluate_i6,
    freeze_ibr1_candidate_lock_receipt,
    freeze_ibr1_combined_gate_receipt,
    freeze_ibr1_result_seal,
    verify_ibr1_result_seal,
)
from ibr1_experiment.model import IBR1_ARCHITECTURE_LOCK


def _self_hashed(value: dict[str, object]) -> dict[str, object]:
    document = deepcopy(value)
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def _rehash(value: dict[str, object]) -> dict[str, object]:
    document = deepcopy(value)
    document.pop("receipt_payload_sha256", None)
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    return document


def _write(path: Path, document: dict[str, object]) -> Path:
    path.write_bytes(canonical_json_bytes(document) + b"\n")
    return path


_EVAL_BLOCK_STARTS = (
    540,
    1699,
    2377,
    3418,
    4066,
    5042,
    5614,
    6650,
    7184,
    8315,
    8873,
    9900,
    10882,
    11801,
    12482,
    13582,
)


def _eval_ordered_indices() -> list[int]:
    return [
        original_index
        for start in _EVAL_BLOCK_STARTS
        for original_index in range(start, start + 32)
    ]


def _eval_support_binding() -> dict[str, object]:
    indices = _eval_ordered_indices()
    indices_sha = canonical_json_sha256(indices)
    return _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": SUPPORT_BINDING_CLASS,
            "family_id": "IBR1",
            "observation": {
                "supports": {
                    "EVAL-FIX": {
                        "rows": len(indices),
                        "ordered_original_indices": indices,
                        "ordered_original_indices_sha256": indices_sha,
                        "row_set_sha256": indices_sha,
                    }
                },
                "inherited_support_contract_payload_sha256": "f" * 64,
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )


def _final_assembly(*, with_eval_support: bool = False) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "family_id": "IBR1",
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "phase": "final",
        "candidate_cap": 1,
        "lambda_freeze_binding": {"verified": True},
        "formal_training_authorized": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    if with_eval_support:
        payload["support_binding"] = _eval_support_binding()
    return _self_hashed(payload)


def _assembly_binding(
    assembly: dict[str, object],
    *,
    path: str = "final_assembly.json",
    file_sha: str = "a" * 64,
) -> dict[str, object]:
    return {
        "path": path,
        "sha256": file_sha,
        "receipt_payload_sha256": assembly["receipt_payload_sha256"],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }


def _cal_evidence() -> dict[str, object]:
    return _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": CAL_NUMERIC_EVIDENCE_CLASS,
            "family_id": "IBR1",
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "support": "CAL",
            "rows": 512,
            "optimizer_updates": 0,
            "geometry_dtype": "torch.float32",
            "zero_init_persistence": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "failures": 0,
            },
            "post_decode_range": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "violations": 0,
                "abs_max": 0.75,
            },
            "realized_delta_reconstruction": {
                "checked_rows": 512,
                "checked_cells": 8192,
                "per_row_shape": [8, 2],
                "failures": 0,
                "error_max": 0.0,
            },
            "prev_free_observation_graph": {
                "checked_rows": 512,
                "failures": 0,
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )


def _sidecar(
    arm: str,
    u_pre: int,
    assembly_binding: dict[str, object],
    *,
    tensor_sha: str,
    checkpoint_file: str = "checkpoint.pt",
    checkpoint_file_sha: str = "e" * 64,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_class": "ibr1_arm_checkpoint_sidecar",
        "family_id": "IBR1",
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "model_class": "ibr1_experiment.model.IBR1AP2Model",
        "adapter_class": (
            "f2_experiment.opentrack_adapter."
            "OpenTrackVLAF2ObservationAdapter"
        ),
        "model_source_sha256": "1" * 64,
        "source_sha256": {
            "ibr1_experiment/checkpoint.py": "2" * 64,
            "ibr1_experiment/model.py": "1" * 64,
        },
        "family_arm": arm,
        "engine_arm": "S-CTRL" if arm == "IBR1-CTRL" else "S-SELF",
        "u_pre": u_pre,
        "checkpoint_tensor_sha256": tensor_sha,
        "final_assembly_receipt": deepcopy(assembly_binding),
        "state_schema": {
            "adapter": {"weight": {"shape": [1], "dtype": "torch.float32"}},
            "model": {"weight": {"shape": [1], "dtype": "torch.float32"}},
        },
        "snapshot_policy": {
            "purpose": "immutable_update_boundary_snapshot",
            "allowed_u_pre": [0, 128],
            "mid_run_resume": "forbidden",
            "optimizer_state_included": False,
            "rng_state_included": False,
        },
        "checkpoint_file": checkpoint_file,
        "checkpoint_file_sha256": checkpoint_file_sha,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


def test_i1_passes_and_update0_tensor_drift_is_a_gate_fail() -> None:
    assembly = _final_assembly()
    binding = _assembly_binding(assembly)
    sidecars = [
        _sidecar("IBR1-CTRL", 0, binding, tensor_sha="b" * 64),
        _sidecar("IBR1-SELF", 0, binding, tensor_sha="b" * 64),
    ]

    passed = evaluate_i1(
        assembly,
        _cal_evidence(),
        sidecars,
        final_assembly_binding=binding,
        non_authority=True,
    )
    assert passed.passed is True
    assert passed.to_dict()["formal_training_authorized"] is False

    drifted = deepcopy(sidecars)
    drifted[1]["checkpoint_tensor_sha256"] = "c" * 64
    failed = evaluate_i1(
        assembly,
        _cal_evidence(),
        drifted,
        final_assembly_binding=binding,
        non_authority=True,
    )
    assert failed.passed is False
    assert failed.checks["update0_checkpoint_tensor_identity"]["passed"] is False

    binding_drift = deepcopy(sidecars)
    binding_drift[1]["final_assembly_receipt"]["sha256"] = "f" * 64
    failed_binding = evaluate_i1(
        assembly,
        _cal_evidence(),
        binding_drift,
        final_assembly_binding=binding,
        non_authority=True,
    )
    assert failed_binding.passed is False

    malformed_binding = deepcopy(sidecars)
    malformed_binding[1]["final_assembly_receipt"]["verified"] = True
    with pytest.raises(IBR1GateContractError, match="keys differ"):
        evaluate_i1(
            assembly,
            _cal_evidence(),
            malformed_binding,
            final_assembly_binding=binding,
            non_authority=True,
        )


def test_i1_requires_exact_standard_or_live_final_assembly_binding(
    tmp_path: Path,
) -> None:
    assembly = _final_assembly()
    assembly_path = _write(tmp_path / "final_assembly.json", assembly)
    binding = _assembly_binding(
        assembly,
        path="final_assembly.json",
        file_sha=hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
    )
    sidecars = [
        _sidecar("IBR1-CTRL", 0, binding, tensor_sha="b" * 64),
        _sidecar("IBR1-SELF", 0, binding, tensor_sha="b" * 64),
    ]

    with pytest.raises(IBR1GateContractError, match="authority requires"):
        evaluate_i1(assembly, _cal_evidence(), sidecars)

    sidecar_paths = [
        _write(tmp_path / f"{sidecar['family_arm']}_update0.json", sidecar)
        for sidecar in sidecars
    ]

    assert evaluate_i1(
        assembly,
        _cal_evidence(),
        sidecars,
        project_root=tmp_path,
        final_assembly_receipt_path=assembly_path,
        update0_checkpoint_sidecar_paths=sidecar_paths,
    ).passed is True

    extra_key = {**binding, "verified": True}
    with pytest.raises(IBR1GateContractError, match="keys differ"):
        evaluate_i1(
            assembly,
            _cal_evidence(),
            sidecars,
            final_assembly_binding=extra_key,
            non_authority=True,
        )

    escaping = {**binding, "path": "../final_assembly.json"}
    with pytest.raises(IBR1GateContractError, match="project-root-relative"):
        evaluate_i1(
            assembly,
            _cal_evidence(),
            sidecars,
            final_assembly_binding=escaping,
            non_authority=True,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("post_decode_range", "abs_max"),
        ("realized_delta_reconstruction", "error_max"),
    ],
)
def test_i1_rejects_negative_maxima(section: str, field: str) -> None:
    assembly = _final_assembly()
    binding = _assembly_binding(assembly)
    sidecars = [
        _sidecar("IBR1-CTRL", 0, binding, tensor_sha="b" * 64),
        _sidecar("IBR1-SELF", 0, binding, tensor_sha="b" * 64),
    ]
    cal = _cal_evidence()
    cal[section][field] = -1e-9
    cal = _rehash(cal)
    with pytest.raises(IBR1GateContractError, match="must be nonnegative"):
        evaluate_i1(
            assembly,
            cal,
            sidecars,
            final_assembly_binding=binding,
            non_authority=True,
        )


def _training_rows() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for engine_arm, family_arm, offset in (
        ("S-CTRL", "IBR1-CTRL", 0),
        ("S-SELF", "IBR1-SELF", 1000),
    ):
        for position in range(256):
            zeros = [[0.0, 0.0] for _ in range(8)]
            records.append(
                {
                    "arm": family_arm,
                    "engine_arm": engine_arm,
                    "row_position": position,
                    "original_row_index": offset + position,
                    "u_pre": position // 2,
                    "row_within_update": position % 2,
                    "branch": "branch2",
                    "prev_source": "logged" if engine_arm == "S-CTRL" else "self",
                    "prev_fy": [0.0, 0.0],
                    "latent_delta_fy": deepcopy(zeros),
                    "cumulative_latent_fy": deepcopy(zeros),
                    "additive_prebound_fy": deepcopy(zeros),
                    "normalizer_fy": [[1.0, 1.0] for _ in range(8)],
                    "raw_fy": deepcopy(zeros),
                    "realized_delta_fy": deepcopy(zeros),
                    "prebound_violation_mask": [
                        [False, False] for _ in range(8)
                    ],
                    "prebound_overshoot_fy": deepcopy(zeros),
                    "boundary_margin_fy": [[1.0, 1.0] for _ in range(8)],
                    "geometry_reconstruction_error": 0.0,
                    "telescoping_reconstruction_error": 0.0,
                }
            )
    return records


def _training_summary(records: list[dict[str, object]]) -> dict[str, object]:
    collector = GeometryCollector(expected_training_rows_per_arm=256)
    collector.training_records = deepcopy(records)
    return collector._validate_training()


def test_i2_recomputes_denominator_and_rejects_summary_or_raw_drift() -> None:
    records = _training_rows()
    summary = _training_summary(records)

    receipt = evaluate_i2(records, summary)
    assert receipt.passed is True
    assert receipt.metrics["arms"]["IBR1-CTRL"][
        "I2_any_axis_denominator"
    ] == 2048
    assert receipt.metrics["arms"]["IBR1-SELF"][
        "overshoot_all_axis_cells"
    ]["support"] == 4096

    summary_drift = deepcopy(summary)
    summary_drift["arms"]["IBR1-CTRL"]["I2_any_axis_denominator"] = 2047
    with pytest.raises(IBR1GateContractError, match="exact raw-row recomputation"):
        evaluate_i2(records, summary_drift)

    raw_drift = deepcopy(records)
    raw_drift[0]["prebound_violation_mask"][0][0] = True
    raw_drift[0]["additive_prebound_fy"][0][0] = 1.5
    raw_drift[0]["prebound_overshoot_fy"][0][0] = 0.5
    with pytest.raises(IBR1GateContractError, match="exact raw-row recomputation"):
        evaluate_i2(raw_drift, summary)

    mask_only = deepcopy(records)
    mask_only[0]["prebound_violation_mask"][0][0] = True
    with pytest.raises(IBR1GateContractError, match="prebound_violation_mask"):
        evaluate_i2(mask_only, _training_summary(mask_only))

    overshoot_only = deepcopy(records)
    overshoot_only[0]["prebound_overshoot_fy"][0][0] = 0.25
    with pytest.raises(IBR1GateContractError, match="prebound_overshoot_fy"):
        evaluate_i2(overshoot_only, _training_summary(overshoot_only))


def _g6_and_gradient_geometry() -> tuple[list[dict[str, object]], dict[str, object]]:
    updates: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    weighted = {"L_cot": 0.1, "L_future": 0.2, "L_verify": 0.2}
    raw = {
        name: value / float(IBR1_FROZEN_AUX_COEFFICIENTS[name])
        for name, value in weighted.items()
    }
    for u_pre in range(128):
        update: dict[str, object] = {
            "u_pre": u_pre,
            "aux_reachable": True,
            "track_reachable": True,
            "cosine_total_track": None,
            "signed_projection": None,
            "aux_track_ratio": None,
            "per_aux_ratios": None,
        }
        if u_pre >= 8:
            update.update(
                {
                    "cosine_total_track": 1.0,
                    "signed_projection": 3.0,
                    "aux_track_ratio": 0.5,
                }
            )
        updates.append(update)
        exact_update = {
            "u_pre": u_pre,
            "aux_reachable": True,
            "track_reachable": True,
            "cosine_total_track": update.get("cosine_total_track"),
            "signed_projection": update.get("signed_projection"),
            "aux_track_ratio": update.get("aux_track_ratio"),
            "per_aux_ratios": None,
        }
        records.append(
            {
                "u_pre": u_pre,
                "engine_arm": "S-CTRL",
                "arm": "IBR1-CTRL",
                "grad_accum": 2,
                "track_grad_norm": 1.0,
                "weighted_aux_grad_norm": 0.5,
                "total_grad_norm": 1.5,
                "weighted_aux_track_dot": 0.5,
                "weighted_aux_track_cosine": 1.0,
                "weighted_aux_signed_projection": 0.5,
                "per_aux_weighted_grad_norm": weighted,
                "per_aux_raw_grad_norm_derived_from_frozen_lambda": raw,
                "per_aux_cosine_to_track": {
                    "L_cot": 1.0,
                    "L_future": 1.0,
                    "L_verify": 1.0,
                },
                "per_aux_signed_projection_to_track": weighted,
                "track_norm_below_eps": False,
                "weighted_aux_norm_below_eps": False,
                "total_norm_below_eps": False,
                "per_aux_norm_below_eps": {
                    "L_cot": False,
                    "L_future": False,
                    "L_verify": False,
                },
                "actual_ratio_denominator": 1.0,
                "per_aux_aggregate_discrepancy_norm": 0.0,
                "per_aux_aggregate_rounding_bound_norm": 0.0,
                "exact_g6_update": exact_update,
            }
        )
    return updates, {
        "schema_version": 1,
        "analysis_class": "ibr1_gradient_geometry",
        "family_id": "IBR1",
        "deciding_arm": "IBR1-CTRL",
        "records": records,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("both", "signed_projection", 2.5, "signed projection"),
        ("geometry", "actual_ratio_denominator", 0.9, "ratio denominator"),
        ("geometry", "track_grad_norm", float("nan"), "finite canonical JSON"),
        (
            "geometry",
            "per_aux_aggregate_discrepancy_norm",
            1e-15,
            "joint per-aux aggregate reconstruction must be exact",
        ),
        (
            "geometry",
            "per_aux_aggregate_rounding_bound_norm",
            1e-15,
            "joint per-aux aggregate reconstruction must be exact",
        ),
    ],
)
def test_i3_cross_checks_projection_ratio_and_nonfinite(
    target: str,
    field: str,
    value: object,
    message: str,
) -> None:
    updates, geometry = _g6_and_gradient_geometry()
    assert evaluate_i3(updates, geometry).passed is True

    if target == "both":
        updates[8][field] = value
        geometry["records"][8]["exact_g6_update"][field] = value
    else:
        geometry["records"][8][field] = value
    with pytest.raises(IBR1GateContractError, match=message):
        evaluate_i3(updates, geometry)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "schema drifted"),
        ("extra", "schema drifted"),
        ("clock", "clock drifted"),
        ("one_ulp", "differs from persisted exact diagnostics"),
        ("signed_zero", "differs from persisted exact diagnostics"),
        ("supplied_extra", "schema drifted"),
        ("supplied_missing", "schema drifted"),
    ],
)
def test_i3_exact_live_g6_mapping_fails_closed(
    mutation: str,
    message: str,
) -> None:
    updates, geometry = _g6_and_gradient_geometry()
    exact = geometry["records"][8]["exact_g6_update"]
    if mutation == "missing":
        del geometry["records"][8]["exact_g6_update"]
    elif mutation == "extra":
        exact["unexpected"] = "not authoritative"
    elif mutation == "clock":
        exact["u_pre"] = 9
    elif mutation == "one_ulp":
        updates[8]["cosine_total_track"] = math.nextafter(1.0, 0.0)
    elif mutation == "signed_zero":
        updates[8]["signed_projection"] = 0.0
        exact["signed_projection"] = -0.0
    elif mutation == "supplied_missing":
        del updates[8]["per_aux_ratios"]
    else:
        updates[8]["unexpected"] = "not authoritative"

    with pytest.raises(IBR1GateContractError, match=message):
        evaluate_i3(updates, geometry)


def test_i3_replay_uses_exact_direct_dot_mapping_without_ulp_reconstruction():
    updates, geometry = _g6_and_gradient_geometry()
    sum_aux = torch.tensor([0.0, 1.0], dtype=torch.float64)
    sum_track = torch.tensor([1.0, 1.0], dtype=torch.float64)
    sum_total = sum_aux + sum_track
    average_aux = sum_aux / 2.0
    average_track = sum_track / 2.0
    average_total = sum_total / 2.0

    aux_norm = float(torch.linalg.vector_norm(average_aux).item())
    track_norm = float(torch.linalg.vector_norm(average_track).item())
    total_norm = float(torch.linalg.vector_norm(average_total).item())
    aux_track_dot = float(torch.dot(average_aux, average_track).item())
    direct_dot = float(torch.dot(sum_total, sum_track).item())
    sum_track_norm = float(torch.linalg.vector_norm(sum_track).item())
    sum_total_norm = float(torch.linalg.vector_norm(sum_total).item())
    live_cosine = direct_dot / (sum_total_norm * sum_track_norm)
    live_projection = direct_dot / sum_track_norm
    old_reconstructed_dot = aux_track_dot + track_norm * track_norm
    old_reconstructed_cosine = old_reconstructed_dot / (
        total_norm * track_norm
    )
    old_reconstructed_projection = (
        old_reconstructed_dot / track_norm
    ) * 2.0

    assert math.nextafter(live_cosine, math.inf) == old_reconstructed_cosine
    assert math.nextafter(live_projection, math.inf) == (
        old_reconstructed_projection
    )

    for u_pre, (update, record) in enumerate(
        zip(updates, geometry["records"])
    ):
        record.update(
            {
                "track_grad_norm": track_norm,
                "weighted_aux_grad_norm": aux_norm,
                "total_grad_norm": total_norm,
                "weighted_aux_track_dot": aux_track_dot,
                "weighted_aux_track_cosine": aux_track_dot
                / (aux_norm * track_norm),
                "weighted_aux_signed_projection": aux_track_dot
                / track_norm,
                "actual_ratio_denominator": track_norm,
            }
        )
        exact = record["exact_g6_update"]
        if u_pre >= 8:
            update.update(
                {
                    "cosine_total_track": live_cosine,
                    "signed_projection": live_projection,
                    "aux_track_ratio": aux_norm / track_norm,
                }
            )
            exact.update(
                {
                    "cosine_total_track": live_cosine,
                    "signed_projection": live_projection,
                    "aux_track_ratio": aux_norm / track_norm,
                }
            )

    production = evaluate_i3(updates, geometry).to_dict()
    replay_updates = _g6_updates_from_gradient_geometry(geometry)
    assert len(replay_updates) == 128
    replay = evaluate_i3(replay_updates, geometry).to_dict()
    assert replay == production
    assert replay["receipt_payload_sha256"] == production[
        "receipt_payload_sha256"
    ]


def _g7_updates() -> list[dict[str, object]]:
    return [
        {
            "u_pre": u_pre,
            "per_method_over_base": {"method_a": 0.1, "method_b": 0.1},
            "total_method_over_base": 0.2,
            "abs_tanh_method_scales": {"method_a": 0.1, "method_b": 0.1},
            "r_prev": 0.1,
            "abs_tanh_s_prev": 0.1,
        }
        for u_pre in range(128)
    ]


def _loss_summary(value: float) -> dict[str, object]:
    strata = ("overall", "change", "turn", "other")
    return {
        "accumulator": "IEEE-754 binary64 math.fsum",
        "means": {stratum: value for stratum in strata},
        "counts": {
            "overall": 512,
            "change": 69,
            "turn": 154,
            "other": 211,
        },
    }


def test_i4_and_i5_preserve_both_arms_and_all_strata() -> None:
    g7 = _g7_updates()
    assert evaluate_i4(g7, deepcopy(g7)).passed is True

    receipt = evaluate_i5(
        {"logged": _loss_summary(0.5), "self": _loss_summary(0.7)},
        {"logged": _loss_summary(0.4), "self": _loss_summary(0.5)},
        {"logged": _loss_summary(0.5), "self": _loss_summary(0.7)},
    )
    assert receipt.passed is True
    inherited = receipt.metrics["inherited_G8"]
    assert set(inherited["metrics"]["support_counts"]) == {
        "overall",
        "change",
        "turn",
        "other",
    }


def _g9_inputs() -> dict[str, object]:
    return {
        "expected_static_resets": 0,
        "observed_static_resets": 0,
        "nonfinite_reset_count": 0,
        "range_violation_count": 0,
        "range_observation_count": 2048,
        "reconstruction_errors": [0.0] * 256,
        "first_quartile_self_errors": [0.5] * 64,
        "last_quartile_self_errors": [0.5] * 64,
    }


def test_i6_one_range_violation_fails_even_when_f2_rate_would_pass() -> None:
    ctrl = _g9_inputs()
    self_inputs = _g9_inputs()
    assert evaluate_i6(ctrl, self_inputs).passed is True

    ctrl["range_violation_count"] = 1
    receipt = evaluate_i6(ctrl, self_inputs)
    assert receipt.passed is False
    assert receipt.checks["IBR1-CTRL.range_violation_count"]["passed"] is False
    assert receipt.metrics["arms"]["IBR1-CTRL"]["inherited_G9"]["passed"] is True


def test_i6_rejects_nested_or_negative_reconstruction_rows() -> None:
    ctrl = _g9_inputs()
    ctrl["reconstruction_errors"] = [[0.0, 0.0] for _ in range(128)]
    with pytest.raises(IBR1GateContractError, match="exactly 256 row scalars"):
        evaluate_i6(ctrl, _g9_inputs())

    ctrl = _g9_inputs()
    ctrl["reconstruction_errors"][7] = -1e-9
    with pytest.raises(IBR1GateContractError, match="must be nonnegative"):
        evaluate_i6(ctrl, _g9_inputs())


def _real_gate_documents(
    *,
    assembly: dict[str, object] | None = None,
    binding: dict[str, object] | None = None,
    update0_sidecars: list[dict[str, object]] | None = None,
    passed: bool = True,
    project_root: Path | None = None,
    update0_sidecar_paths: list[Path] | None = None,
) -> list[dict[str, object]]:
    assembly = assembly or _final_assembly()
    if project_root is not None:
        assembly_path = project_root / "final_assembly.json"
        if not assembly_path.exists():
            _write(assembly_path, assembly)
        binding = binding or _assembly_binding(
            assembly,
            path="final_assembly.json",
            file_sha=hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
        )
    else:
        assembly_path = None
        binding = binding or _assembly_binding(assembly)
    update0_sidecars = update0_sidecars or [
        _sidecar("IBR1-CTRL", 0, binding, tensor_sha="b" * 64),
        _sidecar("IBR1-SELF", 0, binding, tensor_sha="b" * 64),
    ]
    if project_root is not None and update0_sidecar_paths is None:
        update0_sidecar_paths = [
            _write(
                project_root / f"i1_{'pass' if passed else 'fail'}_{sidecar['family_arm']}.json",
                sidecar,
            )
            for sidecar in update0_sidecars
        ]
    i1 = evaluate_i1(
        assembly,
        _cal_evidence(),
        update0_sidecars,
        final_assembly_binding=binding,
        project_root=project_root,
        final_assembly_receipt_path=assembly_path,
        update0_checkpoint_sidecar_paths=update0_sidecar_paths,
        non_authority=project_root is None,
    )
    training = _training_rows()
    i2 = evaluate_i2(training, _training_summary(training))
    g6, geometry = _g6_and_gradient_geometry()
    i3 = evaluate_i3(g6, geometry)
    g7 = _g7_updates()
    i4 = evaluate_i4(g7, deepcopy(g7))
    i5 = evaluate_i5(
        {"logged": _loss_summary(0.5), "self": _loss_summary(0.7)},
        {"logged": _loss_summary(0.4), "self": _loss_summary(0.5)},
        {"logged": _loss_summary(0.5), "self": _loss_summary(0.7)},
    )
    ctrl_g9 = _g9_inputs()
    if not passed:
        ctrl_g9["range_violation_count"] = 1
    i6 = evaluate_i6(ctrl_g9, _g9_inputs())
    return [receipt.to_dict() for receipt in (i1, i2, i3, i4, i5, i6)]


def test_combined_pass_still_forbids_formal_and_negative_forbids_tuning(
    tmp_path: Path,
) -> None:
    passed = build_ibr1_combined_gate_receipt(
        *_real_gate_documents(project_root=tmp_path)
    )
    assert passed["mechanism_pass"] is True
    assert passed["formal_training_authorized"] is False
    assert passed["next_step"] == "independent_review_then_new_preregistration"

    failed_gates = _real_gate_documents(passed=False, project_root=tmp_path)
    failed = build_ibr1_combined_gate_receipt(*failed_gates)
    assert failed["mechanism_pass"] is False
    assert failed["decision"] == "SEAL_STOP"
    assert failed["same_family_retry_authorized"] is False
    assert failed["same_family_seed_change_authorized"] is False
    assert failed["same_family_lambda_change_authorized"] is False
    assert failed["same_family_decode_change_authorized"] is False
    assert failed["same_family_gate_change_authorized"] is False


def test_combined_rejects_non_authority_i1_and_status_decision_drift(
    tmp_path: Path,
) -> None:
    non_authority_gates = _real_gate_documents()
    with pytest.raises(IBR1GateContractError, match="non-authority"):
        build_ibr1_combined_gate_receipt(*non_authority_gates)

    combined = build_ibr1_combined_gate_receipt(
        *_real_gate_documents(project_root=tmp_path)
    )
    for field, value in (("status", "FAIL"), ("decision", "SEAL_STOP")):
        drifted = deepcopy(combined)
        drifted[field] = value
        drifted = _rehash(drifted)
        with pytest.raises(IBR1GateContractError, match="status/decision"):
            freeze_ibr1_combined_gate_receipt(
                tmp_path / f"combined_{field}.json",
                drifted,
            )


@pytest.mark.parametrize(
    "alias",
    [
        "formalTrainingAuthorized",
        "formal_training_allowed",
        "allow_formal_training",
        "formal_run_authorized",
        "open_internal_test",
    ],
)
def test_nested_authority_alias_is_rejected_at_arbitrary_depth(
    alias: str,
    tmp_path: Path,
) -> None:
    gates = _real_gate_documents(project_root=tmp_path)
    tampered = deepcopy(gates[0])
    tampered["contract"]["level1"] = [{"level2": {alias: False}}]
    tampered["receipt_payload_sha256"] = "d" * 64

    with pytest.raises(IBR1GateContractError, match="suspicious or authorizing"):
        build_ibr1_combined_gate_receipt(tampered, *gates[1:])


def test_combined_rejects_empty_or_schema_drifted_gate_checks(
    tmp_path: Path,
) -> None:
    gates = _real_gate_documents(project_root=tmp_path)
    empty = deepcopy(gates[0])
    empty["checks"] = {}
    empty = _rehash(empty)
    with pytest.raises(IBR1GateContractError, match="check names drifted"):
        build_ibr1_combined_gate_receipt(empty, *gates[1:])

    threshold_drift = deepcopy(gates[2])
    threshold_drift["thresholds"]["weighted_aux_over_track_median_max"] = 99.0
    threshold_drift = _rehash(threshold_drift)
    with pytest.raises(IBR1GateContractError, match="frozen thresholds drifted"):
        build_ibr1_combined_gate_receipt(
            gates[0],
            gates[1],
            threshold_drift,
            *gates[3:],
        )

    metric_drift = deepcopy(gates[5])
    metric_drift["metrics"]["arms"]["IBR1-CTRL"][
        "range_violation_count"
    ] = 1
    metric_drift = _rehash(metric_drift)
    with pytest.raises(IBR1GateContractError, match="check/metric drifted"):
        build_ibr1_combined_gate_receipt(*gates[:5], metric_drift)


def _eval_geometry_rows() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for snapshot in (
        "update0_IBR1-SELF",
        "update128_IBR1-CTRL",
        "update128_IBR1-SELF",
    ):
        arm = "IBR1-SELF" if "SELF" in snapshot else "IBR1-CTRL"
        engine_arm = "S-SELF" if arm == "IBR1-SELF" else "S-CTRL"
        for mode in ("logged", "self"):
            for position in range(512):
                for horizon in range(8):
                    for axis in ("forward", "yaw"):
                        records.append(
                            {
                                "arm": arm,
                                "engine_arm": engine_arm,
                                "snapshot": snapshot,
                                "mode": mode,
                                "row_position": position,
                                "original_row_index": position,
                                "horizon": horizon,
                                "axis": axis,
                                "raw_fy": 0.0,
                                "absolute_error": 0.0,
                                "overshoot": 0.0,
                            }
                        )
    return records


def _diagnostics_fixture(root: Path) -> Path:
    training = _training_rows()
    evaluation = _eval_geometry_rows()
    eval_collector = GeometryCollector()
    eval_collector.eval_records = deepcopy(evaluation)
    _g6, gradient = _g6_and_gradient_geometry()
    summary = {
        "schema_version": 1,
        "analysis_class": "ibr1_diagnostics_summary",
        "family_id": "IBR1",
        "training_geometry": _training_summary(training),
        "eval_geometry": eval_collector._validate_eval(),
        "engineering_fail_closed": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    write_diagnostics_bundle(
        root / "diagnostics",
        training_records=training,
        eval_records=evaluation,
        gradient_document=gradient,
        optimizer_document={
            "schema_version": 1,
            "analysis_class": "ibr1_optimizer_geometry",
            "family_id": "IBR1",
            "records": [
                {"kind": "optimizer", "position": index, "value": 0.0}
                for index in range(EXPECTED_OPTIMIZER_RECORDS)
            ],
            "internal_test": "sealed",
            "internal_test_opened": False,
        },
        summary_document=summary,
        lifecycle_bindings={
            "checkpoint_identity": {"verified": True},
            "eval_order_guard_receipt": {"verified": True},
            "final_assembly_receipt": {"verified": True},
            "predictor_identity": {"verified": True},
            "u_pre_identity": {"verified": True},
        },
    )
    return root / "diagnostics" / DIAGNOSTICS_MANIFEST_FILENAME


def _count_receipt(tensor_sha: str) -> dict[str, object]:
    common = {
        "rows": 256,
        "feature_forwards": 256,
        "aux_forwards": 256,
        "head_forwards": 512,
        "track_loss_calls": 512,
        "backward_calls": 256,
        "optimizer_steps": 128,
        "controller_steps": 256,
        "static_resets": 12,
        "nonfinite_resets": 0,
        "branch1_logged_rows": 256,
        "g7_updates": 128,
        "g9_transitions": 256,
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }
    return {
        "schema_version": 1,
        "analysis_class": "f2_paired_runner_count_receipt",
        "architecture_lock": F2_ARCHITECTURE_LOCK,
        "checkpoint_init_sha256": tensor_sha,
        "rows_per_arm": 256,
        "optimizer_updates_per_arm": 128,
        "grad_accum": 2,
        "warmup": "u_pre<16",
        "loss": "L_aux+0.5*L1+0.5*L2",
        "expected_static_resets": 12,
        "arms": {
            "S-CTRL": {
                **common,
                "branch2_logged_rows": 256,
                "branch2_self_rows": 0,
                "g6_updates": 128,
            },
            "S-SELF": {
                **common,
                "branch2_logged_rows": 32,
                "branch2_self_rows": 224,
                "g6_updates": 0,
            },
        },
        "passed": True,
        "status": "PASS",
        "decision": "GO",
    }


def _phase_loss_value(phase_name: str) -> float:
    return {
        "u0_self_logged": 0.5,
        "u0_self_self": 0.7,
        "u128_ctrl_logged": 0.5,
        "u128_ctrl_self": 0.7,
        "u128_self_logged": 0.4,
        "u128_self_self": 0.5,
    }[phase_name]


def _eval_phase_receipt(phase_name: str, mode: str) -> dict[str, object]:
    phase = next(
        candidate
        for candidate in IBR1_EVAL_PHASES
        if candidate.phase == phase_name
    )
    assert mode == phase.mode
    value = _phase_loss_value(phase_name)
    row_losses = [value] * 512
    summary = _loss_summary(value)
    assert summary["means"]["overall"] == math.fsum(row_losses) / 512
    return {
        "schema_version": 1,
        "analysis_class": "f2_eval_fix_snapshot_receipt",
        "architecture_lock": F2_ARCHITECTURE_LOCK,
        **phase.to_dict(),
        "engine_arm": FAMILY_TO_ENGINE_ARM[phase.family_arm],
        "checkpoint_u_pre": 0 if phase.snapshot.startswith("update0_") else 128,
        "support": "EVAL-FIX",
        "rows": 512,
        "mode": mode,
        "static_resets": {"expected": 28, "observed": 28},
        "controller_config": DEFAULT_CONFIG.to_dict(),
        "eval_mode_contract": deepcopy(EVAL_MODE_CONTRACT),
        "row_losses": row_losses,
        "summary": summary,
    }


def _eval_guard(binding: dict[str, object]) -> dict[str, object]:
    indices = _eval_ordered_indices()
    phases = [
        {
            **phase.to_dict(),
            "rows": 512,
            "ordered_original_indices": indices,
            "ordered_original_indices_sha256": canonical_json_sha256(indices),
            "bytes_equal_expected_binding": True,
        }
        for phase in IBR1_EVAL_PHASES
    ]
    return {
        "schema_version": 1,
        "analysis_class": "ibr1_eval_fixed_order_guard_receipt",
        "family_id": "IBR1",
        "formal_training_authorized": False,
        "support": "EVAL-FIX",
        "final_assembly_receipt": binding,
        "phase_order": [phase.phase for phase in IBR1_EVAL_PHASES],
        "rows_per_phase": 512,
        "phases": phases,
        "total_predictor_calls": 3072,
        "expected_total_predictor_calls": 3072,
        "all_phase_mapping_bytes_identical": True,
        "all_phase_mapping_sha256_identical": True,
        "all_phase_mappings_equal_expected_binding": True,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


def _seal_artifacts(
    root: Path,
    *,
    passed: bool,
    with_eval_support: bool = False,
) -> dict[str, object]:
    assembly = _final_assembly(with_eval_support=with_eval_support)
    assembly_path = _write(root / "final_assembly.json", assembly)
    binding = _assembly_binding(
        assembly,
        path="final_assembly.json",
        file_sha=hashlib.sha256(assembly_path.read_bytes()).hexdigest(),
    )
    candidate_path = _write(
        root / "candidate_lock.json",
        build_ibr1_candidate_lock_receipt(),
    )

    sidecar_paths: list[Path] = []
    update0_sidecars: list[dict[str, object]] = []
    update0_sidecar_paths: list[Path] = []
    for arm in ("IBR1-CTRL", "IBR1-SELF"):
        for u_pre in (0, 128):
            tensor_sha = "b" * 64 if u_pre == 0 else (
                "c" * 64 if arm == "IBR1-CTRL" else "d" * 64
            )
            checkpoint_name = f"{arm}_update{u_pre}.pt"
            checkpoint_path = root / checkpoint_name
            checkpoint_path.write_bytes(
                f"synthetic-checkpoint:{arm}:{u_pre}".encode("utf-8")
            )
            checkpoint_sha = hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest()
            sidecar = _sidecar(
                arm,
                u_pre,
                binding,
                tensor_sha=tensor_sha,
                checkpoint_file=checkpoint_name,
                checkpoint_file_sha=checkpoint_sha,
            )
            if u_pre == 0:
                update0_sidecars.append(sidecar)
            sidecar_path = _write(
                root / f"{arm}_update{u_pre}.receipt.json",
                sidecar,
            )
            if u_pre == 0:
                update0_sidecar_paths.append(sidecar_path)
            sidecar_paths.append(sidecar_path)

    count_path = _write(root / "count_receipt.json", _count_receipt("b" * 64))

    eval_phase_paths = {
        phase.phase: _write(
            root / f"eval_{phase.phase}.json",
            _eval_phase_receipt(phase.phase, phase.mode),
        )
        for phase in IBR1_EVAL_PHASES
    }

    eval_guard_path = _write(
        root / "eval_guard.json",
        _eval_guard(binding),
    )
    diagnostics_path = _diagnostics_fixture(root)
    gate_documents = _real_gate_documents(
        assembly=assembly,
        binding=binding,
        update0_sidecars=update0_sidecars,
        passed=passed,
        project_root=root,
        update0_sidecar_paths=update0_sidecar_paths,
    )
    gate_paths = [
        _write(root / f"I{index}.json", document)
        for index, document in enumerate(gate_documents, 1)
    ]
    combined_path = _write(
        root / "combined.json",
        build_ibr1_combined_gate_receipt(*gate_documents),
    )
    return {
        "final_assembly_receipt_path": assembly_path,
        "candidate_lock_receipt_path": candidate_path,
        "checkpoint_sidecar_paths": sidecar_paths,
        "count_receipt_path": count_path,
        "eval_guard_receipt_path": eval_guard_path,
        "eval_phase_receipt_paths": eval_phase_paths,
        "diagnostics_manifest_path": diagnostics_path,
        "gate_receipt_paths": gate_paths,
        "combined_gate_receipt_path": combined_path,
    }


def _replace_synthetic_guard_with_real_finalized_receipt(
    root: Path,
    artifacts: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    final_path = Path(artifacts["final_assembly_receipt_path"])

    def verify_final_assembly(
        project_root: str | Path,
        receipt_path: str | Path,
        *,
        required_phase: str | None = None,
    ) -> dict[str, object]:
        assert Path(project_root).resolve() == root.resolve()
        assert Path(receipt_path).resolve() == final_path.resolve()
        assert required_phase == ASSEMBLY_PHASE_FINAL
        return json.loads(final_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        eval_guard_module,
        "verify_assembly_receipt",
        verify_final_assembly,
    )
    rows = tuple(
        RunnerRow(
            original_row_index=index,
            sequence_id=f"sequence-{index}",
            frame_idx=index,
            mirrored=False,
            logged_prev_action=(0.0, 0.0, 0.0),
            target_actions=torch.zeros(8, 3, dtype=torch.float32),
            observation=object(),
        )
        for index in _eval_ordered_indices()
    )
    guard = IBR1EvalOrderGuard(
        rows,
        project_root=root,
        final_assembly_receipt_path=final_path,
    )

    def predictor(row, prev_fy, *, mode, reset, position):
        del row, prev_fy, mode, reset, position
        return None

    for phase in IBR1_EVAL_PHASES:
        wrapped = guard.wrap_predictor(
            predictor,
            phase=phase.phase,
            snapshot=phase.snapshot,
            family_arm=phase.family_arm,
            mode=phase.mode,
        )
        for position, row in enumerate(rows):
            wrapped(
                row,
                None,
                mode=phase.mode,
                reset=position == 0,
                position=position,
            )
    receipt = guard.finalize()
    _write(Path(artifacts["eval_guard_receipt_path"]), receipt)
    return receipt


@pytest.mark.parametrize("passed", [True, False])
def test_pass_and_negative_seals_bind_all_artifacts_and_verify(
    tmp_path: Path,
    passed: bool,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=passed)
    builder = build_ibr1_pass_seal if passed else build_ibr1_negative_result_seal
    document = builder(tmp_path, **artifacts)
    expected_class = IBR1_PASS_SEAL_CLASS if passed else IBR1_NEGATIVE_SEAL_CLASS
    assert document["analysis_class"] == expected_class
    assert document["formal_training_authorized"] is False
    assert len(document["evidence"]["checkpoint_sidecars"]) == 4
    assert len(document["evidence"]["gate_receipts"]) == 6

    seal_path = _write(tmp_path / "result_seal.json", document)
    assert verify_ibr1_result_seal(tmp_path, seal_path) == document


@pytest.mark.parametrize("passed", [True, False])
def test_real_eval_guard_receipt_builds_freezes_and_verifies_result_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed: bool,
) -> None:
    artifacts = _seal_artifacts(
        tmp_path,
        passed=passed,
        with_eval_support=True,
    )
    eval_guard = _replace_synthetic_guard_with_real_finalized_receipt(
        tmp_path,
        artifacts,
        monkeypatch,
    )
    assert eval_guard["final_assembly_receipt"]["analysis_class"] == (
        ASSEMBLY_RECEIPT_CLASS
    )

    builder = build_ibr1_pass_seal if passed else build_ibr1_negative_result_seal
    built = builder(tmp_path, **artifacts)
    expected_class = IBR1_PASS_SEAL_CLASS if passed else IBR1_NEGATIVE_SEAL_CLASS
    assert built["analysis_class"] == expected_class

    frozen = freeze_ibr1_result_seal(
        tmp_path,
        "real_guard_result_seal.json",
        expected_pass=passed,
        **artifacts,
    )
    assert frozen["mechanism_pass"] is passed
    assert verify_ibr1_result_seal(tmp_path, frozen["path"]) == built


@pytest.mark.parametrize("passed", [True, False])
@pytest.mark.parametrize("mutation", ["missing", "wrong", "extra"])
def test_result_seal_rejects_eval_guard_final_binding_schema_drift(
    tmp_path: Path,
    passed: bool,
    mutation: str,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=passed)
    eval_guard_path = Path(artifacts["eval_guard_receipt_path"])
    eval_guard = json.loads(eval_guard_path.read_text(encoding="utf-8"))
    binding = eval_guard["final_assembly_receipt"]
    if mutation == "missing":
        del binding["analysis_class"]
    elif mutation == "wrong":
        binding["analysis_class"] = "wrong_final_assembly_class"
    else:
        binding["unexpected"] = "not_authoritative"
    _write(eval_guard_path, eval_guard)

    builder = build_ibr1_pass_seal if passed else build_ibr1_negative_result_seal
    with pytest.raises(
        IBR1GateContractError,
        match="EVAL guard binds a different final assembly",
    ):
        builder(tmp_path, **artifacts)


def test_seal_verification_detects_artifact_byte_drift(tmp_path: Path) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=False)
    frozen = freeze_ibr1_result_seal(
        tmp_path,
        "negative_seal.json",
        expected_pass=False,
        **artifacts,
    )
    seal_path = tmp_path / frozen["path"]
    assert verify_ibr1_result_seal(tmp_path, seal_path)["run"][
        "scientific_negative_result"
    ] is True

    gate_path = Path(artifacts["gate_receipt_paths"][0])
    gate_path.write_bytes(gate_path.read_bytes() + b" ")
    with pytest.raises(IBR1GateContractError):
        verify_ibr1_result_seal(tmp_path, seal_path)


def test_seal_builder_rejects_checkpoint_and_diagnostics_bundle_pocs(
    tmp_path: Path,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=False)
    assert build_ibr1_negative_result_seal(tmp_path, **artifacts)["run"][
        "scientific_negative_result"
    ] is True

    sidecar_path = Path(artifacts["checkpoint_sidecar_paths"][0])
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    checkpoint_path = sidecar_path.parent / sidecar["checkpoint_file"]
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_path.write_bytes(checkpoint_bytes + b"drift")
    with pytest.raises(IBR1GateContractError, match="checkpoint file SHA drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    checkpoint_path.write_bytes(checkpoint_bytes)

    sidecar_bytes = sidecar_path.read_bytes()
    del sidecar["checkpoint_file_sha256"]
    sidecar_path.write_bytes(canonical_json_bytes(sidecar) + b"\n")
    with pytest.raises(IBR1GateContractError, match="sidecar.*schema drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    sidecar_path.write_bytes(sidecar_bytes)

    manifest_path = Path(artifacts["diagnostics_manifest_path"])
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    training_path = manifest_path.parent / "training_geometry.jsonl"
    training_bytes = training_path.read_bytes()
    training_path.write_bytes(training_bytes + b"{}\n")
    with pytest.raises(IBR1GateContractError, match="byte count drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    training_path.write_bytes(training_bytes)

    eval_path = manifest_path.parent / "eval_geometry.jsonl"
    eval_bytes = eval_path.read_bytes()
    eval_path.unlink()
    with pytest.raises(IBR1GateContractError, match="eval_geometry.jsonl is missing"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    eval_path.write_bytes(eval_bytes)

    eval_records = [
        json.loads(line)
        for line in eval_bytes.decode("utf-8").splitlines()
    ]
    eval_records[0]["raw_fy"] = 0.25
    drifted_eval_bytes = b"".join(
        canonical_json_bytes(record) + b"\n" for record in eval_records
    )
    eval_path.write_bytes(drifted_eval_bytes)
    eval_content_manifest = deepcopy(manifest)
    eval_entry = eval_content_manifest["artifacts"]["eval_geometry.jsonl"]
    eval_entry["bytes"] = len(drifted_eval_bytes)
    eval_entry["sha256"] = hashlib.sha256(drifted_eval_bytes).hexdigest()
    _write(manifest_path, _rehash(eval_content_manifest))
    with pytest.raises(IBR1GateContractError, match="EVAL summary differs"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    eval_path.write_bytes(eval_bytes)
    manifest_path.write_bytes(manifest_bytes)

    training_records = [
        json.loads(line)
        for line in training_bytes.decode("utf-8").splitlines()
    ]
    training_records[0]["additive_prebound_fy"][0][0] = 1.5
    training_records[0]["prebound_violation_mask"][0][0] = True
    training_records[0]["prebound_overshoot_fy"][0][0] = 0.5
    drifted_training_bytes = b"".join(
        canonical_json_bytes(record) + b"\n" for record in training_records
    )
    training_path.write_bytes(drifted_training_bytes)
    summary_path = manifest_path.parent / "diagnostics_summary.json"
    summary_bytes = summary_path.read_bytes()
    summary_document = json.loads(summary_bytes)
    summary_document["training_geometry"] = _training_summary(training_records)
    _write(summary_path, _rehash(summary_document))
    i2_binding_manifest = deepcopy(manifest)
    for filename, payload in (
        ("training_geometry.jsonl", drifted_training_bytes),
        ("diagnostics_summary.json", summary_path.read_bytes()),
    ):
        entry = i2_binding_manifest["artifacts"][filename]
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
    _write(manifest_path, _rehash(i2_binding_manifest))
    with pytest.raises(IBR1GateContractError, match="bound I2 receipt differs"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    training_path.write_bytes(training_bytes)
    summary_path.write_bytes(summary_bytes)
    manifest_path.write_bytes(manifest_bytes)

    gradient_path = manifest_path.parent / "gradient_geometry.json"
    gradient_bytes = gradient_path.read_bytes()
    gradient_document = json.loads(gradient_bytes)
    gradient_document["records"][8].update(
        {
            "weighted_aux_grad_norm": 0.4,
            "total_grad_norm": 1.4,
            "weighted_aux_track_dot": 0.4,
            "weighted_aux_signed_projection": 0.4,
        }
    )
    gradient_document["records"][8]["exact_g6_update"].update(
        {
            "signed_projection": 2.8,
            "aux_track_ratio": 0.4,
        }
    )
    _write(gradient_path, gradient_document)
    i3_binding_manifest = deepcopy(manifest)
    gradient_payload = gradient_path.read_bytes()
    gradient_entry = i3_binding_manifest["artifacts"]["gradient_geometry.json"]
    gradient_entry["bytes"] = len(gradient_payload)
    gradient_entry["sha256"] = hashlib.sha256(gradient_payload).hexdigest()
    _write(manifest_path, _rehash(i3_binding_manifest))
    with pytest.raises(IBR1GateContractError, match="bound I3 receipt differs"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    gradient_path.write_bytes(gradient_bytes)
    manifest_path.write_bytes(manifest_bytes)

    phase_path = Path(artifacts["eval_phase_receipt_paths"]["u128_self_self"])
    phase_bytes = phase_path.read_bytes()
    phase_document = json.loads(phase_bytes)
    phase_document["row_losses"] = [0.6] * 512
    phase_document["summary"]["means"] = {
        stratum: 0.6 for stratum in ("overall", "change", "turn", "other")
    }
    _write(phase_path, phase_document)
    with pytest.raises(IBR1GateContractError, match="bound I5 receipt differs"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    phase_path.write_bytes(phase_bytes)

    count_path = Path(artifacts["count_receipt_path"])
    count_bytes = count_path.read_bytes()
    count_document = json.loads(count_bytes)
    count_document["status"] = "FAIL"
    _write(count_path, count_document)
    with pytest.raises(IBR1GateContractError, match="count receipt verdict drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    count_path.write_bytes(count_bytes)

    lifecycle_drift = deepcopy(manifest)
    del lifecycle_drift["lifecycle_bindings"]["u_pre_identity"]
    _write(manifest_path, _rehash(lifecycle_drift))
    with pytest.raises(IBR1GateContractError, match="lifecycle binding keys drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    manifest_path.write_bytes(manifest_bytes)

    format_drift = deepcopy(manifest)
    format_drift["artifacts"]["training_geometry.jsonl"]["format"] = "json"
    _write(manifest_path, _rehash(format_drift))
    with pytest.raises(IBR1GateContractError, match="format drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    manifest_path.write_bytes(manifest_bytes)

    sha_drift = deepcopy(manifest)
    sha_drift["artifacts"]["training_geometry.jsonl"]["sha256"] = "0" * 64
    _write(manifest_path, _rehash(sha_drift))
    with pytest.raises(IBR1GateContractError, match="SHA drifted"):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)


@pytest.mark.parametrize("arm", ["S-CTRL", "S-SELF"])
def test_count_receipt_static_resets_are_frozen_to_smk_train(
    tmp_path: Path,
    arm: str,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=False)
    count_path = Path(artifacts["count_receipt_path"])
    baseline = json.loads(count_path.read_bytes())

    self_reported_drift = deepcopy(baseline)
    self_reported_drift["expected_static_resets"] = 11
    for counts in self_reported_drift["arms"].values():
        counts["static_resets"] = 11
    _write(count_path, self_reported_drift)
    with pytest.raises(
        IBR1GateContractError,
        match="frozen SMK-TRAIN count",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)

    arm_drift = deepcopy(baseline)
    arm_drift["arms"][arm]["static_resets"] = 11
    _write(count_path, arm_drift)
    with pytest.raises(
        IBR1GateContractError,
        match=rf"paired runner {arm}\.static_resets drifted",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)


def test_eval_phase_runtime_contracts_reject_six_consistently_drifted_receipts(
    tmp_path: Path,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=False)
    phase_paths = artifacts["eval_phase_receipt_paths"]
    assert isinstance(phase_paths, dict)
    baselines = {
        phase_name: json.loads(Path(path).read_bytes())
        for phase_name, path in phase_paths.items()
    }

    for phase_name, path in phase_paths.items():
        document = deepcopy(baselines[phase_name])
        document["static_resets"] = {"expected": 29, "observed": 29}
        _write(Path(path), document)
    with pytest.raises(
        IBR1GateContractError,
        match="frozen EVAL-FIX count",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)

    for phase_name, path in phase_paths.items():
        document = deepcopy(baselines[phase_name])
        document["controller_config"]["max_abs"] = 0.75
        _write(Path(path), document)
    with pytest.raises(
        IBR1GateContractError,
        match="frozen DEFAULT_CONFIG",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)

    for phase_name, path in phase_paths.items():
        document = deepcopy(baselines[phase_name])
        document["eval_mode_contract"]["grad"] = "drifted"
        _write(Path(path), document)
    with pytest.raises(
        IBR1GateContractError,
        match="frozen EVAL_MODE_CONTRACT",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)


def test_eval_phase_receipts_fail_closed_on_pairing_identity_and_strata_drift(
    tmp_path: Path,
) -> None:
    artifacts = _seal_artifacts(tmp_path, passed=False)
    assert build_ibr1_negative_result_seal(tmp_path, **artifacts)["run"][
        "scientific_negative_result"
    ] is True

    phase_paths = artifacts["eval_phase_receipt_paths"]
    assert isinstance(phase_paths, dict)
    swapped = deepcopy(artifacts)
    swapped_paths = dict(phase_paths)
    swapped_paths["u128_ctrl_logged"], swapped_paths["u128_self_logged"] = (
        swapped_paths["u128_self_logged"],
        swapped_paths["u128_ctrl_logged"],
    )
    swapped["eval_phase_receipt_paths"] = swapped_paths
    with pytest.raises(IBR1GateContractError, match="identity drifted"):
        build_ibr1_negative_result_seal(tmp_path, **swapped)

    target_path = Path(phase_paths["u128_ctrl_logged"])
    target_bytes = target_path.read_bytes()
    identity_drifts = {
        "phase": "u128_self_logged",
        "snapshot": "update128_IBR1-SELF",
        "family_id": "F2",
        "family_arm": "IBR1-SELF",
        "engine_arm": "S-SELF",
        "checkpoint_u_pre": 0,
        "support": "CAL",
        "rows": 511,
        "mode": "self",
    }
    for field, drifted_value in identity_drifts.items():
        document = json.loads(target_bytes)
        document[field] = drifted_value
        _write(target_path, document)
        with pytest.raises(IBR1GateContractError, match="identity drifted"):
            build_ibr1_negative_result_seal(tmp_path, **artifacts)
        target_path.write_bytes(target_bytes)

    for stratum in ("change", "turn", "other"):
        document = json.loads(target_bytes)
        document["summary"]["means"][stratum] += 0.01
        _write(target_path, document)
        with pytest.raises(
            IBR1GateContractError,
            match="summary differs from row_losses and frozen EVAL raw strata",
        ):
            build_ibr1_negative_result_seal(tmp_path, **artifacts)
        target_path.write_bytes(target_bytes)

    document = json.loads(target_bytes)
    document["summary"]["counts"]["change"] += 1
    _write(target_path, document)
    with pytest.raises(
        IBR1GateContractError,
        match="summary differs from row_losses and frozen EVAL raw strata",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    target_path.write_bytes(target_bytes)

    document = json.loads(target_bytes)
    document["row_losses"][11] += 0.1
    document["summary"]["means"]["overall"] = (
        math.fsum(document["row_losses"]) / 512
    )
    _write(target_path, document)
    with pytest.raises(
        IBR1GateContractError,
        match="summary differs from row_losses and frozen EVAL raw strata",
    ):
        build_ibr1_negative_result_seal(tmp_path, **artifacts)
    target_path.write_bytes(target_bytes)


def test_candidate_lock_is_self_hashed_and_frozen_exclusively(
    tmp_path: Path,
) -> None:
    document = build_ibr1_candidate_lock_receipt()
    assert document["analysis_class"] == IBR1_CANDIDATE_LOCK_CLASS
    assert document["candidate_cap"] == 1
    assert document["seed"] == 0
    assert document["package"] == "SA-Hstar"
    assert document["formal_training_authorized"] is False

    path = tmp_path / "candidate_lock.json"
    freeze_ibr1_candidate_lock_receipt(path)
    with pytest.raises(IBR1GateContractError, match="overwrite"):
        freeze_ibr1_candidate_lock_receipt(path)
