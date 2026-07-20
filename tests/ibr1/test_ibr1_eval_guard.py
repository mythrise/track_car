from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

import ibr1_experiment.eval_guard as eval_guard_module
from f2_experiment.runner import RunnerRow
from f2_experiment.support import canonical_json_bytes
from ibr1_experiment.assembly_model import IBR1_CTRL, IBR1_SELF
from ibr1_experiment.authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    SUPPORT_BINDING_CLASS,
)
from ibr1_experiment.eval_guard import (
    FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
    FROZEN_EVAL_ROWS,
    IBR1_EVAL_PHASES,
    IBR1EvalGuardContractError,
    IBR1EvalOrderGuard,
)
from ibr1_experiment.model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


EVAL_BLOCK_STARTS = (
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
EVAL_INDICES = tuple(
    index for start in EVAL_BLOCK_STARTS for index in range(start, start + 32)
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _rehash(document: dict[str, Any]) -> dict[str, Any]:
    result = dict(document)
    result.pop("receipt_payload_sha256", None)
    result["receipt_payload_sha256"] = _sha256(canonical_json_bytes(result))
    return result


def _write_receipt(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _authority_document() -> dict[str, Any]:
    support_binding = _rehash(
        {
            "schema_version": 1,
            "analysis_class": SUPPORT_BINDING_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "observation": {
                "supports": {
                    "EVAL-FIX": {
                        "rows": FROZEN_EVAL_ROWS,
                        "ordered_original_indices": list(EVAL_INDICES),
                        "ordered_original_indices_sha256": (
                            FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
                        ),
                        "row_set_sha256": (
                            FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
                        ),
                    }
                },
                "inherited_support_contract_payload_sha256": "c" * 64,
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    return _rehash(
        {
            "schema_version": 1,
            "receipt_version": 1,
            "analysis_class": ASSEMBLY_RECEIPT_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "phase": ASSEMBLY_PHASE_FINAL,
            "support_binding": support_binding,
            "lambda_freeze_binding": {
                "path": "experiments/windows_cuda_ibr1/lambda_freeze.json",
                "sha256": "a" * 64,
            },
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )


@pytest.fixture
def final_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = (tmp_path / "project").resolve()
    path = root / "experiments/windows_cuda_ibr1/assembly_final.json"
    document = _authority_document()
    _write_receipt(path, document)
    verifier_calls: list[tuple[Path, Path, str | None]] = []

    def verify(project_root, receipt_path, *, required_phase=None, **kwargs):
        del kwargs
        resolved_root = Path(project_root).resolve()
        resolved_path = Path(receipt_path).resolve()
        verifier_calls.append((resolved_root, resolved_path, required_phase))
        assert resolved_root == root
        assert resolved_path == path
        assert required_phase == ASSEMBLY_PHASE_FINAL
        return json.loads(resolved_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(eval_guard_module, "verify_assembly_receipt", verify)
    return {
        "root": root,
        "path": path,
        "document": document,
        "verifier_calls": verifier_calls,
    }


def _row(original_row_index: int) -> RunnerRow:
    return RunnerRow(
        original_row_index=original_row_index,
        sequence_id=f"sequence-{original_row_index}",
        frame_idx=original_row_index,
        mirrored=False,
        logged_prev_action=(0.0, 0.0, 0.0),
        target_actions=torch.zeros(8, 3, dtype=torch.float32),
        observation=object(),
    )


@pytest.fixture
def rows() -> tuple[RunnerRow, ...]:
    assert len(EVAL_INDICES) == FROZEN_EVAL_ROWS
    assert _sha256(canonical_json_bytes(list(EVAL_INDICES))) == (
        FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
    )
    return tuple(_row(index) for index in EVAL_INDICES)


def _guard(rows: tuple[RunnerRow, ...], final_authority) -> IBR1EvalOrderGuard:
    return IBR1EvalOrderGuard(
        rows,
        project_root=final_authority["root"],
        final_assembly_receipt_path=final_authority["path"],
    )


def _wrapper(guard, predictor, binding):
    return guard.wrap_predictor(
        predictor,
        phase=binding.phase,
        snapshot=binding.snapshot,
        family_arm=binding.family_arm,
        mode=binding.mode,
    )


def _run_all_with_same_predictor(guard, rows, predictor):
    results = []
    for binding in IBR1_EVAL_PHASES:
        wrapped = _wrapper(guard, predictor, binding)
        for position, row in enumerate(rows):
            results.append(
                wrapped(
                    row,
                    None,
                    mode=binding.mode,
                    reset=position == 0,
                    position=position,
                )
            )
    return results


def test_final_assembly_binds_fixed_512_row_order_only_receipt(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    sentinel = object()

    def same_predictor(row, prev_fy, *, mode, reset, position):
        del row, prev_fy, mode, reset, position
        return sentinel

    results = _run_all_with_same_predictor(guard, rows, same_predictor)
    receipt = guard.finalize()

    assert len(final_authority["verifier_calls"]) == 1
    assert receipt["rows_per_phase"] == FROZEN_EVAL_ROWS
    assert receipt["total_predictor_calls"] == 6 * FROZEN_EVAL_ROWS
    assert receipt["authority_role"] == "row_order_only"
    assert receipt["formal_training_authorized"] is False
    assert receipt["final_assembly_receipt"] == {
        "path": "experiments/windows_cuda_ibr1/assembly_final.json",
        "sha256": _sha256(final_authority["path"].read_bytes()),
        "receipt_payload_sha256": final_authority["document"][
            "receipt_payload_sha256"
        ],
    }
    support = final_authority["document"]["support_binding"]
    assert receipt["fresh_support_binding"]["receipt_payload_sha256"] == (
        support["receipt_payload_sha256"]
    )
    assert receipt["fresh_support_binding"][
        "eval_fix_ordered_original_indices_sha256"
    ] == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
    assert receipt["all_phase_mapping_bytes_identical"] is True
    assert receipt["all_phase_mapping_sha256_identical"] is True
    assert receipt["all_phase_mappings_equal_expected_binding"] is True
    assert all(
        phase["ordered_original_indices"] == list(EVAL_INDICES)
        for phase in receipt["phases"]
    )
    assert results == [sentinel] * (6 * FROZEN_EVAL_ROWS)
    scope = receipt["identity_scope"]
    assert scope["row_order_verified"] is True
    assert scope["predictor_identity_verified"] is False
    assert scope["checkpoint_identity_verified"] is False
    assert scope["u_pre_identity_verified"] is False
    assert scope["lifecycle_identity_proof_required"] == [
        "predictor_identity",
        "checkpoint_identity",
        "u_pre_identity",
    ]
    assert "lifecycle" in scope["predictor_checkpoint_u_pre_authority"]
    assert receipt["internal_test"] == "sealed"
    assert receipt["internal_test_opened"] is False


def test_reviewer_poc_arbitrary_three_rows_cannot_self_sign(final_authority):
    arbitrary = tuple(_row(index) for index in (101, 205, 309))
    with pytest.raises(IBR1EvalGuardContractError, match="exactly 512"):
        _guard(arbitrary, final_authority)


def test_predictor_baseexception_permanently_faults_guard(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    binding = IBR1_EVAL_PHASES[0]

    def broken_predictor(*args, **kwargs):
        del args, kwargs
        raise KeyboardInterrupt("reviewer PoC")

    wrapped = _wrapper(guard, broken_predictor, binding)
    with pytest.raises(KeyboardInterrupt, match="reviewer PoC"):
        wrapped(rows[0], None, mode=binding.mode, reset=True, position=0)
    with pytest.raises(IBR1EvalGuardContractError, match="permanently faulted"):
        wrapped(rows[0], None, mode=binding.mode, reset=True, position=0)
    with pytest.raises(IBR1EvalGuardContractError, match="permanently faulted"):
        _wrapper(guard, lambda *args, **kwargs: object(), IBR1_EVAL_PHASES[1])
    with pytest.raises(IBR1EvalGuardContractError, match="permanently faulted"):
        guard.finalize()


def test_swapping_ctrl_and_self_fails_before_predictor_runs(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    calls = 0

    def predictor(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    ctrl_phase = IBR1_EVAL_PHASES[2]
    with pytest.raises(IBR1EvalGuardContractError, match="family arm mismatch"):
        guard.wrap_predictor(
            predictor,
            phase=ctrl_phase.phase,
            snapshot=ctrl_phase.snapshot,
            family_arm=IBR1_SELF,
            mode=ctrl_phase.mode,
        )
    assert ctrl_phase.family_arm == IBR1_CTRL
    assert calls == 0


def test_snapshot_binding_must_match_the_fixed_phase(rows, final_authority):
    guard = _guard(rows, final_authority)
    binding = IBR1_EVAL_PHASES[0]
    with pytest.raises(IBR1EvalGuardContractError, match="snapshot mismatch"):
        guard.wrap_predictor(
            lambda *args, **kwargs: object(),
            phase=binding.phase,
            snapshot="update128_IBR1-SELF",
            family_arm=binding.family_arm,
            mode=binding.mode,
        )


def test_swapping_runtime_mode_fails_before_predictor_runs(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    calls = 0

    def predictor(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    binding = IBR1_EVAL_PHASES[0]
    wrapped = _wrapper(guard, predictor, binding)
    with pytest.raises(IBR1EvalGuardContractError, match="runtime mode mismatch"):
        wrapped(rows[0], None, mode="self", reset=True, position=0)
    assert calls == 0


def test_same_cardinality_permuted_mapping_fails_closed(rows, final_authority):
    guard = _guard(rows, final_authority)
    calls = 0

    def predictor(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    binding = IBR1_EVAL_PHASES[0]
    wrapped = _wrapper(guard, predictor, binding)
    permuted = rows[1:] + rows[:1]
    with pytest.raises(IBR1EvalGuardContractError, match="object identity"):
        for position, row in enumerate(permuted):
            wrapped(
                row,
                None,
                mode=binding.mode,
                reset=position == 0,
                position=position,
            )
    assert calls == 0


def test_incremented_substitute_indices_fail_even_on_same_objects(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    binding = IBR1_EVAL_PHASES[0]
    calls = 0

    def predictor(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    wrapped = _wrapper(guard, predictor, binding)
    object.__setattr__(rows[0], "original_row_index", rows[0].original_row_index + 1)
    with pytest.raises(IBR1EvalGuardContractError, match="original_row_index"):
        wrapped(rows[0], None, mode=binding.mode, reset=True, position=0)
    assert calls == 0


def test_missing_and_duplicate_calls_fail_closed(rows, final_authority):
    first = IBR1_EVAL_PHASES[0]
    missing_guard = _guard(rows, final_authority)
    wrapped = _wrapper(missing_guard, lambda *args, **kwargs: object(), first)
    wrapped(rows[0], None, mode=first.mode, reset=True, position=0)
    with pytest.raises(IBR1EvalGuardContractError, match="missing"):
        missing_guard.finalize()

    duplicate_guard = _guard(rows, final_authority)
    wrapped = _wrapper(duplicate_guard, lambda *args, **kwargs: object(), first)
    wrapped(rows[0], None, mode=first.mode, reset=True, position=0)
    with pytest.raises(IBR1EvalGuardContractError, match="phase call index"):
        wrapped(rows[0], None, mode=first.mode, reset=True, position=0)


def test_global_phase_order_rejects_skipping_to_update128_ctrl(
    rows, final_authority
):
    guard = _guard(rows, final_authority)
    ctrl_phase = IBR1_EVAL_PHASES[2]
    calls = 0

    def predictor(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return object()

    wrapped = _wrapper(guard, predictor, ctrl_phase)
    with pytest.raises(IBR1EvalGuardContractError, match="order mismatch"):
        wrapped(rows[0], None, mode=ctrl_phase.mode, reset=True, position=0)
    assert calls == 0


def test_support_sha_drift_inside_verified_assembly_is_rejected(
    rows, final_authority
):
    document = dict(final_authority["document"])
    support_binding = dict(document["support_binding"])
    observation = dict(support_binding["observation"])
    supports = dict(observation["supports"])
    eval_support = dict(supports["EVAL-FIX"])
    eval_support["ordered_original_indices_sha256"] = "0" * 64
    supports["EVAL-FIX"] = eval_support
    observation["supports"] = supports
    support_binding["observation"] = observation
    document["support_binding"] = _rehash(support_binding)
    document = _rehash(document)
    _write_receipt(final_authority["path"], document)

    with pytest.raises(IBR1EvalGuardContractError, match="identity drifted"):
        _guard(rows, final_authority)
