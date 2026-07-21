"""Fail-closed fixed-order guard for the six authoritative IBR1 EVAL passes.

The guard is deliberately independent from the evaluator.  It neither loads
data nor invokes CAL, training, or the internal test.  It first verifies one
final IBR1 assembly receipt, freezes that authority's concrete 512-row
EVAL-FIX sequence, then validates object identity and position mapping before
delegating each predictor call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

from f2_experiment.runner import RunnerRow
from f2_experiment.support import SUPPORT_EXPECTATIONS, canonical_json_bytes

from .assembly_model import IBR1_CTRL, IBR1_SELF, IBR1AssemblyContractError
from .authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    SUPPORT_BINDING_CLASS,
    verify_assembly_receipt,
)
from .model import IBR1_FAMILY_ID


EVAL_SUPPORT_NAME = "EVAL-FIX"
EVAL_GUARD_RECEIPT_CLASS = "ibr1_eval_fixed_order_guard_receipt"
FROZEN_EVAL_ROWS = 512
FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256 = (
    "5123a14dc526dfcef96e73ee838e33b265dee0bff0efe66e36e806540e1922ec"
)


class IBR1EvalGuardContractError(IBR1AssemblyContractError):
    """Raised before prediction when the frozen IBR1 EVAL order drifts."""


@dataclass(frozen=True)
class IBR1EvalPhase:
    """One exact arm/snapshot/mode pass in the authoritative smoke order."""

    phase: str
    snapshot: str
    family_arm: str
    mode: str

    def to_dict(self) -> dict[str, str]:
        return {
            "phase": self.phase,
            "snapshot": self.snapshot,
            "family_id": IBR1_FAMILY_ID,
            "family_arm": self.family_arm,
            "mode": self.mode,
        }


IBR1_EVAL_PHASES = (
    IBR1EvalPhase(
        phase="u0_self_logged",
        snapshot="update0_IBR1-SELF",
        family_arm=IBR1_SELF,
        mode="logged",
    ),
    IBR1EvalPhase(
        phase="u0_self_self",
        snapshot="update0_IBR1-SELF",
        family_arm=IBR1_SELF,
        mode="self",
    ),
    IBR1EvalPhase(
        phase="u128_ctrl_logged",
        snapshot="update128_IBR1-CTRL",
        family_arm=IBR1_CTRL,
        mode="logged",
    ),
    IBR1EvalPhase(
        phase="u128_ctrl_self",
        snapshot="update128_IBR1-CTRL",
        family_arm=IBR1_CTRL,
        mode="self",
    ),
    IBR1EvalPhase(
        phase="u128_self_logged",
        snapshot="update128_IBR1-SELF",
        family_arm=IBR1_SELF,
        mode="logged",
    ),
    IBR1EvalPhase(
        phase="u128_self_self",
        snapshot="update128_IBR1-SELF",
        family_arm=IBR1_SELF,
        mode="self",
    ),
)
_PHASE_BY_NAME = {binding.phase: binding for binding in IBR1_EVAL_PHASES}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1EvalGuardContractError(message)


def _valid_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256",
    )
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _payload_sha256(value: Mapping[str, Any], label: str) -> str:
    payload = dict(value)
    stored = _valid_sha256(
        payload.pop("receipt_payload_sha256", None),
        f"{label} receipt payload SHA",
    )
    _require(
        _sha256(canonical_json_bytes(payload)) == stored,
        f"{label} receipt payload SHA mismatch",
    )
    return stored


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _verified_receipt_file_sha256(
    path: Path, expected_document: Mapping[str, Any], label: str
) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = path.read_bytes()
        observed = json.loads(
            payload.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise IBR1EvalGuardContractError(f"cannot read {label}: {path}") from exc
    _require(
        isinstance(observed, dict) and observed == dict(expected_document),
        f"{label} bytes differ from the verified authority document",
    )
    _require(
        payload == canonical_json_bytes(observed) + b"\n",
        f"{label} is not canonical JSON plus LF",
    )
    return _sha256(payload)


class IBR1EvalOrderGuard:
    """Freeze and enforce one exact six-pass ``position -> row`` mapping."""

    def __init__(
        self,
        expected_eval_rows: Sequence[RunnerRow],
        *,
        project_root: str | Path,
        final_assembly_receipt_path: str | Path,
    ) -> None:
        root = Path(project_root).expanduser().resolve()
        receipt_path = Path(final_assembly_receipt_path).expanduser().resolve()
        final_assembly = verify_assembly_receipt(
            root,
            receipt_path,
            required_phase=ASSEMBLY_PHASE_FINAL,
        )
        _require(
            final_assembly.get("analysis_class") == ASSEMBLY_RECEIPT_CLASS
            and final_assembly.get("phase") == ASSEMBLY_PHASE_FINAL
            and final_assembly.get("family_id") == IBR1_FAMILY_ID
            and final_assembly.get("formal_training_authorized") is False
            and final_assembly.get("internal_test") == "sealed"
            and final_assembly.get("internal_test_opened") is False,
            "verified final IBR1 assembly identity drifted",
        )
        final_assembly_payload_sha = _payload_sha256(
            final_assembly, "final IBR1 assembly"
        )
        final_assembly_file_sha = _verified_receipt_file_sha256(
            receipt_path,
            final_assembly,
            "final IBR1 assembly receipt",
        )
        try:
            relative_receipt_path = receipt_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise IBR1EvalGuardContractError(
                "final IBR1 assembly receipt lies outside project_root"
            ) from exc

        support_binding = _mapping(
            final_assembly.get("support_binding"), "final assembly support binding"
        )
        _require(
            support_binding.get("analysis_class") == SUPPORT_BINDING_CLASS
            and support_binding.get("family_id") == IBR1_FAMILY_ID
            and support_binding.get("formal_training_authorized") is False
            and support_binding.get("internal_test") == "sealed"
            and support_binding.get("internal_test_opened") is False,
            "final assembly support binding identity drifted",
        )
        support_payload_sha = _payload_sha256(
            support_binding, "final assembly support binding"
        )
        support_observation = _mapping(
            support_binding.get("observation"), "fresh support observation"
        )
        supports = _mapping(
            support_observation.get("supports"), "fresh support registry"
        )
        eval_support = _mapping(
            supports.get(EVAL_SUPPORT_NAME), "fresh EVAL-FIX support"
        )
        expectation = SUPPORT_EXPECTATIONS[EVAL_SUPPORT_NAME]
        _require(
            expectation.rows == FROZEN_EVAL_ROWS
            and expectation.sha256
            == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
            "frozen EVAL-FIX SUPPORT_EXPECTATIONS drifted",
        )
        authority_indices_value = eval_support.get("ordered_original_indices")
        _require(
            isinstance(authority_indices_value, Sequence)
            and not isinstance(authority_indices_value, (str, bytes, bytearray)),
            "fresh EVAL-FIX ordered original indices must be a sequence",
        )
        authority_indices = tuple(authority_indices_value)
        _require(
            len(authority_indices) == FROZEN_EVAL_ROWS
            and all(
                isinstance(index, int) and not isinstance(index, bool)
                for index in authority_indices
            )
            and len(set(authority_indices)) == FROZEN_EVAL_ROWS,
            "fresh EVAL-FIX ordered original-index coverage drifted",
        )
        authority_bytes = canonical_json_bytes(list(authority_indices))
        authority_sha = _sha256(authority_bytes)
        _require(
            eval_support.get("rows") == FROZEN_EVAL_ROWS
            and eval_support.get("ordered_original_indices_sha256")
            == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
            and eval_support.get("row_set_sha256")
            == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256
            and authority_sha == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
            "fresh EVAL-FIX ordered support identity drifted",
        )
        _require(
            isinstance(expected_eval_rows, Sequence)
            and not isinstance(expected_eval_rows, (str, bytes, bytearray)),
            "expected eval_rows must be an ordered sequence",
        )
        rows = tuple(expected_eval_rows)
        _require(
            len(rows) == FROZEN_EVAL_ROWS
            and all(isinstance(row, RunnerRow) for row in rows),
            "expected eval_rows must contain exactly 512 RunnerRow objects",
        )
        original_indices = tuple(row.original_row_index for row in rows)
        _require(
            original_indices == authority_indices,
            "expected eval_rows position mapping differs from final assembly support",
        )
        expected_bytes = canonical_json_bytes(list(original_indices))
        expected_sha = _sha256(expected_bytes)
        _require(
            expected_bytes == authority_bytes
            and expected_sha == FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256,
            "expected eval_rows ordered original-index bytes/SHA mismatch",
        )

        self._expected_rows = rows
        self._expected_original_indices = original_indices
        self._expected_original_index_bytes = expected_bytes
        self._support_ordered_original_indices_sha256 = authority_sha
        self._final_assembly_receipt_binding = {
            "path": relative_receipt_path,
            "sha256": final_assembly_file_sha,
            "receipt_payload_sha256": final_assembly_payload_sha,
            "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        }
        self._support_binding_payload_sha256 = support_payload_sha
        self._inherited_support_contract_payload_sha256 = _valid_sha256(
            support_observation.get("inherited_support_contract_payload_sha256"),
            "inherited support contract payload SHA",
        )
        self._calls_by_phase: dict[str, list[int]] = {
            binding.phase: [] for binding in IBR1_EVAL_PHASES
        }
        self._wrapped_phases: set[str] = set()
        self._total_calls = 0
        self._in_flight = False
        self._faulted = False
        self._finalized = False

    @property
    def expected_rows_per_phase(self) -> int:
        return len(self._expected_rows)

    @property
    def support_ordered_original_indices_sha256(self) -> str:
        return self._support_ordered_original_indices_sha256

    def _require_healthy(self) -> None:
        _require(
            not self._faulted,
            "IBR1 EVAL guard is permanently faulted after predictor BaseException",
        )

    def _validate_wrapper_binding(
        self,
        *,
        phase: str,
        snapshot: str,
        family_arm: str,
        mode: str,
    ) -> IBR1EvalPhase:
        self._require_healthy()
        _require(not self._finalized, "IBR1 EVAL guard is already finalized")
        binding = _PHASE_BY_NAME.get(phase)
        _require(binding is not None, f"unknown IBR1 EVAL phase {phase!r}")
        assert binding is not None
        _require(
            snapshot == binding.snapshot,
            f"IBR1 EVAL snapshot mismatch for phase {phase}",
        )
        _require(
            family_arm == binding.family_arm,
            f"IBR1 EVAL family arm mismatch for phase {phase}",
        )
        _require(mode == binding.mode, f"IBR1 EVAL mode mismatch for phase {phase}")
        _require(
            phase not in self._wrapped_phases,
            f"duplicate IBR1 EVAL predictor wrapper for phase {phase}",
        )
        self._wrapped_phases.add(phase)
        return binding

    def _validate_call(
        self,
        binding: IBR1EvalPhase,
        row: RunnerRow,
        *,
        mode: str,
        position: int,
    ) -> None:
        self._require_healthy()
        _require(not self._finalized, "IBR1 EVAL guard is already finalized")
        _require(not self._in_flight, "nested IBR1 EVAL predictor call is forbidden")
        rows_per_phase = len(self._expected_rows)
        expected_total = len(IBR1_EVAL_PHASES) * rows_per_phase
        _require(
            self._total_calls < expected_total,
            "duplicate or excess IBR1 EVAL predictor call",
        )
        expected_phase_index = self._total_calls // rows_per_phase
        expected_binding = IBR1_EVAL_PHASES[expected_phase_index]
        _require(
            binding == expected_binding,
            "IBR1 EVAL phase/snapshot/family-arm/mode order mismatch: "
            f"expected {expected_binding.phase}, observed {binding.phase}",
        )
        expected_position = self._total_calls % rows_per_phase
        _require(
            isinstance(position, Integral) and not isinstance(position, bool),
            "IBR1 EVAL position must be an integer",
        )
        _require(
            int(position) == expected_position,
            "IBR1 EVAL position differs from the phase call index",
        )
        _require(
            mode == expected_binding.mode,
            f"IBR1 EVAL runtime mode mismatch for phase {binding.phase}",
        )
        _require(isinstance(row, RunnerRow), "IBR1 EVAL row must be a RunnerRow")
        expected_row = self._expected_rows[expected_position]
        _require(
            row is expected_row,
            "IBR1 EVAL row object identity differs from frozen eval_rows",
        )
        expected_original_index = self._expected_original_indices[expected_position]
        _require(
            row.original_row_index == expected_original_index,
            "IBR1 EVAL original_row_index differs from the frozen position mapping",
        )

    def _commit_call(self, binding: IBR1EvalPhase, row: RunnerRow) -> None:
        self._calls_by_phase[binding.phase].append(row.original_row_index)
        self._total_calls += 1

    def wrap_predictor(
        self,
        predictor: Callable[..., Any],
        *,
        phase: str,
        snapshot: str,
        family_arm: str,
        mode: str,
    ) -> Callable[..., Any]:
        """Bind one pass and validate every row before invoking ``predictor``."""

        self._require_healthy()
        _require(callable(predictor), "IBR1 EVAL predictor must be callable")
        binding = self._validate_wrapper_binding(
            phase=phase,
            snapshot=snapshot,
            family_arm=family_arm,
            mode=mode,
        )

        def wrapped(
            row: RunnerRow,
            prev_fy: Any,
            *,
            mode: str,
            reset: bool,
            position: int,
        ) -> Any:
            self._validate_call(
                binding,
                row,
                mode=mode,
                position=position,
            )
            self._in_flight = True
            try:
                result = predictor(
                    row,
                    prev_fy,
                    mode=mode,
                    reset=reset,
                    position=position,
                )
            except BaseException:
                self._faulted = True
                raise
            finally:
                self._in_flight = False
            self._commit_call(binding, row)
            return result

        return wrapped

    def finalize(self) -> dict[str, Any]:
        """Prove all six mappings are byte-identical to the support binding."""

        self._require_healthy()
        _require(not self._finalized, "IBR1 EVAL guard is already finalized")
        rows_per_phase = len(self._expected_rows)
        expected_total = len(IBR1_EVAL_PHASES) * rows_per_phase
        _require(
            self._total_calls == expected_total,
            "missing IBR1 EVAL predictor calls: "
            f"expected {expected_total}, observed {self._total_calls}",
        )
        _require(
            self._wrapped_phases == set(_PHASE_BY_NAME),
            "IBR1 EVAL phase wrapper coverage mismatch",
        )

        phase_receipts: list[dict[str, Any]] = []
        phase_bytes: list[bytes] = []
        phase_shas: list[str] = []
        for binding in IBR1_EVAL_PHASES:
            observed = tuple(self._calls_by_phase[binding.phase])
            _require(
                len(observed) == rows_per_phase,
                f"IBR1 EVAL phase {binding.phase} cardinality mismatch",
            )
            observed_bytes = canonical_json_bytes(list(observed))
            observed_sha = _sha256(observed_bytes)
            _require(
                observed_bytes == self._expected_original_index_bytes,
                f"IBR1 EVAL phase {binding.phase} position mapping bytes mismatch",
            )
            _require(
                observed_sha == self._support_ordered_original_indices_sha256,
                f"IBR1 EVAL phase {binding.phase} ordered original-index SHA mismatch",
            )
            phase_bytes.append(observed_bytes)
            phase_shas.append(observed_sha)
            phase_receipts.append(
                {
                    **binding.to_dict(),
                    "rows": len(observed),
                    "ordered_original_indices": list(observed),
                    "ordered_original_indices_sha256": observed_sha,
                    "bytes_equal_expected_binding": True,
                }
            )
        _require(
            len(set(phase_bytes)) == 1,
            "six IBR1 EVAL phase mapping byte streams differ",
        )
        _require(
            len(set(phase_shas)) == 1,
            "six IBR1 EVAL phase mapping SHAs differ",
        )

        self._finalized = True
        return {
            "schema_version": 1,
            "analysis_class": EVAL_GUARD_RECEIPT_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "authority_role": "row_order_only",
            "formal_training_authorized": False,
            "support": EVAL_SUPPORT_NAME,
            "final_assembly_receipt": dict(self._final_assembly_receipt_binding),
            "fresh_support_binding": {
                "receipt_payload_sha256": self._support_binding_payload_sha256,
                "inherited_support_contract_payload_sha256": (
                    self._inherited_support_contract_payload_sha256
                ),
                "eval_fix_rows": FROZEN_EVAL_ROWS,
                "eval_fix_ordered_original_indices_sha256": (
                    self._support_ordered_original_indices_sha256
                ),
            },
            "phase_order": [binding.phase for binding in IBR1_EVAL_PHASES],
            "rows_per_phase": rows_per_phase,
            "phases": phase_receipts,
            "total_predictor_calls": self._total_calls,
            "expected_total_predictor_calls": expected_total,
            "support_ordered_original_indices_sha256": (
                self._support_ordered_original_indices_sha256
            ),
            "all_phase_mapping_bytes_identical": True,
            "all_phase_mapping_sha256_identical": True,
            "all_phase_mappings_equal_expected_binding": True,
            "identity_scope": {
                "row_order_verified": True,
                "predictor_identity_verified": False,
                "checkpoint_identity_verified": False,
                "u_pre_identity_verified": False,
                "lifecycle_identity_proof_required": [
                    "predictor_identity",
                    "checkpoint_identity",
                    "u_pre_identity",
                ],
                "predictor_checkpoint_u_pre_authority": (
                    "must be proven separately by the IBR1 smoke lifecycle"
                ),
            },
            "internal_test": "sealed",
            "internal_test_opened": False,
        }


__all__ = [
    "EVAL_GUARD_RECEIPT_CLASS",
    "EVAL_SUPPORT_NAME",
    "FROZEN_EVAL_ORDERED_ORIGINAL_INDICES_SHA256",
    "FROZEN_EVAL_ROWS",
    "IBR1_EVAL_PHASES",
    "IBR1EvalGuardContractError",
    "IBR1EvalOrderGuard",
    "IBR1EvalPhase",
]
