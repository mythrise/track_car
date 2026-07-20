"""Single-parent authoritative lifecycle for the preregistered IBR1 smoke.

The public entry point keeps the non-serializable authority continuation live
across ``CAL pair -> freeze/final -> capability consumption -> smoke plan``.
It then executes the fixed checkpoint/EVAL/train/EVAL/diagnostics/gate order.
Engineering failures burn the fresh output directory and never manufacture a
scientific PASS/FAIL seal.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

from f2_experiment.assembly import run_eval_fix
from f2_experiment.runner import (
    ARM_ORDER,
    S_CTRL,
    S_SELF,
    RunnerTelemetryHooks,
    run_paired_smoke,
)

from .artifacts import (
    DIAGNOSTICS_MANIFEST_FILENAME,
    write_diagnostics_bundle,
)
from .assembly_model import (
    FAMILY_TO_ENGINE_ARM,
    IBR1_CTRL,
    IBR1_SELF,
)
from .authority import (
    ASSEMBLY_PHASE_BOOTSTRAP,
    ASSEMBLY_PHASE_FINAL,
    canonical_json_bytes,
    canonical_json_sha256,
    exclusive_write_json,
    verify_assembly_receipt,
)
from .cal_pair import (
    consume_final_authority_capability,
    run_live_cal_pair_and_freeze,
)
from .calibration import NUMERIC_EVIDENCE_FILENAME
from .checkpoint import save_ibr1_arm_checkpoint
from .eval_guard import IBR1_EVAL_PHASES, IBR1EvalOrderGuard, IBR1EvalPhase
from .gates import (
    IBR1_GATE_IDS,
    build_ibr1_negative_result_seal,
    build_ibr1_pass_seal,
    build_ibr1_combined_gate_receipt,
    evaluate_i1,
    evaluate_i2,
    evaluate_i3,
    evaluate_i4,
    evaluate_i5,
    evaluate_i6,
    freeze_ibr1_candidate_lock_receipt,
    freeze_ibr1_combined_gate_receipt,
    freeze_ibr1_gate_receipt,
)
from .model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID
from .smoke_model import IBR1SmokePlan, build_ibr1_production_smoke_plan


CANDIDATE_LOCK_FILENAME = "candidate_lock.json"
COUNT_RECEIPT_FILENAME = "count_receipt.json"
EVAL_GUARD_FILENAME = "eval_order_guard_receipt.json"
COMBINED_GATE_FILENAME = "combined_ibr1_gate_receipt.json"
PASS_SEAL_FILENAME = "ibr1_pass_seal.json"
NEGATIVE_SEAL_FILENAME = "ibr1_negative_result_seal.json"
SMOKE_SUMMARY_FILENAME = "smoke_summary.json"
ENGINEERING_FAILURE_FILENAME = "engineering_failure.json"
DIAGNOSTICS_DIRNAME = "diagnostics"

_OFFICIAL_CAL_RUNNER = run_live_cal_pair_and_freeze
_OFFICIAL_CAPABILITY_CONSUMER = consume_final_authority_capability
_OFFICIAL_PLAN_BUILDER = build_ibr1_production_smoke_plan


class IBR1LifecycleError(RuntimeError):
    """Raised when the authoritative smoke cannot preserve its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1LifecycleError(message)


def _inside(root: Path, value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise IBR1LifecycleError(
            f"{label} must stay inside the project root: {resolved}"
        ) from exc
    return resolved


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _sha256_file(path: Path, label: str) -> str:
    _require(path.is_file(), f"{label} is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = path.read_bytes()
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IBR1LifecycleError(f"cannot read {label}: {path}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    _require(
        payload == canonical_json_bytes(value) + b"\n",
        f"{label} is not canonical JSON plus LF",
    )
    return value


def _binding(root: Path, path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "path": _relative(root, path),
        "sha256": _sha256_file(path, path.name),
        "analysis_class": document.get("analysis_class"),
    }
    payload_sha = document.get("receipt_payload_sha256")
    if isinstance(payload_sha, str):
        result["receipt_payload_sha256"] = payload_sha
    return result


def _self_hashed(document: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(document)
    _require(
        "receipt_payload_sha256" not in result,
        "document already contains receipt_payload_sha256",
    )
    result["receipt_payload_sha256"] = canonical_json_sha256(result)
    return result


def _preflight_paths(
    root: Path,
    *,
    bootstrap: Path,
    cal_output: Path,
    freeze_output: Path,
    final_output: Path,
    smoke_output: Path,
) -> None:
    _require(root.is_dir(), f"project root is not a directory: {root}")
    _require(bootstrap.is_file(), f"bootstrap receipt is missing: {bootstrap}")
    destinations = (cal_output, freeze_output, final_output, smoke_output)
    _require(
        len(set(destinations)) == len(destinations),
        "CAL/freeze/final/smoke outputs must be distinct",
    )
    for destination in destinations:
        _require(
            not destination.exists(),
            f"authoritative output already exists: {destination}",
        )
        _require(destination != bootstrap, "an output path aliases the bootstrap")
    for directory in (cal_output, smoke_output):
        for other in destinations:
            if other == directory:
                continue
            _require(
                not other.is_relative_to(directory),
                f"output {other} must not be nested under fresh directory {directory}",
            )


def _write_engineering_failure(
    output: Path,
    *,
    stage: str,
    error: BaseException,
) -> None:
    path = output / ENGINEERING_FAILURE_FILENAME
    if path.exists():
        return
    result_seal_written = any(
        (output / filename).exists()
        for filename in (PASS_SEAL_FILENAME, NEGATIVE_SEAL_FILENAME)
    )
    document = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": "ibr1_engineering_failure_burn",
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
            "valid_scientific_result": False,
            "result_seal_written": result_seal_written,
            "rerun_requires_fresh_output_directory": True,
            "formal_training_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    try:
        exclusive_write_json(path, document)
    except BaseException:
        # Never replace the original engineering failure with audit-marker I/O.
        pass


def _checkpoint_filename(u_pre: int, family_arm: str) -> str:
    return f"checkpoint_update{u_pre}_{family_arm}.pt"


def _save_checkpoints(
    root: Path,
    output: Path,
    plan: IBR1SmokePlan,
    *,
    u_pre: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    results: dict[str, dict[str, Any]] = {}
    targets: dict[str, dict[str, Any]] = {}
    arms = plan.arms
    for engine_arm in ARM_ORDER:
        arm = arms[engine_arm]
        target = arm.checkpoint_identity(u_pre)
        target_receipt = target.identity_receipt()
        result = save_ibr1_arm_checkpoint(
            output / _checkpoint_filename(u_pre, arm.family_arm),
            **target.writer_kwargs(),
        )
        _require(
            result.get("family_arm") == arm.family_arm
            and result.get("engine_arm") == engine_arm
            and result.get("u_pre") == u_pre,
            f"checkpoint writer identity drifted for {arm.family_arm} update {u_pre}",
        )
        results[arm.family_arm] = result
        targets[arm.family_arm] = target_receipt
    if u_pre == 0:
        tensor_shas = {result["tensor_sha256"] for result in results.values()}
        _require(
            tensor_shas == {plan.checkpoint_init_sha256},
            "update-0 checkpoint tensor identities differ from sealed init",
        )
    return results, targets


def _eval_filename(phase: IBR1EvalPhase) -> str:
    return f"eval_phase_{phase.phase}.json"


def _run_eval_phase(
    root: Path,
    output: Path,
    plan: IBR1SmokePlan,
    guard: IBR1EvalOrderGuard,
    phase: IBR1EvalPhase,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    engine_arm = FAMILY_TO_ENGINE_ARM[phase.family_arm]
    arm = plan.arms[engine_arm]
    predictor = arm.eval_predictor_factory(phase.snapshot)
    _require(
        predictor.engine_arm == engine_arm
        and predictor.family_arm == phase.family_arm
        and predictor.snapshot == phase.snapshot
        and predictor.arm_assembly is arm.assembly,
        f"EVAL predictor identity drifted for phase {phase.phase}",
    )
    guarded = guard.wrap_predictor(
        predictor,
        phase=phase.phase,
        snapshot=phase.snapshot,
        family_arm=phase.family_arm,
        mode=phase.mode,
    )
    base_receipt = run_eval_fix(
        eval_rows=plan.eval_rows,
        raw_rows=plan.eval_raw_rows,
        mode=phase.mode,
        predictor=guarded,
        strafe_reset_original_indices=(
            plan.data.eval_strafe_reset_original_indices
        ),
    )
    checkpoint_u_pre = 0 if phase.snapshot.startswith("update0_") else 128
    receipt = {
        **base_receipt,
        **phase.to_dict(),
        "engine_arm": engine_arm,
        "checkpoint_u_pre": checkpoint_u_pre,
    }
    path = output / _eval_filename(phase)
    exclusive_write_json(path, receipt)
    identity = {
        **phase.to_dict(),
        "engine_arm": engine_arm,
        "checkpoint_u_pre": checkpoint_u_pre,
        "receipt": _binding(root, path, receipt),
        "verified": True,
    }
    return receipt, path, identity


def _gate_document(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    _require(isinstance(value, Mapping), "IBR1 gate result is not mapping-like")
    return dict(value)


def _write_gates(
    root: Path,
    output: Path,
    *,
    final_document: Mapping[str, Any],
    final_path: Path,
    cal_numeric_path: Path,
    checkpoints: Mapping[int, Mapping[str, Mapping[str, Any]]],
    paired_result: Any,
    training_records: Sequence[Mapping[str, Any]],
    diagnostics_summary: Mapping[str, Any],
    gradient_document: Mapping[str, Any],
    eval_receipts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    update0_sidecars = {
        arm: Path(checkpoints[0][arm]["sidecar"]) for arm in (IBR1_CTRL, IBR1_SELF)
    }
    update0_documents = {
        arm: _load_canonical_json(path, f"{arm} update-0 checkpoint sidecar")
        for arm, path in update0_sidecars.items()
    }
    cal_numeric = _load_canonical_json(cal_numeric_path, "IBR1 CAL numeric evidence")
    ctrl_result = paired_result.arms[S_CTRL]
    self_result = paired_result.arms[S_SELF]
    snapshot_inputs = {
        "s_self_update0": {
            "logged": eval_receipts["u0_self_logged"]["summary"],
            "self": eval_receipts["u0_self_self"]["summary"],
        },
        "s_self_update128": {
            "logged": eval_receipts["u128_self_logged"]["summary"],
            "self": eval_receipts["u128_self_self"]["summary"],
        },
        "s_ctrl_update128": {
            "logged": eval_receipts["u128_ctrl_logged"]["summary"],
            "self": eval_receipts["u128_ctrl_self"]["summary"],
        },
    }
    receipts = {
        "I1": evaluate_i1(
            final_document,
            cal_numeric,
            update0_documents,
            project_root=root,
            final_assembly_receipt_path=final_path,
            update0_checkpoint_sidecar_paths=update0_sidecars,
        ),
        "I2": evaluate_i2(
            training_records,
            diagnostics_summary["training_geometry"],
        ),
        "I3": evaluate_i3(ctrl_result.g6_updates, gradient_document),
        "I4": evaluate_i4(
            [update.gate_update() for update in ctrl_result.g7_updates],
            [update.gate_update() for update in self_result.g7_updates],
        ),
        "I5": evaluate_i5(**snapshot_inputs),
        "I6": evaluate_i6(
            ctrl_result.g9.gate_kwargs(),
            self_result.g9.gate_kwargs(),
        ),
    }
    _require(set(receipts) == set(IBR1_GATE_IDS), "IBR1 gate set drifted")
    paths: dict[str, Path] = {}
    documents: dict[str, dict[str, Any]] = {}
    for gate_id in IBR1_GATE_IDS:
        path = output / f"gate_{gate_id}.json"
        freeze_ibr1_gate_receipt(path, receipts[gate_id])
        paths[gate_id] = path
        documents[gate_id] = _gate_document(receipts[gate_id])
    combined = build_ibr1_combined_gate_receipt(
        *(receipts[gate_id] for gate_id in IBR1_GATE_IDS)
    )
    combined_path = output / COMBINED_GATE_FILENAME
    freeze_ibr1_combined_gate_receipt(combined_path, combined)
    return combined, paths, documents


def _execute_smoke_plan_body(
    root: Path,
    output: Path,
    *,
    plan: IBR1SmokePlan,
    candidate_lock_path: Path,
    final_path: Path,
    cal_output: Path,
    cal_result: Mapping[str, Any],
    live_authority: Mapping[str, Any],
    close_plan: Callable[[], None],
) -> dict[str, Any]:
    _require(plan.production_context, "IBR1 smoke plan is not production-bound")
    _require(plan.authority_eligible, "IBR1 smoke plan is not authority-eligible")
    _require(plan.formal_training_authorized is False, "smoke plan authorizes formal training")
    _require(
        plan.internal_test == "sealed" and plan.internal_test_opened is False,
        "smoke plan opened the internal test",
    )
    identity_receipt = plan.identity_receipt()
    final_document = verify_assembly_receipt(
        root,
        final_path,
        required_phase=ASSEMBLY_PHASE_FINAL,
    )
    _require(
        dict(final_document) == dict(plan.final_assembly_receipt),
        "smoke plan final assembly differs from live authority",
    )

    checkpoint_results: dict[int, dict[str, dict[str, Any]]] = {}
    checkpoint_targets: dict[int, dict[str, dict[str, Any]]] = {}
    checkpoint_results[0], checkpoint_targets[0] = _save_checkpoints(
        root, output, plan, u_pre=0
    )

    guard = IBR1EvalOrderGuard(
        plan.eval_rows,
        project_root=root,
        final_assembly_receipt_path=final_path,
    )
    eval_receipts: dict[str, dict[str, Any]] = {}
    eval_paths: dict[str, Path] = {}
    predictor_identity: dict[str, dict[str, Any]] = {}

    for phase in IBR1_EVAL_PHASES[:2]:
        receipt, path, binding = _run_eval_phase(root, output, plan, guard, phase)
        eval_receipts[phase.phase] = receipt
        eval_paths[phase.phase] = path
        predictor_identity[phase.phase] = binding

    paired_result = run_paired_smoke(
        plan.smoke_rows,
        callbacks={engine: plan.arms[engine].callbacks for engine in ARM_ORDER},
        hooks=RunnerTelemetryHooks(g6_update=plan.g6_update),
        strafe_reset_original_indices=(
            plan.data.smoke_strafe_reset_original_indices
        ),
        expected_static_reset_original_indices=(
            plan.data.smoke_expected_static_reset_original_indices
        ),
        require_audit_counters=True,
    )
    count_document = paired_result.count_receipt.to_dict()
    count_path = output / COUNT_RECEIPT_FILENAME
    exclusive_write_json(count_path, count_document)
    _require(
        count_document.get("passed") is True,
        "paired runner count receipt failed; this is an engineering failure",
    )

    checkpoint_results[128], checkpoint_targets[128] = _save_checkpoints(
        root, output, plan, u_pre=128
    )
    for phase in IBR1_EVAL_PHASES[2:]:
        receipt, path, binding = _run_eval_phase(root, output, plan, guard, phase)
        eval_receipts[phase.phase] = receipt
        eval_paths[phase.phase] = path
        predictor_identity[phase.phase] = binding

    eval_guard_document = guard.finalize()
    eval_guard_path = output / EVAL_GUARD_FILENAME
    exclusive_write_json(eval_guard_path, eval_guard_document)

    training_records, eval_records, diagnostics_summary = (
        plan.geometry_collector.finalize()
    )
    gradient_document, optimizer_document = plan.gradient_collector.finalize()
    # Hook cleanup is part of engineering validity and must succeed before a
    # scientific gate, summary, or result seal is allowed to exist.  The
    # wrapper owns the once-only call so early failures are cleaned up too.
    close_plan()
    checkpoint_identity = {
        "verified": True,
        "targets": {
            f"{arm}:update{u_pre}": checkpoint_targets[u_pre][arm]
            for u_pre in (0, 128)
            for arm in (IBR1_CTRL, IBR1_SELF)
        },
        "artifacts": {
            f"{arm}:update{u_pre}": {
                "path": _relative(root, Path(checkpoint_results[u_pre][arm]["path"])),
                "file_sha256": checkpoint_results[u_pre][arm]["file_sha256"],
                "sidecar": _relative(root, Path(checkpoint_results[u_pre][arm]["sidecar"])),
                "sidecar_sha256": checkpoint_results[u_pre][arm]["sidecar_sha256"],
                "tensor_sha256": checkpoint_results[u_pre][arm]["tensor_sha256"],
            }
            for u_pre in (0, 128)
            for arm in (IBR1_CTRL, IBR1_SELF)
        },
    }
    eval_guard_binding = {
        **_binding(root, eval_guard_path, eval_guard_document),
        "verified": True,
    }
    lifecycle_bindings = {
        "checkpoint_identity": checkpoint_identity,
        "eval_order_guard_receipt": eval_guard_binding,
        "final_assembly_receipt": {
            **dict(plan.final_assembly_receipt_binding),
            "verified": True,
        },
        "predictor_identity": {
            "verified": True,
            "phases": predictor_identity,
        },
        "u_pre_identity": {
            "verified": True,
            "phase_to_u_pre": {
                phase.phase: (
                    0 if phase.snapshot.startswith("update0_") else 128
                )
                for phase in IBR1_EVAL_PHASES
            },
        },
    }
    diagnostics_result = write_diagnostics_bundle(
        output / DIAGNOSTICS_DIRNAME,
        training_records=training_records,
        eval_records=eval_records,
        gradient_document=gradient_document,
        optimizer_document=optimizer_document,
        summary_document=diagnostics_summary,
        lifecycle_bindings=lifecycle_bindings,
    )
    diagnostics_manifest_path = (
        output / DIAGNOSTICS_DIRNAME / DIAGNOSTICS_MANIFEST_FILENAME
    )

    cal_numeric_path = cal_output / "main" / NUMERIC_EVIDENCE_FILENAME
    combined, gate_paths, gate_documents = _write_gates(
        root,
        output,
        final_document=final_document,
        final_path=final_path,
        cal_numeric_path=cal_numeric_path,
        checkpoints=checkpoint_results,
        paired_result=paired_result,
        training_records=training_records,
        diagnostics_summary=diagnostics_summary,
        gradient_document=gradient_document,
        eval_receipts=eval_receipts,
    )
    combined_path = output / COMBINED_GATE_FILENAME
    mechanism_pass = bool(combined.get("mechanism_pass"))
    sidecar_paths = {
        f"{arm}:update{u_pre}": checkpoint_results[u_pre][arm]["sidecar"]
        for u_pre in (0, 128)
        for arm in (IBR1_CTRL, IBR1_SELF)
    }
    seal_kwargs = {
        "final_assembly_receipt_path": final_path,
        "candidate_lock_receipt_path": candidate_lock_path,
        "checkpoint_sidecar_paths": sidecar_paths,
        "count_receipt_path": count_path,
        "eval_guard_receipt_path": eval_guard_path,
        "eval_phase_receipt_paths": eval_paths,
        "diagnostics_manifest_path": diagnostics_manifest_path,
        "gate_receipt_paths": gate_paths,
        "combined_gate_receipt_path": combined_path,
    }
    seal_path = output / (
        PASS_SEAL_FILENAME if mechanism_pass else NEGATIVE_SEAL_FILENAME
    )
    if mechanism_pass:
        seal_document = build_ibr1_pass_seal(root, **seal_kwargs)
    else:
        seal_document = build_ibr1_negative_result_seal(root, **seal_kwargs)
    seal_result = {
        "path": _relative(root, seal_path),
        "sha256": hashlib.sha256(
            canonical_json_bytes(seal_document) + b"\n"
        ).hexdigest(),
        "analysis_class": seal_document["analysis_class"],
        "receipt_payload_sha256": seal_document["receipt_payload_sha256"],
        "mechanism_pass": mechanism_pass,
        "formal_training_authorized": False,
    }

    printable_cal_result = {
        key: value
        for key, value in cal_result.items()
        if key != "final_authority_capability"
    }
    summary = _self_hashed(
        {
            "schema_version": 1,
            "analysis_class": "ibr1_authoritative_smoke_summary",
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "run": {
                "valid_input": True,
                "engineering_failure": False,
                "mechanism_pass": mechanism_pass,
                "scientific_negative_result": not mechanism_pass,
                "status": "PASS" if mechanism_pass else "FAIL",
                "decision": "MECHANISM_PASS" if mechanism_pass else "SEAL_STOP",
            },
            "live_final_authority": dict(live_authority),
            "cal_pair": printable_cal_result,
            "smoke_plan": identity_receipt,
            "candidate_lock": _binding(
                root,
                candidate_lock_path,
                _load_canonical_json(candidate_lock_path, "candidate lock"),
            ),
            "checkpoints": checkpoint_identity,
            "count_receipt": _binding(root, count_path, count_document),
            "eval_guard": eval_guard_binding,
            "eval_phase_receipts": {
                phase: _binding(root, eval_paths[phase], eval_receipts[phase])
                for phase in eval_paths
            },
            "diagnostics": diagnostics_result,
            "gate_receipts": {
                gate_id: {
                    **_binding(root, gate_paths[gate_id], gate_documents[gate_id]),
                    "passed": gate_documents[gate_id]["passed"],
                }
                for gate_id in IBR1_GATE_IDS
            },
            "combined_gate_receipt": _binding(
                root, combined_path, _load_canonical_json(combined_path, "combined gate")
            ),
            "result_seal": seal_result,
            "formal_training_authorized": False,
            "same_family_retry_authorized": False,
            "internal_test": "sealed",
            "internal_test_opened": False,
        }
    )
    summary_path = output / SMOKE_SUMMARY_FILENAME
    summary_sha = exclusive_write_json(summary_path, summary)
    result = {
        "path": str(summary_path),
        "sha256": summary_sha,
        "receipt_payload_sha256": summary["receipt_payload_sha256"],
        "analysis_class": summary["analysis_class"],
        "mechanism_pass": mechanism_pass,
        "status": summary["run"]["status"],
        "decision": summary["run"]["decision"],
        "result_seal": seal_result,
        "formal_training_authorized": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }
    # This is deliberately the final authoritative write.  Every cleanup,
    # artifact validation/build, and summary write has already succeeded, and
    # exclusive_write_json removes a partial file if its write/fsync fails.
    exclusive_write_json(seal_path, seal_document)
    return result


def _execute_smoke_plan(
    root: Path,
    output: Path,
    *,
    plan: IBR1SmokePlan,
    candidate_lock_path: Path,
    final_path: Path,
    cal_output: Path,
    cal_result: Mapping[str, Any],
    live_authority: Mapping[str, Any],
) -> dict[str, Any]:
    closed = False

    def close_plan() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        plan.close()

    try:
        return _execute_smoke_plan_body(
            root,
            output,
            plan=plan,
            candidate_lock_path=candidate_lock_path,
            final_path=final_path,
            cal_output=cal_output,
            cal_result=cal_result,
            live_authority=live_authority,
            close_plan=close_plan,
        )
    except BaseException:
        # Early execution failures still clean up hooks, while failures after
        # the explicit close point never trigger a second close attempt.
        close_plan()
        raise


def run_authoritative_smoke(
    project_root: str | Path,
    *,
    bootstrap_receipt_path: str | Path,
    cal_output_dir: str | Path,
    freeze_output_path: str | Path,
    final_output_path: str | Path,
    smoke_output_dir: str | Path,
) -> dict[str, Any]:
    """Run the sole live CAL-to-smoke candidate in one parent process."""

    root = Path(project_root).expanduser().resolve()
    bootstrap = _inside(root, bootstrap_receipt_path, "bootstrap receipt")
    cal_output = _inside(root, cal_output_dir, "CAL output directory")
    freeze_output = _inside(root, freeze_output_path, "lambda freeze output")
    final_output = _inside(root, final_output_path, "final assembly output")
    smoke_output = _inside(root, smoke_output_dir, "smoke output directory")
    _preflight_paths(
        root,
        bootstrap=bootstrap,
        cal_output=cal_output,
        freeze_output=freeze_output,
        final_output=final_output,
        smoke_output=smoke_output,
    )

    # Validate the immutable bootstrap (including the inherited F2 token
    # ledger compatibility anchor) before creating any burn directory or
    # candidate-lock evidence.  The CAL pair performs the same verification
    # immediately before worker spawn to close the remaining TOCTOU window.
    verify_assembly_receipt(
        root,
        bootstrap,
        required_phase=ASSEMBLY_PHASE_BOOTSTRAP,
    )

    stage = "create_smoke_output"
    os.makedirs(smoke_output.parent, exist_ok=True)
    smoke_output.mkdir(exist_ok=False)
    candidate_lock_path = smoke_output / CANDIDATE_LOCK_FILENAME
    try:
        stage = "freeze_candidate_lock"
        freeze_ibr1_candidate_lock_receipt(candidate_lock_path)

        stage = "live_cal_pair_freeze_final"
        cal_result = _OFFICIAL_CAL_RUNNER(
            root,
            bootstrap_receipt_path=bootstrap,
            output_dir=cal_output,
            freeze_output_path=freeze_output,
            final_output_path=final_output,
        )
        capability = cal_result.get("final_authority_capability")
        _require(capability is not None, "live CAL returned no final capability")

        stage = "consume_final_authority_capability"
        live_authority = _OFFICIAL_CAPABILITY_CONSUMER(
            capability,
            project_root=root,
            final_receipt_path=final_output,
        )

        stage = "build_production_smoke_plan"
        plan = _OFFICIAL_PLAN_BUILDER(
            root,
            final_output,
            final_authority_capability=capability,
        )
        _require(isinstance(plan, IBR1SmokePlan), "plan builder returned no IBR1 plan")
        stage = "execute_authoritative_smoke"
        return _execute_smoke_plan(
            root,
            smoke_output,
            plan=plan,
            candidate_lock_path=candidate_lock_path,
            final_path=final_output,
            cal_output=cal_output,
            cal_result=cal_result,
            live_authority=live_authority,
        )
    except BaseException as exc:
        _write_engineering_failure(smoke_output, stage=stage, error=exc)
        raise


__all__ = [
    "IBR1LifecycleError",
    "run_authoritative_smoke",
]
