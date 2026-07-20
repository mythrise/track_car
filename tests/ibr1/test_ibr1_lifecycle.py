from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ibr1_experiment import lifecycle
from ibr1_experiment import cal_pair
from ibr1_experiment.authority import exclusive_write_json
from ibr1_experiment.eval_guard import IBR1_EVAL_PHASES


def _bootstrap(root: Path) -> Path:
    path = root / "bootstrap.json"
    path.write_text("{}\n", encoding="utf-8")
    return path


def _paths(root: Path) -> dict[str, Path]:
    return {
        "bootstrap": _bootstrap(root),
        "cal": root / "cal_pair",
        "freeze": root / "freeze.json",
        "final": root / "final.json",
        "smoke": root / "smoke",
    }


def test_eval_phase_receipt_seals_exact_phase_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    phase = IBR1_EVAL_PHASES[0]
    output = tmp_path / "smoke"
    output.mkdir()
    assembly = object()
    predictor = SimpleNamespace(
        engine_arm="S-SELF",
        family_arm=phase.family_arm,
        snapshot=phase.snapshot,
        arm_assembly=assembly,
    )
    arm = SimpleNamespace(
        assembly=assembly,
        eval_predictor_factory=lambda snapshot: predictor,
    )
    plan = SimpleNamespace(
        arms={"S-SELF": arm},
        eval_rows=(object(),),
        eval_raw_rows=(object(),),
        data=SimpleNamespace(eval_strafe_reset_original_indices=frozenset()),
    )
    guarded = object()

    class FakeGuard:
        @staticmethod
        def wrap_predictor(value: object, **identity: Any) -> object:
            assert value is predictor
            assert identity == {
                "phase": phase.phase,
                "snapshot": phase.snapshot,
                "family_arm": phase.family_arm,
                "mode": phase.mode,
            }
            return guarded

    base_receipt = {
        "schema_version": 1,
        "analysis_class": "f2_eval_fix_snapshot_receipt",
        "architecture_lock": "L1+D2+AP2+F2",
        "support": "EVAL-FIX",
        "rows": 512,
        "mode": phase.mode,
        "static_resets": {"expected": 0, "observed": 0},
        "controller_config": {},
        "eval_mode_contract": {},
        "row_losses": [0.0] * 512,
        "summary": {
            "accumulator": "IEEE-754 binary64 math.fsum",
            "means": {
                name: 0.0 for name in ("overall", "change", "turn", "other")
            },
            "counts": {
                "overall": 512,
                "change": 1,
                "turn": 1,
                "other": 510,
            },
        },
    }

    def fake_run_eval_fix(**kwargs: Any) -> dict[str, Any]:
        assert kwargs == {
            "eval_rows": plan.eval_rows,
            "raw_rows": plan.eval_raw_rows,
            "mode": phase.mode,
            "predictor": guarded,
            "strafe_reset_original_indices": frozenset(),
        }
        return base_receipt

    monkeypatch.setattr(lifecycle, "run_eval_fix", fake_run_eval_fix)
    receipt, path, identity = lifecycle._run_eval_phase(
        tmp_path, output, plan, FakeGuard(), phase
    )

    expected_keys = set(base_receipt) | {
        "phase",
        "snapshot",
        "family_id",
        "family_arm",
        "engine_arm",
        "checkpoint_u_pre",
    }
    assert set(receipt) == expected_keys
    assert len(receipt) == 17
    assert receipt == json.loads(path.read_text(encoding="utf-8"))
    assert identity == {
        **phase.to_dict(),
        "engine_arm": "S-SELF",
        "checkpoint_u_pre": 0,
        "receipt": identity["receipt"],
        "verified": True,
    }
    assert receipt["checkpoint_u_pre"] == 0


def test_live_cal_capability_and_plan_stay_in_one_parent_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    events: list[Any] = []
    capability = object()

    class FakePlan:
        def close(self) -> None:
            events.append("close")

    plan = FakePlan()

    def run_cal(root: Path, **kwargs: Any) -> dict[str, Any]:
        events.append(("cal", root, kwargs))
        return {
            "analysis_class": "fake_live_cal",
            "final_authority_capability": capability,
            "formal_training_authorized": False,
        }

    def consume(observed: object, **kwargs: Any) -> dict[str, Any]:
        events.append(("consume", observed, kwargs))
        assert observed is capability
        return {
            "analysis_class": "fake_consumed_capability",
            "formal_training_authorized": False,
        }

    def build_plan(root: Path, final: Path, **kwargs: Any) -> FakePlan:
        events.append(("plan", root, final, kwargs))
        assert kwargs["final_authority_capability"] is capability
        return plan

    def execute(root: Path, output: Path, **kwargs: Any) -> dict[str, Any]:
        events.append(("execute", root, output, kwargs))
        assert kwargs["plan"] is plan
        assert kwargs["live_authority"]["formal_training_authorized"] is False
        kwargs["plan"].close()
        return {
            "analysis_class": "fake_smoke_summary",
            "formal_training_authorized": False,
        }

    monkeypatch.setattr(lifecycle, "_OFFICIAL_CAL_RUNNER", run_cal)
    monkeypatch.setattr(lifecycle, "_OFFICIAL_CAPABILITY_CONSUMER", consume)
    monkeypatch.setattr(lifecycle, "_OFFICIAL_PLAN_BUILDER", build_plan)
    monkeypatch.setattr(lifecycle, "_execute_smoke_plan", execute)
    monkeypatch.setattr(lifecycle, "IBR1SmokePlan", FakePlan)

    result = lifecycle.run_authoritative_smoke(
        tmp_path,
        bootstrap_receipt_path=paths["bootstrap"],
        cal_output_dir=paths["cal"],
        freeze_output_path=paths["freeze"],
        final_output_path=paths["final"],
        smoke_output_dir=paths["smoke"],
    )

    assert result["formal_training_authorized"] is False
    assert [event if isinstance(event, str) else event[0] for event in events] == [
        "cal",
        "consume",
        "plan",
        "execute",
        "close",
    ]
    assert (paths["smoke"] / lifecycle.CANDIDATE_LOCK_FILENAME).is_file()
    assert not (paths["smoke"] / lifecycle.ENGINEERING_FAILURE_FILENAME).exists()


def test_engineering_failure_burns_directory_without_result_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)

    def fail_cal(*args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        raise RuntimeError("worker failed")

    monkeypatch.setattr(lifecycle, "_OFFICIAL_CAL_RUNNER", fail_cal)
    with pytest.raises(RuntimeError, match="worker failed"):
        lifecycle.run_authoritative_smoke(
            tmp_path,
            bootstrap_receipt_path=paths["bootstrap"],
            cal_output_dir=paths["cal"],
            freeze_output_path=paths["freeze"],
            final_output_path=paths["final"],
            smoke_output_dir=paths["smoke"],
        )

    failure = json.loads(
        (paths["smoke"] / lifecycle.ENGINEERING_FAILURE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert failure["stage"] == "live_cal_pair_freeze_final"
    assert failure["valid_scientific_result"] is False
    assert failure["result_seal_written"] is False
    assert failure["formal_training_authorized"] is False
    assert not (paths["smoke"] / lifecycle.SMOKE_SUMMARY_FILENAME).exists()
    assert not (paths["smoke"] / lifecycle.PASS_SEAL_FILENAME).exists()
    assert not (paths["smoke"] / lifecycle.NEGATIVE_SEAL_FILENAME).exists()


def test_engineering_failure_marker_reports_existing_result_seal(
    tmp_path: Path,
) -> None:
    output = tmp_path / "smoke"
    output.mkdir()
    exclusive_write_json(
        output / lifecycle.PASS_SEAL_FILENAME,
        {"analysis_class": "test_preexisting_result_seal"},
    )

    lifecycle._write_engineering_failure(
        output,
        stage="test_post_seal_failure",
        error=RuntimeError("post-seal failure"),
    )

    failure = json.loads(
        (output / lifecycle.ENGINEERING_FAILURE_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    assert failure["result_seal_written"] is True


def test_existing_output_fails_before_live_cal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths["smoke"].mkdir()
    called = False

    def run_cal(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        del args, kwargs
        called = True
        return {}

    monkeypatch.setattr(lifecycle, "_OFFICIAL_CAL_RUNNER", run_cal)
    with pytest.raises(lifecycle.IBR1LifecycleError, match="already exists"):
        lifecycle.run_authoritative_smoke(
            tmp_path,
            bootstrap_receipt_path=paths["bootstrap"],
            cal_output_dir=paths["cal"],
            freeze_output_path=paths["freeze"],
            final_output_path=paths["final"],
            smoke_output_dir=paths["smoke"],
        )
    assert called is False


def test_capability_registry_ignores_writable_slot_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_path = tmp_path / "final.json"
    final_document = {
        "analysis_class": "ibr1_assembly_source_receipt",
        "phase": "final",
        "receipt_payload_sha256": "a" * 64,
    }
    exclusive_write_json(final_path, final_document)
    final_binding = {
        "path": str(final_path),
        "sha256": hashlib.sha256(final_path.read_bytes()).hexdigest(),
        "receipt_payload_sha256": "a" * 64,
        "analysis_class": "ibr1_assembly_source_receipt",
        "phase": "final",
    }
    attestation = {
        "parent_pid": os.getpid(),
        "parent_challenge": "b" * 64,
        "workers": {
            "main": {"pid": 11},
            "reproduction": {"pid": 12},
        },
    }
    direct = cal_pair.FinalAuthorityCapability(
        cal_pair._FINAL_CAPABILITY_SECRET,
        root=tmp_path.resolve(),
        freeze_path=tmp_path / "freeze.json",
        final_path=final_path.resolve(),
        final_binding=final_binding,
        attestation=attestation,
    )
    with pytest.raises(cal_pair.IBR1CalPairError, match="fresh"):
        cal_pair.consume_final_authority_capability(
            direct,
            project_root=tmp_path,
            final_receipt_path=final_path,
        )
    capability = cal_pair._mint_final_authority_capability(
        cal_pair._FINAL_CAPABILITY_SECRET,
        root=tmp_path.resolve(),
        freeze_path=tmp_path / "freeze.json",
        final_path=final_path.resolve(),
        final_binding=final_binding,
        attestation=attestation,
    )
    monkeypatch.setattr(
        cal_pair.authority,
        "verify_assembly_receipt",
        lambda *args, **kwargs: final_document,
    )

    cal_pair.consume_final_authority_capability(
        capability,
        project_root=tmp_path,
        final_receipt_path=final_path,
    )
    capability._consumed = False
    capability._root = tmp_path / "forged-root"
    with pytest.raises(cal_pair.IBR1CalPairError, match="fresh"):
        cal_pair.consume_final_authority_capability(
            capability,
            project_root=tmp_path,
            final_receipt_path=final_path,
        )

    claim = cal_pair.claim_consumed_final_authority_for_smoke(
        capability,
        project_root=tmp_path,
        final_receipt_path=final_path,
    )
    assert claim["authority_eligible"] is True
    capability._smoke_claimed = False
    with pytest.raises(cal_pair.IBR1CalPairError, match="freshly consumed"):
        cal_pair.claim_consumed_final_authority_for_smoke(
            capability,
            project_root=tmp_path,
            final_receipt_path=final_path,
        )


@pytest.mark.parametrize(
    "failure_mode",
    (None, "summary_write", "close"),
    ids=("success", "summary-write-failure", "close-failure"),
)
@pytest.mark.parametrize(
    "mechanism_pass",
    (True, False),
    ids=("pass-seal", "negative-seal"),
)
def test_execute_plan_seals_only_after_close_and_summary_succeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str | None,
    mechanism_pass: bool,
) -> None:
    output = tmp_path / "smoke"
    output.mkdir()
    candidate_lock = output / lifecycle.CANDIDATE_LOCK_FILENAME
    lifecycle.freeze_ibr1_candidate_lock_receipt(candidate_lock)
    final_path = tmp_path / "final.json"
    final_document = {
        "analysis_class": "ibr1_assembly_source_receipt",
        "phase": "final",
        "formal_training_authorized": False,
    }
    final_path.write_text("{}\n", encoding="utf-8")
    cal_output = tmp_path / "cal"
    (cal_output / "main").mkdir(parents=True)

    phase_order: list[str] = []
    finalization_events: list[str] = []
    write_json = lifecycle.exclusive_write_json

    def ordered_write(path: str | Path, value: Any) -> str:
        filename = Path(path).name
        finalization_events.append(filename)
        if (
            failure_mode == "summary_write"
            and filename == lifecycle.SMOKE_SUMMARY_FILENAME
        ):
            raise OSError("summary write failed")
        return write_json(path, value)

    monkeypatch.setattr(lifecycle, "exclusive_write_json", ordered_write)

    class FakeCollector:
        def __init__(self, value: Any) -> None:
            self.value = value

        def finalize(self) -> Any:
            return self.value

    class FakePlan:
        production_context = True
        authority_eligible = True
        formal_training_authorized = False
        internal_test = "sealed"
        internal_test_opened = False
        final_assembly_receipt = final_document
        final_assembly_receipt_binding = {
            "path": "final.json",
            "sha256": "a" * 64,
            "receipt_payload_sha256": "b" * 64,
            "analysis_class": "ibr1_assembly_source_receipt",
        }
        checkpoint_init_sha256 = "c" * 64
        eval_rows: tuple[Any, ...] = ()
        eval_raw_rows: tuple[Any, ...] = ()
        smoke_rows: tuple[Any, ...] = ()
        g6_update = staticmethod(lambda event: event)
        arms = {"S-CTRL": SimpleNamespace(callbacks=object()), "S-SELF": SimpleNamespace(callbacks=object())}
        data = SimpleNamespace(
            smoke_strafe_reset_original_indices=frozenset(),
            smoke_expected_static_reset_original_indices=frozenset(),
        )
        geometry_collector = FakeCollector(((), (), {"training_geometry": {}}))
        gradient_collector = FakeCollector(({}, {}))

        @staticmethod
        def identity_receipt() -> dict[str, Any]:
            return {
                "analysis_class": "fake_plan",
                "formal_training_authorized": False,
            }

        @staticmethod
        def close() -> None:
            finalization_events.append("plan.close")
            if failure_mode == "close":
                raise RuntimeError("plan close failed")

    fake_plan = FakePlan()

    monkeypatch.setattr(
        lifecycle,
        "verify_assembly_receipt",
        lambda *args, **kwargs: final_document,
    )
    monkeypatch.setattr(
        lifecycle,
        "_save_checkpoints",
        lambda *args, u_pre, **kwargs: (
            {
                arm: {
                    "path": str(output / f"{arm}-{u_pre}.pt"),
                    "file_sha256": "1" * 64,
                    "tensor_sha256": "c" * 64,
                    "sidecar": str(output / f"{arm}-{u_pre}.receipt.json"),
                    "sidecar_sha256": "2" * 64,
                }
                for arm in ("IBR1-CTRL", "IBR1-SELF")
            },
            {
                arm: {"family_arm": arm, "u_pre": u_pre}
                for arm in ("IBR1-CTRL", "IBR1-SELF")
            },
        ),
    )

    class FakeGuard:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        @staticmethod
        def finalize() -> dict[str, Any]:
            return {"analysis_class": "ibr1_eval_fixed_order_guard_receipt"}

    monkeypatch.setattr(lifecycle, "IBR1EvalOrderGuard", FakeGuard)

    def eval_phase(*args: Any, **kwargs: Any):
        del kwargs
        phase = args[-1]
        phase_order.append(phase.phase)
        document = {
            "analysis_class": "f2_eval_fix_snapshot_receipt",
            "summary": {
                "accumulator": "IEEE-754 binary64 math.fsum",
                "means": {name: 0.0 for name in ("overall", "change", "turn", "other")},
                "counts": {name: 1 for name in ("overall", "change", "turn", "other")},
            },
        }
        path = output / f"{phase.phase}.json"
        exclusive_write_json(path, document)
        return document, path, {"phase": phase.phase, "verified": True}

    monkeypatch.setattr(lifecycle, "_run_eval_phase", eval_phase)
    count = SimpleNamespace(
        to_dict=lambda: {
            "analysis_class": "f2_paired_runner_count_receipt",
            "passed": True,
        }
    )
    paired = SimpleNamespace(count_receipt=count, arms={})
    monkeypatch.setattr(lifecycle, "run_paired_smoke", lambda *args, **kwargs: paired)

    def diagnostics(path: Path, **kwargs: Any) -> dict[str, Any]:
        del kwargs
        path.mkdir()
        manifest = path / lifecycle.DIAGNOSTICS_MANIFEST_FILENAME
        exclusive_write_json(manifest, {"analysis_class": "ibr1_diagnostics_manifest"})
        return {"manifest": {"filename": manifest.name}}

    monkeypatch.setattr(lifecycle, "write_diagnostics_bundle", diagnostics)

    def gates(*args: Any, **kwargs: Any):
        del args, kwargs
        paths: dict[str, Path] = {}
        documents: dict[str, dict[str, Any]] = {}
        for gate_id in lifecycle.IBR1_GATE_IDS:
            document = {
                "analysis_class": "ibr1_gate_receipt",
                "passed": mechanism_pass,
            }
            path = output / f"gate_{gate_id}.json"
            exclusive_write_json(path, document)
            paths[gate_id] = path
            documents[gate_id] = document
        combined = {
            "analysis_class": "ibr1_combined_gate_receipt",
            "mechanism_pass": mechanism_pass,
        }
        exclusive_write_json(output / lifecycle.COMBINED_GATE_FILENAME, combined)
        return combined, paths, documents

    monkeypatch.setattr(lifecycle, "_write_gates", gates)
    seal_document = {
        "analysis_class": (
            "ibr1_authoritative_smoke_pass_seal"
            if mechanism_pass
            else "ibr1_authoritative_smoke_negative_result_seal"
        ),
        "receipt_payload_sha256": "d" * 64,
        "formal_training_authorized": False,
    }
    monkeypatch.setattr(
        lifecycle,
        (
            "build_ibr1_pass_seal"
            if mechanism_pass
            else "build_ibr1_negative_result_seal"
        ),
        lambda *args, **kwargs: {
            **seal_document,
            "receipt_payload_sha256": "d" * 64,
        },
    )

    def execute() -> dict[str, Any]:
        return lifecycle._execute_smoke_plan(
            tmp_path,
            output,
            plan=fake_plan,
            candidate_lock_path=candidate_lock,
            final_path=final_path,
            cal_output=cal_output,
            cal_result={"final_authority_capability": object()},
            live_authority={"formal_training_authorized": False},
        )

    if failure_mode == "summary_write":
        with pytest.raises(OSError, match="summary write failed"):
            execute()
    elif failure_mode == "close":
        with pytest.raises(RuntimeError, match="plan close failed"):
            execute()
    else:
        result = execute()

    assert phase_order == [phase.phase for phase in IBR1_EVAL_PHASES]
    assert finalization_events.count("plan.close") == 1
    expected_seal = (
        lifecycle.PASS_SEAL_FILENAME
        if mechanism_pass
        else lifecycle.NEGATIVE_SEAL_FILENAME
    )
    if failure_mode is not None:
        assert not (output / lifecycle.PASS_SEAL_FILENAME).exists()
        assert not (output / lifecycle.NEGATIVE_SEAL_FILENAME).exists()
        assert not (output / lifecycle.SMOKE_SUMMARY_FILENAME).exists()
        return

    assert result["mechanism_pass"] is mechanism_pass
    assert finalization_events[-1] == expected_seal
    assert finalization_events.index("plan.close") < finalization_events.index(
        lifecycle.SMOKE_SUMMARY_FILENAME
    )
    assert (output / expected_seal).is_file()
    summary = json.loads((output / lifecycle.SMOKE_SUMMARY_FILENAME).read_text())
    assert summary["formal_training_authorized"] is False
    assert summary["internal_test"] == "sealed"
    assert summary["result_seal"] == result["result_seal"]
