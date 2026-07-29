import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from f2_experiment.assembly import (
    ADJUDICATION_AMENDMENT1_ID,
    ADJUDICATION_AMENDMENT1_SHA256,
    ADJUDICATION_SHA256,
    ASSEMBLY_RECEIPT_CLASS,
    ASSEMBLY_RECEIPT_VERSION,
    ASSEMBLY_SCHEMA_VERSION,
    CACHE_BINDING_MODE,
    CAL_AUDIT_RECEIPT_CLASS,
    FORENSIC_GATE_CLASS,
    FORENSIC_GATES_CLASS,
    LAMBDA_FREEZE_CLASS,
    LAMBDA_MECHANISM,
    CalRowAudit,
    F2AssemblyContractError,
    G_LEGACY_MAP,
    LIFECYCLE_ORDER,
    SmokeArmAssembly,
    SmokeAssemblyPlan,
    assert_g7_updates_carry_prev_scale,
    build_assembly_receipt,
    build_gate_receipts_from_artifacts,
    build_support_reset_plan,
    freeze_assembly_receipt,
    load_arm_checkpoint_verified,
    run_cal_audit,
    run_eval_fix,
    run_eval_snapshot_command,
    run_production_smoke,
    save_arm_checkpoint,
    verify_assembly_receipt,
    verify_cal_lambda_authority,
    _validate_cal_context_receipt,
    _verify_smoke_plan_cal_context,
)
from f2_experiment.cli import SOURCE_FILES, TRANSITIVE_SOURCE_FILES
from f2_experiment.controller import clamp_stage
from f2_experiment.assembly_data import TokenHashLedger
from f2_experiment.evaluation import G6Update
from f2_experiment.model import AP2_HORIZON, AP2Prediction
from f2_experiment.runner import (
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    AuxForwardResult,
    FeatureForwardResult,
    HeadForwardResult,
    RunnerG7Update,
    RunnerRow,
)
from f2_experiment.reproducibility import CUDA_REPRODUCIBILITY_SETTINGS
from f2_experiment.support import (
    ARCHITECTURE_LOCK,
    FROZEN_TRAIN_SHA256,
    INTERNAL_TEST_POLICY,
    canonical_json_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = '"""Integration placeholder for a pending W1 assembly file."""\n'
FAKE_TOKEN_LEDGER = TokenHashLedger(
    entries={"train/fake.vfine.pt": "0" * 64}
)
FAKE_ASSETS = {
    "base_hf": {"artifact_sha256": "ff" * 32},
    "vision_cache": {"manifest_sha256": "aa" * 32},
    "token_ledger_sha256": FAKE_TOKEN_LEDGER.ledger_sha256,
    "token_ledger_file_count": FAKE_TOKEN_LEDGER.token_files,
}


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _link_test_directory(destination: Path, source: Path) -> None:
    """Create a directory link without requiring Windows symlink privilege."""

    if os.name != "nt":
        destination.symlink_to(source, target_is_directory=True)
        return
    result = subprocess.run(
        [
            "cmd.exe",
            "/d",
            "/c",
            "mklink",
            "/J",
            str(destination),
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not destination.is_dir():
        raise RuntimeError(
            "failed to create Windows test junction "
            f"{destination} -> {source}: {result.stdout} {result.stderr}"
        )


def _bound_root(tmp_path):
    """Hybrid root: real data/approvals/third_party plus the live f2 sources.

    Assembly files that other implementation waves have not landed yet are
    represented by deterministic placeholders so that the receipt contract
    (12 source bindings) is exercised end to end today and keeps passing
    unchanged once the real files exist.
    """

    root = tmp_path / "root"
    root.mkdir()
    for name in ("data", "experiments", "third_party", "tests"):
        _link_test_directory(root / name, PROJECT_ROOT / name)
    (root / "f2_experiment").mkdir()
    for relative in SOURCE_FILES:
        source = PROJECT_ROOT / relative
        destination = root / relative
        if source.is_file():
            destination.write_bytes(source.read_bytes())
        else:
            destination.write_text(PLACEHOLDER, encoding="utf-8")
    return root


def _grouped_rows(total, groups, *, start_index=1000):
    rows = []
    current_group = -1
    frame_idx = 0
    for position in range(total):
        group = position * groups // total
        if group != current_group:
            current_group = group
            frame_idx = 0
        else:
            frame_idx += 1
        rows.append(
            RunnerRow(
                original_row_index=start_index + position,
                sequence_id=f"sequence-{group}",
                frame_idx=frame_idx,
                mirrored=False,
                logged_prev_action=(0.0, 0.0, 0.0),
                target_actions=torch.zeros(AP2_HORIZON, 3),
                observation={"token": position},
                aux_targets={"aux": position},
            )
        )
    return tuple(rows)


def _raw_rows(total):
    raw = []
    for position in range(total):
        if position % 3 == 0:
            step0 = [0.5, 0.0, 0.0]
            transition = "other"
        elif position % 3 == 1:
            step0 = [0.0, 0.0, 0.0]
            transition = "turn_onset"
        else:
            step0 = [0.0, 0.0, 0.0]
            transition = "cruise"
        raw.append(
            {
                "prev_action": [0.0, 0.0, 0.0],
                "step_actions": [step0] + [[0.0, 0.0, 0.0]] * 7,
                "transition_type": transition,
            }
        )
    return tuple(raw)


def _bounded_future(raw_actions):
    bounded = []
    for step in range(1, AP2_HORIZON):
        bounded.append(
            clamp_stage(
                (
                    float(raw_actions[0, step, 0].item()),
                    float(raw_actions[0, step, 2].item()),
                )
            )
        )
    return raw_actions.new_tensor([bounded])


def _prediction(prev_fy, *, delta=0.001, strafe=False):
    delta_fy = prev_fy.new_zeros((1, AP2_HORIZON, 2))
    delta_fy[:, 0, 0] = delta
    raw_fy = prev_fy.unsqueeze(-2) + torch.cumsum(delta_fy, dim=-2)
    raw = torch.stack(
        (
            raw_fy[..., 0],
            torch.zeros_like(raw_fy[..., 0]),
            raw_fy[..., 1],
        ),
        dim=-1,
    )
    if strafe:
        raw = raw.clone()
        raw[0, 0, 1] = 0.1
    return AP2Prediction(
        delta_fy=delta_fy,
        raw_actions=raw,
        bounded_future_actions=_bounded_future(raw),
    )


def _g7_telemetry():
    return {
        "per_method_over_base": {"polar": torch.tensor([0.1])},
        "total_method_over_base": torch.tensor([0.1]),
        "abs_tanh_method_scales": {"polar": torch.tensor(0.2)},
        "abs_tanh_s_prev": torch.tensor(0.1),
        "r_prev": torch.tensor([0.3]),
    }


def _g6_update(event):
    in_window = event.u_pre >= 8
    return G6Update(
        u_pre=event.u_pre,
        aux_reachable=True,
        track_reachable=in_window,
        cosine_total_track=0.7 if in_window else None,
        signed_projection=1.0 if in_window else None,
        aux_track_ratio=1.0 if in_window else None,
    )


FAKE_SMOKE_ASSETS = {
    "train": {"sha256": "ab" * 32},
    "base_hf": {"path": "/frozen/base", "artifact_sha256": "cd" * 32},
    "token_ledger_sha256": FAKE_TOKEN_LEDGER.ledger_sha256,
    "token_ledger_file_count": FAKE_TOKEN_LEDGER.token_files,
}
# Deliberately NOT the real frozen values: the authority chain must be
# validated by receipt-chain consistency, never by hardcoded lambdas.
FAKE_LAMBDA = {"L_cot": 0.111, "L_future": 0.0456, "L_verify": 0.5}


def _write_static_assembly_receipt_stub(path, *, marker):
    document = {
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "schema_version": ASSEMBLY_SCHEMA_VERSION,
        "receipt_version": ASSEMBLY_RECEIPT_VERSION,
        "architecture_lock": ARCHITECTURE_LOCK,
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
        "stub_marker": marker,
    }
    document["receipt_payload_sha256"] = canonical_json_sha256(document)
    path.write_text(
        json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
    )
    return document


def _stub_receipt(tmp_path, name="assembly_receipt_stub.json"):
    path = tmp_path / name
    static_document = _write_static_assembly_receipt_stub(path, marker=name)
    document = {
        "receipt_payload_sha256": static_document["receipt_payload_sha256"],
        "probe_surface": "base.proj",
        "block_mode": "bstar",
        "smoke_package": "SA-Hstar",
        "asset_binding": dict(FAKE_SMOKE_ASSETS),
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }

    def verifier(root, receipt_path):
        return document

    verifier.document = document
    return path, verifier


def _authority_chain(
    tmp_path,
    receipt_path,
    document,
    *,
    lambda_values=None,
    direct=True,
    adopt=True,
    bind_freeze=True,
    amendment_sha=ADJUDICATION_AMENDMENT1_SHA256,
):
    """Write a fake CAL receipt + lambda-freeze receipt and wire the chain."""

    values = dict(FAKE_LAMBDA) if lambda_values is None else dict(lambda_values)
    if direct:
        bootstrap_path = receipt_path
    else:
        bootstrap_path = tmp_path / "bootstrap_assembly_receipt_stub.json"
        _write_static_assembly_receipt_stub(
            bootstrap_path, marker="pre-freeze-bootstrap"
        )
    bootstrap_document = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    bootstrap_file_sha = hashlib.sha256(bootstrap_path.read_bytes()).hexdigest()
    cal_path = tmp_path / "cal_audit_receipt_stub.json"
    cal_document = {
        "analysis_class": CAL_AUDIT_RECEIPT_CLASS,
        "rows": 512,
        "optimizer_updates": 0,
        "support": "CAL",
        "package": "SA-Hstar",
        "cal_context": {
            "seed": 0,
            "device": "cpu",
            "package": "SA-Hstar",
            "probe_surface": "base.proj",
            "initialization": "seeded test initialization",
            "checkpoint_init_sha256": "c" * 64,
        },
        "token_ledger_binding": {
            "anchor": "trust_on_first_read_at_freeze",
            "sha256": FAKE_TOKEN_LEDGER.ledger_sha256,
            "file_count": FAKE_TOKEN_LEDGER.token_files,
        },
        "assembly_receipt_sha256": bootstrap_file_sha,
        "assembly_receipt_payload_sha256": bootstrap_document[
            "receipt_payload_sha256"
        ],
        "amendment_binding": {"sha256": amendment_sha},
        "gradient_calibration": {
            "aux_grad_norm_median_min": 0.5,
            "per_aux_grad_norm_median": {
                "L_cot": 2.25,
                "L_future": 0.75,
                "L_verify": 0.5,
            },
            "probe_surface": "base.proj",
        },
        "lambda_calibration": {
            "mechanism": LAMBDA_MECHANISM,
            "proposed_lambda": values,
        },
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    cal_path.write_text(
        json.dumps(cal_document, sort_keys=True), encoding="utf-8"
    )
    cal_file_sha = hashlib.sha256(cal_path.read_bytes()).hexdigest()
    reproduction_path = tmp_path / "cal_audit_receipt_stub_repro.json"
    reproduction_path.write_bytes(cal_path.read_bytes())
    freeze_path = tmp_path / "lambda_freeze_receipt_stub.json"
    freeze_document = {
        "schema_version": 1,
        "analysis_class": LAMBDA_FREEZE_CLASS,
        "mechanism": (
            LAMBDA_MECHANISM
            + ", computed on base.proj over 512 CAL rows at zero updates"
        ),
        "frozen_values": values,
        "evidence": {
            "cal_audit_receipt": {
                "path": str(cal_path),
                "sha256": cal_file_sha if adopt else "2" * 64,
                "seed": cal_document["cal_context"]["seed"],
                "device": cal_document["cal_context"]["device"],
                "checkpoint_init_sha256": cal_document["cal_context"][
                    "checkpoint_init_sha256"
                ],
            },
            "deterministic_reproduction": {
                "path": str(reproduction_path),
                "sha256": cal_file_sha,
                "comparison": "byte-identical to main CAL receipt",
            },
            "bootstrap_assembly_receipt": {
                "path": str(bootstrap_path),
                "sha256": bootstrap_file_sha,
                "receipt_payload_sha256": bootstrap_document[
                    "receipt_payload_sha256"
                ],
            },
        },
        "internal_test": INTERNAL_TEST_POLICY,
        "internal_test_opened": False,
    }
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    if bind_freeze:
        document["lambda_freeze_binding"] = {
            "path": str(freeze_path),
            "sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
            "analysis_class": LAMBDA_FREEZE_CLASS,
        }
    return cal_path


def _smoke_authority_kwargs(tmp_path, **chain_kwargs):
    """Common fail-closed wiring for run_production_smoke tests."""

    receipt_path, verifier = _stub_receipt(tmp_path)
    cal_path = _authority_chain(
        tmp_path, receipt_path, verifier.document, **chain_kwargs
    )
    return {
        "receipt_path": receipt_path,
        "verifier": verifier,
        "cal_audit_receipt_path": cal_path,
        "asset_verifier": lambda root: dict(FAKE_SMOKE_ASSETS),
        "aux_coefficients": dict(FAKE_LAMBDA),
    }


class LifecycleArm:
    """Runner-compatible fake arm with EVAL and checkpoint accessors."""

    def __init__(self, arm, eval_losses):
        self.arm = arm
        self.updates_done = 0
        self.current_mode = None
        self.checkpoint_state = {
            "head.bias": torch.tensor([1.0, 2.0]),
            "head.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
        self.eval_losses = eval_losses

    def callbacks(self):
        return ArmCallbacks(
            checkpoint_state=self.checkpoint_state,
            feature_forward=self.feature_forward,
            aux_forward=self.aux_forward,
            head_forward=self.head_forward,
            track_loss=self.track_loss,
            backward=self.backward,
            optimizer_step=self.optimizer_step,
            audit_counters=lambda: {
                "expert_future_leak_count": 0,
                "self_state_expert_overwrite_count": 0,
            },
        )

    def feature_forward(self, observation, event):
        return FeatureForwardResult(
            value={"row": event.row_position},
            reference_tensor=torch.ones(1, 4),
        )

    def aux_forward(self, features, aux_targets, event):
        return AuxForwardResult(loss=torch.tensor(2.0))

    def head_forward(self, features, prev_fy, event):
        return HeadForwardResult(
            prediction=_prediction(prev_fy),
            g7_telemetry=_g7_telemetry(),
        )

    def track_loss(self, prediction, target, event):
        return torch.tensor(1.0 if event.branch == "branch1" else 3.0)

    def backward(self, event):
        return None

    def optimizer_step(self, event):
        self.updates_done += 1

    def eval_predictor(self, row, prev_tensor, *, mode, reset, position):
        self.current_mode = mode
        return _prediction(prev_tensor)

    def eval_loss_fn(self, prediction, target):
        phase = "update0" if self.updates_done == 0 else "update128"
        return self.eval_losses[(phase, self.current_mode)]

    def checkpoint_payload(self):
        return {
            "model": self.checkpoint_state,
            "optimizer": {"state": {}, "param_groups": []},
        }

    def assembly(self):
        return SmokeArmAssembly(
            callbacks=self.callbacks(),
            eval_predictor=self.eval_predictor,
            checkpoint_payload=self.checkpoint_payload,
            eval_loss_fn=self.eval_loss_fn,
        )


def _smoke_plan(arms):
    smoke_rows = _grouped_rows(256, 12, start_index=2000)
    eval_rows = _grouped_rows(512, 28, start_index=10_000)
    reset_plan = build_support_reset_plan(smoke_rows, frozenset())
    expected = frozenset(
        row.original_row_index
        for row, reasons in zip(smoke_rows, reset_plan)
        if reasons
    )
    return SmokeAssemblyPlan(
        smoke_rows=smoke_rows,
        eval_rows=eval_rows,
        eval_raw_rows=_raw_rows(512),
        strafe_reset_original_indices=frozenset(),
        expected_static_reset_original_indices=expected,
        arms={S_CTRL: arms[S_CTRL].assembly(), S_SELF: arms[S_SELF].assembly()},
        g6_update=_g6_update,
        g6_fallback_evidence=lambda: {
            "deciding_block_mode": "bstar",
            "block_mode": "bstar",
            "fallback_per_aux_ratio_max": 0.75,
            "per_aux_ratio_series": tuple(
                {
                    "u_pre": u_pre,
                    "ratios": {
                        "L_cot": 0.5,
                        "L_future": 0.5,
                        "L_verify": 0.5,
                    },
                }
                for u_pre in range(128)
            ),
        },
        seed=0,
        device="cpu",
        checkpoint_init_sha256="c" * 64,
    )


PASS_SELF_LOSSES = {
    ("update0", "logged"): 1.0,
    ("update0", "self"): 1.5,
    ("update128", "logged"): 0.9,
    ("update128", "self"): 1.0,
}
PASS_CTRL_LOSSES = {
    ("update0", "logged"): 1.0,
    ("update0", "self"): 1.5,
    ("update128", "logged"): 0.9,
    ("update128", "self"): 1.1,
}
FAIL_SELF_LOSSES = {
    ("update0", "logged"): 1.0,
    ("update0", "self"): 1.5,
    ("update128", "logged"): 0.9,
    ("update128", "self"): 1.6,
}


def test_smoke_plan_reproducibility_must_match_cal_context():
    arms = {
        S_CTRL: LifecycleArm(S_CTRL, PASS_CTRL_LOSSES),
        S_SELF: LifecycleArm(S_SELF, PASS_SELF_LOSSES),
    }
    plan = _smoke_plan(arms)
    cal_context = {
        "seed": 0,
        "device": "cpu",
        "checkpoint_init_sha256": "c" * 64,
    }
    _verify_smoke_plan_cal_context(plan, cal_context)

    mismatches = (
        ("seed", 1, "seed differs"),
        ("device", "cuda:0", "device differs"),
        ("checkpoint_init_sha256", "d" * 64, "init SHA differs"),
        (
            "cuda_reproducibility",
            {"unexpected": True},
            "must not claim CUDA",
        ),
    )
    for field, value, match in mismatches:
        mutated = SmokeAssemblyPlan(**{**plan.__dict__, field: value})
        with pytest.raises(F2AssemblyContractError, match=match):
            _verify_smoke_plan_cal_context(mutated, cal_context)


# ---------------------------------------------------------------------------
# Assembly source receipt v4
# ---------------------------------------------------------------------------


LAMBDA_FREEZE_RELATIVE = (
    "experiments/collected_v1_main/external_reviews/"
    "20260719_f2_seeded_cal_lambda_freeze_receipt.json"
)


def test_build_freeze_verify_assembly_receipt_roundtrip(tmp_path):
    root = _bound_root(tmp_path)
    output = tmp_path / "assembly_receipt_v1.json"
    result = freeze_assembly_receipt(
        root,
        output,
        asset_binding=FAKE_ASSETS,
        lambda_freeze_receipt_path=LAMBDA_FREEZE_RELATIVE,
    )
    assert len(result["sha256"]) == 64
    document = verify_assembly_receipt(root, output)
    assert len(document["tests_sha256"]) == 11
    assert all(
        not Path(relative).name.startswith("._")
        for relative in document["tests_sha256"]
    )
    # P1-1: the PRIMARY lambda-freeze receipt joins the approvals chain.
    freeze_sha = hashlib.sha256(
        (PROJECT_ROOT / LAMBDA_FREEZE_RELATIVE).read_bytes()
    ).hexdigest()
    assert document["lambda_freeze_binding"] == {
        "path": LAMBDA_FREEZE_RELATIVE,
        "sha256": freeze_sha,
        "analysis_class": LAMBDA_FREEZE_CLASS,
    }
    assert document["approval_sha256"]["fable_f2_lambda_freeze"] == freeze_sha
    assert len(document["approval_sha256"]) == 7
    assert set(document["source_sha256"]) == {str(path) for path in SOURCE_FILES}
    assert len(document["source_sha256"]) == 12
    assert set(document["transitive_source_sha256"]) == {
        str(path) for path in TRANSITIVE_SOURCE_FILES
    }
    assert len(document["transitive_source_sha256"]) == 15
    assert document["namespace_packages"] == [
        "third_party",
        "third_party.OpenTrackVLA",
    ]
    assert "tests/f2/test_assembly_lifecycle.py" in document["tests_sha256"]
    assert document["adjudication_binding"]["sha256"] == ADJUDICATION_SHA256
    amendment = document["adjudication_amendment_binding"]
    assert amendment["sha256"] == ADJUDICATION_AMENDMENT1_SHA256
    assert amendment["amendment_id"] == ADJUDICATION_AMENDMENT1_ID
    assert amendment["amends"] == "rulings.b_lambda_policy"
    assert document["lambda_policy"]["mechanism"] == LAMBDA_MECHANISM
    assert document["data_binding"]["train"]["sha256"] == FROZEN_TRAIN_SHA256
    assert document["asset_binding"] == FAKE_ASSETS
    assert document["asset_binding"]["token_ledger_sha256"] == (
        FAKE_TOKEN_LEDGER.ledger_sha256
    )
    assert document["asset_binding"]["token_ledger_file_count"] == 1
    assert document["optimizer_contract"]["optimizer"] == "AdamW"
    assert document["optimizer_contract"]["base_lr"] == 2e-5
    assert document["optimizer_contract"]["head_lr"] == 3e-4
    assert document["optimizer_contract"]["grad_clip_norm"] == 1.0
    assert document["optimizer_contract"]["betas"] == [0.9, 0.999]
    assert document["probe_surface"] == "base.proj"
    assert document["block_mode"] == "bstar"
    assert document["smoke_package"] == "SA-Hstar"
    assert document["gate_contract_changes"] == [
        "G7Update.abs_tanh_s_prev",
        "evaluate_g7.prev_scale_saturation_rate",
    ]
    # PRIMARY incremental adjudication: manifest-binding cache mode only.
    assert document["cache_binding_mode"] == CACHE_BINDING_MODE
    assert "internal-test seal" in document["cache_binding_reason"]
    assert document["internal_test_opened"] is False

    missing_ledger = dict(FAKE_ASSETS)
    missing_ledger.pop("token_ledger_sha256")
    with pytest.raises(F2AssemblyContractError, match="token_ledger_sha256"):
        build_assembly_receipt(root, asset_binding=missing_ledger)

    with pytest.raises(F2AssemblyContractError, match="refusing to overwrite"):
        freeze_assembly_receipt(root, output, asset_binding=FAKE_ASSETS)

    verify_assembly_receipt(root, output, asset_verifier=lambda _: FAKE_ASSETS)
    with pytest.raises(F2AssemblyContractError, match="asset binding"):
        verify_assembly_receipt(
            root, output, asset_verifier=lambda _: {"tampered": True}
        )


def test_receipt_rejects_token_payload_reread_under_internal_test_seal(tmp_path):
    root = _bound_root(tmp_path)
    sealed_violation = {
        "vision_cache": {
            "cache_manifest_sha256": "aa" * 32,
            "token_payload_verified": True,
        }
    }
    with pytest.raises(F2AssemblyContractError, match="internal-test seal"):
        build_assembly_receipt(root, asset_binding=sealed_violation)


def test_verify_assembly_receipt_fails_closed_on_tamper(tmp_path):
    root = _bound_root(tmp_path)
    output = tmp_path / "assembly_receipt_v1.json"
    freeze_assembly_receipt(root, output, asset_binding=FAKE_ASSETS)

    target = root / "f2_experiment" / "support.py"
    original = target.read_bytes()
    target.write_bytes(original + b"\n# tampered\n")
    with pytest.raises(F2AssemblyContractError, match="source bindings"):
        verify_assembly_receipt(root, output)
    target.write_bytes(original)
    verify_assembly_receipt(root, output)

    document = json.loads(output.read_text(encoding="utf-8"))
    document["smoke_package"] = "SA-B0"
    output.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(F2AssemblyContractError, match="payload SHA"):
        verify_assembly_receipt(root, output)


# ---------------------------------------------------------------------------
# CAL zero-update audit
# ---------------------------------------------------------------------------


def _cal_loader(rows):
    def loader(root, support_name, token_ledger):
        assert support_name == "CAL"
        assert token_ledger is FAKE_TOKEN_LEDGER
        return rows, None, frozenset()

    loader.token_ledger = FAKE_TOKEN_LEDGER
    return loader


def _cal_auditor(track=0.0, aux=(0.4, 4.05, 0.1), *, parity_fail_at=None, leak_at=None):
    def auditor(row, reasons, position):
        return CalRowAudit(
            step0_parity=position != parity_fail_at,
            prev_free=position != leak_at,
            aux_grad_norms={
                "L_cot": aux[0],
                "L_future": aux[1],
                "L_verify": aux[2],
            },
            track_grad_norm=track,
        )

    auditor.context_receipt = lambda: {
        "seed": 0,
        "device": "cpu",
        "package": "SA-Hstar",
        "probe_surface": "base.proj",
        "initialization": "seeded test initialization",
        "checkpoint_init_sha256": "c" * 64,
    }
    return auditor


def test_run_cal_audit_freezes_lambda_proposal_and_legacy_map(tmp_path):
    receipt_path, verifier = _stub_receipt(tmp_path)
    output_dir = tmp_path / "cal_audit_v1"
    result = run_cal_audit(
        tmp_path,
        receipt_path=receipt_path,
        output_dir=output_dir,
        rows_loader=_cal_loader(_grouped_rows(512, 30)),
        row_auditor=_cal_auditor(),
        verifier=verifier,
    )
    receipt = result["receipt"]
    assert receipt["optimizer_updates"] == 0
    assert receipt["cal_context"] == {
        "seed": 0,
        "device": "cpu",
        "package": "SA-Hstar",
        "probe_surface": "base.proj",
        "initialization": "seeded test initialization",
        "checkpoint_init_sha256": "c" * 64,
    }
    assert receipt["token_ledger_binding"] == {
        "anchor": "trust_on_first_read_at_freeze",
        "sha256": FAKE_TOKEN_LEDGER.ledger_sha256,
        "file_count": 1,
    }
    assert "no optimizer object" in receipt["zero_update_proof"]
    assert receipt["static_reset_receipt"]["observed"] == 30
    assert receipt["step0_parity"]["checked_rows"] == 512
    assert receipt["prev_free_graph_audit"]["failures"] == 0
    # Amendment 1: AP2 zero-init positive proof plus receipt binding.
    assert receipt["ap2_zero_init_proof"]["checked_rows"] == 512
    assert receipt["ap2_zero_init_proof"]["violations"] == 0
    assert receipt["ap2_zero_init_proof"]["track_grad_norm_max"] == 0.0
    assert "smoke G6" in receipt["ap2_zero_init_proof"][
        "track_reachability_enforcement"
    ] or "G6 gate" in receipt["ap2_zero_init_proof"][
        "track_reachability_enforcement"
    ]
    amendment = receipt["amendment_binding"]
    assert amendment["sha256"] == ADJUDICATION_AMENDMENT1_SHA256
    assert amendment["amendment_id"] == ADJUDICATION_AMENDMENT1_ID
    assert amendment["amends"] == "rulings.b_lambda_policy"
    # Amended mechanism (f2-adjudication-amendment-1):
    # lambda_i = 0.5 * min_j(median||g_aux_j||) / median||g_aux_i||,
    # each capped at 1.0, 3 significant digits; min median here is 0.1.
    assert receipt["lambda_calibration"]["mechanism"] == LAMBDA_MECHANISM
    assert "min_j" in receipt["lambda_calibration"]["mechanism"]
    assert receipt["lambda_calibration"]["proposed_lambda"] == {
        "L_cot": 0.125,     # 0.5*0.1/0.4  = 0.125
        "L_future": 0.0123,  # 0.5*0.1/4.05 = 0.012345679 -> 3 sig digits
        "L_verify": 0.5,    # weakest auxiliary receives exactly 0.5
    }
    assert receipt["lambda_calibration"]["amendment_id"] == (
        ADJUDICATION_AMENDMENT1_ID
    )
    assert receipt["lambda_calibration"]["status"] == "proposal"
    assert receipt["gradient_calibration"]["aux_grad_norm_median_min"] == 0.1
    assert "track_grad_norm_median" not in receipt["gradient_calibration"]
    assert receipt["g_legacy_map"] == G_LEGACY_MAP
    assert "smoke G6" in receipt["g_legacy_map"]["G3_subordination_ratio"]
    assert "f2-adjudication-amendment-1" in receipt["g_legacy_map"][
        "G3_subordination_ratio"
    ]
    written = json.loads(
        (output_dir / "cal_audit_receipt_v1.json").read_text(encoding="utf-8")
    )
    assert written == receipt

    with pytest.raises(F2AssemblyContractError, match="not empty"):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=output_dir,
            rows_loader=_cal_loader(_grouped_rows(512, 30)),
            row_auditor=_cal_auditor(),
            verifier=verifier,
        )


def test_cuda_cal_context_requires_complete_reproducibility_receipt():
    context = {
        "seed": 0,
        "device": "cuda:0",
        "package": "SA-Hstar",
        "probe_surface": "base.proj",
        "initialization": "seeded test initialization",
        "checkpoint_init_sha256": "c" * 64,
    }
    with pytest.raises(F2AssemblyContractError, match="must be a mapping"):
        _validate_cal_context_receipt(
            context,
            package="SA-Hstar",
            probe_surface="base.proj",
        )

    cuda_reproducibility = {
        **CUDA_REPRODUCIBILITY_SETTINGS,
        "torch_version": "2.6.0+cu124",
        "cuda_runtime": "12.4",
    }
    normalized = _validate_cal_context_receipt(
        {**context, "cuda_reproducibility": cuda_reproducibility},
        package="SA-Hstar",
        probe_surface="base.proj",
    )
    assert normalized["cuda_reproducibility"] == dict(
        sorted(cuda_reproducibility.items())
    )


def test_run_cal_audit_fails_closed_on_parity_leak_track_and_aux(tmp_path):
    receipt_path, verifier = _stub_receipt(tmp_path)

    with pytest.raises(F2AssemblyContractError, match="HS6_STEP0_PARITY"):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_parity",
            rows_loader=_cal_loader(_grouped_rows(512, 30)),
            row_auditor=_cal_auditor(parity_fail_at=7),
            verifier=verifier,
        )
    assert not (tmp_path / "cal_parity" / "cal_audit_receipt_v1.json").exists()

    with pytest.raises(F2AssemblyContractError, match="PREV_GRAPH_LEAK"):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_leak",
            rows_loader=_cal_loader(_grouped_rows(512, 30)),
            row_auditor=_cal_auditor(leak_at=100),
            verifier=verifier,
        )

    # Amendment 1: any nonzero track gradient on base.proj at zero updates
    # disproves the frozen AP2 zero-init contract and must fail closed.
    with pytest.raises(
        F2AssemblyContractError, match="AP2_ZERO_INIT_VIOLATION"
    ):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_track_nonzero",
            rows_loader=_cal_loader(_grouped_rows(512, 30)),
            row_auditor=_cal_auditor(track=0.05),
            verifier=verifier,
        )
    assert not (
        tmp_path / "cal_track_nonzero" / "cal_audit_receipt_v1.json"
    ).exists()

    # Zero-median auxiliaries still escalate (never defaulted).
    with pytest.raises(
        F2AssemblyContractError, match="L_cot gradient median is zero"
    ):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_aux_zero",
            rows_loader=_cal_loader(_grouped_rows(512, 30)),
            row_auditor=_cal_auditor(aux=(0.0, 4.05, 0.1)),
            verifier=verifier,
        )


def test_run_cal_audit_rejects_wrong_row_or_reset_counts(tmp_path):
    receipt_path, verifier = _stub_receipt(tmp_path)
    with pytest.raises(F2AssemblyContractError, match="exactly 512 rows"):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_short",
            rows_loader=_cal_loader(_grouped_rows(511, 30)),
            row_auditor=_cal_auditor(),
            verifier=verifier,
        )
    with pytest.raises(F2AssemblyContractError, match="static resets"):
        run_cal_audit(
            tmp_path,
            receipt_path=receipt_path,
            output_dir=tmp_path / "cal_resets",
            rows_loader=_cal_loader(_grouped_rows(512, 29)),
            row_auditor=_cal_auditor(),
            verifier=verifier,
        )


# ---------------------------------------------------------------------------
# EVAL-FIX executor
# ---------------------------------------------------------------------------


def test_default_eval_loss_aligns_target_with_prediction(monkeypatch):
    import f2_experiment.assembly as assembly_module

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    raw = torch.zeros((1, AP2_HORIZON, 3), device=device, dtype=torch.float32)
    prediction = AP2Prediction(
        delta_fy=torch.zeros(
            (1, AP2_HORIZON, 2), device=device, dtype=torch.float32
        ),
        raw_actions=raw,
        bounded_future_actions=raw.clone(),
    )
    observed = {}

    def fake_track_loss(given_prediction, target):
        observed["prediction"] = given_prediction
        observed["target_device"] = target.device
        observed["target_dtype"] = target.dtype
        return type("Loss", (), {"total": torch.tensor(1.25)})()

    monkeypatch.setattr(assembly_module, "ap2_track_loss", fake_track_loss)
    loss = assembly_module._default_eval_loss(
        prediction, torch.zeros((1, AP2_HORIZON, 3), dtype=torch.float64)
    )
    assert loss == 1.25
    assert observed["prediction"] is prediction
    assert observed["target_device"] == prediction.raw_actions.device
    assert observed["target_dtype"] == prediction.raw_actions.dtype


class RecordingPredictor:
    def __init__(self, *, delta=0.0, strafe_at=None):
        self.delta = delta
        self.strafe_at = strafe_at
        self.calls = []

    def __call__(self, row, prev_tensor, *, mode, reset, position):
        self.calls.append(
            (
                mode,
                position,
                reset,
                (float(prev_tensor[0, 0]), float(prev_tensor[0, 1])),
            )
        )
        return _prediction(
            prev_tensor,
            delta=self.delta,
            strafe=position == self.strafe_at,
        )


def test_run_eval_fix_logged_and_self_recurrence_and_strata(tmp_path):
    eval_rows = _grouped_rows(512, 28)
    raw_rows = _raw_rows(512)

    logged = RecordingPredictor(delta=0.4)
    logged_result = run_eval_fix(
        eval_rows=eval_rows,
        raw_rows=raw_rows,
        mode="logged",
        predictor=logged,
        strafe_reset_original_indices=frozenset(),
    )
    assert all(call[3] == (0.0, 0.0) for call in logged.calls)
    assert len(logged_result["row_losses"]) == 512
    assert logged_result["static_resets"] == {"expected": 28, "observed": 28}

    rolled = RecordingPredictor(delta=0.4)
    self_result = run_eval_fix(
        eval_rows=eval_rows,
        raw_rows=raw_rows,
        mode="self",
        predictor=rolled,
        strafe_reset_original_indices=frozenset(),
    )
    # Reset rows re-seed from the logged prev; the next row sees the
    # controller-filtered value (clamp -> rate limit -> EMA => prev+0.2).
    assert rolled.calls[0][2] is True
    assert rolled.calls[0][3] == (0.0, 0.0)
    assert rolled.calls[1][3][0] == pytest.approx(0.2)
    assert rolled.calls[2][3][0] == pytest.approx(0.4)
    # Reset rows are identical across modes (same prev, same predictor).
    assert self_result["row_losses"][0] == pytest.approx(
        logged_result["row_losses"][0]
    )
    # Non-reset rows diverge in self mode.
    assert self_result["row_losses"][1] != pytest.approx(
        logged_result["row_losses"][1]
    )
    summary = self_result["summary"]
    assert summary["counts"] == {
        "overall": 512,
        "change": 171,
        "turn": 171,
        "other": 171,
    }
    assert all(value >= 0.0 for value in summary["means"].values())


def test_run_eval_fix_rejects_bad_budget_and_strafe(tmp_path):
    raw_rows = _raw_rows(512)
    with pytest.raises(F2AssemblyContractError, match="exactly 512 rows"):
        run_eval_fix(
            eval_rows=_grouped_rows(511, 28),
            raw_rows=raw_rows,
            mode="logged",
            predictor=RecordingPredictor(),
            strafe_reset_original_indices=frozenset(),
        )
    with pytest.raises(F2AssemblyContractError, match="nonzero strafe"):
        run_eval_fix(
            eval_rows=_grouped_rows(512, 28),
            raw_rows=raw_rows,
            mode="logged",
            predictor=RecordingPredictor(strafe_at=3),
            strafe_reset_original_indices=frozenset(),
        )


# ---------------------------------------------------------------------------
# Checkpoint save/load fail-closed
# ---------------------------------------------------------------------------


def test_checkpoint_save_load_roundtrip_exclusive_and_tamper(tmp_path):
    state = {"w": torch.arange(4, dtype=torch.float32)}
    path = tmp_path / "checkpoint_update0_S-CTRL.pt"
    info = save_arm_checkpoint(
        path,
        arm=S_CTRL,
        model_state=state,
        optimizer_state={"state": {}, "param_groups": []},
        u_pre=0,
        assembly_receipt_sha256="a" * 64,
    )
    assert Path(info["sidecar"]).is_file()
    payload = load_arm_checkpoint_verified(
        path,
        expected_assembly_receipt_sha256="a" * 64,
        expected_arm=S_CTRL,
        expected_u_pre=0,
        expected_state_keys=["w"],
    )
    assert torch.equal(payload["model"]["w"], state["w"])
    assert payload["checkpoint_init_sha256"] == info["state_sha256"]

    with pytest.raises(F2AssemblyContractError, match="refusing to overwrite"):
        save_arm_checkpoint(
            path,
            arm=S_CTRL,
            model_state=state,
            optimizer_state={},
            u_pre=0,
            assembly_receipt_sha256="a" * 64,
        )
    with pytest.raises(
        F2AssemblyContractError, match="different assembly receipt"
    ):
        load_arm_checkpoint_verified(
            path, expected_assembly_receipt_sha256="b" * 64
        )
    with pytest.raises(F2AssemblyContractError, match="tensor set"):
        load_arm_checkpoint_verified(
            path,
            expected_assembly_receipt_sha256="a" * 64,
            expected_state_keys=["w", "extra"],
        )
    with pytest.raises(F2AssemblyContractError, match="u_pre"):
        load_arm_checkpoint_verified(
            path,
            expected_assembly_receipt_sha256="a" * 64,
            expected_u_pre=128,
        )

    data = path.read_bytes()
    path.write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    with pytest.raises(F2AssemblyContractError, match="sidecar receipt"):
        load_arm_checkpoint_verified(
            path, expected_assembly_receipt_sha256="a" * 64
        )


# ---------------------------------------------------------------------------
# HS8 prev-scale presence
# ---------------------------------------------------------------------------


def _runner_g7(abs_tanh_s_prev):
    return RunnerG7Update(
        u_pre=0,
        per_method_over_base={"polar": (0.1,)},
        total_method_over_base=(0.1,),
        abs_tanh_method_scales={"polar": 0.2},
        r_prev=(0.3,),
        abs_tanh_s_prev=abs_tanh_s_prev,
        head_observations=4,
    )


def test_assert_g7_updates_carry_prev_scale_fails_closed():
    assert_g7_updates_carry_prev_scale([_runner_g7(0.1)])
    with pytest.raises(F2AssemblyContractError, match="PREV_SCALE_MISSING"):
        assert_g7_updates_carry_prev_scale([_runner_g7(None)])
    with pytest.raises(F2AssemblyContractError, match="PREV_SCALE_INVALID"):
        assert_g7_updates_carry_prev_scale([_runner_g7(1.5)])
    with pytest.raises(F2AssemblyContractError, match="empty"):
        assert_g7_updates_carry_prev_scale([])


# ---------------------------------------------------------------------------
# Full lifecycle orchestration
# ---------------------------------------------------------------------------


EXPECTED_SMOKE_ARTIFACTS = (
    "checkpoint_update0_S-CTRL.pt",
    "checkpoint_update0_S-CTRL.pt.receipt.json",
    "checkpoint_update0_S-SELF.pt",
    "checkpoint_update0_S-SELF.pt.receipt.json",
    "eval_fix_update0_S-SELF.json",
    "count_receipt.json",
    "checkpoint_update128_S-CTRL.pt",
    "checkpoint_update128_S-CTRL.pt.receipt.json",
    "checkpoint_update128_S-SELF.pt",
    "checkpoint_update128_S-SELF.pt.receipt.json",
    "eval_fix_update128_S-CTRL.json",
    "eval_fix_update128_S-SELF.json",
    "g6_receipt.json",
    "g7_receipt_S-CTRL.json",
    "g7_receipt_S-SELF.json",
    "g8_receipt.json",
    "g9_receipt_S-CTRL.json",
    "g9_receipt_S-SELF.json",
    "combined_smoke_gate_receipt.json",
    "gate_inputs.json",
    "smoke_summary.json",
)


def test_run_production_smoke_pass_lifecycle_and_forensic_rebuild(tmp_path):
    arms = {
        S_CTRL: LifecycleArm(S_CTRL, PASS_CTRL_LOSSES),
        S_SELF: LifecycleArm(S_SELF, PASS_SELF_LOSSES),
    }
    kwargs = _smoke_authority_kwargs(tmp_path)
    output = tmp_path / "smoke_v1"
    summary = run_production_smoke(
        tmp_path,
        output_dir=output,
        plan_builder=lambda root, doc: _smoke_plan(arms),
        **kwargs,
    )
    assert summary["passed"] is True
    assert summary["status"] == "PASS"
    assert summary["formal_training_authorized"] is True
    assert summary["next_step"] == "external_review"
    assert summary["lifecycle_order"] == list(LIFECYCLE_ORDER)
    # P1-1: the CAL -> lambda-freeze -> receipt authority chain is recorded.
    authority = summary["cal_lambda_authority"]
    assert authority["assembly_receipt_chain"] == "direct"
    assert authority["frozen_lambda"] == FAKE_LAMBDA
    assert authority["mechanism"] == LAMBDA_MECHANISM
    assert authority["cal_context"]["seed"] == 0
    assert authority["cal_context"]["checkpoint_init_sha256"] == "c" * 64
    # P3: reproducibility metadata from the plan.
    assert summary["seed"] == 0
    assert summary["device"] == "cpu"
    assert summary["cuda_reproducibility"] is None
    assert summary["gates"] == {
        "G6": True,
        "G7": {S_CTRL: True, S_SELF: True},
        "G8": True,
        "G9": {S_CTRL: True, S_SELF: True},
        "combined": True,
    }
    assert summary["g6_fallback_report"] == {
        "deciding": False,
        "passed": True,
        "status": "PASS",
        "ratio_median": {
            "L_cot": 0.5,
            "L_future": 0.5,
            "L_verify": 0.5,
        },
    }
    for name in EXPECTED_SMOKE_ARTIFACTS:
        assert (output / name).is_file(), name
    combined = json.loads(
        (output / "combined_smoke_gate_receipt.json").read_text(encoding="utf-8")
    )
    assert combined["formal_training_authorized"] is True
    assert combined["gate_order"] == ["G6", "G7", "G8", "G9"]
    g6_document = json.loads(
        (output / "g6_receipt.json").read_text(encoding="utf-8")
    )
    assert g6_document["contract"]["deciding_block_mode"] == "bstar"
    assert g6_document["fallback_per_aux"]["role"] == (
        "reported_non_deciding"
    )
    assert g6_document["fallback_per_aux"]["passed"] is True
    # P1-4: the production combined receipt declares live provenance.
    assert combined["provenance"]["source"] == "run_production_smoke"
    assert combined["provenance"]["authoritative"] is True
    assert (
        combined["provenance"]["assembly_receipt_sha256"]
        == summary["assembly_receipt"]["sha256"]
    )

    # Both arms trained through exactly 128 optimizer updates.
    assert arms[S_CTRL].updates_done == 128
    assert arms[S_SELF].updates_done == 128
    # Bit-identical init and receipt-bound checkpoints.
    update0 = summary["checkpoints"]["update0"]
    assert update0[S_CTRL]["state_sha256"] == update0[S_SELF]["state_sha256"]
    payload = load_arm_checkpoint_verified(
        output / "checkpoint_update0_S-CTRL.pt",
        expected_assembly_receipt_sha256=summary["assembly_receipt"]["sha256"],
        expected_arm=S_CTRL,
        expected_u_pre=0,
    )
    assert payload["arm"] == S_CTRL

    # A finished (even successful) smoke directory can never be reused.
    with pytest.raises(F2AssemblyContractError, match="not empty"):
        run_production_smoke(
            tmp_path,
            output_dir=output,
            plan_builder=lambda root, doc: _smoke_plan(arms),
            **kwargs,
        )

    # Forensic rebuild reproduces the gate verdict but is NEVER
    # authoritative (P1-4): distinct analysis classes, hard-false
    # authorization, and explicit non-authoritative provenance.
    rebuild = build_gate_receipts_from_artifacts(
        output, output_dir=tmp_path / "rebuild"
    )
    assert rebuild["passed"] is True
    assert rebuild["formal_training_authorized"] is False
    assert rebuild["forensic_rebuild_not_authoritative"] is True
    for name in (
        "g6_receipt.json",
        "g7_receipt_S-CTRL.json",
        "g7_receipt_S-SELF.json",
        "g8_receipt.json",
        "g9_receipt_S-CTRL.json",
        "g9_receipt_S-SELF.json",
        "combined_smoke_gate_receipt.json",
    ):
        assert (tmp_path / "rebuild" / name).is_file(), name
    rebuilt_g6 = json.loads(
        (tmp_path / "rebuild" / "g6_receipt.json").read_text(encoding="utf-8")
    )
    assert rebuilt_g6["analysis_class"] == FORENSIC_GATE_CLASS
    assert rebuilt_g6["forensic_rebuild_not_authoritative"] is True
    rebuilt_combined = json.loads(
        (tmp_path / "rebuild" / "combined_smoke_gate_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert rebuilt_combined["analysis_class"] == FORENSIC_GATES_CLASS
    assert rebuilt_combined["passed"] is True
    assert rebuilt_combined["formal_training_authorized"] is False
    assert rebuilt_combined["forensic_rebuild_not_authoritative"] is True
    assert rebuilt_combined["provenance"]["authoritative"] is False
    assert (
        rebuilt_combined["provenance"]["source"]
        == "build_gate_receipts_from_artifacts"
    )
    assert rebuilt_combined["provenance"]["eval_snapshot_overrides"] == {
        "update0_S-SELF": None,
        "update128_S-SELF": None,
        "update128_S-CTRL": None,
    }


def test_run_production_smoke_gate_fail_seals_negative_result(tmp_path):
    arms = {
        S_CTRL: LifecycleArm(S_CTRL, PASS_CTRL_LOSSES),
        S_SELF: LifecycleArm(S_SELF, FAIL_SELF_LOSSES),
    }
    kwargs = _smoke_authority_kwargs(tmp_path)
    output = tmp_path / "smoke_fail"
    summary = run_production_smoke(
        tmp_path,
        output_dir=output,
        plan_builder=lambda root, doc: _smoke_plan(arms),
        **kwargs,
    )
    # A failing gate never raises: the negative result is sealed instead.
    assert summary["passed"] is False
    assert summary["status"] == "FAIL"
    assert summary["decision"] == "STOP"
    assert summary["formal_training_authorized"] is False
    assert summary["next_step"] == (
        "seal_negative_result_no_retry_no_gate_tuning"
    )
    assert summary["gates"]["G8"] is False
    assert summary["gates"]["G6"] is True
    for name in EXPECTED_SMOKE_ARTIFACTS:
        assert (output / name).is_file(), name
    combined = json.loads(
        (output / "combined_smoke_gate_receipt.json").read_text(encoding="utf-8")
    )
    assert combined["passed"] is False
    assert combined["formal_training_authorized"] is False
    g8 = json.loads((output / "g8_receipt.json").read_text(encoding="utf-8"))
    assert g8["passed"] is False


def test_run_production_smoke_refuses_dirty_output_dir(tmp_path):
    kwargs = _smoke_authority_kwargs(tmp_path)
    output = tmp_path / "smoke_dirty"
    output.mkdir()
    (output / "stale.txt").write_text("burned evidence", encoding="utf-8")
    with pytest.raises(F2AssemblyContractError, match="not empty"):
        run_production_smoke(
            tmp_path,
            output_dir=output,
            plan_builder=lambda root, doc: pytest.fail(
                "plan builder must not run for a dirty output dir"
            ),
            **kwargs,
        )
    assert (output / "stale.txt").read_text(encoding="utf-8") == "burned evidence"


def test_run_production_smoke_compares_live_assets_with_receipt_binding(tmp_path):
    kwargs = _smoke_authority_kwargs(tmp_path)
    drifted = dict(FAKE_SMOKE_ASSETS)
    drifted["base_hf"] = {"path": "/frozen/base", "artifact_sha256": "ee" * 32}
    kwargs["asset_verifier"] = lambda root: drifted
    with pytest.raises(
        F2AssemblyContractError, match="ASSET_BINDING_MISMATCH.*base_hf"
    ):
        run_production_smoke(
            tmp_path,
            output_dir=tmp_path / "smoke_assets",
            plan_builder=lambda root, doc: pytest.fail(
                "plan builder must not run when the live assets differ from "
                "the receipt binding"
            ),
            **kwargs,
        )


def test_run_production_smoke_requires_matching_lambda_literals(tmp_path):
    kwargs = _smoke_authority_kwargs(tmp_path)
    drifted = dict(FAKE_LAMBDA)
    drifted["L_future"] = 0.999
    kwargs["aux_coefficients"] = drifted
    with pytest.raises(
        F2AssemblyContractError, match="lambda value mismatch for L_future"
    ):
        run_production_smoke(
            tmp_path,
            output_dir=tmp_path / "smoke_lambda",
            plan_builder=lambda root, doc: pytest.fail(
                "plan builder must not run when the lambda literals do not "
                "match the CAL/freeze chain"
            ),
            **kwargs,
        )


def test_verify_cal_lambda_authority_chain_and_fail_closed_branches(tmp_path):
    # Supersede chain: CAL bound to an older receipt is still traceable
    # through the PRIMARY lambda-freeze receipt bound in the current one.
    receipt_path, verifier = _stub_receipt(tmp_path)
    cal_path = _authority_chain(
        tmp_path, receipt_path, verifier.document, direct=False
    )
    receipt_file_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    authority = verify_cal_lambda_authority(
        tmp_path,
        cal_audit_receipt_path=cal_path,
        receipt_document=verifier.document,
        receipt_file_sha256=receipt_file_sha,
        aux_coefficients=dict(FAKE_LAMBDA),
    )
    assert authority["assembly_receipt_chain"] == "via_lambda_freeze_supersede"
    assert authority["frozen_lambda"] == FAKE_LAMBDA
    assert authority["amendment_id"] == ADJUDICATION_AMENDMENT1_ID

    # Missing lambda-freeze binding in the assembly receipt: forbidden.
    (tmp_path / "unbound").mkdir()
    unbound_path, unbound_verifier = _stub_receipt(
        tmp_path / "unbound", name="receipt.json"
    )
    cal_unbound = _authority_chain(
        tmp_path / "unbound",
        unbound_path,
        unbound_verifier.document,
        bind_freeze=False,
    )
    with pytest.raises(
        F2AssemblyContractError, match="does not bind a PRIMARY lambda freeze"
    ):
        verify_cal_lambda_authority(
            tmp_path / "unbound",
            cal_audit_receipt_path=cal_unbound,
            receipt_document=unbound_verifier.document,
            receipt_file_sha256="0" * 64,
            aux_coefficients=dict(FAKE_LAMBDA),
        )

    # The freeze receipt must adopt exactly this CAL receipt.
    (tmp_path / "orphan").mkdir()
    orphan_path, orphan_verifier = _stub_receipt(
        tmp_path / "orphan", name="receipt.json"
    )
    cal_orphan = _authority_chain(
        tmp_path / "orphan",
        orphan_path,
        orphan_verifier.document,
        adopt=False,
    )
    with pytest.raises(
        F2AssemblyContractError, match="does not adopt this CAL"
    ):
        verify_cal_lambda_authority(
            tmp_path / "orphan",
            cal_audit_receipt_path=cal_orphan,
            receipt_document=orphan_verifier.document,
            receipt_file_sha256="0" * 64,
            aux_coefficients=dict(FAKE_LAMBDA),
        )

    # A CAL receipt with a wrong amendment binding is rejected.
    (tmp_path / "amend").mkdir()
    amend_path, amend_verifier = _stub_receipt(
        tmp_path / "amend", name="receipt.json"
    )
    cal_amend = _authority_chain(
        tmp_path / "amend",
        amend_path,
        amend_verifier.document,
        amendment_sha="9" * 64,
    )
    with pytest.raises(
        F2AssemblyContractError, match="amendment binding mismatch"
    ):
        verify_cal_lambda_authority(
            tmp_path / "amend",
            cal_audit_receipt_path=cal_amend,
            receipt_document=amend_verifier.document,
            receipt_file_sha256="0" * 64,
            aux_coefficients=dict(FAKE_LAMBDA),
        )


def test_verify_cal_lambda_authority_rejects_same_proposal_with_raw_cal_drift(
    tmp_path,
):
    receipt_path, verifier = _stub_receipt(tmp_path)
    cal_path = _authority_chain(
        tmp_path, receipt_path, verifier.document, direct=False
    )
    freeze_path = Path(verifier.document["lambda_freeze_binding"]["path"])
    freeze_document = json.loads(freeze_path.read_text(encoding="utf-8"))
    reproduction_path = Path(
        freeze_document["evidence"]["deterministic_reproduction"]["path"]
    )
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    reproduction["gradient_calibration"]["per_aux_grad_norm_median"][
        "L_verify"
    ] = 0.5001
    reproduction["gradient_calibration"]["aux_grad_norm_median_min"] = 0.5001
    assert (
        reproduction["lambda_calibration"]["proposed_lambda"]
        == FAKE_LAMBDA
    )
    reproduction_path.write_text(
        json.dumps(reproduction, sort_keys=True), encoding="utf-8"
    )
    freeze_document["evidence"]["deterministic_reproduction"][
        "sha256"
    ] = hashlib.sha256(reproduction_path.read_bytes()).hexdigest()
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        F2AssemblyContractError, match="not byte-identical"
    ):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )


@pytest.mark.parametrize(
    ("field", "drifted"),
    [
        ("seed", 1),
        ("device", "cuda:0"),
        ("checkpoint_init_sha256", "d" * 64),
    ],
)
def test_verify_cal_lambda_authority_rejects_copied_context_drift(
    tmp_path, field, drifted
):
    receipt_path, verifier = _stub_receipt(tmp_path)
    cal_path = _authority_chain(tmp_path, receipt_path, verifier.document)
    freeze_path = Path(verifier.document["lambda_freeze_binding"]["path"])
    freeze_document = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_document["evidence"]["cal_audit_receipt"][field] = drifted
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()

    with pytest.raises(
        F2AssemblyContractError, match=repr(field)
    ):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )


def test_verify_cal_lambda_authority_rehashes_bootstrap_and_checks_seal(
    tmp_path,
):
    receipt_path, verifier = _stub_receipt(tmp_path)
    cal_path = _authority_chain(tmp_path, receipt_path, verifier.document)
    freeze_path = Path(verifier.document["lambda_freeze_binding"]["path"])
    freeze_document = json.loads(freeze_path.read_text(encoding="utf-8"))
    freeze_document["internal_test_opened"] = True
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    with pytest.raises(F2AssemblyContractError, match="internal-test seal"):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )

    freeze_document["internal_test_opened"] = False
    freeze_document.pop("internal_test")
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        F2AssemblyContractError, match="must record internal_test=sealed"
    ):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )

    freeze_document["internal_test"] = INTERNAL_TEST_POLICY
    bootstrap_evidence = freeze_document["evidence"][
        "bootstrap_assembly_receipt"
    ]
    bootstrap_payload_sha = bootstrap_evidence.pop("receipt_payload_sha256")
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    with pytest.raises(
        F2AssemblyContractError, match="bootstrap payload evidence differs"
    ):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )

    bootstrap_evidence["receipt_payload_sha256"] = bootstrap_payload_sha
    freeze_path.write_text(json.dumps(freeze_document), encoding="utf-8")
    verifier.document["lambda_freeze_binding"]["sha256"] = hashlib.sha256(
        freeze_path.read_bytes()
    ).hexdigest()
    bootstrap_path = Path(
        freeze_document["evidence"]["bootstrap_assembly_receipt"]["path"]
    )
    bootstrap_path.write_text(
        bootstrap_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(
        F2AssemblyContractError,
        match="bootstrap assembly evidence SHA differs",
    ):
        verify_cal_lambda_authority(
            tmp_path,
            cal_audit_receipt_path=cal_path,
            receipt_document=verifier.document,
            receipt_file_sha256=hashlib.sha256(
                receipt_path.read_bytes()
            ).hexdigest(),
            aux_coefficients=dict(FAKE_LAMBDA),
        )


def test_run_eval_snapshot_command_verifies_checkpoint_binding(tmp_path):
    receipt_path, verifier = _stub_receipt(tmp_path)
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    state = {"w": torch.arange(4, dtype=torch.float32)}
    checkpoint = tmp_path / "checkpoint_update0_S-SELF.pt"
    save_arm_checkpoint(
        checkpoint,
        arm=S_SELF,
        model_state=state,
        optimizer_state={},
        u_pre=0,
        assembly_receipt_sha256=receipt_sha,
    )
    eval_rows = _grouped_rows(512, 28)
    raw_rows = _raw_rows(512)

    def loader(root, support_name, token_ledger):
        assert support_name == "EVAL-FIX"
        assert token_ledger is FAKE_TOKEN_LEDGER
        return eval_rows, raw_rows, frozenset()

    loader.token_ledger = FAKE_TOKEN_LEDGER
    predictor = RecordingPredictor()
    result = run_eval_snapshot_command(
        tmp_path,
        receipt_path=receipt_path,
        arm=S_SELF,
        snapshot=0,
        checkpoint_path=checkpoint,
        output_dir=tmp_path / "eval0",
        verifier=verifier,
        rows_loader=loader,
        predictor_builder=lambda root, doc, arm, payload: predictor,
    )
    document = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert document["arm"] == S_SELF
    assert document["snapshot"] == "update0"
    assert document["token_ledger_binding"]["sha256"] == (
        FAKE_TOKEN_LEDGER.ledger_sha256
    )
    assert set(document["logged"]) == {"accumulator", "means", "counts"}
    assert len(predictor.calls) == 1024  # logged + self over 512 rows

    with pytest.raises(F2AssemblyContractError, match="belongs to"):
        run_eval_snapshot_command(
            tmp_path,
            receipt_path=receipt_path,
            arm=S_CTRL,
            snapshot=0,
            checkpoint_path=checkpoint,
            output_dir=tmp_path / "eval0_wrong_arm",
            verifier=verifier,
            rows_loader=loader,
            predictor_builder=lambda root, doc, arm, payload: predictor,
        )
