from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import ibr1_experiment.checkpoint as checkpoint_module
from f2_experiment.assembly_model import (
    ArmAssembly,
    FROZEN_AUX_COEFFICIENTS,
    OptimizerContract,
    build_package,
)
from f2_experiment.runner import S_CTRL, S_SELF, checkpoint_init_sha256
from ibr1_experiment.assembly_model import (
    F2SealedInitEvidence,
    IBR1PairedArms,
    IBR1_CTRL,
    IBR1_SELF,
    build_ibr1_paired_arms,
)
from ibr1_experiment.authority import (
    ASSEMBLY_PHASE_BOOTSTRAP,
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
    IBR1AuthorityError,
)
from ibr1_experiment.checkpoint import (
    CHECKPOINT_PAYLOAD_CLASS,
    CHECKPOINT_SIDECAR_CLASS,
    IBR1CheckpointContractError,
    IBR1EvaluationSnapshot,
    checkpoint_tensor_sha256,
    load_ibr1_arm_checkpoint_verified,
    save_ibr1_arm_checkpoint,
)
from ibr1_experiment.model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LIVE_F2_EVIDENCE: F2SealedInitEvidence | None = None


@pytest.fixture(autouse=True)
def _stub_live_assembly_verifier(monkeypatch):
    def verify(project_root, receipt_path, *, required_phase=None, **kwargs):
        del project_root, kwargs
        assert required_phase == ASSEMBLY_PHASE_FINAL
        return _read_json(Path(receipt_path))

    monkeypatch.setattr(checkpoint_module, "verify_assembly_receipt", verify)

    def read_live_f2(project_root):
        del project_root
        assert _LIVE_F2_EVIDENCE is not None
        return _LIVE_F2_EVIDENCE

    monkeypatch.setattr(
        checkpoint_module, "read_sealed_f2_init_evidence", read_live_f2
    )


class DummyLLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mix = nn.Linear(d_model, d_model, bias=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.mix.weight.dtype


class DummyOfficialBase(nn.Module):
    def __init__(self, input_dim: int = 6, d_model: int = 8) -> None:
        super().__init__()
        self.D = d_model
        self.cfg = SimpleNamespace(use_angle_tvi=False)
        self.proj = nn.Linear(input_dim, d_model)
        self.llm = DummyLLM(d_model)
        self.act_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

    def _interleave_tvi(self, *args, **kwargs):
        del kwargs
        return args[0]

    def _embed_text(self, instructions, device):
        batch_size = len(instructions)
        return (
            self.text_token.expand(batch_size, -1, -1).to(device),
            torch.ones(batch_size, 2, dtype=torch.long, device=device),
        )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(init_sha: str) -> F2SealedInitEvidence:
    return F2SealedInitEvidence(
        primary_path="primary.json",
        primary_sha256="0" * 64,
        negative_adoption_path="adoption.json",
        negative_adoption_sha256="1" * 64,
        negative_seal_path="seal.json",
        negative_seal_sha256="2" * 64,
        smoke_summary_path="summary.json",
        smoke_summary_sha256="3" * 64,
        seed=0,
        checkpoint_init_sha256=init_sha,
    )


def _f2_init_sha(base: nn.Module) -> str:
    torch.manual_seed(0)
    modules = build_package(
        "SA-Hstar",
        base,
        device="cpu",
        aux_coefficients=dict(FROZEN_AUX_COEFFICIENTS or {}),
    )
    return checkpoint_init_sha256(modules.full_state_dict())


def _paired(*, d_model: int = 8) -> IBR1PairedArms:
    global _LIVE_F2_EVIDENCE
    torch.manual_seed(37)
    base = DummyOfficialBase(d_model=d_model)
    expected_sha = _f2_init_sha(copy.deepcopy(base))
    paired = build_ibr1_paired_arms(
        copy.deepcopy(base),
        seed=0,
        device="cpu",
        contract=OptimizerContract(),
        parent_f2_evidence=_evidence(expected_sha),
    )
    _LIVE_F2_EVIDENCE = paired.parent_f2_evidence
    return paired


def _activate_live_evidence(paired: IBR1PairedArms) -> None:
    global _LIVE_F2_EVIDENCE
    _LIVE_F2_EVIDENCE = paired.parent_f2_evidence


def _final_receipt(
    tmp_path: Path,
    *,
    phase: str = ASSEMBLY_PHASE_FINAL,
    freeze: object = "default",
) -> tuple[Path, str]:
    if freeze == "default":
        freeze = {
            "path": "experiments/windows_cuda_ibr1/"
            "ibr1_lambda_adoption_freeze_v1.json",
            "sha256": "a" * 64,
        }
    path = tmp_path / "ibr1_assembly_final.json"
    _write_json(
        path,
        {
            "schema_version": 1,
            "analysis_class": ASSEMBLY_RECEIPT_CLASS,
            "family_id": IBR1_FAMILY_ID,
            "architecture_lock": IBR1_ARCHITECTURE_LOCK,
            "phase": phase,
            "lambda_freeze_binding": freeze,
            "internal_test": "sealed",
            "internal_test_opened": False,
            "receipt_payload_sha256": "b" * 64,
        },
    )
    return path, _file_sha256(path)


@dataclass(frozen=True)
class SavedCheckpoint:
    path: Path
    paired: IBR1PairedArms
    arm: ArmAssembly
    engine_arm: str
    receipt_path: Path
    receipt_sha256: str
    info: Mapping[str, Any]


def _save(
    tmp_path: Path,
    *,
    paired: IBR1PairedArms | None = None,
    arm: ArmAssembly | None = None,
    engine_arm: str = S_CTRL,
    u_pre: int = 0,
) -> SavedCheckpoint:
    tmp_path.mkdir(parents=True, exist_ok=True)
    if paired is None:
        paired = _paired()
    if arm is None:
        arm = paired.arms[engine_arm]
    _activate_live_evidence(paired)
    receipt_path, receipt_sha = _final_receipt(tmp_path)
    path = tmp_path / f"checkpoint_update{u_pre}_{engine_arm}.pt"
    info = save_ibr1_arm_checkpoint(
        path,
        paired_arms=paired,
        arm_assembly=arm,
        engine_arm=engine_arm,
        u_pre=u_pre,
        final_assembly_receipt_path=receipt_path,
        final_assembly_receipt_sha256=receipt_sha,
        project_root=PROJECT_ROOT,
    )
    return SavedCheckpoint(
        path=path,
        paired=paired,
        arm=arm,
        engine_arm=engine_arm,
        receipt_path=receipt_path,
        receipt_sha256=receipt_sha,
        info=info,
    )


def _load(
    saved: SavedCheckpoint,
    *,
    paired: IBR1PairedArms | None = None,
    arm: ArmAssembly | None = None,
    engine_arm: str | None = None,
    u_pre: int | None = None,
    checkpoint_sha256: str | None = None,
    sidecar_sha256: str | None = None,
) -> IBR1EvaluationSnapshot:
    selected_paired = saved.paired if paired is None else paired
    _activate_live_evidence(selected_paired)
    return load_ibr1_arm_checkpoint_verified(
        saved.path,
        paired_arms=selected_paired,
        expected_arm_assembly=saved.arm if arm is None else arm,
        expected_engine_arm=(
            saved.engine_arm if engine_arm is None else engine_arm
        ),
        expected_u_pre=saved.info["u_pre"] if u_pre is None else u_pre,
        expected_checkpoint_file_sha256=(
            saved.info["file_sha256"]
            if checkpoint_sha256 is None
            else checkpoint_sha256
        ),
        expected_sidecar_sha256=(
            saved.info["sidecar_sha256"]
            if sidecar_sha256 is None
            else sidecar_sha256
        ),
        expected_final_assembly_receipt_path=saved.receipt_path,
        expected_final_assembly_receipt_sha256=saved.receipt_sha256,
        project_root=PROJECT_ROOT,
    )


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in module.state_dict().items()
    }


def _assert_state_equal(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
) -> None:
    assert set(observed) == set(expected)
    for name in expected:
        assert torch.equal(observed[name].detach().cpu(), expected[name]), name


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, Mapping):
        assert set(left) == set(right)
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for l_value, r_value in zip(left, right):
            _assert_nested_equal(l_value, r_value)
    else:
        assert left == right


def _rewrite_payload_and_sidecar(
    saved: SavedCheckpoint,
    mutate,
) -> tuple[str, str]:
    payload = torch.load(saved.path, map_location="cpu", weights_only=True)
    mutate(payload)
    with saved.path.open("wb") as handle:
        torch.save(payload, handle)
    file_sha = _file_sha256(saved.path)
    sidecar_path = Path(saved.info["sidecar"])
    sidecar = _read_json(sidecar_path)
    sidecar["checkpoint_file_sha256"] = file_sha
    sidecar["checkpoint_tensor_sha256"] = payload[
        "checkpoint_tensor_sha256"
    ]
    _write_json(sidecar_path, sidecar)
    return file_sha, _file_sha256(sidecar_path)


def test_checkpoint_tensor_sha_reuses_f2_full_state_domain() -> None:
    adapter_state = {
        "weight": torch.tensor([[1.0, -2.0]], dtype=torch.float32),
        "counter": torch.tensor([3], dtype=torch.int64),
    }
    model_state = {
        "bias": torch.tensor([0.25, -0.5], dtype=torch.float64),
    }
    expected = checkpoint_init_sha256(
        {
            **{f"adapter.{name}": tensor for name, tensor in adapter_state.items()},
            **{f"model.{name}": tensor for name, tensor in model_state.items()},
        }
    )
    observed = checkpoint_tensor_sha256(
        adapter_state=adapter_state,
        model_state=model_state,
    )
    assert observed == expected

    drifted_model = {name: tensor.clone() for name, tensor in model_state.items()}
    drifted_model["bias"][0] += 1.0
    assert checkpoint_tensor_sha256(
        adapter_state=adapter_state,
        model_state=drifted_model,
    ) != expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_tensor_sha_is_cpu_cuda_invariant() -> None:
    adapter_cpu = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    model_cpu = {"scale": torch.tensor(0.5, dtype=torch.float32)}
    expected = checkpoint_tensor_sha256(
        adapter_state=adapter_cpu,
        model_state=model_cpu,
    )
    assert checkpoint_tensor_sha256(
        adapter_state={name: tensor.cuda() for name, tensor in adapter_cpu.items()},
        model_state={name: tensor.cuda() for name, tensor in model_cpu.items()},
    ) == expected


def test_update0_saved_tensor_hash_matches_sealed_init_domain(tmp_path: Path) -> None:
    paired = _paired()
    ctrl = _save(
        tmp_path / "ctrl",
        paired=paired,
        arm=paired.arms[S_CTRL],
        engine_arm=S_CTRL,
        u_pre=0,
    )
    self_arm = _save(
        tmp_path / "self",
        paired=paired,
        arm=paired.arms[S_SELF],
        engine_arm=S_SELF,
        u_pre=0,
    )
    assert {
        ctrl.info["tensor_sha256"],
        self_arm.info["tensor_sha256"],
    } == {paired.checkpoint_init_sha256}

    payload = torch.load(ctrl.path, map_location="cpu", weights_only=True)
    assert payload["checkpoint_tensor_sha256"] == paired.checkpoint_init_sha256


def test_loader_returns_cpu_eval_snapshot_without_mutating_live_state(tmp_path):
    saved = _save(tmp_path)
    saved_adapter = _clone_state(saved.arm.modules.adapter)
    saved_model = _clone_state(saved.arm.modules.model)

    trainable = list(saved.arm.modules.trainable_parameters())
    loss = sum(parameter.float().sum() for parameter in trainable)
    loss.backward()
    saved.arm.optimizer.step()
    saved.arm.optimizer.zero_grad(set_to_none=True)
    live_adapter = _clone_state(saved.arm.modules.adapter)
    live_model = _clone_state(saved.arm.modules.model)
    live_optimizer = copy.deepcopy(saved.arm.optimizer.state_dict())
    live_rng = torch.random.get_rng_state().clone()

    snapshot = _load(saved)
    assert isinstance(snapshot, IBR1EvaluationSnapshot)
    assert snapshot.evaluation_only is True
    assert snapshot.resume_supported is False
    assert snapshot.optimizer_state_included is False
    assert snapshot.rng_state_included is False
    assert all(t.device.type == "cpu" for t in snapshot.adapter_state.values())
    assert all(t.device.type == "cpu" for t in snapshot.model_state.values())
    _assert_state_equal(snapshot.adapter_state, saved_adapter)
    _assert_state_equal(snapshot.model_state, saved_model)
    _assert_state_equal(saved.arm.modules.adapter.state_dict(), live_adapter)
    _assert_state_equal(saved.arm.modules.model.state_dict(), live_model)
    _assert_nested_equal(saved.arm.optimizer.state_dict(), live_optimizer)
    assert torch.equal(torch.random.get_rng_state(), live_rng)
    assert saved.arm.modules.adapter.base is saved.arm.modules.base

    public_model = snapshot.model_state
    first_name = next(iter(public_model))
    authoritative_value = snapshot.model_state[first_name].clone()
    with pytest.raises(TypeError):
        public_model["new"] = torch.zeros(1)
    public_model[first_name].add_(9.0)
    assert torch.equal(snapshot.model_state[first_name], authoritative_value)
    materialized = snapshot.materialize_model_state()
    materialized[first_name].sub_(3.0)
    assert torch.equal(snapshot.model_state[first_name], authoritative_value)
    with pytest.raises(TypeError):
        snapshot.final_assembly_receipt["sha256"] = "0" * 64


def test_external_file_and_sidecar_anchors_reject_coordinated_rewrite(
    tmp_path, monkeypatch
):
    saved = _save(tmp_path)
    original_file_sha = saved.info["file_sha256"]
    original_sidecar_sha = saved.info["sidecar_sha256"]

    def mutate(payload):
        name = next(iter(payload["model_state"]))
        payload["model_state"][name] = payload["model_state"][name].clone()
        payload["model_state"][name].reshape(-1)[0] += 1.0
        payload["checkpoint_tensor_sha256"] = checkpoint_tensor_sha256(
            adapter_state=payload["adapter_state"],
            model_state=payload["model_state"],
        )

    new_file_sha, new_sidecar_sha = _rewrite_payload_and_sidecar(saved, mutate)
    assert new_file_sha != original_file_sha
    assert new_sidecar_sha != original_sidecar_sha

    def forbidden_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.load must not run after anchor drift")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(
        IBR1CheckpointContractError, match="sidecar.*external lifecycle anchor"
    ):
        _load(
            saved,
            checkpoint_sha256=original_file_sha,
            sidecar_sha256=original_sidecar_sha,
        )


def test_checkpoint_file_anchor_is_independent_of_sidecar_anchor(
    tmp_path, monkeypatch
):
    saved = _save(tmp_path)
    payload = torch.load(saved.path, map_location="cpu", weights_only=True)
    payload["analysis_class"] = "ibr1_arm_checkpoint_payload_rewritten"
    with saved.path.open("wb") as handle:
        torch.save(payload, handle)
    assert _file_sha256(saved.path) != saved.info["file_sha256"]

    def forbidden_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.load must not run after file anchor drift")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(
        IBR1CheckpointContractError, match="file.*external lifecycle anchor"
    ):
        _load(saved)


def test_update0_payload_is_rebound_to_live_sealed_init_even_with_new_anchors(
    tmp_path
):
    saved = _save(tmp_path)

    def mutate(payload):
        name = next(iter(payload["model_state"]))
        payload["model_state"][name] = payload["model_state"][name].clone()
        payload["model_state"][name].reshape(-1)[0] += 0.5
        payload["checkpoint_tensor_sha256"] = checkpoint_tensor_sha256(
            adapter_state=payload["adapter_state"],
            model_state=payload["model_state"],
        )

    file_sha, sidecar_sha = _rewrite_payload_and_sidecar(saved, mutate)
    with pytest.raises(
        IBR1CheckpointContractError, match="payload differs.*live sealed F2 init"
    ):
        _load(
            saved,
            checkpoint_sha256=file_sha,
            sidecar_sha256=sidecar_sha,
        )


def test_paired_object_identity_rejects_relabel_and_swapped_arm(tmp_path):
    paired = _paired()
    ctrl = paired.arms[S_CTRL]
    relabeled_receipt = {
        **dict(ctrl.parameter_receipt),
        "family_arm": IBR1_SELF,
        "engine_arm": S_SELF,
    }
    reconstructed = ArmAssembly(
        modules=ctrl.modules,
        optimizer=ctrl.optimizer,
        parameter_receipt=relabeled_receipt,
    )
    receipt, receipt_sha = _final_receipt(tmp_path)
    with pytest.raises(IBR1CheckpointContractError, match="exact paired_arms"):
        save_ibr1_arm_checkpoint(
            tmp_path / "relabel.pt",
            paired_arms=paired,
            arm_assembly=reconstructed,
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )
    with pytest.raises(IBR1CheckpointContractError, match="exact paired_arms"):
        save_ibr1_arm_checkpoint(
            tmp_path / "swapped.pt",
            paired_arms=paired,
            arm_assembly=paired.arms[S_SELF],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )


def test_paired_mapping_parent_and_disjointness_are_revalidated(tmp_path):
    paired = _paired()
    receipt, receipt_sha = _final_receipt(tmp_path)
    drifted_mapping = replace(
        paired,
        arm_mapping={S_CTRL: IBR1_SELF, S_SELF: IBR1_CTRL},
    )
    with pytest.raises(IBR1CheckpointContractError, match="mapping drifted"):
        save_ibr1_arm_checkpoint(
            tmp_path / "mapping.pt",
            paired_arms=drifted_mapping,
            arm_assembly=drifted_mapping.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    drifted_parent = replace(paired, checkpoint_init_sha256="f" * 64)
    with pytest.raises(IBR1CheckpointContractError, match="live F2 evidence"):
        save_ibr1_arm_checkpoint(
            tmp_path / "parent.pt",
            paired_arms=drifted_parent,
            arm_assembly=drifted_parent.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    fake_evidence = replace(
        paired.parent_f2_evidence,
        smoke_summary_sha256="e" * 64,
    )
    forged_parent = replace(paired, parent_f2_evidence=fake_evidence)
    _activate_live_evidence(paired)
    with pytest.raises(
        IBR1CheckpointContractError, match="differs from the live sealed chain"
    ):
        save_ibr1_arm_checkpoint(
            tmp_path / "forged_parent.pt",
            paired_arms=forged_parent,
            arm_assembly=forged_parent.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    drifted_device = replace(paired, device="cuda:0")
    _activate_live_evidence(paired)
    with pytest.raises(IBR1CheckpointContractError, match="declared device"):
        save_ibr1_arm_checkpoint(
            tmp_path / "device.pt",
            paired_arms=drifted_device,
            arm_assembly=drifted_device.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    ctrl = paired.arms[S_CTRL]
    fake_self = ArmAssembly(
        modules=ctrl.modules,
        optimizer=ctrl.optimizer,
        parameter_receipt={
            **dict(ctrl.parameter_receipt),
            "family_arm": IBR1_SELF,
            "engine_arm": S_SELF,
        },
    )
    shared = replace(paired, arms={S_CTRL: ctrl, S_SELF: fake_self})
    with pytest.raises(IBR1CheckpointContractError, match="share optimizer"):
        save_ibr1_arm_checkpoint(
            tmp_path / "shared.pt",
            paired_arms=shared,
            arm_assembly=shared.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )


@pytest.mark.parametrize(
    "drift",
    ["lr", "betas", "group_order", "parameter_receipt"],
)
def test_optimizer_and_parameter_receipt_are_mechanically_rebuilt(
    tmp_path, drift
):
    paired = _paired()
    arm = paired.arms[S_CTRL]
    if drift == "lr":
        arm.optimizer.param_groups[0]["lr"] *= 2.0
    elif drift == "betas":
        arm.optimizer.param_groups[0]["betas"] = (0.8, 0.99)
    elif drift == "group_order":
        arm.optimizer.param_groups[:] = list(reversed(arm.optimizer.param_groups))
    else:
        arm.parameter_receipt["contract"]["base_lr"] = 9e-5
    receipt, receipt_sha = _final_receipt(tmp_path)
    with pytest.raises(
        IBR1CheckpointContractError,
        match="receipt differs|optimizer .*drifted",
    ):
        save_ibr1_arm_checkpoint(
            tmp_path / f"optimizer_{drift}.pt",
            paired_arms=paired,
            arm_assembly=arm,
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )


def test_partial_sidecar_failure_rolls_back_and_same_path_can_retry(
    tmp_path, monkeypatch
):
    paired = _paired()
    receipt, receipt_sha = _final_receipt(tmp_path)
    path = tmp_path / "retry.pt"
    original_writer = checkpoint_module._write_sidecar_bytes

    def fail_after_partial_write(handle, encoded):
        handle.write(encoded[:17])
        raise OSError("injected partial sidecar failure")

    monkeypatch.setattr(
        checkpoint_module, "_write_sidecar_bytes", fail_after_partial_write
    )
    with pytest.raises(IBR1CheckpointContractError, match="write.*sidecar"):
        save_ibr1_arm_checkpoint(
            path,
            paired_arms=paired,
            arm_assembly=paired.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )
    assert not path.exists()
    assert not path.with_name(path.name + ".receipt.json").exists()

    monkeypatch.setattr(
        checkpoint_module, "_write_sidecar_bytes", original_writer
    )
    info = save_ibr1_arm_checkpoint(
        path,
        paired_arms=paired,
        arm_assembly=paired.arms[S_CTRL],
        engine_arm=S_CTRL,
        u_pre=0,
        final_assembly_receipt_path=receipt,
        final_assembly_receipt_sha256=receipt_sha,
        project_root=PROJECT_ROOT,
    )
    assert path.is_file()
    assert Path(info["sidecar"]).is_file()


def test_f2_sidecar_and_payload_are_rejected_even_with_new_anchors(tmp_path):
    saved = _save(tmp_path)
    sidecar_path = Path(saved.info["sidecar"])
    sidecar = _read_json(sidecar_path)
    sidecar["analysis_class"] = "f2_arm_checkpoint_receipt"
    sidecar["family_id"] = "F2"
    _write_json(sidecar_path, sidecar)
    with pytest.raises(IBR1CheckpointContractError, match="not an IBR1"):
        _load(saved, sidecar_sha256=_file_sha256(sidecar_path))

    other = _save(tmp_path / "payload")

    def make_f2(payload):
        payload["family_id"] = "F2"

    file_sha, sidecar_sha = _rewrite_payload_and_sidecar(other, make_f2)
    with pytest.raises(IBR1CheckpointContractError, match="different family"):
        _load(
            other,
            checkpoint_sha256=file_sha,
            sidecar_sha256=sidecar_sha,
        )


@pytest.mark.parametrize("drift", ["key", "shape", "dtype"])
def test_schema_drift_fails_before_torch_load(tmp_path, monkeypatch, drift):
    saved = _save(tmp_path)
    if drift == "shape":
        target = _paired(d_model=10)
        target_arm = target.arms[S_CTRL]
    else:
        target = saved.paired
        target_arm = saved.arm
    if drift == "key":
        target_arm.modules.model.register_buffer("schema_drift", torch.zeros(1))
    elif drift == "dtype":
        target_arm.modules.adapter.double()
        target_arm.modules.model.double()

    def forbidden_load(*args, **kwargs):
        del args, kwargs
        raise AssertionError("torch.load must not run after schema drift")

    monkeypatch.setattr(torch, "load", forbidden_load)
    with pytest.raises(
        IBR1CheckpointContractError, match="key, shape, or dtype schema"
    ):
        _load(saved, paired=target, arm=target_arm)


def test_update_boundary_and_final_authority_contracts(tmp_path, monkeypatch):
    paired = _paired()
    receipt, receipt_sha = _final_receipt(tmp_path)
    with pytest.raises(IBR1CheckpointContractError, match="mid-run"):
        save_ibr1_arm_checkpoint(
            tmp_path / "update64.pt",
            paired_arms=paired,
            arm_assembly=paired.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=64,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    with torch.no_grad():
        next(paired.arms[S_CTRL].modules.model.parameters()).add_(1.0)
    with pytest.raises(IBR1CheckpointContractError, match="live sealed F2 init"):
        save_ibr1_arm_checkpoint(
            tmp_path / "drifted_update0.pt",
            paired_arms=paired,
            arm_assembly=paired.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    other_arm_drift = _paired()
    with torch.no_grad():
        next(other_arm_drift.arms[S_SELF].modules.model.parameters()).add_(1.0)
    _activate_live_evidence(other_arm_drift)
    with pytest.raises(
        IBR1CheckpointContractError, match="S-SELF state differs.*live sealed"
    ):
        save_ibr1_arm_checkpoint(
            tmp_path / "only_ctrl_is_correct.pt",
            paired_arms=other_arm_drift,
            arm_assembly=other_arm_drift.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    receipt, receipt_sha = _final_receipt(
        bootstrap_dir, phase=ASSEMBLY_PHASE_BOOTSTRAP
    )
    fresh = _paired()
    with pytest.raises(IBR1CheckpointContractError, match="final, not bootstrap"):
        save_ibr1_arm_checkpoint(
            bootstrap_dir / "bootstrap.pt",
            paired_arms=fresh,
            arm_assembly=fresh.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )

    def reject_live(*args, **kwargs):
        del args, kwargs
        raise IBR1AuthorityError("not live")

    monkeypatch.setattr(checkpoint_module, "verify_assembly_receipt", reject_live)
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    receipt, receipt_sha = _final_receipt(live_dir)
    with pytest.raises(IBR1CheckpointContractError, match="live authority"):
        save_ibr1_arm_checkpoint(
            live_dir / "shallow.pt",
            paired_arms=fresh,
            arm_assembly=fresh.arms[S_CTRL],
            engine_arm=S_CTRL,
            u_pre=0,
            final_assembly_receipt_path=receipt,
            final_assembly_receipt_sha256=receipt_sha,
            project_root=PROJECT_ROOT,
        )


def test_sidecar_binds_sources_arms_tensor_and_final_payload_sha(tmp_path):
    saved = _save(tmp_path)
    sidecar = _read_json(Path(saved.info["sidecar"]))
    assert sidecar["analysis_class"] == CHECKPOINT_SIDECAR_CLASS
    assert sidecar["family_id"] == IBR1_FAMILY_ID
    assert sidecar["family_arm"] == IBR1_CTRL
    assert sidecar["engine_arm"] == S_CTRL
    assert sidecar["checkpoint_file_sha256"] == saved.info["file_sha256"]
    assert sidecar["checkpoint_tensor_sha256"] == saved.info["tensor_sha256"]
    assert sidecar["final_assembly_receipt"]["receipt_payload_sha256"] == (
        "b" * 64
    )
    assert set(sidecar["source_sha256"]) == {
        "ibr1_experiment/model.py",
        "ibr1_experiment/checkpoint.py",
    }
    payload = torch.load(saved.path, map_location="cpu", weights_only=True)
    assert payload["analysis_class"] == CHECKPOINT_PAYLOAD_CLASS
    assert "optimizer" not in payload and "rng_state" not in payload
