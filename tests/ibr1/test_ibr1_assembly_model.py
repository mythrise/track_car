from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f2_experiment.assembly_model import (
    FROZEN_AUX_COEFFICIENTS,
    OptimizerContract,
    build_package,
)
from f2_experiment.runner import S_CTRL, S_SELF, checkpoint_init_sha256
from ibr1_experiment.assembly_model import (
    ENGINE_TO_FAMILY_ARM,
    F2SealedInitEvidence,
    IBR1AssemblyContractError,
    IBR1_CTRL,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    IBR1_SELF,
    build_ibr1_package,
    build_ibr1_paired_arms,
    read_sealed_f2_init_evidence,
)
from ibr1_experiment.model import IBR1AP2Model, IBR1_ARCHITECTURE_LOCK


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def _f2_dummy_init_sha(base: nn.Module) -> str:
    torch.manual_seed(0)
    modules = build_package(
        "SA-Hstar",
        base,
        device="cpu",
        aux_coefficients=dict(FROZEN_AUX_COEFFICIENTS or {}),
    )
    return checkpoint_init_sha256(modules.full_state_dict())


def test_reads_seed0_init_through_the_sealed_f2_authority_chain():
    evidence = read_sealed_f2_init_evidence(PROJECT_ROOT)
    assert evidence.seed == 0
    assert evidence.checkpoint_init_sha256 == (
        "74f838e314dd6f3b208dbed23e0e7a92dcdde09bb7d79b0ce8c1147d9b251e54"
    )
    assert evidence.negative_seal_sha256 == (
        "b85585c8232f65c75d5958abb7d51d7624db4031c9adec86ee570d0a5b7378e7"
    )


def test_ibr1_package_preserves_f2_state_keys_bytes_and_rng_consumption():
    torch.manual_seed(17)
    base = DummyOfficialBase()
    f2_base = copy.deepcopy(base)
    ibr1_base = copy.deepcopy(base)

    torch.manual_seed(0)
    f2 = build_package(
        "SA-Hstar",
        f2_base,
        device="cpu",
        aux_coefficients=dict(FROZEN_AUX_COEFFICIENTS or {}),
    )
    f2_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(0)
    ibr1 = build_ibr1_package(ibr1_base, device="cpu")
    ibr1_rng = torch.random.get_rng_state().clone()

    assert isinstance(ibr1.model, IBR1AP2Model)
    assert ibr1.adapter.base is ibr1.base
    assert set(f2.full_state_dict()) == set(ibr1.full_state_dict())
    for key in f2.full_state_dict():
        assert torch.equal(f2.full_state_dict()[key], ibr1.full_state_dict()[key])
    assert torch.equal(f2_rng, ibr1_rng)
    assert checkpoint_init_sha256(f2.full_state_dict()) == checkpoint_init_sha256(
        ibr1.full_state_dict()
    )

    proj_ids = {id(parameter) for parameter in ibr1.base.proj.parameters()}
    assert all(parameter.requires_grad for parameter in ibr1.base.proj.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in ibr1.base.parameters()
        if id(parameter) not in proj_ids
    )


def test_paired_arms_are_bit_identical_but_fully_disjoint():
    torch.manual_seed(41)
    base = DummyOfficialBase()
    expected_sha = _f2_dummy_init_sha(copy.deepcopy(base))
    paired = build_ibr1_paired_arms(
        copy.deepcopy(base),
        seed=0,
        device="cpu",
        contract=OptimizerContract(),
        parent_f2_evidence=_evidence(expected_sha),
    )

    assert paired.checkpoint_init_sha256 == expected_sha
    assert paired.architecture_lock == IBR1_ARCHITECTURE_LOCK
    assert paired.arm_mapping == {S_CTRL: IBR1_CTRL, S_SELF: IBR1_SELF}
    assert set(paired.public_arms()) == {IBR1_CTRL, IBR1_SELF}
    ctrl = paired.arms[S_CTRL]
    self_arm = paired.arms[S_SELF]
    assert ctrl.optimizer is not self_arm.optimizer
    ctrl_state = ctrl.modules.full_state_dict()
    self_state = self_arm.modules.full_state_dict()
    ctrl_ids = {id(value) for value in ctrl_state.values()}
    self_ids = {id(value) for value in self_state.values()}
    assert ctrl_ids.isdisjoint(self_ids)
    assert checkpoint_init_sha256(ctrl.modules.full_state_dict()) == expected_sha
    assert checkpoint_init_sha256(self_arm.modules.full_state_dict()) == expected_sha
    for engine_arm, assembly in paired.arms.items():
        receipt = assembly.parameter_receipt
        assert receipt["analysis_class"] == "ibr1_arm_optimizer_parameter_receipt"
        assert receipt["engine_arm"] == engine_arm
        assert receipt["family_arm"] == ENGINE_TO_FAMILY_ARM[engine_arm]
        assert receipt["architecture_lock"] == IBR1_ARCHITECTURE_LOCK

    identity = paired.identity_receipt()
    assert identity["arm_mapping_to_engine"] == {
        IBR1_CTRL: S_CTRL,
        IBR1_SELF: S_SELF,
    }
    assert identity["parameters_shared_between_arms"] is False
    assert identity["optimizers_shared_between_arms"] is False
    assert identity["internal_test_opened"] is False


def test_paired_assembly_rejects_parent_init_or_seed_drift():
    torch.manual_seed(7)
    base = DummyOfficialBase()
    expected_sha = _f2_dummy_init_sha(copy.deepcopy(base))
    with pytest.raises(IBR1AssemblyContractError, match="initialization differs"):
        build_ibr1_paired_arms(
            copy.deepcopy(base),
            seed=0,
            device="cpu",
            contract=OptimizerContract(),
            parent_f2_evidence=_evidence("f" * 64),
        )
    with pytest.raises(IBR1AssemblyContractError, match="seed differs"):
        build_ibr1_paired_arms(
            copy.deepcopy(base),
            seed=1,
            device="cpu",
            contract=OptimizerContract(),
            parent_f2_evidence=_evidence(expected_sha),
        )
    drifted = dict(IBR1_FROZEN_AUX_COEFFICIENTS)
    drifted["L_cot"] = 0.02
    with pytest.raises(IBR1AssemblyContractError, match="frozen values"):
        build_ibr1_paired_arms(
            copy.deepcopy(base),
            seed=0,
            device="cpu",
            contract=OptimizerContract(),
            parent_f2_evidence=_evidence(expected_sha),
            aux_coefficients=drifted,
        )


def test_aux_coefficients_fail_closed_on_key_or_value_drift():
    with pytest.raises(IBR1AssemblyContractError, match="exactly"):
        build_ibr1_package(
            DummyOfficialBase(),
            device="cpu",
            aux_coefficients={"L_cot": 0.0195},
        )
    bad = dict(IBR1_FROZEN_AUX_COEFFICIENTS)
    bad["L_verify"] = float("nan")
    with pytest.raises(IBR1AssemblyContractError, match="finite"):
        build_ibr1_package(
            DummyOfficialBase(), device="cpu", aux_coefficients=bad
        )
