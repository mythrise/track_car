from __future__ import annotations

import copy
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.optim.optimizer as torch_optimizer
from torch import nn

from f2_experiment.assembly_model import (
    ArmExecutor,
    OptimizerContract,
    build_package,
)
from f2_experiment.runner import S_CTRL, S_SELF, RunnerRow, checkpoint_init_sha256
from f2_experiment.assembly_data import TokenHashLedger
from f2_experiment.reproducibility import CUDA_REPRODUCIBILITY_SETTINGS
from f2_experiment.support import canonical_json_sha256
from ibr1_experiment.assembly_model import (
    IBR1PairedArms,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    build_ibr1_paired_arms,
)
from ibr1_experiment.authority import (
    ASSEMBLY_PHASE_FINAL,
    ASSEMBLY_RECEIPT_CLASS,
)
from ibr1_experiment.diagnostics import EVAL_SNAPSHOTS
from ibr1_experiment.model import IBR1_ARCHITECTURE_LOCK, IBR1_FAMILY_ID
from ibr1_experiment.smoke_model import (
    IBR1SmokeContractError,
    IBR1SmokeData,
    IBR1SmokePlan,
    build_ibr1_production_smoke_plan,
    build_ibr1_smoke_plan_from_components,
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


def _paired() -> IBR1PairedArms:
    torch.manual_seed(41)
    base = DummyOfficialBase()
    expected_base = copy.deepcopy(base)
    torch.manual_seed(0)
    expected_modules = build_package(
        "SA-Hstar",
        expected_base,
        device="cpu",
        aux_coefficients=dict(IBR1_FROZEN_AUX_COEFFICIENTS),
    )
    expected_sha = checkpoint_init_sha256(expected_modules.full_state_dict())
    from ibr1_experiment.assembly_model import F2SealedInitEvidence

    evidence = F2SealedInitEvidence(
        primary_path="primary.json",
        primary_sha256="0" * 64,
        negative_adoption_path="adoption.json",
        negative_adoption_sha256="1" * 64,
        negative_seal_path="seal.json",
        negative_seal_sha256="2" * 64,
        smoke_summary_path="summary.json",
        smoke_summary_sha256="3" * 64,
        seed=0,
        checkpoint_init_sha256=expected_sha,
    )
    return build_ibr1_paired_arms(
        base,
        seed=0,
        device="cpu",
        contract=OptimizerContract(),
        parent_f2_evidence=evidence,
    )


def _row(index: int) -> RunnerRow:
    return RunnerRow(
        original_row_index=index,
        sequence_id=f"sequence-{index // 32}",
        frame_idx=index % 32,
        mirrored=False,
        logged_prev_action=(0.0, 0.0, 0.0),
        target_actions=torch.zeros(8, 3),
        observation=object(),
        aux_targets={},
    )


def _data() -> IBR1SmokeData:
    smoke = tuple(_row(index) for index in range(256))
    evaluation = tuple(_row(10_000 + index) for index in range(512))
    ledger = TokenHashLedger({"tokens/example.pt": "a" * 64})
    return IBR1SmokeData(
        smoke_rows=smoke,
        eval_rows=evaluation,
        eval_raw_rows=tuple({"original_row_index": row.original_row_index} for row in evaluation),
        smoke_strafe_reset_original_indices=frozenset(),
        eval_strafe_reset_original_indices=frozenset(),
        smoke_expected_static_reset_original_indices=frozenset(range(12)),
        eval_expected_static_reset_original_indices=frozenset(range(10_000, 10_028)),
        token_ledger=ledger,
        token_ledger_binding={
            "anchor": "unit-test",
            "sha256": ledger.ledger_sha256,
            "file_count": ledger.token_files,
        },
        support_order={
            "SMK-TRAIN": tuple(row.original_row_index for row in smoke),
            "EVAL-FIX": tuple(row.original_row_index for row in evaluation),
        },
    )


def _receipt() -> dict[str, object]:
    return {
        "schema_version": 1,
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
        "family_id": IBR1_FAMILY_ID,
        "architecture_lock": IBR1_ARCHITECTURE_LOCK,
        "phase": ASSEMBLY_PHASE_FINAL,
        "lambda_freeze_binding": {"path": "freeze.json", "sha256": "a" * 64},
        "receipt_payload_sha256": "b" * 64,
        "formal_training_authorized": False,
        "internal_test": "sealed",
        "internal_test_opened": False,
    }


def _production_receipt_for_data(
    tmp_path: Path, data: IBR1SmokeData
) -> dict[str, object]:
    document = _receipt()
    document["support_binding"] = {
        "observation": {
            "supports": {
                name: {
                    "ordered_original_indices": list(data.support_order[name])
                }
                for name in ("SMK-TRAIN", "EVAL-FIX")
            }
        }
    }
    document["asset_binding"] = {
        "observation": {
            "base_hf": {"path": str(tmp_path / "base-hf")},
            "token_ledger_sha256": data.token_ledger.ledger_sha256,
            "token_ledger_file_count": data.token_ledger.token_files,
        }
    }
    return document


def _patch_unit_support_shas(monkeypatch, smoke_model, data: IBR1SmokeData):
    expectations = dict(smoke_model.SUPPORT_EXPECTATIONS)
    for name in ("SMK-TRAIN", "EVAL-FIX"):
        expectations[name] = replace(
            expectations[name],
            sha256=canonical_json_sha256(list(data.support_order[name])),
        )
    monkeypatch.setattr(smoke_model, "SUPPORT_EXPECTATIONS", expectations)


def _plan(
    tmp_path: Path,
    *,
    paired: IBR1PairedArms | None = None,
    data: IBR1SmokeData | None = None,
):
    paired = paired or _paired()
    document = _receipt()
    binding = {
        "path": "final.json",
        "sha256": "c" * 64,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }
    return build_ibr1_smoke_plan_from_components(
        project_root=tmp_path,
        final_assembly_receipt_path=tmp_path / "final.json",
        final_assembly_receipt=document,
        final_assembly_receipt_binding=binding,
        paired_arms=paired,
        data=data or _data(),
        cuda_reproducibility=None,
    )


def _plan_init_kwargs(plan: IBR1SmokePlan) -> dict[str, object]:
    return {
        item.name: getattr(plan, item.name)
        for item in fields(IBR1SmokePlan)
        if item.init
    }


def _register_unit_production_plan(component, smoke_model):
    constructor_kwargs = _plan_init_kwargs(component)
    constructor_kwargs["cuda_reproducibility"] = {
        **dict(CUDA_REPRODUCIBILITY_SETTINGS),
        "torch_version": "unit-test-torch",
        "cuda_runtime": "unit-test-cuda",
    }
    provenance = smoke_model._ProductionSmokeProvenance(
        key=smoke_model._PRODUCTION_SMOKE_PROVENANCE_KEY,
        project_root=component.project_root,
        final_receipt_path=component.final_assembly_receipt_path,
        final_receipt_file_sha256="c" * 64,
        final_receipt_payload_sha256="b" * 64,
        final_receipt_analysis_class=ASSEMBLY_RECEIPT_CLASS,
        smoke_data_object_id=id(component.data),
        smoke_data_identity_sha256="d" * 64,
    )
    production = smoke_model._IBR1ProductionSmokePlan(**constructor_kwargs)
    smoke_model._register_production_plan(production, provenance)
    return production


def test_component_assembly_is_pure_and_binds_exact_disjoint_arms(tmp_path: Path):
    paired = _paired()
    tensor_snapshots = {
        engine_arm: {
            name: tensor.detach().clone()
            for name, tensor in paired.arms[engine_arm]
            .modules.full_state_dict()
            .items()
        }
        for engine_arm in (S_CTRL, S_SELF)
    }
    optimizer_snapshots = {
        engine_arm: copy.deepcopy(paired.arms[engine_arm].optimizer.state_dict())
        for engine_arm in (S_CTRL, S_SELF)
    }
    plan = _plan(tmp_path, paired=paired)
    try:
        assert plan.geometry_collector.training_records == []
        assert plan.geometry_collector.eval_records == []
        assert plan.gradient_collector.gradient_records == []
        assert plan.gradient_collector.optimizer_records == []
        assert plan.identity_receipt()["callbacks_executed_during_assembly"] is False
        assert plan.identity_receipt()["authority_eligible"] is False
        assert plan.authority_eligible is False
        assert plan.production_context is False
        reset_sets = plan.identity_receipt()["reset_sets"]
        assert reset_sets["SMK-TRAIN"]["strafe"] == {
            "count": 0,
            "sorted_original_indices": [],
            "canonical_sha256": canonical_json_sha256([]),
        }
        assert reset_sets["SMK-TRAIN"]["expected_static"] == {
            "count": 12,
            "sorted_original_indices": list(range(12)),
            "canonical_sha256": canonical_json_sha256(list(range(12))),
        }
        assert reset_sets["EVAL-FIX"]["expected_static"] == {
            "count": 28,
            "sorted_original_indices": list(range(10_000, 10_028)),
            "canonical_sha256": canonical_json_sha256(
                list(range(10_000, 10_028))
            ),
        }
        assert plan.formal_training_authorized is False
        assert plan.internal_test_opened is False
        assert set(plan.public_arms()) == {"IBR1-CTRL", "IBR1-SELF"}

        ctrl = plan.arms[S_CTRL]
        self_arm = plan.arms[S_SELF]
        assert ctrl.modules is plan.paired_arms.arms[S_CTRL].modules
        assert self_arm.modules is plan.paired_arms.arms[S_SELF].modules
        assert ctrl.optimizer is not self_arm.optimizer
        assert ctrl.executor.g6 is plan.g6
        assert self_arm.executor.g6 is None
        assert ctrl.optimizer_diagnostics.modules is ctrl.modules
        assert self_arm.optimizer_diagnostics.modules is self_arm.modules
        assert ctrl.callbacks is not self_arm.callbacks
        assert ctrl.executor is not self_arm.executor
        assert ctrl.optimizer.state_dict()["state"] == {}
        assert self_arm.optimizer.state_dict()["state"] == {}
        for engine_arm in (S_CTRL, S_SELF):
            assert paired.arms[engine_arm].optimizer.state_dict() == (
                optimizer_snapshots[engine_arm]
            )
            live = paired.arms[engine_arm].modules.full_state_dict()
            assert set(live) == set(tensor_snapshots[engine_arm])
            assert all(
                torch.equal(live[name], tensor_snapshots[engine_arm][name])
                for name in live
            )
    finally:
        plan.close()


def test_component_factory_never_calls_forward_callback_or_step(
    tmp_path: Path, monkeypatch
):
    paired = _paired()
    calls: list[str] = []
    for method_name in (
        "feature_forward",
        "aux_forward",
        "head_forward",
        "track_loss",
        "backward",
        "optimizer_step",
    ):
        original = getattr(ArmExecutor, method_name)

        def guarded(self, *args, _name=method_name, _original=original, **kwargs):
            calls.append(_name)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ArmExecutor, method_name, guarded)

    original_step = torch.optim.AdamW.step

    def guarded_step(self, *args, **kwargs):
        calls.append("adamw.step")
        return original_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", guarded_step)
    forward_handles = []
    for engine_arm in (S_CTRL, S_SELF):
        modules = paired.arms[engine_arm].modules
        for module_name, module in (
            ("base", modules.base),
            ("adapter", modules.adapter),
            ("model", modules.model),
        ):
            forward_handles.append(
                module.register_forward_pre_hook(
                    lambda _module, _inputs, name=(engine_arm, module_name): (
                        calls.append(f"forward:{name[0]}:{name[1]}")
                    )
                )
            )
    plan = _plan(tmp_path, paired=paired)
    try:
        assert calls == []
    finally:
        plan.close()
        for handle in forward_handles:
            handle.remove()


def test_diagnostics_hooks_are_removed_idempotently(tmp_path: Path):
    plan = _plan(tmp_path)
    ctrl_optimizer = plan.arms[S_CTRL].optimizer
    self_optimizer = plan.arms[S_SELF].optimizer
    assert len(ctrl_optimizer._optimizer_step_pre_hooks) == 1
    assert len(ctrl_optimizer._optimizer_step_post_hooks) == 1
    assert len(self_optimizer._optimizer_step_pre_hooks) == 1
    assert len(self_optimizer._optimizer_step_post_hooks) == 1
    plan.close()
    plan.close()
    assert len(ctrl_optimizer._optimizer_step_pre_hooks) == 0
    assert len(ctrl_optimizer._optimizer_step_post_hooks) == 0
    assert len(self_optimizer._optimizer_step_pre_hooks) == 0
    assert len(self_optimizer._optimizer_step_post_hooks) == 0


def test_eval_predictor_factory_is_fresh_and_snapshot_scoped(tmp_path: Path):
    plan = _plan(tmp_path)
    try:
        self_factory = plan.arms[S_SELF].eval_predictor_factory
        first = self_factory("update0_IBR1-SELF")
        second = self_factory("update128_IBR1-SELF")
        assert first.raw_predictor is not second.raw_predictor
        assert first.raw_predictor.arm is plan.arms[S_SELF].modules
        assert second.raw_predictor.arm is plan.arms[S_SELF].modules
        with pytest.raises(IBR1SmokeContractError, match="not valid"):
            plan.arms[S_CTRL].eval_predictor_factory("update0_IBR1-SELF")
        assert set(EVAL_SNAPSHOTS) == {
            "update0_IBR1-SELF",
            "update128_IBR1-CTRL",
            "update128_IBR1-SELF",
        }
    finally:
        plan.close()


def test_checkpoint_identity_accessor_returns_live_writer_arguments(tmp_path: Path):
    plan = _plan(tmp_path)
    try:
        target = plan.arms[S_CTRL].checkpoint_identity(0)
        target_128 = plan.arms[S_CTRL].checkpoint_identity(128)
        assert target.arm_assembly is plan.paired_arms.arms[S_CTRL]
        assert target.writer_kwargs()["paired_arms"] is plan.paired_arms
        assert target.writer_kwargs()["arm_assembly"] is target.arm_assembly
        assert target_128.u_pre == 128
        assert target.identity_receipt()["checkpoint_written"] is False
        assert not (tmp_path / "final.json").exists()
        with pytest.raises(IBR1SmokeContractError, match="one of"):
            plan.arms[S_CTRL].checkpoint_identity(1)
    finally:
        plan.close()


def test_smoke_data_fails_closed_on_order_reset_or_ledger_drift():
    data = _data()
    with pytest.raises(IBR1SmokeContractError, match="support order keys"):
        IBR1SmokeData(
            **{
                **data.__dict__,
                "support_order": {
                    "EVAL-FIX": data.support_order["EVAL-FIX"],
                    "SMK-TRAIN": data.support_order["SMK-TRAIN"],
                },
            }
        )
    with pytest.raises(IBR1SmokeContractError, match="static reset count"):
        IBR1SmokeData(
            **{
                **data.__dict__,
                "smoke_expected_static_reset_original_indices": frozenset(),
            }
        )
    with pytest.raises(IBR1SmokeContractError, match="ledger object"):
        IBR1SmokeData(
            **{
                **data.__dict__,
                "token_ledger_binding": {
                    "anchor": "unit-test",
                    "sha256": "f" * 64,
                    "file_count": data.token_ledger.token_files,
                },
            }
        )


def test_component_rejects_shared_arm_modules_or_optimizer(tmp_path: Path):
    paired = _paired()
    ctrl = paired.arms[S_CTRL]
    self_arm = paired.arms[S_SELF]
    forged_cases = (
        replace(paired, arms={S_CTRL: ctrl, S_SELF: ctrl}),
        replace(
            paired,
            arms={
                S_CTRL: ctrl,
                S_SELF: replace(self_arm, modules=ctrl.modules),
            },
        ),
        replace(
            paired,
            arms={
                S_CTRL: ctrl,
                S_SELF: replace(self_arm, optimizer=ctrl.optimizer),
            },
        ),
    )
    for forged in forged_cases:
        with pytest.raises(IBR1SmokeContractError):
            _plan(tmp_path, paired=forged)


def test_component_rejects_shared_descendant_named_module(tmp_path: Path):
    paired = _paired()
    ctrl_model = paired.arms[S_CTRL].modules.model
    self_model = paired.arms[S_SELF].modules.model
    self_model.action_head.trunk[1] = ctrl_model.action_head.trunk[1]
    with pytest.raises(IBR1SmokeContractError, match="named_modules"):
        _plan(tmp_path, paired=paired)


@pytest.mark.parametrize("alias_kind", ["parameter_data", "partial_view", "buffer_data"])
def test_component_rejects_cross_arm_storage_aliases(
    tmp_path: Path, alias_kind: str
):
    paired = _paired()
    ctrl = paired.arms[S_CTRL].modules
    self_arm = paired.arms[S_SELF].modules
    if alias_kind == "parameter_data":
        self_arm.model.action_head.forward_branch.bias.data = (
            ctrl.model.action_head.forward_branch.bias.data
        )
    elif alias_kind == "partial_view":
        ctrl_bias = ctrl.model.fusion.head_norm.bias
        self_bias = self_arm.model.fusion.head_norm.bias
        assert torch.count_nonzero(ctrl_bias).item() == 0
        backing = torch.zeros(
            ctrl_bias.numel() + 1,
            dtype=ctrl_bias.dtype,
            device=ctrl_bias.device,
        )
        ctrl_bias.data = backing[:-1].view_as(ctrl_bias)
        self_bias.data = backing[1:].view_as(self_bias)
        assert ctrl_bias.data_ptr() != self_bias.data_ptr()
    else:
        self_arm.adapter.expert_future_leak_count.data = (
            ctrl.adapter.expert_future_leak_count.data
        )
    with pytest.raises(IBR1SmokeContractError, match="storage"):
        _plan(tmp_path, paired=paired)


@pytest.mark.parametrize(
    "register_hook",
    [
        torch_optimizer.register_optimizer_step_pre_hook,
        torch_optimizer.register_optimizer_step_post_hook,
    ],
)
def test_component_rejects_process_global_optimizer_hooks(
    tmp_path: Path, register_hook
):
    handle = register_hook(lambda *args, **kwargs: None)
    try:
        with pytest.raises(IBR1SmokeContractError, match="global optimizer"):
            _plan(tmp_path)
    finally:
        handle.remove()


def test_component_factory_cannot_claim_production_authority(tmp_path: Path):
    paired = _paired()
    document = _receipt()
    binding = {
        "path": "final.json",
        "sha256": "c" * 64,
        "receipt_payload_sha256": document["receipt_payload_sha256"],
        "analysis_class": ASSEMBLY_RECEIPT_CLASS,
    }
    with pytest.raises(TypeError, match="_production_context"):
        build_ibr1_smoke_plan_from_components(
            project_root=tmp_path,
            final_assembly_receipt_path=tmp_path / "final.json",
            final_assembly_receipt=document,
            final_assembly_receipt_binding=binding,
            paired_arms=paired,
            data=_data(),
            cuda_reproducibility=None,
            _production_context=True,
        )


def test_component_plan_capability_cannot_be_upgraded_or_copied(
    tmp_path: Path
):
    from ibr1_experiment import smoke_model

    plan = _plan(tmp_path)
    replaced_plan = None
    try:
        for attribute in (
            "_production_provenance",
            "_production_class_marker",
            "authority_eligible",
            "production_context",
        ):
            with pytest.raises(AttributeError):
                setattr(plan, attribute, object())
            with pytest.raises(AttributeError):
                object.__setattr__(plan, attribute, object())

        with pytest.raises(TypeError):
            plan.__class__ = smoke_model._IBR1ProductionSmokePlan

        assert copy.copy(plan) is plan
        assert copy.deepcopy(plan) is plan
        assert plan.identity_receipt()["authority_eligible"] is False

        replaced_plan = replace(plan)
        assert type(replaced_plan) is IBR1SmokePlan
        assert replaced_plan is not plan
        assert replaced_plan.identity_receipt()["authority_eligible"] is False
        with pytest.raises((TypeError, ValueError)):
            replace(plan, _production_provenance=object())

        constructor_kwargs = _plan_init_kwargs(plan)
        with pytest.raises(TypeError):
            IBR1SmokePlan(
                **constructor_kwargs,
                _production_provenance=object(),
            )
    finally:
        if replaced_plan is not None:
            replaced_plan.close()
        plan.close()


def test_registered_production_plan_revalidates_and_close_stays_idempotent(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    component = _plan(tmp_path)
    production = None
    live = {"valid": True}
    validations: list[str] = []

    def validate_live(*args, **kwargs):
        del args, kwargs
        validations.append("validate")
        if not live["valid"]:
            raise IBR1SmokeContractError("unit live provenance drift")

    monkeypatch.setattr(
        smoke_model, "_validate_production_smoke_provenance", validate_live
    )
    monkeypatch.setattr(smoke_model, "IBR1_SMOKE_DEVICE", "cpu")
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    constructor_kwargs = _plan_init_kwargs(component)
    constructor_kwargs["cuda_reproducibility"] = {
        **dict(CUDA_REPRODUCIBILITY_SETTINGS),
        "torch_version": "unit-test-torch",
        "cuda_runtime": "unit-test-cuda",
    }
    provenance = smoke_model._ProductionSmokeProvenance(
        key=smoke_model._PRODUCTION_SMOKE_PROVENANCE_KEY,
        project_root=component.project_root,
        final_receipt_path=component.final_assembly_receipt_path,
        final_receipt_file_sha256="c" * 64,
        final_receipt_payload_sha256="b" * 64,
        final_receipt_analysis_class=ASSEMBLY_RECEIPT_CLASS,
        smoke_data_object_id=id(component.data),
        smoke_data_identity_sha256="d" * 64,
    )
    try:
        production = smoke_model._IBR1ProductionSmokePlan(
            **constructor_kwargs
        )
        smoke_model._register_production_plan(production, provenance)
        registration_validations = len(validations)
        assert registration_validations == 1
        assert production.identity_receipt()["authority_eligible"] is True
        assert len(validations) > registration_validations
        before_arms = len(validations)
        assert production.arms is component.arms
        assert len(validations) == before_arms + 1

        live["valid"] = False
        with pytest.raises(IBR1SmokeContractError, match="live provenance drift"):
            production.identity_receipt()
        with pytest.raises(IBR1SmokeContractError, match="live provenance drift"):
            _ = production.arms

        production.close()
        production.close()
        for engine_arm in (S_CTRL, S_SELF):
            optimizer = production._arms[engine_arm].optimizer
            assert len(optimizer._optimizer_step_pre_hooks) == 0
            assert len(optimizer._optimizer_step_post_hooks) == 0
    finally:
        if production is not None:
            smoke_model._PRODUCTION_PLAN_PROVENANCE.pop(production, None)
            production.close()
        component.close()


def test_registered_production_rejects_all_plan_field_replacements(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    component = _plan(tmp_path)
    production = None
    monkeypatch.setattr(
        smoke_model, "_validate_production_smoke_provenance", lambda *args: None
    )
    monkeypatch.setattr(smoke_model, "IBR1_SMOKE_DEVICE", "cpu")
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    try:
        production = _register_unit_production_plan(component, smoke_model)
        replacements = {
            "project_root": str(tmp_path / "other-root"),
            "final_assembly_receipt_path": str(tmp_path / "other.json"),
            "final_assembly_receipt": {},
            "final_assembly_receipt_binding": {},
            "paired_arms": object(),
            "data": object(),
            "geometry_collector": object(),
            "gradient_collector": object(),
            "g6": object(),
            "_arms": {},
            "optimizer_contract": object(),
            "seed": 1,
            "device": "cuda:1",
            "checkpoint_init_sha256": "e" * 64,
            "cuda_reproducibility": None,
            "base_load_report": {"drift": True},
            "formal_training_authorized": True,
            "internal_test": "opened",
            "internal_test_opened": True,
        }
        for setter in (setattr, object.__setattr__):
            for name, replacement in replacements.items():
                original = getattr(production, name)
                setter(production, name, replacement)
                with pytest.raises(
                    IBR1SmokeContractError,
                    match=rf"field '{name}' drifted",
                ):
                    production.identity_receipt()
                object.__setattr__(production, name, original)
        assert production.authority_eligible is True
    finally:
        if production is not None:
            production.close()
            smoke_model._PRODUCTION_PLAN_PROVENANCE.pop(production, None)
            smoke_model._PRODUCTION_PLAN_REGISTRATIONS.pop(production, None)
        component.close()


def test_registered_production_field_drift_fails_every_public_entry(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    component = _plan(tmp_path)
    production = None
    monkeypatch.setattr(
        smoke_model, "_validate_production_smoke_provenance", lambda *args: None
    )
    monkeypatch.setattr(smoke_model, "IBR1_SMOKE_DEVICE", "cpu")
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    try:
        production = _register_unit_production_plan(component, smoke_model)
        original_data = production.data
        entries = (
            lambda: production.identity_receipt(),
            lambda: production.production_context,
            lambda: production.authority_eligible,
            lambda: production.arms,
            lambda: production.smoke_rows,
            lambda: production.eval_rows,
            lambda: production.eval_raw_rows,
            lambda: production.g6_update,
        )
        for setter in (setattr, object.__setattr__):
            setter(production, "data", object())
            for enter in entries:
                with pytest.raises(
                    IBR1SmokeContractError, match="field 'data' drifted"
                ):
                    enter()
            object.__setattr__(production, "data", original_data)
    finally:
        if production is not None:
            production.close()
            smoke_model._PRODUCTION_PLAN_PROVENANCE.pop(production, None)
            smoke_model._PRODUCTION_PLAN_REGISTRATIONS.pop(production, None)
        component.close()


def test_registered_production_close_uses_original_registered_hooks(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    component = _plan(tmp_path)
    production = None
    monkeypatch.setattr(
        smoke_model, "_validate_production_smoke_provenance", lambda *args: None
    )
    monkeypatch.setattr(smoke_model, "IBR1_SMOKE_DEVICE", "cpu")
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    try:
        production = _register_unit_production_plan(component, smoke_model)
        original_optimizers = tuple(
            production._arms[engine_arm].optimizer
            for engine_arm in (S_CTRL, S_SELF)
        )
        production._arms = {}
        object.__setattr__(production, "_closed", True)

        production.close()
        production.close()

        for optimizer in original_optimizers:
            assert len(optimizer._optimizer_step_pre_hooks) == 0
            assert len(optimizer._optimizer_step_post_hooks) == 0
    finally:
        if production is not None:
            production.close()
            smoke_model._PRODUCTION_PLAN_PROVENANCE.pop(production, None)
            smoke_model._PRODUCTION_PLAN_REGISTRATIONS.pop(production, None)
        component.close()


@pytest.mark.parametrize("drift", ["lr", "betas", "group_order"])
def test_component_rebuilds_and_rejects_optimizer_contract_drift(
    tmp_path: Path, drift: str
):
    paired = _paired()
    optimizer = paired.arms[S_CTRL].optimizer
    if drift == "lr":
        optimizer.param_groups[0]["lr"] = 9e-3
    elif drift == "betas":
        optimizer.param_groups[0]["betas"] = (0.8, 0.99)
    else:
        optimizer.param_groups.reverse()
    with pytest.raises(IBR1SmokeContractError, match="AdamW"):
        _plan(tmp_path, paired=paired)


def test_component_rejects_declared_device_drift(tmp_path: Path):
    paired = replace(_paired(), device="cuda:0")
    with pytest.raises(IBR1SmokeContractError, match="declared device"):
        _plan(tmp_path, paired=paired)


def test_component_rejects_double_optimizer_hook_wiring(tmp_path: Path):
    paired = _paired()
    first = _plan(tmp_path, paired=paired)
    try:
        with pytest.raises(IBR1SmokeContractError, match="double wiring"):
            _plan(tmp_path, paired=paired)
    finally:
        first.close()


def test_production_cpu_fails_before_base_loader(tmp_path: Path, monkeypatch):
    from ibr1_experiment import smoke_model

    receipt_path = tmp_path / "final.json"
    receipt_path.write_text("{}", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: _receipt(),
    )
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        smoke_model,
        "load_base_checkpoint",
        lambda *args, **kwargs: calls.append("base") or (None, {}),
    )
    monkeypatch.setattr(
        smoke_model,
        "load_ibr1_smoke_data",
        lambda *args, **kwargs: calls.append("data") or _data(),
    )
    with pytest.raises(IBR1SmokeContractError, match="requires CUDA"):
        build_ibr1_production_smoke_plan(tmp_path, receipt_path)
    assert calls == []


def test_mocked_production_success_uses_fixed_authority_context(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    receipt_path = tmp_path / "final.json"
    receipt_path.write_text("{}", encoding="utf-8")
    raw_data = _data()
    data = replace(
        raw_data,
        token_ledger_binding={
            "anchor": "final_assembly_receipt.asset_binding.observation",
            "sha256": raw_data.token_ledger.ledger_sha256,
            "file_count": raw_data.token_ledger.token_files,
        },
    )
    document = _production_receipt_for_data(tmp_path, data)
    _patch_unit_support_shas(monkeypatch, smoke_model, data)
    cuda_receipt = {
        **dict(CUDA_REPRODUCIBILITY_SETTINGS),
        "torch_version": "unit-test-torch",
        "cuda_runtime": "unit-test-cuda",
    }
    calls: list[str] = []
    base = object()
    evidence = SimpleNamespace(checkpoint_init_sha256="d" * 64)
    paired = SimpleNamespace(checkpoint_init_sha256="d" * 64)
    sentinel = object()
    captured: dict[str, object] = {}

    def build_paired(observed_base, **kwargs):
        calls.append("paired")
        assert observed_base is base
        captured.update(kwargs)
        return paired

    def build_components(**kwargs):
        calls.append("components")
        captured.update({f"component_{key}": value for key, value in kwargs.items()})
        return sentinel

    def load_data(root, observed_document):
        calls.append("data")
        return smoke_model._attach_smoke_data_load_provenance(
            Path(root).resolve(),
            observed_document,
            data,
            train_sha256=smoke_model.FROZEN_TRAIN_SHA256,
        )

    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: calls.append("verify") or document,
    )
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        smoke_model,
        "configure_cuda_reproducibility",
        lambda: calls.append("cuda") or cuda_receipt,
    )
    monkeypatch.setattr(
        smoke_model,
        "load_ibr1_smoke_data",
        load_data,
    )
    monkeypatch.setattr(
        smoke_model,
        "load_base_checkpoint",
        lambda path: calls.append("base") or (base, {"path": str(path)}),
    )
    monkeypatch.setattr(
        smoke_model,
        "read_sealed_f2_init_evidence",
        lambda root: calls.append("evidence") or evidence,
    )
    monkeypatch.setattr(smoke_model, "build_ibr1_paired_arms", build_paired)
    monkeypatch.setattr(
        smoke_model,
        "build_ibr1_smoke_plan_from_components",
        lambda **kwargs: pytest.fail(
            f"production called public component factory: {kwargs}"
        ),
    )
    monkeypatch.setattr(
        smoke_model,
        "_build_ibr1_smoke_plan_from_components",
        build_components,
    )

    result = build_ibr1_production_smoke_plan(tmp_path, receipt_path)
    assert result is sentinel
    assert calls == [
        "verify",
        "cuda",
        "data",
        "base",
        "evidence",
        "paired",
        "components",
    ]
    assert captured["seed"] == 0
    assert captured["device"] == "cuda:0"
    assert isinstance(captured["contract"], OptimizerContract)
    assert captured["parent_f2_evidence"] is evidence
    assert captured["component_paired_arms"] is paired
    assert captured["component_data"]._load_provenance is not None
    provenance = captured["component_production_provenance"]
    assert provenance.key is smoke_model._PRODUCTION_SMOKE_PROVENANCE_KEY


def test_production_rejects_missing_receipt_even_with_fake_verifier(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    receipt_path = tmp_path / "missing-final.json"
    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: _receipt(),
    )
    with pytest.raises(IBR1SmokeContractError, match="missing"):
        build_ibr1_production_smoke_plan(tmp_path, receipt_path)


def test_production_rejects_global_optimizer_post_hook_before_authority_read(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    authority_calls: list[str] = []
    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: authority_calls.append("verify") or _receipt(),
    )
    handle = torch_optimizer.register_optimizer_step_post_hook(
        lambda *args, **kwargs: None
    )
    try:
        with pytest.raises(IBR1SmokeContractError, match="global optimizer"):
            build_ibr1_production_smoke_plan(
                tmp_path, tmp_path / "final.json"
            )
    finally:
        handle.remove()
    assert authority_calls == []


def test_production_rejects_fake_data_without_loader_provenance(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    receipt_path = tmp_path / "final.json"
    receipt_path.write_text("{}", encoding="utf-8")
    document = _receipt()
    document["asset_binding"] = {
        "observation": {"base_hf": {"path": str(tmp_path / "base-hf")}}
    }
    base_calls: list[str] = []
    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        smoke_model,
        "configure_cuda_reproducibility",
        lambda: {
            **dict(CUDA_REPRODUCIBILITY_SETTINGS),
            "torch_version": "unit-test-torch",
            "cuda_runtime": "unit-test-cuda",
        },
    )
    monkeypatch.setattr(smoke_model, "load_ibr1_smoke_data", lambda *args: _data())
    monkeypatch.setattr(
        smoke_model,
        "load_base_checkpoint",
        lambda *args: base_calls.append("base"),
    )
    with pytest.raises(IBR1SmokeContractError, match="loader provenance"):
        build_ibr1_production_smoke_plan(tmp_path, receipt_path)
    assert base_calls == []


def test_production_rejects_reset_drift_after_frozen_loading(
    tmp_path: Path, monkeypatch
):
    from ibr1_experiment import smoke_model

    receipt_path = tmp_path / "final.json"
    receipt_path.write_text("{}", encoding="utf-8")
    raw_data = _data()
    data = replace(
        raw_data,
        token_ledger_binding={
            "anchor": "final_assembly_receipt.asset_binding.observation",
            "sha256": raw_data.token_ledger.ledger_sha256,
            "file_count": raw_data.token_ledger.token_files,
        },
    )
    document = _production_receipt_for_data(tmp_path, data)
    _patch_unit_support_shas(monkeypatch, smoke_model, data)
    loaded = smoke_model._attach_smoke_data_load_provenance(
        tmp_path.resolve(),
        document,
        data,
        train_sha256=smoke_model.FROZEN_TRAIN_SHA256,
    )
    drifted = replace(
        loaded,
        smoke_expected_static_reset_original_indices=frozenset(range(1, 13)),
    )
    monkeypatch.setattr(
        smoke_model,
        "verify_assembly_receipt",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(smoke_model.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        smoke_model,
        "configure_cuda_reproducibility",
        lambda: {
            **dict(CUDA_REPRODUCIBILITY_SETTINGS),
            "torch_version": "unit-test-torch",
            "cuda_runtime": "unit-test-cuda",
        },
    )
    monkeypatch.setattr(
        smoke_model, "load_ibr1_smoke_data", lambda *args: drifted
    )
    with pytest.raises(IBR1SmokeContractError, match="reset sets drifted"):
        build_ibr1_production_smoke_plan(tmp_path, receipt_path)
