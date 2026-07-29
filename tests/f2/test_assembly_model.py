from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f2_experiment.assembly import CalRowAudit, SmokeAssemblyPlan
from f2_experiment.assembly_data import (
    AuxTargetPacket,
    COARSE_TOKEN_COUNT,
    FINE_TOKEN_COUNT,
    FROZEN_BASE_HF_DIR_DEFAULT,
    ObservationPacket,
    VISION_FEATURE_DIM,
    build_train_token_ledger,
    observation_packet_from_fields,
    verify_frozen_assets,
)
from f2_experiment.assembly_model import (
    ArmExecutor,
    CalRowAuditor,
    ConfidenceTIMObservationAdapter,
    EvalRowPredictor,
    F2AssemblyContractError,
    G6Instrument,
    NullTIMObservationAdapter,
    OptimizerContract,
    PACKAGE_AUX_COMPONENTS,
    PACKAGE_NAMES,
    SMOKE_PACKAGE,
    audit_cal_row,
    build_arm_callbacks,
    build_arm_optimizer,
    build_eval_row_predictor,
    build_eval_row_predictor_from_checkpoint,
    build_package,
    build_paired_arms,
    build_production_smoke_plan,
    make_backward_callback,
    make_optimizer_step_callback,
)
from f2_experiment.evaluation import evaluate_g7
from f2_experiment.model import AP2_HORIZON
from f2_experiment.opentrack_adapter import (
    F2ObservationContractError,
    OpenTrackVLAF2ObservationAdapter,
)
from f2_experiment.runner import (
    BackwardEvent,
    HeadEvent,
    OptimizerUpdateEvent,
    RowEvent,
    RunnerRow,
    RunnerTelemetryHooks,
    S_CTRL,
    S_SELF,
    SMOKE_ROWS,
    checkpoint_init_sha256,
    run_paired_smoke,
)


class DummyLLM(nn.Module):
    """Linear stand-in with cross-position pooling.

    The mean-pool term matters: it makes the last-position ``h_act`` depend on
    the visual token positions, mirroring real attention so that the track
    loss can reach ``base.proj`` the way it does with the official LLM.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mix = nn.Linear(d_model, d_model, bias=False)
        self.last_sequence_length: int | None = None

    @property
    def dtype(self) -> torch.dtype:
        return self.mix.weight.dtype

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
    ) -> SimpleNamespace:
        assert attention_mask.shape == inputs_embeds.shape[:2]
        assert output_hidden_states is True
        assert use_cache is False
        self.last_sequence_length = int(inputs_embeds.shape[1])
        mixed = self.mix(inputs_embeds)
        pooled = mixed.mean(dim=1, keepdim=True)
        return SimpleNamespace(last_hidden_state=mixed + pooled)


class DummyOfficialBase(nn.Module):
    """Official-interface stand-in mirroring the adapter test double."""

    def __init__(
        self, input_dim: int = VISION_FEATURE_DIM, d_model: int = 8
    ) -> None:
        super().__init__()
        self.D = d_model
        self.cfg = SimpleNamespace(use_angle_tvi=False)
        self.proj = nn.Linear(input_dim, d_model)
        self.llm = DummyLLM(d_model)
        self.act_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.text_token = nn.Parameter(torch.randn(1, 2, d_model) * 0.02)

    def _interleave_tvi(
        self,
        tokens: torch.Tensor,
        t_idx: torch.Tensor,
        kind_id: int,
        yaw_per_frame: torch.Tensor | None = None,
        use_angle: bool = False,
    ) -> torch.Tensor:
        assert t_idx.shape[:2] == tokens.shape[:2]
        assert kind_id in (0, 1)
        assert yaw_per_frame is None
        assert use_angle is False
        return tokens

    def _embed_text(
        self, instructions: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(instructions)
        return (
            self.text_token.expand(batch_size, -1, -1).to(device),
            torch.ones(batch_size, 2, dtype=torch.long, device=device),
        )


AUX_COEFFICIENTS = {
    "SA-B0": {},
    "SA-B1": {"L_cot": 0.5},
    "SA-Hstar": {"L_cot": 0.5, "L_future": 0.5, "L_verify": 0.5},
}


def _observation_fields(seed: int = 0) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "coarse_tokens": torch.randn(
            COARSE_TOKEN_COUNT, VISION_FEATURE_DIM, generator=generator
        ),
        "coarse_tidx": torch.zeros(COARSE_TOKEN_COUNT, dtype=torch.long),
        "fine_tokens": torch.randn(
            FINE_TOKEN_COUNT, VISION_FEATURE_DIM, generator=generator
        ),
        "fine_tidx": torch.ones(FINE_TOKEN_COUNT, dtype=torch.long),
        "instruction": "follow the person",
    }


def _observation(seed: int = 0) -> ObservationPacket:
    return observation_packet_from_fields(_observation_fields(seed))


def _aux_targets() -> dict[str, torch.Tensor]:
    targets: dict[str, torch.Tensor] = {
        "polar_theta_idx": torch.zeros(1, dtype=torch.long),
        "polar_dist_idx": torch.zeros(1, dtype=torch.long),
        "polar_invalid": torch.zeros(1),
    }
    for horizon in (4, 8, 16):
        targets[f"fut_valid_{horizon}"] = torch.ones(1, dtype=torch.bool)
        targets[f"fut_vis_{horizon}"] = torch.ones(1)
        targets[f"fut_theta_idx_{horizon}"] = torch.zeros(1, dtype=torch.long)
        targets[f"fut_dist_idx_{horizon}"] = torch.zeros(1, dtype=torch.long)
    return targets


def _build(package: str, *, seed: int = 11):
    torch.manual_seed(seed)
    base = DummyOfficialBase()
    return build_package(
        package,
        base,
        device="cpu",
        aux_coefficients=AUX_COEFFICIENTS[package],
    )


def _row_event(*, reset: bool = True, row_position: int = 0) -> RowEvent:
    return RowEvent(
        arm=S_CTRL,
        row_position=row_position,
        original_row_index=1000 + row_position,
        u_pre=row_position // 2,
        reset=reset,
        reset_reasons=("stream_first",) if reset else (),
    )


def _head_event(branch: str, *, row_position: int = 0) -> HeadEvent:
    return HeadEvent(
        arm=S_CTRL,
        row_position=row_position,
        original_row_index=1000 + row_position,
        u_pre=row_position // 2,
        branch=branch,
        prev_source="logged",
    )


def _update_event(u_pre: int, *, arm: str = S_CTRL) -> OptimizerUpdateEvent:
    return OptimizerUpdateEvent(
        arm=arm,
        u_pre=u_pre,
        row_positions=(2 * u_pre, 2 * u_pre + 1),
        original_row_indices=(2 * u_pre, 2 * u_pre + 1),
        row_loss_values=(1.0, 1.0),
        mean_loss=1.0,
    )


# ---------------------------------------------------------------------------
# Optimizer contract (adjudication ruling f)
# ---------------------------------------------------------------------------


def test_optimizer_contract_matches_adjudicated_frozen_literals():
    contract = OptimizerContract()
    assert contract.base_lr == 2e-5
    assert contract.head_lr == 3e-4
    assert contract.weight_decay == 1e-4
    assert contract.grad_clip == 1.0
    assert contract.betas == (0.9, 0.999)
    assert contract.eps == 1e-8
    payload = contract.to_dict()
    assert payload["optimizer"] == "AdamW"
    assert payload["grad_clip_norm"] == 1.0

    with pytest.raises(F2AssemblyContractError):
        OptimizerContract(base_lr=-1.0)
    with pytest.raises(F2AssemblyContractError):
        OptimizerContract(grad_clip=0.0)
    with pytest.raises(F2AssemblyContractError):
        OptimizerContract(betas=(0.9, 1.0))


# ---------------------------------------------------------------------------
# Package factories (blocker 11)
# ---------------------------------------------------------------------------


def test_smoke_package_is_sa_hstar_and_all_three_packages_forward():
    assert SMOKE_PACKAGE == "SA-Hstar"
    for package in PACKAGE_NAMES:
        arm = _build(package)
        adapter = arm.adapter
        state = adapter.init_state(1, "cpu")
        observation = _observation()
        output = adapter.encode_step(
            observation.coarse_tokens.unsqueeze(0),
            observation.coarse_tidx.unsqueeze(0),
            observation.fine_tokens.unsqueeze(0),
            observation.fine_tidx.unsqueeze(0),
            [observation.instruction],
            state,
        )
        assert output["h_act"].shape == (1, 8)
        assert set(output["method_features"]) == set(adapter.method_dims)
        prev = torch.zeros(1, 2)
        model_output = arm.model(
            output["base_features"],
            prev,
            method_features=output["method_features"],
            method_alphas=output["method_alphas"],
        )
        assert model_output.prediction.raw_actions.shape == (1, AP2_HORIZON, 3)
        assert torch.equal(
            model_output.prediction.raw_fy,
            prev.unsqueeze(-2).expand(1, AP2_HORIZON, 2),
        )


def test_sa_b0_null_tim_slots_are_bitwise_zero_with_matched_token_count():
    hstar = _build("SA-Hstar")
    b0 = _build("SA-B0")
    observation = _observation()

    hstar_output = hstar.adapter.encode_step(
        observation.coarse_tokens.unsqueeze(0),
        observation.coarse_tidx.unsqueeze(0),
        observation.fine_tokens.unsqueeze(0),
        observation.fine_tidx.unsqueeze(0),
        [observation.instruction],
        hstar.adapter.init_state(1, "cpu"),
    )
    hstar_sequence = hstar.base.llm.last_sequence_length

    b0_output = b0.adapter.encode_step(
        observation.coarse_tokens.unsqueeze(0),
        observation.coarse_tidx.unsqueeze(0),
        observation.fine_tokens.unsqueeze(0),
        observation.fine_tidx.unsqueeze(0),
        [observation.instruction],
        b0.adapter.init_state(1, "cpu"),
    )
    b0_sequence = b0.base.llm.last_sequence_length

    assert b0_sequence == hstar_sequence
    slots = b0_output["null_tim_slots"]
    assert slots.shape == (1, b0.adapter.n_tim_slots, 8)
    assert b0.adapter.n_tim_slots == hstar.adapter.tim.n_tokens == 4
    assert torch.count_nonzero(slots).item() == 0
    assert b0_output["method_features"] == {}
    assert b0.adapter.method_dims == {}
    assert b0_output["new_state"] == {}
    parameter_names = [name for name, _ in b0.adapter.named_parameters()]
    assert all(name.startswith("base.") for name in parameter_names)
    assert hstar_output["method_features"]  # SA-Hstar keeps method streams.


def test_sa_b1_deletes_future_event_orchestrator_q_and_zero_fills_q_slot():
    b1 = _build("SA-B1")
    assert isinstance(b1.adapter, ConfidenceTIMObservationAdapter)
    parameter_names = [name for name, _ in b1.adapter.named_parameters()]
    for forbidden in ("future.", "events.", "orchestrator.", "self_correctness."):
        assert not any(forbidden in name for name in parameter_names)
    assert b1.adapter.method_dims == {"polar": 8, "tim_q": 10}

    state = b1.adapter.init_state(1, "cpu")
    assert "evt" not in state
    observation = _observation()
    output = b1.adapter.encode_step(
        observation.coarse_tokens.unsqueeze(0),
        observation.coarse_tidx.unsqueeze(0),
        observation.fine_tokens.unsqueeze(0),
        observation.fine_tidx.unsqueeze(0),
        [observation.instruction],
        state,
    )
    assert set(output["method_features"]) == {"polar", "tim_q"}
    assert output["method_features"]["tim_q"].shape == (1, 10)
    assert torch.count_nonzero(output["method_features"]["tim_q"][..., -1]).item() == 0
    assert torch.count_nonzero(output["new_state"]["pending_q_write"]).item() == 0
    assert "q_write" not in output

    losses = b1.adapter.compute_aux_losses(output, _aux_targets())
    assert set(losses) == {"loss", "L_aux", "L_cot"}
    assert torch.equal(losses["L_aux"], losses["L_cot"])


def test_ap2_head_and_prev_stream_structure_identical_across_packages():
    arms = {package: _build(package) for package in PACKAGE_NAMES}
    shared_names: dict[str, set[str]] = {}
    method_names: dict[str, set[str]] = {}
    for package, arm in arms.items():
        names = {name for name, _ in arm.model.named_parameters()}
        method = {
            name
            for name in names
            if name.startswith("fusion.method_projections")
            or name.startswith("fusion.method_scales")
        }
        shared_names[package] = names - method
        method_names[package] = method
    assert (
        shared_names["SA-B0"]
        == shared_names["SA-B1"]
        == shared_names["SA-Hstar"]
    )
    assert method_names["SA-B0"] == set()
    assert {name.split(".")[2] for name in method_names["SA-B1"]} == {
        "polar",
        "tim_q",
    }
    assert {name.split(".")[2] for name in method_names["SA-Hstar"]} == {
        "polar",
        "tim_q",
        "future",
        "event",
    }


def test_aux_coefficient_contract_fails_closed():
    torch.manual_seed(0)
    base = DummyOfficialBase()
    with pytest.raises(F2AssemblyContractError, match="keys must be exactly"):
        build_package(
            "SA-Hstar", base, device="cpu", aux_coefficients={"L_cot": 0.5}
        )
    with pytest.raises(F2AssemblyContractError, match=r"\(0,1\]"):
        build_package(
            "SA-B1", base, device="cpu", aux_coefficients={"L_cot": 1.5}
        )
    with pytest.raises(F2AssemblyContractError, match="positive"):
        build_package(
            "SA-B1", base, device="cpu", aux_coefficients={"L_cot": 0.0}
        )
    with pytest.raises(F2AssemblyContractError, match="package"):
        build_package("SA-H0", base, device="cpu", aux_coefficients={})


# ---------------------------------------------------------------------------
# Paired arms and optimizer receipts (blocker 7)
# ---------------------------------------------------------------------------


def test_build_paired_arms_bit_identical_sha_and_disjoint_parameters():
    torch.manual_seed(3)
    base = DummyOfficialBase()
    paired = build_paired_arms(
        base,
        package="SA-Hstar",
        seed=7,
        device="cpu",
        contract=OptimizerContract(),
        aux_coefficients=AUX_COEFFICIENTS["SA-Hstar"],
    )
    assert set(paired.arms) == {S_CTRL, S_SELF}
    assert paired.seed == 7
    ctrl = paired.arms[S_CTRL]
    self_arm = paired.arms[S_SELF]
    ctrl_sha = checkpoint_init_sha256(ctrl.modules.full_state_dict())
    self_sha = checkpoint_init_sha256(self_arm.modules.full_state_dict())
    assert ctrl_sha == self_sha == paired.checkpoint_init_sha256
    assert len(paired.checkpoint_init_sha256) == 64

    ctrl_ids = {id(parameter) for parameter in ctrl.modules.trainable_parameters()}
    self_ids = {
        id(parameter) for parameter in self_arm.modules.trainable_parameters()
    }
    assert ctrl_ids.isdisjoint(self_ids)
    assert ctrl.optimizer is not self_arm.optimizer
    assert ctrl.modules.adapter.base is ctrl.modules.base
    assert self_arm.modules.adapter.base is self_arm.modules.base
    assert ctrl.parameter_receipt["groups"] == self_arm.parameter_receipt["groups"]


def test_build_arm_optimizer_membership_lr_wd_receipt_and_freeze_contract():
    arm = _build("SA-Hstar")
    contract = OptimizerContract()
    optimizer, receipt = build_arm_optimizer(arm, contract)

    assert not arm.base.llm.mix.weight.requires_grad
    assert not arm.base.act_token.requires_grad
    assert not arm.base.text_token.requires_grad
    assert arm.base.proj.weight.requires_grad

    groups = receipt["groups"]
    assert receipt["group_order"] == [
        "base_proj",
        "adapter_and_ordinary_head",
        "method_layerscales",
        "prev_layerscale",
    ]
    assert groups["base_proj"]["lr"] == 2e-5
    assert groups["base_proj"]["weight_decay"] == 1e-4
    assert "adapter.base.proj.weight" in groups["base_proj"]["parameter_names"]
    assert groups["adapter_and_ordinary_head"]["lr"] == 3e-4
    assert groups["adapter_and_ordinary_head"]["weight_decay"] == 1e-4
    assert groups["method_layerscales"]["weight_decay"] == 0.0
    assert groups["method_layerscales"]["parameter_count"] == 4
    assert groups["prev_layerscale"]["parameter_names"] == ["model.fusion.s_prev"]
    assert groups["prev_layerscale"]["weight_decay"] == 0.0

    assert receipt["coverage_exact"] is True
    assert receipt["missing_from_groups"] == []
    assert receipt["nontrainable_in_groups"] == []
    assert receipt["trainable_base_modules"] == ["proj"]
    assert receipt["trainable_parameter_count"] == sum(
        group["parameter_count"] for group in groups.values()
    )
    assert receipt["contract"] == contract.to_dict()

    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_ids = {id(parameter) for parameter in arm.trainable_parameters()}
    assert optimizer_ids == trainable_ids

    b0 = _build("SA-B0")
    _b0_optimizer, b0_receipt = build_arm_optimizer(b0, contract)
    assert b0_receipt["group_order"] == [
        "base_proj",
        "adapter_and_ordinary_head",
        "prev_layerscale",
    ]


def test_build_arm_optimizer_fails_closed_on_unfrozen_base_parameter():
    arm = _build("SA-Hstar")
    arm.base.llm.mix.weight.requires_grad_(True)
    with pytest.raises(F2AssemblyContractError, match="missing from optimizer"):
        build_arm_optimizer(arm, OptimizerContract())


# ---------------------------------------------------------------------------
# G6 instrumentation (blocker 8)
# ---------------------------------------------------------------------------


def _instrument_case(a, b, *, aux_scale=1.0, rows=9):
    weight = nn.Parameter(torch.zeros(2))
    instrument = G6Instrument((weight,), rows_per_update=1)
    a_tensor = torch.tensor(a)
    b_tensor = torch.tensor(b)
    update = None
    for u_pre in range(rows):
        aux_loss = aux_scale * (weight * a_tensor).sum()
        track = (weight * b_tensor).sum()
        instrument.observe_row(aux_loss=aux_loss, track1=track, track2=track)
        update = instrument.emit_update(_update_event(u_pre))
    return update


def test_g6_instrument_geometry_reachability_and_projection_signs():
    aligned = _instrument_case((2.0, 0.0), (1.0, 0.0))
    assert aligned.u_pre == 8
    assert aligned.aux_reachable and aligned.track_reachable
    assert aligned.cosine_total_track == pytest.approx(1.0)
    assert aligned.signed_projection == pytest.approx(3.0)
    assert aligned.aux_track_ratio == pytest.approx(2.0)
    assert aligned.per_aux_ratios is None

    dominant_anti = _instrument_case((-2.0, 0.0), (1.0, 0.0))
    assert dominant_anti.cosine_total_track == pytest.approx(-1.0)
    assert dominant_anti.signed_projection == pytest.approx(-1.0)

    orthogonal = _instrument_case((0.0, 1.0), (1.0, 0.0))
    assert orthogonal.cosine_total_track == pytest.approx(1.0 / 2.0 ** 0.5)
    assert orthogonal.signed_projection == pytest.approx(1.0)


def test_g6_instrument_window_boundary_and_unreachable_losses():
    early = _instrument_case((1.0, 0.0), (1.0, 0.0), rows=8)
    assert early.u_pre == 7
    assert early.cosine_total_track is None
    assert early.signed_projection is None
    assert early.aux_track_ratio is None

    zero_aux = _instrument_case((0.0, 0.0), (1.0, 0.0))
    assert not zero_aux.aux_reachable
    assert zero_aux.track_reachable
    assert zero_aux.cosine_total_track == pytest.approx(1.0)

    weight = nn.Parameter(torch.zeros(2))
    instrument = G6Instrument((weight,), rows_per_update=1)
    constant_aux = torch.tensor(0.5)  # no graph: zero-vector evidence
    track = (weight * torch.tensor([1.0, 0.0])).sum()
    instrument.observe_row(aux_loss=constant_aux, track1=track, track2=track)
    update = instrument.emit_update(_update_event(0))
    assert not update.aux_reachable
    assert update.track_reachable


def test_g6_instrument_lambda_scaling_keeps_cos_sign_and_projection():
    for aux_scale in (1.0, 10.0):
        aligned = _instrument_case((2.0, 0.0), (1.0, 0.0), aux_scale=aux_scale)
        assert aligned.cosine_total_track == pytest.approx(1.0)
        orthogonal = _instrument_case(
            (0.0, 1.0), (1.0, 0.0), aux_scale=aux_scale
        )
        assert orthogonal.signed_projection == pytest.approx(1.0)
        assert orthogonal.cosine_total_track > 0.0
        assert orthogonal.aux_track_ratio == pytest.approx(aux_scale)


def test_g6_instrument_row_and_clock_discipline_fail_closed():
    weight = nn.Parameter(torch.zeros(2))
    instrument = G6Instrument((weight,), rows_per_update=1)
    with pytest.raises(F2AssemblyContractError, match="aggregated 0 rows"):
        instrument.emit_update(_update_event(0))

    track = (weight * torch.tensor([1.0, 0.0])).sum()
    instrument.observe_row(aux_loss=track, track1=track, track2=track)
    with pytest.raises(F2AssemblyContractError, match="more times"):
        instrument.observe_row(aux_loss=track, track1=track, track2=track)

    with pytest.raises(F2AssemblyContractError, match="S-CTRL only"):
        instrument.emit_update(_update_event(0, arm=S_SELF))
    with pytest.raises(F2AssemblyContractError, match="clock discontinuity"):
        instrument.emit_update(_update_event(5))
    emitted = instrument.emit_update(_update_event(0))
    assert emitted.u_pre == 0

    with pytest.raises(F2AssemblyContractError, match="probe"):
        G6Instrument(())
    with pytest.raises(F2AssemblyContractError, match="block_mode"):
        G6Instrument((weight,), block_mode="hybrid")


def test_g6_instrument_bstar_collects_separate_per_aux_diagnostics():
    weight = nn.Parameter(torch.zeros(2))
    bstar = G6Instrument((weight,), rows_per_update=1)
    bstar_update = None
    for u_pre in range(9):
        track = (weight * torch.tensor([1.0, 0.0])).sum()
        aux = (weight * torch.tensor([0.0, 1.0])).sum()
        cot = (weight * torch.tensor([0.0, 0.5])).sum()
        bstar.observe_row(
            aux_loss=aux,
            track1=track,
            track2=track,
            per_aux_losses={"L_cot": cot},
        )
        bstar_update = bstar.emit_update(_update_event(u_pre))

    assert bstar_update.aux_track_ratio == pytest.approx(1.0)
    assert bstar_update.per_aux_ratios is None
    fallback = bstar.fallback_evidence()
    assert fallback["deciding_block_mode"] == "bstar"
    assert fallback["block_mode"] == "bstar"
    assert len(fallback["per_aux_ratio_series"]) == 9
    assert fallback["per_aux_ratio_series"][-1] == {
        "u_pre": 8,
        "ratios": pytest.approx({"L_cot": 0.5}),
    }

    per_aux = G6Instrument((weight,), block_mode="per_aux", rows_per_update=1)
    track = (weight * torch.tensor([1.0, 0.0])).sum()
    aux = (weight * torch.tensor([0.0, 1.0])).sum()
    with pytest.raises(F2AssemblyContractError, match="requires a nonempty"):
        per_aux.observe_row(aux_loss=aux, track1=track, track2=track)

    update = None
    for u_pre in range(9):
        aux = (weight * torch.tensor([0.0, 1.0])).sum()
        cot = (weight * torch.tensor([0.0, 0.5])).sum()
        track = (weight * torch.tensor([1.0, 0.0])).sum()
        per_aux.observe_row(
            aux_loss=aux,
            track1=track,
            track2=track,
            per_aux_losses={"L_cot": cot},
        )
        update = per_aux.emit_update(_update_event(u_pre))
    assert update.aux_track_ratio is None
    assert update.per_aux_ratios == pytest.approx({"L_cot": 0.5})


# ---------------------------------------------------------------------------
# Observation whitelist and executor discipline (blockers 5/10 wiring)
# ---------------------------------------------------------------------------


def _executor(package: str = "SA-Hstar", *, g6: G6Instrument | None = None):
    arm = _build(package)
    optimizer, _receipt = build_arm_optimizer(arm, OptimizerContract())
    return ArmExecutor(arm, optimizer, OptimizerContract(), g6=g6)


def test_observation_whitelist_blocks_expert_fields_and_unknown_keys():
    executor = _executor()
    event = _row_event()

    poisoned = _observation_fields()
    poisoned["step_actions"] = [[0.1, 0.0, 0.0]] * 8
    with pytest.raises(
        F2AssemblyContractError,
        match="OBSERVATION_LEAK: forbidden observation keys",
    ):
        observation_packet_from_fields(poisoned)

    unknown = _observation_fields()
    unknown["mystery"] = 1
    with pytest.raises(
        F2AssemblyContractError,
        match="OBSERVATION_LEAK: unknown observation keys",
    ):
        observation_packet_from_fields(unknown)

    incomplete = _observation_fields()
    del incomplete["instruction"]
    with pytest.raises(F2AssemblyContractError, match="fields are missing"):
        observation_packet_from_fields(incomplete)

    raw_fields = _observation_fields()
    for untrusted in (raw_fields, SimpleNamespace(**raw_fields)):
        with pytest.raises(
            F2AssemblyContractError,
            match="OBSERVATION_LEAK:.*ObservationPacket",
        ):
            executor.feature_forward(untrusted, event)

    packet = _observation()
    assert packet.coarse_tokens.unsqueeze(0).shape == (
        1,
        COARSE_TOKEN_COUNT,
        VISION_FEATURE_DIM,
    )
    assert packet.fine_tokens.unsqueeze(0).shape == (
        1,
        FINE_TOKEN_COUNT,
        VISION_FEATURE_DIM,
    )
    result = executor.feature_forward(packet, event)
    assert result.reference_tensor.shape == (1, 8)


def test_executor_records_leak_counter_when_labels_alias_the_observation():
    executor = _executor()
    observation = _observation()
    features = executor.feature_forward(observation, _row_event())
    targets = _aux_targets()
    targets["polar_theta_idx"] = observation.coarse_tidx[:1]
    with pytest.raises(F2AssemblyContractError, match="aliased into the observation"):
        executor.aux_forward(features.value, targets, _row_event())
    counters = executor.arm.adapter.audit_counters()
    assert counters["expert_future_leak_count"] == 1
    with pytest.raises(F2ObservationContractError, match="nonzero"):
        executor.arm.adapter.assert_audit_counters_clean()


def test_executor_row_flow_updates_parameters_and_clears_scratch():
    executor = _executor("SA-Hstar")
    g6 = G6Instrument(
        tuple(executor.arm.base.proj.parameters()), rows_per_update=1
    )
    executor.g6 = g6
    callbacks = executor.callbacks()
    assert callbacks.audit_counters == executor.arm.adapter.audit_counters
    assert make_backward_callback(executor) == executor.backward
    assert make_optimizer_step_callback(executor) == executor.optimizer_step

    observation = _observation()
    features = executor.feature_forward(observation, _row_event())
    aux = executor.aux_forward(features.value, _aux_targets(), _row_event())
    prev = torch.zeros(1, 2)
    branch1 = executor.head_forward(features.value, prev, _head_event("branch1"))
    branch2 = executor.head_forward(features.value, prev, _head_event("branch2"))
    target = torch.full((1, AP2_HORIZON, 3), 0.2)
    track1 = executor.track_loss(branch1.prediction, target, _head_event("branch1"))
    track2 = executor.track_loss(branch2.prediction, target, _head_event("branch2"))

    with pytest.raises(F2AssemblyContractError, match="recorded twice"):
        executor.track_loss(branch1.prediction, target, _head_event("branch1"))
    with pytest.raises(F2AssemblyContractError, match="residue"):
        executor.feature_forward(observation, _row_event(row_position=1, reset=False))

    row_loss = aux.loss + 0.5 * track1 + 0.5 * track2
    proj_before = executor.arm.base.proj.weight.detach().clone()
    head_before = (
        executor.arm.model.action_head.forward_branch.weight.detach().clone()
    )
    executor.backward(
        BackwardEvent(
            arm=S_CTRL,
            row_position=0,
            original_row_index=1000,
            u_pre=0,
            unscaled_loss=row_loss,
            scaled_loss=row_loss / 2,
        )
    )
    executor.optimizer_step(_update_event(0))

    assert not torch.equal(proj_before, executor.arm.base.proj.weight)
    assert not torch.equal(
        head_before, executor.arm.model.action_head.forward_branch.weight
    )
    assert all(
        parameter.grad is None
        for parameter in executor.arm.trainable_parameters()
    )
    update = g6.emit_update(_update_event(0))
    assert update.aux_reachable
    with pytest.raises(F2AssemblyContractError, match="missing captured"):
        executor.backward(
            BackwardEvent(
                arm=S_CTRL,
                row_position=1,
                original_row_index=1001,
                u_pre=0,
                unscaled_loss=row_loss.detach(),
                scaled_loss=row_loss.detach(),
            )
        )


# ---------------------------------------------------------------------------
# Full paired smoke on the dummy base (integration of blockers 7/8/11)
# ---------------------------------------------------------------------------


def _integration_rows() -> list[RunnerRow]:
    rows = []
    target = torch.tensor([[0.2, 0.0, -0.1]] * AP2_HORIZON)
    for position in range(SMOKE_ROWS):
        rows.append(
            RunnerRow(
                original_row_index=1000 + position,
                sequence_id="sequence-a",
                frame_idx=position,
                mirrored=False,
                logged_prev_action=(0.0, 0.0, 0.0),
                target_actions=target,
                observation=_observation(23 + position),
                aux_targets=_aux_targets(),
            )
        )
    return rows


def test_full_paired_smoke_with_production_assembly_on_dummy_base():
    torch.manual_seed(5)
    base = DummyOfficialBase()
    contract = OptimizerContract()
    paired = build_paired_arms(
        base,
        package=SMOKE_PACKAGE,
        seed=0,
        device="cpu",
        contract=contract,
        aux_coefficients=AUX_COEFFICIENTS["SA-Hstar"],
    )
    ctrl = paired.arms[S_CTRL]
    self_arm = paired.arms[S_SELF]
    g6 = G6Instrument(tuple(ctrl.modules.base.proj.parameters()))
    ctrl_callbacks, _ctrl_executor = build_arm_callbacks(
        ctrl.modules, ctrl.optimizer, contract, g6=g6
    )
    self_callbacks, _self_executor = build_arm_callbacks(
        self_arm.modules, self_arm.optimizer, contract
    )
    llm_before = ctrl.modules.base.llm.mix.weight.detach().clone()
    proj_before = ctrl.modules.base.proj.weight.detach().clone()

    result = run_paired_smoke(
        _integration_rows(),
        callbacks={S_CTRL: ctrl_callbacks, S_SELF: self_callbacks},
        hooks=RunnerTelemetryHooks(g6_update=g6.emit_update),
        strafe_reset_original_indices=frozenset(),
        expected_static_reset_original_indices=frozenset({1000}),
    )

    assert result.count_receipt.passed
    assert result.checkpoint_init_sha256 == paired.checkpoint_init_sha256
    receipt_payload = result.count_receipt.to_dict()
    for arm_name in (S_CTRL, S_SELF):
        assert receipt_payload["arms"][arm_name]["expert_future_leak_count"] == 0
        assert (
            receipt_payload["arms"][arm_name]["self_state_expert_overwrite_count"]
            == 0
        )

    g6_updates = result.arms[S_CTRL].g6_updates
    assert [update.u_pre for update in g6_updates] == list(range(128))
    assert all(update.aux_reachable for update in g6_updates)
    assert not g6_updates[0].track_reachable
    assert sum(update.track_reachable for update in g6_updates if update.u_pre >= 8) == 120
    for update in g6_updates:
        in_window = update.u_pre >= 8
        assert (update.cosine_total_track is not None) == in_window
        assert (update.signed_projection is not None) == in_window
        assert (update.aux_track_ratio is not None) == in_window
        assert update.per_aux_ratios is None

    g7_receipt = evaluate_g7(
        [update.gate_update() for update in result.arms[S_SELF].g7_updates]
    )
    assert g7_receipt.passed
    assert g7_receipt.checks["prev_scale_saturation_rate"]["passed"]
    assert g7_receipt.metrics["prev_scale_saturation_rate"] == 0.0

    g9 = result.arms[S_SELF].g9
    assert g9.nonfinite_reset_count == 0
    assert max(abs(value) for value in g9.reconstruction_errors) == 0.0

    assert torch.equal(llm_before, ctrl.modules.base.llm.mix.weight)
    assert not torch.equal(proj_before, ctrl.modules.base.proj.weight)
    ctrl_sha = checkpoint_init_sha256(ctrl.modules.full_state_dict())
    self_sha = checkpoint_init_sha256(self_arm.modules.full_state_dict())
    assert ctrl_sha != paired.checkpoint_init_sha256
    assert ctrl_sha != self_sha


# ---------------------------------------------------------------------------
# W3 integration seams: CAL auditor, EVAL predictors, production smoke plan
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_REAL_ASSETS_PRESENT = (
    FROZEN_BASE_HF_DIR_DEFAULT.is_dir()
    and (PROJECT_ROOT / "data/collected_v1/vision_cache/cache_manifest.json").is_file()
    and (PROJECT_ROOT / "data/collected_v1/datasets/train.jsonl").is_file()
)


def _runner_row(position: int = 0, *, seed: int = 0) -> RunnerRow:
    return RunnerRow(
        original_row_index=1000 + position,
        sequence_id="sequence-a",
        frame_idx=position,
        mirrored=False,
        logged_prev_action=(0.2, 0.0, -0.1),
        target_actions=torch.tensor([[0.2, 0.0, -0.1]] * AP2_HORIZON),
        observation=_observation(seed + position),
        aux_targets=_aux_targets(),
    )


def test_assembly_contract_error_is_unified_across_the_three_modules():
    from f2_experiment.assembly import F2AssemblyContractError as lifecycle_error
    from f2_experiment.assembly_data import (
        F2AssemblyContractError as data_error,
    )

    assert lifecycle_error is data_error
    assert F2AssemblyContractError is data_error


def test_eval_row_predictor_persistence_recurrence_and_pass_discipline():
    arm = _build("SA-Hstar")
    predictor = build_eval_row_predictor(arm)
    assert isinstance(predictor, EvalRowPredictor)
    row0 = _runner_row(0)
    row1 = _runner_row(1)
    prev = torch.tensor([[0.2, -0.1]])
    with torch.no_grad():
        first = predictor(row0, prev, mode="logged", reset=True, position=0)
        assert torch.equal(
            first.raw_fy, prev.unsqueeze(-2).expand(1, AP2_HORIZON, 2)
        )
        second = predictor(row1, prev, mode="logged", reset=False, position=1)
        assert second.raw_actions.shape == (1, AP2_HORIZON, 3)
        with pytest.raises(F2AssemblyContractError, match="pass order"):
            predictor(row1, prev, mode="self", reset=False, position=2)
        with pytest.raises(F2AssemblyContractError, match="pass order"):
            predictor(row1, prev, mode="logged", reset=False, position=3)
        with pytest.raises(F2AssemblyContractError, match="start with a reset"):
            predictor(row0, prev, mode="self", reset=False, position=0)
        restart = predictor(row0, prev, mode="self", reset=True, position=0)
        assert torch.equal(
            restart.raw_fy, prev.unsqueeze(-2).expand(1, AP2_HORIZON, 2)
        )
    assert not any(
        module.training
        for module in (arm.base, arm.adapter, arm.model)
    )


def test_cal_row_auditor_zero_update_reports_and_position_discipline():
    arm = _build("SA-Hstar")
    sha_before = checkpoint_init_sha256(arm.full_state_dict())
    auditor = CalRowAuditor(arm)
    first = auditor(_runner_row(0), ("stream_first",), 0)
    assert isinstance(first, CalRowAudit)
    assert first.step0_parity
    assert first.prev_free
    assert set(first.aux_grad_norms) == {"L_cot", "L_future", "L_verify"}
    assert first.aux_grad_norms["L_cot"] > 0.0
    # Zero-init AP2 branches: the track loss cannot reach base.proj before
    # the first optimizer update, so a faithful CAL audit reports exactly
    # 0.0 here.  This pins the run_cal_audit track-median>0 incompatibility
    # that is escalated for adjudication rather than defaulted away.
    assert first.track_grad_norm == 0.0
    second = auditor(_runner_row(1), (), 1)
    assert second.track_grad_norm == 0.0
    with pytest.raises(F2AssemblyContractError, match="discontinuity"):
        auditor(_runner_row(2), (), 3)
    # Zero-update proof: no parameter ever moved.
    assert checkpoint_init_sha256(arm.full_state_dict()) == sha_before


def test_cal_row_auditor_detects_parity_break_and_package_contract():
    arm = _build("SA-Hstar")
    with torch.no_grad():
        arm.model.action_head.forward_branch.weight.add_(0.05)
    audit = CalRowAuditor(arm)(_runner_row(0), ("stream_first",), 0)
    assert not audit.step0_parity
    assert audit.track_grad_norm > 0.0
    assert audit.prev_free

    with pytest.raises(F2AssemblyContractError, match="reset"):
        CalRowAuditor(_build("SA-Hstar"))(_runner_row(0), (), 0)
    with pytest.raises(F2AssemblyContractError, match="aux block set"):
        CalRowAuditor(_build("SA-B1"))


def test_audit_cal_row_seam_requires_position_zero_start(monkeypatch):
    import f2_experiment.assembly_model as assembly_model

    monkeypatch.setattr(assembly_model, "_ACTIVE_CAL_AUDITOR", None)
    with pytest.raises(F2AssemblyContractError, match="position 0"):
        audit_cal_row(_runner_row(3), (), 3)
    with pytest.raises(F2AssemblyContractError, match="nonnegative"):
        audit_cal_row(_runner_row(0), (), -1)


def test_build_eval_row_predictor_from_checkpoint_round_trip_and_tamper():
    arm = _build("SA-Hstar", seed=31)
    with torch.no_grad():
        arm.model.action_head.forward_branch.weight.add_(0.01)
        arm.base.proj.weight.add_(0.005)
    payload = {
        "model": {
            name: tensor.clone()
            for name, tensor in arm.full_state_dict().items()
        },
        "optimizer": {},
        "u_pre": 128,
        "arm": S_SELF,
        "checkpoint_init_sha256": checkpoint_init_sha256(
            arm.full_state_dict()
        ),
    }

    torch.manual_seed(99)
    predictor = build_eval_row_predictor_from_checkpoint(
        PROJECT_ROOT,
        {"smoke_package": "SA-Hstar"},
        S_SELF,
        payload,
        device="cpu",
        base=DummyOfficialBase(),
    )
    rebuilt_sha = checkpoint_init_sha256(predictor.arm.full_state_dict())
    assert rebuilt_sha == payload["checkpoint_init_sha256"]

    row = _runner_row(0)
    prev = torch.tensor([[0.1, -0.05]])
    with torch.no_grad():
        original = EvalRowPredictor(arm)(
            row, prev, mode="logged", reset=True, position=0
        )
        rebuilt = predictor(row, prev, mode="logged", reset=True, position=0)
    assert torch.equal(original.raw_actions, rebuilt.raw_actions)

    with pytest.raises(F2AssemblyContractError, match="belongs to"):
        build_eval_row_predictor_from_checkpoint(
            PROJECT_ROOT,
            {"smoke_package": "SA-Hstar"},
            S_CTRL,
            payload,
            device="cpu",
            base=DummyOfficialBase(),
        )

    tampered = dict(payload)
    tampered["model"] = dict(payload["model"])
    tampered["model"]["model.fusion.s_prev"] = (
        payload["model"]["model.fusion.s_prev"] + 0.5
    )
    with pytest.raises(F2AssemblyContractError, match="does not match"):
        torch.manual_seed(7)
        build_eval_row_predictor_from_checkpoint(
            PROJECT_ROOT,
            {"smoke_package": "SA-Hstar"},
            S_SELF,
            tampered,
            device="cpu",
            base=DummyOfficialBase(),
        )

    rogue = dict(payload)
    rogue["model"] = dict(payload["model"])
    rogue["model"]["rogue.weight"] = torch.zeros(1)
    with pytest.raises(F2AssemblyContractError, match="unknown prefix"):
        build_eval_row_predictor_from_checkpoint(
            PROJECT_ROOT,
            {"smoke_package": "SA-Hstar"},
            S_SELF,
            rogue,
            device="cpu",
            base=DummyOfficialBase(),
        )


def test_frozen_aux_coefficients_pin_windows_cuda_cal_v3_values(
    tmp_path, monkeypatch
):
    import f2_experiment.assembly_model as assembly_model_module

    assert dict(assembly_model_module.FROZEN_AUX_COEFFICIENTS or {}) == {
        "L_cot": 0.0195,
        "L_future": 0.34,
        "L_verify": 0.5,
    }
    monkeypatch.setattr(
        assembly_model_module, "FROZEN_AUX_COEFFICIENTS", None
    )
    with pytest.raises(F2AssemblyContractError, match="not frozen"):
        build_production_smoke_plan(
            tmp_path, {"smoke_package": "SA-Hstar"}
        )


def test_build_production_smoke_plan_fail_closed_contracts(tmp_path):
    with pytest.raises(F2AssemblyContractError, match="ruling d"):
        build_production_smoke_plan(tmp_path, {"smoke_package": "SA-B0"})
    with pytest.raises(F2AssemblyContractError, match="bstar"):
        build_production_smoke_plan(
            tmp_path, {"smoke_package": "SA-Hstar", "block_mode": "per_aux"}
        )
    with pytest.raises(F2AssemblyContractError, match="OptimizerContract"):
        build_production_smoke_plan(
            tmp_path,
            {
                "smoke_package": "SA-Hstar",
                "optimizer_contract": {"base_lr": 1.0},
            },
            aux_coefficients=AUX_COEFFICIENTS["SA-Hstar"],
        )
    with pytest.raises(F2AssemblyContractError, match="train JSONL"):
        build_production_smoke_plan(
            tmp_path,
            {"smoke_package": "SA-Hstar"},
            aux_coefficients=AUX_COEFFICIENTS["SA-Hstar"],
        )


@pytest.mark.skipif(
    not _REAL_ASSETS_PRESENT,
    reason="frozen real assets (base HF checkpoint / vision cache / train "
    "JSONL) are unavailable",
)
def test_production_smoke_plan_constructs_on_real_frozen_assets():
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asset_binding = verify_frozen_assets(
        PROJECT_ROOT, verify_token_payload=False
    )
    token_ledger = build_train_token_ledger(PROJECT_ROOT)
    asset_binding = {
        **asset_binding,
        "token_ledger_sha256": token_ledger.ledger_sha256,
        "token_ledger_file_count": token_ledger.token_files,
    }
    receipt_document = {
        "smoke_package": "SA-Hstar",
        "block_mode": "bstar",
        "asset_binding": asset_binding,
        "optimizer_contract": {
            "base_lr": 2e-5,
            "head_lr": 3e-4,
            "weight_decay": 1e-4,
            "grad_clip_norm": 1.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
    }
    # aux_coefficients here are a test-only override: the production path
    # reads the ruling-b FROZEN_AUX_COEFFICIENTS source literals and fails
    # closed while they are None.
    plan = build_production_smoke_plan(
        PROJECT_ROOT,
        receipt_document,
        device=test_device,
        aux_coefficients={"L_cot": 0.5, "L_future": 0.5, "L_verify": 0.5},
    )
    assert isinstance(plan, SmokeAssemblyPlan)
    assert len(plan.smoke_rows) == 256
    assert len(plan.eval_rows) == 512
    assert len(plan.eval_raw_rows) == 512
    assert all(isinstance(row, RunnerRow) for row in plan.smoke_rows)
    assert isinstance(plan.smoke_rows[0].observation, ObservationPacket)
    assert isinstance(plan.smoke_rows[0].aux_targets, AuxTargetPacket)
    assert len(plan.expected_static_reset_original_indices) == 12
    smoke_indices = {row.original_row_index for row in plan.smoke_rows}
    assert not (plan.strafe_reset_original_indices & smoke_indices)
    assert set(plan.arms) == {S_CTRL, S_SELF}
    assert callable(plan.g6_update)
    for arm_name in (S_CTRL, S_SELF):
        assert plan.arms[arm_name].callbacks.audit_counters is not None

    # Bit-identical paired-arm proof through the checkpoint payload seam.
    payload_ctrl = plan.arms[S_CTRL].checkpoint_payload()
    payload_self = plan.arms[S_SELF].checkpoint_payload()
    assert set(payload_ctrl) >= {"model", "optimizer"}
    sha_ctrl = checkpoint_init_sha256(payload_ctrl["model"])
    sha_self = checkpoint_init_sha256(payload_self["model"])
    assert sha_ctrl == sha_self

    # One real EVAL-FIX forward through the frozen Qwen base: the zero-init
    # AP2 head must reproduce exact raw persistence on the first row.
    first_row = plan.eval_rows[0]
    prev = torch.tensor(
        [
            [
                first_row.logged_prev_action[0],
                first_row.logged_prev_action[2],
            ]
        ],
        device=test_device,
    )
    with torch.no_grad():
        prediction = plan.arms[S_SELF].eval_predictor(
            first_row, prev, mode="logged", reset=True, position=0
        )
    assert torch.equal(
        prediction.raw_fy, prev.unsqueeze(-2).expand(1, AP2_HORIZON, 2)
    )


def test_adapter_variants_share_the_frozen_full_adapter_machinery():
    hstar = _build("SA-Hstar")
    b1 = _build("SA-B1")
    b0 = _build("SA-B0")
    assert isinstance(hstar.adapter, OpenTrackVLAF2ObservationAdapter)
    assert isinstance(b1.adapter, OpenTrackVLAF2ObservationAdapter)
    assert isinstance(b0.adapter, NullTIMObservationAdapter)
    for arm in (hstar, b1, b0):
        assert set(PACKAGE_AUX_COMPONENTS[arm.package]) == set(
            arm.aux_coefficients
        )
        assert arm.adapter.audit_counters() == {
            "expert_future_leak_count": 0,
            "self_state_expert_overwrite_count": 0,
        }
