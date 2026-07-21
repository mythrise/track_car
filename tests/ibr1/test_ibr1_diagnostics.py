from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from f2_experiment.assembly_model import OptimizerContract, build_arm_optimizer
from f2_experiment.model import AP2_HORIZON
from f2_experiment.runner import (
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    HeadEvent,
    HeadForwardResult,
    OptimizerUpdateEvent,
    RunnerRow,
)
from ibr1_experiment.assembly_model import (
    IBR1_CTRL,
    IBR1_FROZEN_AUX_COEFFICIENTS,
    IBR1_SELF,
    build_ibr1_package,
)
from ibr1_experiment.diagnostics import (
    GeometryCollector,
    GradientDiagnosticsCollector,
    IBR1DiagnosticsContractError,
    IBR1G6Instrument,
    OptimizerDiagnosticsHandle,
    wrap_eval_predictor,
    wrap_training_head_forward,
)
from ibr1_experiment.model import IBR1Prediction, normalized_cumulative_decode


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


def _update_event(arm: str, u_pre: int = 0) -> OptimizerUpdateEvent:
    return OptimizerUpdateEvent(
        arm=arm,
        u_pre=u_pre,
        row_positions=(2 * u_pre, 2 * u_pre + 1),
        original_row_indices=(1000 + 2 * u_pre, 1001 + 2 * u_pre),
        row_loss_values=(1.0, 1.0),
        mean_loss=1.0,
    )


def _prediction(prev: torch.Tensor, violation_horizons: int = 0) -> IBR1Prediction:
    cumulative = torch.zeros(1, AP2_HORIZON, 2, dtype=torch.float32)
    for horizon in range(violation_horizons):
        cumulative[0, horizon, 0] = 2.0
    latent = torch.cat(
        (cumulative[:, :1], cumulative[:, 1:] - cumulative[:, :-1]), dim=1
    )
    raw_fy, realized, cumulative_live, prebound = normalized_cumulative_decode(
        prev, latent
    )
    raw_actions = torch.stack(
        (
            raw_fy[..., 0],
            torch.zeros_like(raw_fy[..., 0]),
            raw_fy[..., 1],
        ),
        dim=-1,
    )
    return IBR1Prediction(
        delta_fy=realized,
        raw_actions=raw_actions,
        bounded_future_actions=raw_actions[..., 1:, :].detach(),
        latent_delta_fy=latent,
        cumulative_latent_fy=cumulative_live,
        additive_prebound_fy=prebound,
        normalizer_fy=1.0 + torch.abs(cumulative_live),
        prebound_violation_mask=torch.abs(prebound) > 1.0,
    )


def _row(position: int) -> RunnerRow:
    return RunnerRow(
        original_row_index=5000 + position,
        sequence_id="sequence-a",
        frame_idx=position,
        mirrored=False,
        logged_prev_action=(0.0, 0.0, 0.0),
        target_actions=torch.zeros(AP2_HORIZON, 3),
        observation=object(),
        aux_targets={},
    )


def _callbacks(optimizer_step) -> ArmCallbacks:
    def unused(*args, **kwargs):
        del args, kwargs
        raise AssertionError("unused callback")

    return ArmCallbacks(
        checkpoint_state={"weight": torch.zeros(1)},
        feature_forward=unused,
        aux_forward=unused,
        head_forward=unused,
        track_loss=unused,
        backward=unused,
        optimizer_step=optimizer_step,
    )


def _run_g6_update(collector: GradientDiagnosticsCollector):
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    instrument = IBR1G6Instrument((weight,), collector)
    for _ in range(2):
        track = (weight * torch.tensor([2.0, 0.0])).sum()
        components = {
            name: (weight * torch.tensor([0.0, coefficient])).sum()
            for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
        }
        instrument.observe_row(
            aux_loss=sum(components.values()),
            track1=track,
            track2=track,
            per_aux_losses=components,
        )
    return instrument.emit_update(_update_event(S_CTRL))


def test_g6_absolute_geometry_uses_two_row_mean_and_preserves_gate_update():
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    gate_update = _run_g6_update(collector)
    assert gate_update.u_pre == 0
    assert gate_update.aux_reachable
    assert gate_update.track_reachable
    record = collector.gradient_records[0]
    assert record["track_grad_norm"] == pytest.approx(2.0)
    assert record["weighted_aux_grad_norm"] == pytest.approx(0.8595)
    assert record["total_grad_norm"] == pytest.approx(
        (2.0**2 + 0.8595**2) ** 0.5
    )
    assert record["weighted_aux_track_dot"] == pytest.approx(0.0)
    assert record["weighted_aux_track_cosine"] == pytest.approx(0.0)
    assert record["weighted_aux_signed_projection"] == pytest.approx(0.0)
    for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items():
        assert record["per_aux_weighted_grad_norm"][name] == pytest.approx(
            coefficient
        )
        assert record[
            "per_aux_raw_grad_norm_derived_from_frozen_lambda"
        ][name] == pytest.approx(1.0)


def test_g6_zero_track_uses_finite_sentinels_and_flags():
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    instrument = IBR1G6Instrument((weight,), collector)
    for _ in range(2):
        zero_track = weight.sum() * 0.0
        components = {
            name: (weight * torch.tensor([0.0, coefficient])).sum()
            for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
        }
        instrument.observe_row(
            aux_loss=sum(components.values()),
            track1=zero_track,
            track2=zero_track,
            per_aux_losses=components,
        )
    instrument.emit_update(_update_event(S_CTRL))
    record = collector.gradient_records[0]
    assert record["track_norm_below_eps"] is True
    assert record["actual_ratio_denominator"] == pytest.approx(1e-12)
    assert record["weighted_aux_track_cosine"] == 0.0
    assert record["weighted_aux_signed_projection"] == 0.0
    assert all(value == 0.0 for value in record["per_aux_cosine_to_track"].values())


def test_joint_per_aux_reconstruction_ignores_independent_vjp_rounding_but_rejects_omission():
    event = _update_event(S_CTRL)
    per_aux = {
        "L_cot": torch.tensor([1.0], dtype=torch.float64),
        "L_future": torch.tensor([1e-7], dtype=torch.float64),
        "L_verify": torch.tensor([-1.0], dtype=torch.float64),
    }
    aggregate_fp32 = (
        torch.tensor([1.0], dtype=torch.float32)
        + torch.tensor([1e-7], dtype=torch.float32)
        + torch.tensor([-1.0], dtype=torch.float32)
    ).to(torch.float64)
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    collector.observe_contributions(
        event,
        g_track=torch.zeros(1, dtype=torch.float64),
        g_aux=aggregate_fp32,
        g_aux_joint=aggregate_fp32.clone(),
        per_aux=per_aux,
    )
    record = collector.gradient_records[0]
    assert record["per_aux_aggregate_discrepancy_norm"] == 0.0
    assert record["per_aux_aggregate_rounding_bound_norm"] == 0.0

    omitted = dict(per_aux)
    omitted["L_verify"] = torch.zeros(1, dtype=torch.float64)
    omitted_joint = sum(omitted.values(), torch.zeros_like(aggregate_fp32))
    rejecting = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    with pytest.raises(IBR1DiagnosticsContractError, match="reconstruct"):
        rejecting.observe_contributions(
            event,
            g_track=torch.zeros(1, dtype=torch.float64),
            g_aux=aggregate_fp32,
            g_aux_joint=omitted_joint,
            per_aux=omitted,
        )


def test_g6_joint_vjp_is_exact_when_deep_independent_vjps_round():
    torch.manual_seed(0)
    probe = nn.Parameter(torch.randn(16, 16) * 0.02)
    frozen_projection = torch.randn(16, 16) * 0.02
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=2, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((probe,), collector)

    for u_pre in range(2):
        for _ in range(2):
            inputs = torch.randn(2, 16)
            hidden = torch.nn.functional.gelu(
                torch.nn.functional.linear(inputs, probe)
            )
            shared = torch.nn.functional.gelu(
                torch.nn.functional.linear(hidden, frozen_projection)
            )
            heads = tuple(
                torch.randn(classes, 16) * 0.02 for classes in (17, 11, 5)
            )
            targets = torch.arange(2)
            raw_losses = {
                "L_cot": torch.nn.functional.cross_entropy(
                    shared @ heads[0].T, targets % 17
                ),
                "L_future": torch.nn.functional.cross_entropy(
                    shared @ heads[1].T, targets % 11
                ),
                "L_verify": ((shared @ heads[2].T) ** 2).mean(),
            }
            weighted = {
                name: IBR1_FROZEN_AUX_COEFFICIENTS[name] * raw_losses[name]
                for name in raw_losses
            }
            aggregate = sum(weighted.values(), shared.sum() * 0.0)
            track = shared.square().mean()
            instrument.observe_row(
                aux_loss=aggregate,
                track1=track,
                track2=track,
                per_aux_losses=weighted,
            )

        instrument.emit_update(_update_event(S_CTRL, u_pre))

    assert len(collector.gradient_records) == 2
    for record in collector.gradient_records:
        assert record["per_aux_aggregate_discrepancy_norm"] == 0.0
        assert record["per_aux_aggregate_rounding_bound_norm"] == 0.0


def test_g6_joint_vjp_rejects_same_value_gradient_substitution():
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((weight,), collector)
    components = {
        name: (weight * torch.tensor([0.0, coefficient])).sum()
        for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
    }
    aggregate = sum(components.values())
    tampered = dict(components)
    tampered["L_verify"] = tampered["L_verify"].detach() + weight.sum() * 0.0
    track = weight.sum() * 0.0

    with pytest.raises(IBR1DiagnosticsContractError, match="row aggregate"):
        instrument.observe_row(
            aux_loss=aggregate,
            track1=track,
            track2=track,
            per_aux_losses=tampered,
        )


def test_g6_joint_vjp_rejects_detached_graph_connected_zero():
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((weight,), collector)
    components = {
        name: (weight * torch.tensor([0.0, coefficient])).sum()
        for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
    }
    components["L_future"] = weight.sum() * 0.0
    aggregate = sum(components.values())
    tampered = dict(components)
    tampered["L_future"] = tampered["L_future"].detach()
    track = weight.sum() * 0.0

    with pytest.raises(IBR1DiagnosticsContractError, match="graph-connected"):
        instrument.observe_row(
            aux_loss=aggregate,
            track1=track,
            track2=track,
            per_aux_losses=tampered,
        )


def test_g6_joint_vjp_rejects_detached_leaf_with_requires_grad():
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((weight,), collector)
    components = {
        name: (weight * torch.tensor([0.0, coefficient])).sum()
        for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
    }
    components["L_future"] = weight.sum() * 0.0
    aggregate = sum(components.values())
    tampered = dict(components)
    tampered["L_future"] = tampered["L_future"].detach().requires_grad_()
    track = weight.sum() * 0.0

    with pytest.raises(IBR1DiagnosticsContractError, match="reach at least one"):
        instrument.observe_row(
            aux_loss=aggregate,
            track1=track,
            track2=track,
            per_aux_losses=tampered,
        )


def test_g6_joint_vjp_rejects_detached_aggregate_under_exact_cancellation():
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((weight,), collector)
    positive = weight.sum()
    components = {
        "L_cot": positive,
        "L_future": -positive,
        "L_verify": weight.sum() * 0.0,
    }
    aggregate = sum(components.values())
    detached_aggregate = aggregate.detach().requires_grad_()
    track = weight.sum() * 0.0

    with pytest.raises(IBR1DiagnosticsContractError, match="reach at least one"):
        instrument.observe_row(
            aux_loss=detached_aggregate,
            track1=track,
            track2=track,
            per_aux_losses=components,
        )


def test_g6_joint_vjp_rejects_global_loss_dtype_drift():
    weight = nn.Parameter(torch.tensor([1.0, 2.0]))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    instrument = IBR1G6Instrument((weight,), collector)
    components = {
        name: (weight * torch.tensor([0.0, coefficient])).sum().double()
        for name, coefficient in IBR1_FROZEN_AUX_COEFFICIENTS.items()
    }
    aggregate = sum(components.values())
    track = weight.sum() * 0.0

    with pytest.raises(IBR1DiagnosticsContractError, match="float32 probe device"):
        instrument.observe_row(
            aux_loss=aggregate,
            track1=track,
            track2=track,
            per_aux_losses=components,
        )


def _observe_optimizer_arm(
    collector: GradientDiagnosticsCollector,
    modules,
    engine_arm: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    optimizer, _receipt = build_arm_optimizer(modules, OptimizerContract())
    for parameter in modules.trainable_parameters():
        parameter.grad = torch.full_like(parameter, 100.0)

    before = {
        name: parameter.detach().clone()
        for name, parameter in modules.named_full_parameters()
        if parameter.requires_grad
    }

    def original(event):
        assert event.arm == engine_arm
        torch.nn.utils.clip_grad_norm_(
            list(modules.trainable_parameters()), 1.0
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    handle = OptimizerDiagnosticsHandle(
        _callbacks(original),
        optimizer=optimizer,
        modules=modules,
        collector=collector,
        engine_arm=engine_arm,
    )
    handle.callbacks.optimizer_step(_update_event(engine_arm))
    handle.close()
    after = {
        name: parameter.detach().clone()
        for name, parameter in modules.named_full_parameters()
        if parameter.requires_grad
    }
    return before, after


def test_optimizer_hooks_observe_real_clip_step_and_parameter_updates():
    torch.manual_seed(3)
    ctrl_modules = build_ibr1_package(DummyOfficialBase(), device="cpu")
    self_modules = copy.deepcopy(ctrl_modules)
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    _run_g6_update(collector)
    ctrl_before, ctrl_after = _observe_optimizer_arm(
        collector, ctrl_modules, S_CTRL
    )
    _observe_optimizer_arm(collector, self_modules, S_SELF)

    gradient_document, optimizer_document = collector.finalize()
    assert len(gradient_document["records"]) == 1
    assert len(optimizer_document["records"]) == 2
    ctrl_record = next(
        record
        for record in optimizer_document["records"]
        if record["engine_arm"] == S_CTRL
    )
    assert ctrl_record["pre_clip_full_grad_norm"] > 1.0
    assert ctrl_record["post_clip_full_grad_norm"] <= 1.000001
    assert ctrl_record["base_proj_parameter_update_norm"] > 0.0
    assert ctrl_record["action_head_parameter_update_norm"] > 0.0

    base_names = {
        name
        for name in ctrl_before
        if name.startswith("adapter.base.proj.")
    }
    base_update = sum(
        torch.sum(
            (ctrl_after[name].double() - ctrl_before[name].double()) ** 2
        )
        for name in base_names
    ).sqrt()
    assert ctrl_record["base_proj_parameter_update_norm"] == pytest.approx(
        float(base_update.item())
    )


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert set(left) == set(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_tree_equal(left_item, right_item)
    else:
        assert left == right


def test_optimizer_diagnostics_do_not_change_parameters_state_or_rng():
    torch.manual_seed(29)
    baseline_modules = build_ibr1_package(DummyOfficialBase(), device="cpu")
    observed_modules = copy.deepcopy(baseline_modules)
    contract = OptimizerContract()
    baseline_optimizer, _ = build_arm_optimizer(baseline_modules, contract)
    observed_optimizer, _ = build_arm_optimizer(observed_modules, contract)
    for baseline, observed in zip(
        baseline_modules.trainable_parameters(),
        observed_modules.trainable_parameters(),
    ):
        gradient = torch.linspace(
            -2.0,
            2.0,
            baseline.numel(),
            dtype=baseline.dtype,
        ).reshape_as(baseline)
        baseline.grad = gradient.clone()
        observed.grad = gradient.clone()

    def step(modules, optimizer):
        def callback(event):
            assert event.arm == S_CTRL
            torch.nn.utils.clip_grad_norm_(
                list(modules.trainable_parameters()), contract.grad_clip
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        return callback

    rng_before = torch.random.get_rng_state().clone()
    step(baseline_modules, baseline_optimizer)(_update_event(S_CTRL))
    collector = GradientDiagnosticsCollector(
        expected_gradient_updates=1, expected_optimizer_updates_per_arm=1
    )
    handle = OptimizerDiagnosticsHandle(
        _callbacks(step(observed_modules, observed_optimizer)),
        optimizer=observed_optimizer,
        modules=observed_modules,
        collector=collector,
        engine_arm=S_CTRL,
    )
    handle.callbacks.optimizer_step(_update_event(S_CTRL))
    handle.close()
    rng_after = torch.random.get_rng_state().clone()

    assert torch.equal(rng_before, rng_after)
    baseline_state = baseline_modules.full_state_dict()
    observed_state = observed_modules.full_state_dict()
    assert set(baseline_state) == set(observed_state)
    for key in baseline_state:
        assert torch.equal(baseline_state[key], observed_state[key])
    _assert_tree_equal(
        baseline_optimizer.state_dict(), observed_optimizer.state_dict()
    )
    assert all(
        parameter.grad is None
        for parameter in observed_modules.trainable_parameters()
    )


def _populate_training_geometry(collector: GeometryCollector, arm: str, count: int):
    remaining = count
    for position in range(256):
        row_count = min(8, remaining)
        remaining -= row_count
        prev = torch.zeros(1, 2, dtype=torch.float32)
        result = HeadForwardResult(
            prediction=_prediction(prev, row_count), g7_telemetry={}
        )
        collector.observe_training(
            result,
            prev,
            HeadEvent(
                arm=arm,
                row_position=position,
                original_row_index=10000 + position,
                u_pre=position // 2,
                branch="branch2",
                prev_source="logged" if arm == S_CTRL else "self",
            ),
        )
    assert remaining == 0


def _populate_eval_geometry(collector: GeometryCollector):
    snapshot_arms = {
        "update0_IBR1-SELF": IBR1_SELF,
        "update128_IBR1-CTRL": IBR1_CTRL,
        "update128_IBR1-SELF": IBR1_SELF,
    }
    for snapshot, family_arm in snapshot_arms.items():
        for mode in ("logged", "self"):
            for position in range(2):
                prev = torch.zeros(1, 2, dtype=torch.float32)
                prediction = _prediction(
                    prev,
                    1 if snapshot.startswith("update128") else 0,
                )
                collector.observe_eval(
                    prediction,
                    prev,
                    _row(position),
                    family_arm=family_arm,
                    snapshot=snapshot,
                    mode=mode,
                    position=position,
                )


def test_i2_exact_boundary_quantiles_u0_bucket_and_eval_pair_coverage():
    collector = GeometryCollector(
        expected_training_rows_per_arm=256,
        expected_eval_rows_per_snapshot_mode=2,
    )
    _populate_training_geometry(collector, S_CTRL, 102)
    _populate_training_geometry(collector, S_SELF, 103)
    _populate_eval_geometry(collector)

    training_records, eval_records, summary = collector.finalize()
    assert len(training_records) == 512
    assert len(eval_records) == 192
    training = summary["training_geometry"]
    ctrl = training["arms"][IBR1_CTRL]
    self_arm = training["arms"][IBR1_SELF]
    assert ctrl["I2_any_axis_denominator"] == 2048
    assert ctrl["I2_any_axis_violation_count"] == 102
    assert ctrl["I2_pass"] is True
    assert self_arm["I2_any_axis_violation_count"] == 103
    assert self_arm["I2_pass"] is False
    assert training["I2_pass"] is False
    assert "0-7" in ctrl["time_bin_violation_counts"]
    assert ctrl["overshoot_all_axis_cells"]["p50"] == 0.0
    assert ctrl["overshoot_violating_only_descriptive"]["support"] == 102
    assert len(ctrl["arm_by_axis_by_horizon_by_update_counts"]) == 4096
    assert ctrl["overshoot_quantiles_by_axis"]["forward"]["support"] == 2048
    assert ctrl["overshoot_quantiles_by_horizon"]["0"]["support"] == 512
    assert ctrl["overshoot_quantiles_by_u_pre"]["0"]["support"] == 32
    assert ctrl["overshoot_quantiles_by_time_bin"]["0-7"]["support"] == 256
    assert set(ctrl["row_within_update_violation_counts"]) == {"0", "1"}

    eval_summary = summary["eval_geometry"]
    assert eval_summary["records"] == 192
    assert eval_summary["self_update0_to_update128_pair_count"] == 64
    assert eval_summary["adds_scientific_threshold"] is False


def test_training_and_eval_wrappers_preserve_original_return_identity():
    training_collector = GeometryCollector(
        expected_training_rows_per_arm=1,
        expected_eval_rows_per_snapshot_mode=1,
    )
    prev = torch.zeros(1, 2, dtype=torch.float32)
    result = HeadForwardResult(prediction=_prediction(prev), g7_telemetry={})

    def head_forward(features, previous, event):
        del features, previous, event
        return result

    callbacks = _callbacks(lambda event: None)
    callbacks = ArmCallbacks(
        checkpoint_state=callbacks.checkpoint_state,
        feature_forward=callbacks.feature_forward,
        aux_forward=callbacks.aux_forward,
        head_forward=head_forward,
        track_loss=callbacks.track_loss,
        backward=callbacks.backward,
        optimizer_step=callbacks.optimizer_step,
    )
    wrapped_callbacks = wrap_training_head_forward(callbacks, training_collector)
    branch1 = wrapped_callbacks.head_forward(
        None,
        prev,
        HeadEvent(S_CTRL, 0, 1, 0, "branch1", "logged"),
    )
    assert branch1 is result
    assert training_collector.training_records == []
    branch2 = wrapped_callbacks.head_forward(
        None,
        prev,
        HeadEvent(S_CTRL, 0, 1, 0, "branch2", "logged"),
    )
    assert branch2 is result
    assert len(training_collector.training_records) == 1

    def predictor(row, previous, *, mode, reset, position):
        del row, previous, mode, reset, position
        return result.prediction

    wrapped_predictor = wrap_eval_predictor(
        predictor,
        training_collector,
        family_arm=IBR1_SELF,
        snapshot="update0_IBR1-SELF",
    )
    prediction = wrapped_predictor(
        _row(0), prev, mode="logged", reset=True, position=0
    )
    assert prediction is result.prediction
    assert len(training_collector.eval_records) == 16


def test_geometry_finalize_fails_closed_on_missing_rows():
    collector = GeometryCollector(
        expected_training_rows_per_arm=1,
        expected_eval_rows_per_snapshot_mode=1,
    )
    with pytest.raises(IBR1DiagnosticsContractError, match="cardinality"):
        collector.finalize()


def test_i2_rejects_a_prediction_mask_that_underreports_prebound_violations():
    collector = GeometryCollector(
        expected_training_rows_per_arm=1,
        expected_eval_rows_per_snapshot_mode=1,
    )
    prev = torch.zeros(1, 2, dtype=torch.float32)
    prediction = _prediction(prev, 1)
    forged = replace(
        prediction,
        prebound_violation_mask=torch.zeros_like(
            prediction.prebound_violation_mask
        ),
    )
    with pytest.raises(IBR1DiagnosticsContractError, match="differs"):
        collector.observe_training(
            HeadForwardResult(prediction=forged, g7_telemetry={}),
            prev,
            HeadEvent(S_CTRL, 0, 1, 0, "branch2", "logged"),
        )


def test_eval_finalize_rejects_mode_or_position_coverage_substitution():
    collector = GeometryCollector(
        expected_training_rows_per_arm=256,
        expected_eval_rows_per_snapshot_mode=2,
    )
    _populate_training_geometry(collector, S_CTRL, 0)
    _populate_training_geometry(collector, S_SELF, 0)
    _populate_eval_geometry(collector)
    for record in collector.eval_records:
        if (
            record["snapshot"] == "update0_IBR1-SELF"
            and record["mode"] == "logged"
            and record["row_position"] == 1
        ):
            record["mode"] = "self"
    with pytest.raises(
        IBR1DiagnosticsContractError,
        match="duplicate EVAL geometry cell|snapshot/mode cardinality",
    ):
        collector.finalize()
