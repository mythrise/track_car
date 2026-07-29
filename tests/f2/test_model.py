import math

import pytest
import torch
from torch import nn

from f2_experiment import controller as controller_core
from f2_experiment.model import (
    ACTION_MAX_ABS,
    AP2_HIDDEN_DIM,
    AP2_HORIZON,
    PREV_HIDDEN_DIM,
    AP2DeltaHead,
    BoundedContextFusion,
    F2AP2Model,
    F2ModelContractError,
    ap2_track_loss,
    assert_prev_free_tensor,
    assert_prev_free_tensors,
    assert_step0_controlled_axis_persistence,
    controlled_axis_targets,
    unit_l2,
)


def _method_inputs(batch: int = 2):
    return {
        "polar": torch.randn(batch, 3),
        "tim": torch.randn(batch, 5),
    }


def test_unit_l2_implements_the_frozen_safe_unit_operation():
    value = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    normalized = unit_l2(value)
    assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]))
    assert torch.equal(normalized[1], torch.zeros(2))


def test_context_contract_is_generic_in_d_and_zero_initialized():
    torch.manual_seed(2)
    fusion = BoundedContextFusion(16, {"polar": 3, "tim": 5})
    base = torch.randn(2, 16)
    prev = torch.tensor([[0.25, -0.5], [-0.1, 0.2]])
    composition = fusion.compose_context(base, _method_inputs())
    fused = fusion.condition_on_prev(composition, prev)

    assert fusion.prev_projection[0].in_features == 2
    assert fusion.prev_projection[0].out_features == PREV_HIDDEN_DIM
    assert fusion.prev_projection[2].out_features == 16
    assert isinstance(fusion.head_norm, nn.LayerNorm)
    assert fusion.head_norm.normalized_shape == (16,)
    assert fusion.head_norm.elementwise_affine is True
    assert torch.equal(fusion.s_prev, torch.zeros_like(fusion.s_prev))
    assert all(
        torch.equal(scale, torch.zeros_like(scale))
        for scale in fusion.method_scales.values()
    )

    expected_norm = torch.full((2,), math.sqrt(16))
    assert torch.allclose(composition.base_stream_norm, expected_norm, atol=1e-5)
    assert torch.allclose(composition.x_norm, expected_norm, atol=1e-5)
    assert all(
        torch.count_nonzero(stream).item() == 0
        for stream in composition.method_streams.values()
    )
    assert torch.count_nonzero(fused.prev_stream).item() == 0
    assert torch.count_nonzero(fused.telemetry.r_prev).item() == 0
    assert fused.head_input.shape == (2, 16)

    telemetry = fused.telemetry.as_dict()
    assert set(telemetry["per_method_over_base"]) == {"polar", "tim"}
    assert set(telemetry["abs_tanh_method_scales"]) == {"polar", "tim"}
    assert set(telemetry["layerscale_saturation_rate_both"]) == {
        "method",
        "prev",
    }


def test_method_and_prev_streams_have_constructive_g7_bounds():
    torch.manual_seed(5)
    fusion = BoundedContextFusion(32, {"polar": 3, "tim": 5})
    with torch.no_grad():
        for scale in fusion.method_scales.values():
            scale.fill_(20.0)
        fusion.s_prev.fill_(20.0)

    fused = fusion(
        torch.randn(4, 32),
        torch.rand(4, 2) * 2.0 - 1.0,
        _method_inputs(batch=4),
        {"polar": 1.0, "tim": 1.0},
    )
    telemetry = fused.telemetry
    assert all(
        torch.all(ratio <= 0.5001)
        for ratio in telemetry.per_method_over_base.values()
    )
    assert torch.all(telemetry.total_method_over_base <= 1.0001)
    assert torch.all(telemetry.r_prev <= 0.5001)
    assert telemetry.method_saturation_fraction.item() == pytest.approx(1.0)
    assert telemetry.prev_saturation_indicator.item() == pytest.approx(1.0)
    assert telemetry.abs_tanh_s_prev.item() == pytest.approx(1.0)


def test_orchestrator_alpha_sum_must_preserve_the_total_method_bound():
    fusion = BoundedContextFusion(8, {"polar": 2, "tim": 2, "future": 2})
    features = {name: torch.randn(1, 2) for name in fusion.method_dims}
    with pytest.raises(F2ModelContractError, match="ALPHA_SUM"):
        fusion(
            torch.randn(1, 8),
            torch.zeros(1, 2),
            features,
            {"polar": 1.0, "tim": 1.0, "future": 1.0},
        )


@pytest.mark.parametrize(
    "base",
    [torch.zeros(1, 8), torch.full((1, 8), float("nan"))],
)
def test_base_near_zero_and_nonfinite_are_hard_stops(base):
    fusion = BoundedContextFusion(8)
    with pytest.raises(F2ModelContractError):
        fusion(base, torch.zeros(1, 2))


def test_method_and_previous_action_contracts_fail_closed():
    fusion = BoundedContextFusion(8, {"polar": 2})
    base = torch.randn(1, 8)
    with pytest.raises(F2ModelContractError, match="NONFINITE"):
        fusion(
            base,
            torch.zeros(1, 2),
            {"polar": torch.tensor([[float("inf"), 0.0]])},
        )
    with pytest.raises(F2ModelContractError, match="OUTSIDE_FROZEN_DOMAIN"):
        fusion(base, torch.tensor([[ACTION_MAX_ABS + 0.01, 0.0]]), {"polar": torch.zeros(1, 2)})
    with pytest.raises(F2ModelContractError, match="NONFINITE"):
        fusion(
            base,
            torch.tensor([[float("nan"), 0.0]]),
            {"polar": torch.zeros(1, 2)},
        )


def test_ap2_zero_init_is_exact_two_axis_persistence_and_uses_shared_clamp(
    monkeypatch,
):
    calls = []
    original_clamp_stage = controller_core.clamp_stage

    def recording_clamp(raw_fy, max_abs=ACTION_MAX_ABS):
        calls.append(tuple(raw_fy))
        return original_clamp_stage(raw_fy, max_abs=max_abs)

    monkeypatch.setattr(controller_core, "clamp_stage", recording_clamp)
    head = AP2DeltaHead(12)
    prev = torch.tensor([[0.25, -0.5], [-0.75, 0.125]])
    prediction = head(torch.randn(2, 12), prev)

    assert head.trunk[0].out_features == AP2_HIDDEN_DIM
    assert head.forward_branch.out_features == AP2_HORIZON
    assert head.yaw_branch.out_features == AP2_HORIZON
    assert torch.count_nonzero(head.forward_branch.weight).item() == 0
    assert torch.count_nonzero(head.forward_branch.bias).item() == 0
    assert torch.count_nonzero(head.yaw_branch.weight).item() == 0
    assert torch.count_nonzero(head.yaw_branch.bias).item() == 0
    assert prediction.delta_fy.shape == (2, AP2_HORIZON, 2)
    assert prediction.raw_actions.shape == (2, AP2_HORIZON, 3)
    assert prediction.bounded_future_actions.shape == (2, AP2_HORIZON - 1, 3)
    assert len(calls) == 2 * (AP2_HORIZON - 1)
    assert prediction.bounded_future_actions.requires_grad is False
    assert_step0_controlled_axis_persistence(prediction, prev)

    for batch_index in range(2):
        for future_index in range(AP2_HORIZON - 1):
            expected = original_clamp_stage(tuple(prev[batch_index].tolist()))
            assert torch.equal(
                prediction.bounded_future_actions[batch_index, future_index],
                torch.tensor(expected),
            )


def test_ap2_uses_unbounded_deltas_and_cumulative_raw_reconstruction():
    head = AP2DeltaHead(4)
    with torch.no_grad():
        head.forward_branch.bias.fill_(0.25)
        head.yaw_branch.bias.fill_(-0.5)
    prev = torch.tensor([[0.1, 0.2]])
    prediction = head(torch.zeros(1, 4), prev)

    steps = torch.arange(1, AP2_HORIZON + 1, dtype=torch.float32)
    assert torch.allclose(prediction.raw_actions[0, :, 0], 0.1 + 0.25 * steps)
    assert torch.allclose(prediction.raw_actions[0, :, 2], 0.2 - 0.5 * steps)
    assert torch.count_nonzero(prediction.raw_actions[..., 1]).item() == 0
    assert prediction.raw_actions.abs().max() > 1.0
    assert prediction.bounded_future_actions.abs().max() <= 1.0


def test_track_loss_uses_raw_actions_and_only_controlled_axes():
    head = AP2DeltaHead(4)
    with torch.no_grad():
        head.forward_branch.bias.fill_(2.0)
        head.yaw_branch.bias.fill_(-2.0)
    prediction = head(torch.zeros(1, 4), torch.zeros(1, 2))
    target = prediction.raw_actions.detach().clone()
    target[..., 1] = 1000.0

    loss = ap2_track_loss(prediction, target)
    assert prediction.raw_actions.abs().max() > 1.0
    assert not torch.allclose(
        prediction.raw_actions[..., 0],
        torch.cat(
            (
                prediction.raw_actions[..., :1, 0],
                prediction.bounded_future_actions[..., 0],
            ),
            dim=-1,
        ),
    )
    assert loss.forward.item() == pytest.approx(0.0)
    assert loss.yaw.item() == pytest.approx(0.0)
    assert loss.total.item() == pytest.approx(0.0)


def test_raw_track_loss_backpropagates_to_both_delta_branches():
    head = AP2DeltaHead(6)
    with torch.no_grad():
        head.forward_branch.bias.fill_(0.1)
        head.yaw_branch.bias.fill_(-0.1)
    prediction = head(torch.randn(2, 6), torch.zeros(2, 2))
    loss = ap2_track_loss(prediction, torch.zeros_like(prediction.raw_actions))
    loss.total.backward()

    assert head.forward_branch.bias.grad is not None
    assert head.yaw_branch.bias.grad is not None
    assert torch.count_nonzero(head.forward_branch.bias.grad).item() > 0
    assert torch.count_nonzero(head.yaw_branch.bias.grad).item() > 0
    assert prediction.bounded_future_actions.grad_fn is None


def test_prev_free_graph_audit_rejects_even_zero_valued_dependencies():
    fusion = BoundedContextFusion(8)
    base = torch.randn(2, 8, requires_grad=True)
    prev = torch.randn(2, 2, requires_grad=True)
    composition = fusion.compose_context(base)

    assert_prev_free_tensor(composition.x, prev, label="x_ctx")
    assert_prev_free_tensors(
        {"x_ctx": composition.x, "L_aux": composition.x.square().mean()},
        prev,
    )
    leaked = composition.x + 0.0 * prev.sum()
    with pytest.raises(F2ModelContractError, match="PREV_GRAPH_LEAK"):
        assert_prev_free_tensor(leaked, prev, label="x_ctx")


def test_optimizer_groups_are_exact_and_keep_scales_at_zero_weight_decay():
    model = F2AP2Model(16, {"polar": 3, "tim": 5})
    groups = model.optimizer_parameter_groups()
    by_name = {group["name"]: group for group in groups}
    assert set(by_name) == {
        "ordinary_head",
        "method_layerscales",
        "prev_layerscale",
    }
    assert by_name["ordinary_head"]["weight_decay"] == pytest.approx(1e-4)
    assert by_name["method_layerscales"]["weight_decay"] == pytest.approx(0.0)
    assert by_name["prev_layerscale"]["weight_decay"] == pytest.approx(0.0)
    assert by_name["prev_layerscale"]["params"] == [model.fusion.s_prev]

    grouped_ids = [
        id(parameter)
        for group in groups
        for parameter in group["params"]
    ]
    assert len(grouped_ids) == len(set(grouped_ids))
    assert set(grouped_ids) == {id(parameter) for parameter in model.parameters()}


def test_two_axis_helpers_never_claim_full_vector_identity():
    actions = torch.tensor([[[0.2, 0.7, -0.4]]])
    controlled = controlled_axis_targets(actions)
    assert controlled.shape == (1, 1, 2)
    assert torch.equal(controlled, torch.tensor([[[0.2, -0.4]]]))


def test_loss_fails_closed_on_zero_mask_support():
    prediction = AP2DeltaHead(4)(torch.zeros(1, 4), torch.zeros(1, 2))
    with pytest.raises(F2ModelContractError, match="zero support"):
        ap2_track_loss(
            prediction,
            torch.zeros_like(prediction.raw_actions),
            torch.zeros(1, AP2_HORIZON),
        )
