from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from f2_experiment.model import (
    AP2_HORIZON,
    F2AP2Model,
    ap2_track_loss,
)
from ibr1_experiment.model import (
    IBR1AP2Model,
    IBR1ModelContractError,
    IBR1NormalizedBoundedHead,
    normalized_cumulative_decode,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_init_is_numerically_exact_persistence_including_boundaries(dtype):
    head = IBR1NormalizedBoundedHead(4).to(dtype=dtype)
    context = torch.randn(5, 4, dtype=dtype)
    prev = torch.tensor(
        [[-1.0, 1.0], [-0.5, 0.25], [0.0, 0.0], [0.75, -0.9], [1.0, -1.0]],
        dtype=dtype,
    )

    prediction = head(context, prev)
    expected = prev.unsqueeze(1).expand(-1, AP2_HORIZON, -1)

    assert torch.equal(prediction.raw_fy, expected)
    assert torch.count_nonzero(prediction.latent_delta_fy).item() == 0
    assert torch.count_nonzero(prediction.delta_fy).item() == 0
    assert not prediction.prebound_violation_mask.any()


def test_known_latent_sequence_matches_closed_form_and_telescopes():
    prev = torch.tensor([[0.25, -0.5]])
    latent = torch.tensor(
        [
            [
                [0.1, -0.2],
                [0.2, 0.1],
                [-0.4, 0.3],
                [0.0, -0.2],
                [0.5, 0.0],
                [-0.1, -0.4],
                [0.2, 0.2],
                [-0.3, 0.1],
            ]
        ]
    )

    raw, realized, cumulative, prebound = normalized_cumulative_decode(prev, latent)
    expected_cumulative = torch.cumsum(latent, dim=1)
    expected = (prev.unsqueeze(1) + expected_cumulative) / (
        1.0 + torch.abs(expected_cumulative)
    )

    assert torch.equal(cumulative, expected_cumulative)
    assert torch.equal(prebound, prev.unsqueeze(1) + expected_cumulative)
    assert torch.allclose(raw, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(
        prev.unsqueeze(1) + torch.cumsum(realized, dim=1),
        raw,
        atol=1e-7,
        rtol=0.0,
    )


def test_extreme_latent_values_remain_bounded_and_strafe_is_zero():
    head = IBR1NormalizedBoundedHead(3)
    with torch.no_grad():
        head.forward_branch.bias.copy_(
            torch.tensor([100.0, -100.0] * 4)
        )
        head.yaw_branch.bias.copy_(
            torch.tensor([-100.0, 100.0] * 4)
        )
    prediction = head(torch.zeros(2, 3), torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))

    assert torch.all(torch.abs(prediction.raw_fy) <= 1.0)
    assert torch.count_nonzero(prediction.raw_actions[..., 1]).item() == 0
    assert torch.all(prediction.normalizer_fy >= 1.0)


@pytest.mark.parametrize(
    ("previous", "target", "expected_gradient_sign"),
    [(1.0, 0.0, 1), (-1.0, 0.0, -1)],
)
def test_zero_point_gradient_can_move_boundary_action_inward(
    previous, target, expected_gradient_sign
):
    prev = torch.tensor([[previous, 0.0]])
    latent = torch.zeros(1, AP2_HORIZON, 2, requires_grad=True)
    raw, _realized, _cumulative, _prebound = normalized_cumulative_decode(
        prev, latent
    )
    loss = (raw[0, 0, 0] - target).square()
    loss.backward()

    gradient = float(latent.grad[0, 0, 0].item())
    assert gradient * expected_gradient_sign > 0.0


def test_model_parameter_bytes_and_names_match_f2_at_same_seed():
    torch.manual_seed(123)
    f2_model = F2AP2Model(16, {"tim": 5})
    f2_rng_after = torch.get_rng_state().clone()
    torch.manual_seed(123)
    ibr1_model = IBR1AP2Model(16, {"tim": 5})
    ibr1_rng_after = torch.get_rng_state().clone()

    f2_state = f2_model.state_dict()
    ibr1_state = ibr1_model.state_dict()
    assert f2_state.keys() == ibr1_state.keys()
    for name in f2_state:
        f2_bytes = f2_state[name].detach().cpu().contiguous().numpy().tobytes()
        ibr1_bytes = (
            ibr1_state[name].detach().cpu().contiguous().numpy().tobytes()
        )
        assert f2_bytes == ibr1_bytes, name
    assert torch.equal(f2_rng_after, ibr1_rng_after)


def test_track_loss_reads_raw_actions_not_detached_future_telemetry():
    head = IBR1NormalizedBoundedHead(4)
    prediction = head(torch.randn(1, 4), torch.tensor([[0.2, -0.1]]))
    target = torch.zeros(1, AP2_HORIZON, 3)
    reference = ap2_track_loss(prediction, target).total
    changed = replace(
        prediction,
        bounded_future_actions=torch.full_like(
            prediction.bounded_future_actions, 999.0
        ),
    )

    assert torch.equal(reference, ap2_track_loss(changed, target).total)


def test_invalid_prev_and_nonfinite_latent_fail_closed():
    latent = torch.zeros(1, AP2_HORIZON, 2)
    with pytest.raises(IBR1ModelContractError, match="OUTSIDE_FROZEN_DOMAIN"):
        normalized_cumulative_decode(torch.tensor([[1.01, 0.0]]), latent)

    latent[0, 0, 0] = float("nan")
    with pytest.raises(IBR1ModelContractError, match="NONFINITE"):
        normalized_cumulative_decode(torch.zeros(1, 2), latent)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_geometry_fails_closed(dtype):
    with pytest.raises(IBR1ModelContractError, match="float32 production"):
        normalized_cumulative_decode(
            torch.zeros(1, 2, dtype=dtype),
            torch.zeros(1, AP2_HORIZON, 2, dtype=dtype),
        )


def test_mixed_dtype_geometry_fails_closed_without_silent_cast():
    with pytest.raises(IBR1ModelContractError, match="same dtype"):
        normalized_cumulative_decode(
            torch.zeros(1, 2, dtype=torch.float64),
            torch.zeros(1, AP2_HORIZON, 2, dtype=torch.float32),
        )
