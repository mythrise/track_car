from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn


OPEN_TRACK_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "OpenTrackVLA"
if str(OPEN_TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_TRACK_ROOT))

from harness.core.event_sampling import (  # noqa: E402
    compute_event_sampling_weights,
    weighted_event_fraction,
)
from harness.core.step_planner import (  # noqa: E402
    DeltaVelocityHead,
    StepActionHead,
    step_action_track_loss,
)
from harness.harness_wrapper import PFEMHarness  # noqa: E402


class DummyBase(nn.Module):
    def __init__(self, d_model=8, n_waypoints=8):
        super().__init__()
        self.D = d_model
        self.cfg = SimpleNamespace(n_waypoints=n_waypoints)
        self.proj = nn.Linear(1, 1)
        self.planner = nn.Linear(d_model, n_waypoints * 3)
        self.register_buffer("alpha_task", torch.ones(1, 1, 3))


class RaisingPlanner(nn.Module):
    def forward(self, _ctx):
        raise AssertionError("absolute planner must not run in step_action mode")


class FixedPlanner(nn.Module):
    def __init__(self, n_waypoints, value=0.25):
        super().__init__()
        self.n_waypoints = n_waypoints
        self.value = value

    def forward(self, ctx):
        return torch.full(
            (ctx.size(0), self.n_waypoints, 3),
            self.value,
            dtype=ctx.dtype,
            device=ctx.device,
        )


def test_step_action_head_decouples_axes_and_freezes_strafe():
    head = StepActionHead(16, n_steps=8)
    output = head(torch.randn(3, 16))
    assert output.shape == (3, 8, 3)
    assert torch.count_nonzero(output[..., 1]) == 0
    assert torch.all(output[..., (0, 2)].abs() <= 1.0)
    assert isinstance(head.trunk[0], nn.LayerNorm)
    assert head.trunk[1].out_features == 256


def test_step_action_mode_never_calls_base_planner_and_rebuilds_waypoints():
    base = DummyBase(n_waypoints=2)
    base.planner = RaisingPlanner()
    harness = PFEMHarness(base, label_mode="step_action", dt=0.1)

    class FixedHead(nn.Module):
        def forward(self, ctx):
            return torch.tensor(
                [[[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]]],
                dtype=ctx.dtype,
                device=ctx.device,
            ).expand(ctx.size(0), -1, -1)

    harness.step_action_head = FixedHead()
    waypoints, step_actions = harness._predict_tracking(
        torch.zeros(1, 8),
        torch.ones(1),
        {"mode": torch.zeros(1, dtype=torch.long), "alpha_verify": torch.ones(1)},
        torch.full((1, 2, 3), 100.0),
    )
    assert step_actions.shape == (1, 2, 3)
    np.testing.assert_allclose(waypoints[0, 0].detach().numpy(), [0.1, 0.0, 0.1], atol=1e-6)
    assert float(waypoints.abs().max()) < 1.0


def test_absolute_mode_preserves_planner_and_inference_residual_behavior():
    base = DummyBase(n_waypoints=2)
    base.planner = FixedPlanner(2)
    harness = PFEMHarness(base, label_mode="absolute")
    ctx = torch.zeros(1, 8)
    delta = torch.full((1, 2, 3), 0.1)
    orch = {"mode": torch.ones(1, dtype=torch.long), "alpha_verify": torch.full((1,), 0.5)}

    harness.train()
    training_waypoints, step_actions = harness._predict_tracking(
        ctx, torch.ones(1), orch, delta
    )
    assert step_actions is None
    assert torch.allclose(training_waypoints, torch.full((1, 2, 3), 0.25))

    harness.eval()
    inference_waypoints, _ = harness._predict_tracking(ctx, torch.ones(1), orch, delta)
    assert torch.allclose(inference_waypoints, torch.full((1, 2, 3), 0.30))


def test_step_action_loss_masks_invalid_steps_and_weights_yaw_and_aux():
    pred = torch.zeros(1, 2, 3)
    target = torch.tensor([[[1.0, 0.0, 1.0], [100.0, 0.0, 100.0]]])
    pred_delta = torch.zeros_like(pred)
    target_delta = torch.tensor([[[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]])
    total, forward, yaw, delta = step_action_track_loss(
        pred,
        target,
        torch.tensor([[True, False]]),
        lambda_yaw=2.0,
        pred_delta_vel=pred_delta,
        target_delta_vel=target_delta,
    )
    assert forward.item() == pytest.approx(0.5)
    assert yaw.item() == pytest.approx(0.5)
    assert delta.item() == pytest.approx(1.0 / 6.0)
    assert total.item() == pytest.approx(0.5 + 2.0 * 0.5 + 0.2 / 6.0)


def test_aux_head_range_and_prev_action_context_noise():
    aux_head = DeltaVelocityHead(8, n_steps=4)
    assert torch.all(aux_head(torch.randn(2, 8)).abs() <= 2.0)

    harness = PFEMHarness(DummyBase(), label_mode="step_action", aux_delta_vel=True)
    assert harness.context_proj.in_features == 8 * 7 + 64
    previous = torch.zeros(2, 3)
    harness.eval()
    clean = harness._condition_prev_action(previous)
    harness.train()
    torch.manual_seed(7)
    noisy = harness._condition_prev_action(previous)
    assert not torch.allclose(clean, noisy)


def test_event_sampling_reaches_40_percent_when_cap_allows_and_never_exceeds_10x():
    labels = ["steady_forward"] * 80
    for event in ("turn_onset", "sustained_turn", "turn_exit", "other"):
        labels.extend([event] * 5)
    weights = compute_event_sampling_weights(labels)
    assert weighted_event_fraction(labels, weights) == pytest.approx(0.4)
    assert max(weights) <= 10.0

    capped = compute_event_sampling_weights(["steady_forward"] * 100 + ["turn_onset"])
    assert max(capped) == 10.0
