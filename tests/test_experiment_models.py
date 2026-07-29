from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn as nn


OPEN_TRACK_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "OpenTrackVLA"
if str(OPEN_TRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(OPEN_TRACK_ROOT))

from harness.baseline_adapter import OpenTrackVLABaselineAdapter  # noqa: E402
from harness.base_repro.tim import TIM, roi_pool_candidate  # noqa: E402
from harness.sequence_state import (  # noqa: E402
    continues_sequence,
    detach_state,
    sample_sequence_key,
)
from harness.trackvla_lite import (  # noqa: E402
    FactorizedPolarReasoningToken,
    TrackVLAPlusPlusLite,
)
from harness.harness_wrapper import PFEMHarness  # noqa: E402
from model import CrossModalityProjector, MPSStableLayerNorm  # noqa: E402
from scripts.eval_offline import (  # noqa: E402
    checkpoint_model_family,
    validate_experiment_identity,
)


def test_mps_stable_layer_norm_keeps_projector_state_dict_keys():
    projector = CrossModalityProjector(16, 8)
    assert isinstance(projector.net[0], MPSStableLayerNorm)
    assert "net.0.weight" in projector.state_dict()
    assert "net.0.bias" in projector.state_dict()


def test_manual_layer_norm_matches_native_cpu_gradients():
    torch.manual_seed(7)
    reference = nn.LayerNorm(16)
    stable = MPSStableLayerNorm(16)
    stable.load_state_dict(reference.state_dict())
    x_reference = torch.randn(3, 5, 16, requires_grad=True)
    x_stable = x_reference.detach().clone().requires_grad_(True)
    upstream = torch.randn_like(x_reference)

    y_reference = reference(x_reference)
    y_stable = stable._manual_forward(x_stable)
    (y_reference * upstream).sum().backward()
    (y_stable * upstream).sum().backward()

    assert torch.allclose(y_stable, y_reference, atol=2e-6, rtol=2e-6)
    assert torch.allclose(x_stable.grad, x_reference.grad, atol=3e-6, rtol=3e-6)
    assert torch.allclose(stable.weight.grad, reference.weight.grad, atol=3e-6, rtol=3e-6)
    assert torch.allclose(stable.bias.grad, reference.bias.grad, atol=3e-6, rtol=3e-6)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS is unavailable"
)
def test_mps_stable_layer_norm_affine_gradients_match_cpu_reference():
    torch.manual_seed(11)
    reference = nn.LayerNorm(1536)
    stable = MPSStableLayerNorm(1536).to("mps")
    stable.load_state_dict(reference.state_dict())
    inputs = [
        torch.randn(1, 124, 1536),
        torch.randn(1, 64, 1536),
        torch.randn(1, 64, 1536),
    ]
    upstream = [torch.randn_like(value) * 0.01 for value in inputs]

    reference_loss = sum(
        (reference(value) * grad).sum() for value, grad in zip(inputs, upstream)
    )
    stable_loss = sum(
        (stable(value.to("mps")) * grad.to("mps")).sum()
        for value, grad in zip(inputs, upstream)
    )
    reference_loss.backward()
    stable_loss.backward()

    assert torch.allclose(
        stable.weight.grad.cpu(), reference.weight.grad, atol=2e-3, rtol=2e-3
    )
    assert torch.allclose(
        stable.bias.grad.cpu(), reference.bias.grad, atol=2e-3, rtol=2e-3
    )


class DummyNative(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        coarse_tokens,
        coarse_tidx,
        fine_tokens,
        fine_tidx,
        instructions,
        yaw_hist=None,
        yaw_curr=None,
    ):
        del coarse_tidx, fine_tokens, fine_tidx, instructions, yaw_hist, yaw_curr
        return coarse_tokens[:, :2, :3] * self.scale


class DummyLiteBase(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.D = d_model
        self.cfg = SimpleNamespace(n_waypoints=2, use_angle_tvi=False)
        self.proj = nn.Linear(4, d_model)
        self.planner = nn.Linear(d_model, 6)
        self.llm = nn.Linear(d_model, d_model)
        self.act_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("alpha_task", torch.ones(1, 1, 3))


def test_native_baseline_adapter_adds_no_state_or_action_logic():
    adapter = OpenTrackVLABaselineAdapter(DummyNative())
    coarse = torch.ones(1, 2, 3)
    output = adapter.forward_step(
        coarse,
        torch.zeros(1, 2, dtype=torch.long),
        torch.zeros(1, 1, 3),
        torch.zeros(1, 1, dtype=torch.long),
        ["follow"],
    )
    assert torch.allclose(output["waypoints"], torch.full((1, 2, 3), 2.0))
    assert output["new_state"] == {}
    assert list(adapter.named_parameters()) == [("base.scale", adapter.base.scale)]


def test_trackvla_lite_contains_only_polar_tim_and_reason_feedback():
    model = TrackVLAPlusPlusLite(DummyLiteBase(), expected_history=31)
    assert model.expected_history == 31
    assert model.tim.n_tokens == 4
    for forbidden in ("future", "verifier", "events", "orch"):
        assert not hasattr(model, forbidden)
    assert all(parameter.requires_grad for parameter in model.base.proj.parameters())
    assert all(parameter.requires_grad for parameter in model.base.planner.parameters())
    assert not any(parameter.requires_grad for parameter in model.base.llm.parameters())
    assert torch.count_nonzero(model.action_delta[-1].weight) == 0
    assert torch.count_nonzero(model.action_delta[-1].bias) == 0

    polar_only = TrackVLAPlusPlusLite(DummyLiteBase(), use_tim=False)
    assert polar_only.tim is None
    polar_state = polar_only.init_state(1, "cpu")
    assert polar_state["tim"] is None
    assert polar_state["has_pending"].item() is False


def test_trackvla_lite_tim_only_applies_a_previous_pending_candidate():
    model = TrackVLAPlusPlusLite(DummyLiteBase(), expected_history=31)
    state = model.init_state(1, "cpu")
    unchanged = model._apply_pending_update(state)
    assert torch.count_nonzero(unchanged["mem"]) == 0

    state["pending_candidate"] = torch.ones_like(state["pending_candidate"])
    state["pending_confidence"] = torch.tensor([0.8])
    state["pending_invalid"] = torch.tensor([False])
    state["has_pending"] = torch.tensor([True])
    updated = model._apply_pending_update(state)
    assert updated["initialized"].item() is True
    assert torch.count_nonzero(updated["mem"]) > 0


def test_pfem_tim_only_applies_a_previous_pending_candidate():
    model = PFEMHarness(
        DummyLiteBase(),
        use_future=False,
        use_verifier=False,
        use_events=False,
        use_orchestrator=False,
    )
    state = model.init_state(1, "cpu")
    unchanged = model._apply_pending_tim_update(state)
    assert torch.count_nonzero(unchanged["mem"]) == 0

    state["pending_candidate"] = torch.ones_like(state["pending_candidate"])
    state["pending_confidence"] = torch.tensor([0.8])
    state["pending_q_write"] = torch.tensor([1.0])
    state["pending_invalid"] = torch.tensor([False])
    state["has_pending"] = torch.tensor([True])
    updated = model._apply_pending_tim_update(state)
    assert updated["initialized"].item() is True
    assert torch.count_nonzero(updated["mem"]) > 0


def test_factorized_polar_token_is_differentiable_feedback():
    module = FactorizedPolarReasoningToken(8, n_theta=3, n_dist=2)
    cot = {
        "theta_logits": torch.randn(2, 3, requires_grad=True),
        "dist_logits": torch.randn(2, 2, requires_grad=True),
        "invalid_logit": torch.randn(2, requires_grad=True),
    }
    token = module(cot)
    assert token.shape == (2, 8)
    token.square().mean().backward()
    assert cot["theta_logits"].grad is not None
    assert cot["dist_logits"].grad is not None
    assert cot["invalid_logit"].grad is not None


def test_rolling_sequence_only_continues_for_adjacent_clean_frames():
    assert continues_sequence(("clip", 10, False), ("clip", 11, False))
    assert not continues_sequence(("clip", 10, False), ("clip", 12, False))
    assert not continues_sequence(("clip", 10, False), ("other", 11, False))
    assert not continues_sequence(("clip", 10, False), ("clip", 11, True))
    state = {"x": torch.ones(1, requires_grad=True)}
    detached = detach_state(state)
    assert detached["x"].requires_grad is False

    batch = {
        "sequence_id": ["clean_chunk"],
        "chunk_id": ["stats_chunk"],
        "clip_id": ["clip_3"],
        "frame_idx": torch.tensor([11]),
        "mirrored": torch.tensor([False]),
    }
    assert sample_sequence_key(batch) == ("clean_chunk", 11, False)


def test_paper_tim_counts_invalid_as_zero_confidence_without_initializing_memory():
    tim = TIM(4, n_tokens=2)
    state = tim.init_state(1, "cpu")
    invalid_state = tim.update(
        state,
        torch.ones(1, 2, 4),
        torch.tensor([0.9]),
        torch.ones(1),
        invalid_mask=torch.tensor([True]),
        count_invalid_in_average=True,
    )
    assert invalid_state["C_cnt"].item() == 1
    assert invalid_state["C_avg"].item() == 0.0
    assert invalid_state["initialized"].item() is False
    assert torch.count_nonzero(invalid_state["mem"]) == 0

    valid_state = tim.update(
        invalid_state,
        torch.ones(1, 2, 4),
        torch.tensor([0.8]),
        torch.ones(1),
        invalid_mask=torch.tensor([False]),
        count_invalid_in_average=True,
    )
    assert valid_state["C_cnt"].item() == 2
    assert valid_state["C_avg"].item() == pytest.approx(0.4)
    assert valid_state["initialized"].item() is True


def test_polar_roi_pool_maps_angle_to_8x8_image_columns():
    column_ids = torch.arange(8).view(1, 1, 8, 1).expand(1, 8, 8, 1)
    features = column_ids.reshape(1, 64, 1).float()
    center = roi_pool_candidate(features, torch.tensor([30]), 60, n_tokens=4)
    left = roi_pool_candidate(features, torch.tensor([25]), 60, n_tokens=4)
    right = roi_pool_candidate(features, torch.tensor([34]), 60, n_tokens=4)
    assert center.shape == (1, 4, 1)
    assert torch.all(center == 4)
    assert torch.all(left == 0)
    assert torch.all(right == 7)


def test_collated_temporal_fields_are_available_to_future_loss():
    # Dataset-level tensor plumbing is covered without loading image tokens by
    # checking the collate contract directly.
    from model import collate_batch

    base = {
        "coarse_tokens": torch.zeros(1, 2),
        "coarse_tidx": torch.zeros(1, dtype=torch.long),
        "fine_tokens": torch.zeros(1, 2),
        "fine_tidx": torch.zeros(1, dtype=torch.long),
        "yaw_hist": torch.zeros(1),
        "yaw_curr": torch.zeros(1),
        "waypoints": torch.zeros(1, 3),
        "step_actions": torch.zeros(1, 3),
        "delta_vel": torch.zeros(1, 3),
        "prev_action": torch.zeros(3),
        "transition_type": "steady_forward",
        "episode": "e",
        "sequence_id": "s",
        "chunk_id": "c",
        "clip_id": "p",
        "frame_idx": 0,
        "mirrored": False,
        "valid_mask": torch.ones(1, dtype=torch.bool),
        "instruction": "follow",
        "current_path": "frame.jpg",
        "polar_theta_idx": torch.tensor(0),
        "polar_dist_idx": torch.tensor(0),
        "polar_invalid": torch.tensor(0.0),
    }
    for horizon in (4, 8, 16):
        base[f"fut_valid_{horizon}"] = torch.tensor(True)
        base[f"fut_vis_{horizon}"] = torch.tensor(1.0)
        base[f"fut_theta_idx_{horizon}"] = torch.tensor(2)
        base[f"fut_dist_idx_{horizon}"] = torch.tensor(3)
    batch = collate_batch([base])
    assert batch["fut_valid_8"].tolist() == [True]
    assert batch["fut_theta_idx_16"].tolist() == [2]


def test_pfem_ablation_switches_are_explicit_and_shape_preserving():
    from harness.harness_wrapper import PFEMHarness

    model = PFEMHarness(
        DummyLiteBase(),
        label_mode="absolute",
        use_cot_loss=False,
        use_tim=False,
        use_future=False,
        use_verifier=False,
        use_events=False,
        use_orchestrator=False,
    )
    assert model.use_cot_loss is False
    assert model.use_tim is False
    assert model.use_future is False
    assert model.use_verifier is False
    assert model.use_events is False
    assert model.use_orchestrator is False


def test_checkpoint_family_requires_explicit_supported_metadata():
    with pytest.raises(ValueError, match="missing explicit model_family"):
        checkpoint_model_family({"meta": {}})
    assert (
        checkpoint_model_family({"meta": {"model_family": "trackvla_pp_lite"}})
        == "trackvla_pp_lite"
    )


def test_headline_experiment_ids_bind_family_and_memory_mode():
    assert validate_experiment_identity(
        {
            "experiment_id": "B1",
            "model_family": "trackvla_pp_lite",
            "state_mode": "rolling",
        }
    )
    assert validate_experiment_identity(
        {
            "experiment_id": "H0-S",
            "model_family": "pfem_harness",
            "state_mode": "stateless",
        }
    )
    with pytest.raises(ValueError, match="state_mode=rolling"):
        validate_experiment_identity(
            {
                "experiment_id": "H0",
                "model_family": "pfem_harness",
                "state_mode": "stateless",
            }
        )
