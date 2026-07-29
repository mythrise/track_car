from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
from torch import nn

from f2_experiment.model import assert_prev_free_tensors
from f2_experiment.opentrack_adapter import (
    DifferentiablePolarToken,
    F2ObservationContractError,
    OpenTrackVLAF2ObservationAdapter,
    PrevFreeFutureModule,
    SelfCorrectnessHead,
)
from third_party.OpenTrackVLA.harness.base_repro.polar_cot import PolarCoTHead
from third_party.OpenTrackVLA.harness.base_repro.tim import TIM
from third_party.OpenTrackVLA.harness.core.event_bank import CognitiveEventBank
from third_party.OpenTrackVLA.harness.core.orchestrator import Orchestrator


class DummyLLM(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.mix = nn.Linear(d_model, d_model, bias=False)

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
        return SimpleNamespace(last_hidden_state=self.mix(inputs_embeds))


class DummyOfficialBase(nn.Module):
    """Small official-interface stand-in; it does not load model weights."""

    def __init__(self, input_dim: int = 5, d_model: int = 8) -> None:
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


def _adapter(batch_size: int = 2) -> tuple[OpenTrackVLAF2ObservationAdapter, dict]:
    del batch_size
    torch.manual_seed(19)
    adapter = OpenTrackVLAF2ObservationAdapter(
        DummyOfficialBase(), tim_tokens=2, event_slots=3
    )
    return adapter, adapter.init_state(2, "cpu")


def _inputs(
    batch_size: int = 2,
    *,
    requires_grad: bool = False,
) -> dict[str, Any]:
    coarse = torch.randn(batch_size, 3, 5, requires_grad=requires_grad)
    fine = torch.randn(batch_size, 4, 5, requires_grad=requires_grad)
    return {
        "coarse_tokens": coarse,
        "coarse_tidx": torch.arange(3).expand(batch_size, -1),
        "fine_tokens": fine,
        "fine_tidx": torch.zeros(batch_size, 4, dtype=torch.long),
        "instructions": ["follow the person"] * batch_size,
    }


def _encode(
    adapter: OpenTrackVLAF2ObservationAdapter,
    state: Mapping[str, Any],
    inputs: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    values = _inputs() if inputs is None else dict(inputs)
    return adapter.encode_step(
        values["coarse_tokens"],
        values["coarse_tidx"],
        values["fine_tokens"],
        values["fine_tidx"],
        values["instructions"],
        state,
        **kwargs,
    )


def _tree_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tree_tensors(child)


def _clone_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(child) for key, child in value.items()}
    return value


def _assert_tree_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
        return
    if isinstance(left, Mapping):
        assert set(left) == set(right)
        for key in left:
            _assert_tree_equal(left[key], right[key])


def _targets(output: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    batch_size = output["h_act"].shape[0]
    theta = output["cot"]["theta_logits"].argmax(dim=-1).detach()
    distance = output["cot"]["dist_logits"].argmax(dim=-1).detach()
    targets: dict[str, torch.Tensor] = {
        "polar_theta_idx": theta,
        "polar_dist_idx": distance,
        "polar_invalid": torch.zeros(batch_size),
    }
    for horizon in (4, 8, 16):
        targets[f"fut_valid_{horizon}"] = torch.ones(
            batch_size, dtype=torch.bool
        )
        targets[f"fut_vis_{horizon}"] = torch.ones(batch_size)
        targets[f"fut_theta_idx_{horizon}"] = torch.zeros(
            batch_size, dtype=torch.long
        )
        targets[f"fut_dist_idx_{horizon}"] = torch.zeros(
            batch_size, dtype=torch.long
        )
    return targets


def test_adapter_reuses_official_modules_but_has_no_action_verifier_path():
    adapter, _ = _adapter()
    assert adapter.architecture_lock == "L1+D2+AP2+F2"
    assert isinstance(adapter.cot, PolarCoTHead)
    assert isinstance(adapter.tim, TIM)
    assert isinstance(adapter.events, CognitiveEventBank)
    assert isinstance(adapter.orchestrator, Orchestrator)
    assert isinstance(adapter.polar_token, DifferentiablePolarToken)
    assert isinstance(adapter.future, PrevFreeFutureModule)
    assert isinstance(adapter.self_correctness, SelfCorrectnessHead)
    assert adapter.orchestrator.alpha_mlp[-1].out_features == 3
    assert not hasattr(adapter.future, "act_proj")
    assert not hasattr(adapter.self_correctness, "delta_head")
    assert not any("verifier" in name or "delta" in name for name, _ in adapter.named_parameters())
    assert "prev_action" not in inspect.signature(adapter.encode_step).parameters
    assert "last_action" not in inspect.signature(adapter.future.forward).parameters
    assert adapter.method_dims == {
        "polar": 8,
        "tim_q": 10,
        "future": 8,
        "event": 8,
    }


def test_encode_step_emits_exact_prev_free_feature_and_alpha_contract():
    adapter, state = _adapter()
    output = _encode(adapter, state)

    assert output["base_features"] is output["h_act"]
    assert output["h_act"].shape == (2, 8)
    assert list(output["method_features"]) == [
        "polar",
        "tim_q",
        "future",
        "event",
    ]
    assert output["method_features"]["polar"].shape == (2, 8)
    assert output["method_features"]["tim_q"].shape == (2, 10)
    assert output["method_features"]["future"].shape == (2, 8)
    assert output["method_features"]["event"].shape == (2, 8)
    assert torch.equal(
        output["method_alphas"]["polar"], torch.ones(2)
    )
    alpha_sum = sum(
        output["method_alphas"][name]
        for name in ("tim_q", "event", "future")
    )
    assert torch.allclose(alpha_sum, torch.ones_like(alpha_sum))
    assert "alpha_verify" not in output["orchestrator"]
    assert "delta" not in output
    assert "waypoints" not in output
    assert torch.equal(
        output["new_state"]["pending_q_write"], output["q_write"].detach()
    )
    assert all(
        not tensor.requires_grad and tensor.grad_fn is None
        for tensor in _tree_tensors(output["new_state"])
    )
    assert output["audit_counters"] == {
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }


def test_soft_polar_is_differentiable_but_q_is_detached_from_all_features():
    adapter, state = _adapter()
    output = _encode(adapter, state)
    polar_gradient = torch.autograd.grad(
        output["method_features"]["polar"].square().mean(),
        output["cot"]["theta_logits"],
        retain_graph=True,
    )[0]
    assert polar_gradient is not None
    assert torch.count_nonzero(polar_gradient) > 0

    context_outputs = [
        *output["method_features"].values(),
        *output["method_alphas"].values(),
    ]
    q_parameters = tuple(adapter.self_correctness.parameters())
    for context_output in context_outputs:
        if not context_output.requires_grad:
            assert context_output.grad_fn is None
            continue
        gradients = torch.autograd.grad(
            context_output.sum(),
            q_parameters,
            allow_unused=True,
            retain_graph=True,
        )
        assert all(gradient is None for gradient in gradients)


def test_observation_and_auxiliary_graphs_do_not_contain_previous_action():
    adapter, state = _adapter()
    previous_action = torch.randn(2, 3, requires_grad=True)
    # Even a malformed caller-supplied state graph is cut at the step boundary.
    state["pending_q_write"] = torch.sigmoid(previous_action[:, 0])
    state["has_pending"] = torch.ones(2, dtype=torch.bool)
    output = _encode(adapter, state)
    losses = adapter.compute_aux_losses(output, _targets(output))
    assert_prev_free_tensors(
        {
            "h_act": output["h_act"],
            "method_polar": output["method_features"]["polar"],
            "method_tim_q": output["method_features"]["tim_q"],
            "method_future": output["method_features"]["future"],
            "method_event": output["method_features"]["event"],
            "L_aux": losses["L_aux"],
        },
        previous_action,
    )


def test_shared_perception_state_is_delayed_detached_and_resettable():
    adapter, state = _adapter()
    first_inputs = _inputs(requires_grad=True)
    first = _encode(adapter, state, first_inputs)
    second = _encode(adapter, first["new_state"], _inputs())
    assert second["new_state"]["tim"]["C_cnt"].tolist() == [1, 1]
    cross_step_gradient = torch.autograd.grad(
        second["h_act"].sum(),
        first_inputs["fine_tokens"],
        allow_unused=True,
    )[0]
    assert cross_step_gradient is None

    reset = _encode(
        adapter,
        second["new_state"],
        _inputs(),
        reset_mask=torch.tensor([True, False]),
    )
    assert reset["new_state"]["tim"]["C_cnt"].tolist() == [0, 2]


def test_aux_losses_are_exact_sum_and_future_expert_stays_label_side():
    adapter, state = _adapter()
    output = _encode(adapter, state)
    targets = _targets(output)
    expert_visibility = targets["fut_vis_4"].requires_grad_()
    state_before = _clone_tree(output["new_state"])

    losses = adapter.compute_aux_losses(output, targets)
    assert torch.equal(
        losses["L_aux"],
        losses["L_cot"] + losses["L_future"] + losses["L_verify"],
    )
    assert torch.isfinite(losses["L_aux"])
    _assert_tree_equal(output["new_state"], state_before)
    assert torch.autograd.grad(
        losses["L_aux"], expert_visibility, allow_unused=True, retain_graph=True
    )[0] is None

    q_gradient = torch.autograd.grad(
        losses["L_verify"],
        adapter.self_correctness.q_head.weight,
        retain_graph=True,
    )[0]
    future_gradient = torch.autograd.grad(
        losses["L_future"],
        adapter.future.visibility_heads[0].weight,
        retain_graph=True,
    )[0]
    assert torch.count_nonzero(q_gradient) > 0
    assert torch.count_nonzero(future_gradient) > 0
    assert adapter.audit_counters() == {
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }


def test_forbidden_dataflow_counter_hooks_fail_closed():
    adapter, _ = _adapter()
    adapter.assert_audit_counters_clean()
    adapter.record_expert_future_leak()
    adapter.record_self_state_expert_overwrite(2)
    assert adapter.audit_counters() == {
        "expert_future_leak_count": 1,
        "self_state_expert_overwrite_count": 2,
    }
    with pytest.raises(F2ObservationContractError, match="nonzero"):
        adapter.assert_audit_counters_clean()


def test_state_shape_and_reset_contract_fail_closed():
    adapter, state = _adapter()
    with pytest.raises(F2ObservationContractError, match="reset_mask"):
        _encode(adapter, state, reset_mask=torch.ones(1, dtype=torch.bool))
    bad_state = dict(state)
    bad_state.pop("evt")
    with pytest.raises(F2ObservationContractError, match="state keys"):
        _encode(adapter, bad_state)
