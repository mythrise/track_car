import math

import pytest

from f2_experiment.controller import (
    ACTION_FILTER_ESTIMAND,
    CONTROLLED_AXES,
    PARITY_NAME,
    ActionFilterConfig,
    ActionFilterController,
    ControllerContractError,
    assert_controlled_axis_parity,
    bind_controller_identity,
    clamp_stage,
    controlled_axes,
    controller_contract,
    scatter_controlled_action,
)


def test_controller_contract_is_the_fable_action_filter_proxy():
    contract = controller_contract()
    assert ACTION_FILTER_ESTIMAND == (
        "fixed_logged_vision_post_action_filter_self_rollout_proxy"
    )
    assert contract["architecture_lock"] == "L1+D2+AP2+F2"
    assert contract["estimand_key"] == ACTION_FILTER_ESTIMAND
    assert contract["not_deployment_sent"] is True
    assert contract["controlled_axes"] == [0, 2]
    assert contract["stages"] == ["finite_check", "clamp", "rate_limit", "ema"]
    assert contract["shared_across"] == ["SA-B0", "SA-B1", "SA-H*"]


def test_reset_scatter_uses_only_forward_and_yaw():
    controller = ActionFilterController()
    state = controller.reset((0.25, 0.75, -0.5))
    assert state.prev_cmd == (0.25, 0.0, -0.5)
    assert state.ticks == 0


def test_step_applies_clamp_rate_limit_and_ema_in_order():
    controller = ActionFilterController()
    state = controller.reset((0.0, 0.0, 0.0))
    next_state, transition = controller.step(state, (1.2, -1.2))
    assert transition.scattered == (1.2, 0.0, -1.2)
    assert transition.bounded == (1.0, 0.0, -1.0)
    assert transition.rate_limited == (0.4, 0.0, -0.4)
    assert transition.filtered == (0.2, 0.0, -0.2)
    assert transition.sent_action == transition.filtered
    assert transition.next_prev_fy == (0.2, -0.2)
    assert next_state.prev_cmd == transition.filtered
    assert next_state.ticks == 1


def test_second_step_rate_limit_is_relative_to_filtered_state():
    controller = ActionFilterController()
    state = controller.reset((0.0, 0.0, 0.0))
    state, _first = controller.step(state, (1.0, 1.0))
    state, second = controller.step(state, (1.0, 1.0))
    assert second.prior_cmd == (0.2, 0.0, 0.2)
    assert second.rate_limited == pytest.approx((0.6, 0.0, 0.6))
    assert second.filtered == pytest.approx((0.4, 0.0, 0.4))
    assert state.ticks == 2


def test_clamp_stage_is_the_only_k1_to_k7_bounding_rule():
    assert clamp_stage((2.0, -3.0)) == (1.0, 0.0, -1.0)
    controller = ActionFilterController()
    assert controller.clamp_horizon(((2.0, -3.0), (0.25, -0.5))) == (
        (1.0, 0.0, -1.0),
        (0.25, 0.0, -0.5),
    )


def test_nonfinite_raw_action_is_a_hard_stop():
    controller = ActionFilterController()
    state = controller.reset((0.0, 0.0, 0.0))
    with pytest.raises(ControllerContractError, match="CTRL_NONFINITE"):
        controller.step(state, (math.nan, 0.0))


def test_reset_rejects_logged_prev_outside_frozen_control_domain():
    controller = ActionFilterController()
    with pytest.raises(ControllerContractError, match="frozen control domain"):
        controller.reset((1.1, 0.0, 0.0))


def test_controlled_axis_parity_never_claims_full_vector_identity():
    assert CONTROLLED_AXES == (0, 2)
    assert PARITY_NAME == "controlled_axis_raw_persistence"
    logged = (0.2, 0.9, -0.4)
    assert controlled_axes(logged) == (0.2, -0.4)
    assert scatter_controlled_action((0.2, -0.4)) == (0.2, 0.0, -0.4)
    assert_controlled_axis_parity(logged, (0.2, -0.4))
    with pytest.raises(ControllerContractError, match=PARITY_NAME):
        assert_controlled_axis_parity(logged, (0.2, -0.3))


def test_controller_config_freezes_rate_limit_per_step():
    config = ActionFilterConfig(
        max_abs=1.0,
        max_action_rate=4.0,
        dt=0.1,
        ema_prev_weight=0.5,
    )
    assert config.max_step_delta == pytest.approx(0.4)
    with pytest.raises(ControllerContractError):
        ActionFilterConfig(dt=0.0)
    with pytest.raises(ControllerContractError):
        ActionFilterConfig(dt="0.1")


def test_controller_source_identity_must_be_explicit_sha256():
    binding = bind_controller_identity("a" * 64)
    assert binding["controller_source_sha256"] == "a" * 64
    assert len(binding["controller_config_contract_sha256"]) == 64
    with pytest.raises(ControllerContractError, match="source SHA-256"):
        bind_controller_identity("not-a-sha")
