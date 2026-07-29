import math

import pytest
import torch

from f2_experiment.controller import clamp_stage
from f2_experiment.evaluation import G6Update, evaluate_g7, evaluate_g9
from f2_experiment.model import AP2_HORIZON, AP2Prediction
from f2_experiment.runner import (
    GRAD_ACCUM,
    SMOKE_ROWS,
    SMOKE_UPDATES,
    S_CTRL,
    S_SELF,
    ArmCallbacks,
    AuxForwardResult,
    FeatureForwardResult,
    HeadForwardResult,
    RunnerContractError,
    RunnerNonfiniteActionError,
    RunnerRow,
    RunnerTelemetryHooks,
    checkpoint_init_sha256,
    run_paired_smoke,
)


def _rows(*, discontinuities=False):
    rows = []
    for position in range(SMOKE_ROWS):
        if not discontinuities:
            sequence_id = "sequence-a"
            frame_idx = position
            mirrored = False
        elif position < 100:
            sequence_id = "sequence-a"
            frame_idx = position
            mirrored = False
        elif position < 150:
            sequence_id = "sequence-b"
            frame_idx = position - 100
            mirrored = False
        elif position < 200:
            sequence_id = "sequence-b"
            frame_idx = position - 100
            mirrored = True
        else:
            sequence_id = "sequence-b"
            frame_idx = position - 99
            mirrored = True
        rows.append(
            RunnerRow(
                original_row_index=1000 + position,
                sequence_id=sequence_id,
                frame_idx=frame_idx,
                mirrored=mirrored,
                logged_prev_action=(0.0, 0.0, 0.0),
                target_actions=torch.zeros(AP2_HORIZON, 3),
                observation={"visible_token": position},
                aux_targets={"future_label": position % 3},
            )
        )
    return rows


def _bounded_future(raw_actions):
    bounded = []
    for step in range(1, AP2_HORIZON):
        bounded.append(
            clamp_stage(
                (
                    float(raw_actions[0, step, 0].item()),
                    float(raw_actions[0, step, 2].item()),
                )
            )
        )
    return raw_actions.new_tensor([bounded])


def _prediction(prev_fy, *, nonfinite=False, strafe=False):
    delta = prev_fy.new_zeros((1, AP2_HORIZON, 2))
    delta[:, 0, 0] = 0.001
    raw_fy = prev_fy.unsqueeze(-2) + torch.cumsum(delta, dim=-2)
    raw = torch.stack(
        (
            raw_fy[..., 0],
            torch.zeros_like(raw_fy[..., 0]),
            raw_fy[..., 1],
        ),
        dim=-1,
    )
    if nonfinite:
        raw = raw.clone()
        raw[0, 0, 0] = float("nan")
    if strafe:
        raw = raw.clone()
        raw[0, 0, 1] = 0.1
    bounded = (
        torch.zeros(1, AP2_HORIZON - 1, 3)
        if nonfinite
        else _bounded_future(raw)
    )
    return AP2Prediction(
        delta_fy=delta,
        raw_actions=raw,
        bounded_future_actions=bounded,
    )


def _g7(*, omit_r_prev=False):
    telemetry = {
        "per_method_over_base": {"polar": torch.tensor([0.1])},
        "total_method_over_base": torch.tensor([0.1]),
        "abs_tanh_method_scales": {"polar": torch.tensor(0.2)},
        "abs_tanh_s_prev": torch.tensor(0.1),
    }
    if not omit_r_prev:
        telemetry["r_prev"] = torch.tensor([0.3])
    return telemetry


class FakeArm:
    def __init__(
        self,
        arm,
        *,
        checkpoint_value=1.0,
        nonfinite_at=None,
        omit_r_prev=False,
        strafe_at=None,
        audit_counts=None,
    ):
        self.arm = arm
        self.checkpoint_state = {
            "head.bias": torch.tensor([checkpoint_value, 2.0]),
            "head.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        }
        self.nonfinite_at = nonfinite_at
        self.omit_r_prev = omit_r_prev
        self.strafe_at = strafe_at
        self.audit_counts = audit_counts
        self.audit_counter_reads = 0
        self.feature_events = []
        self.aux_events = []
        self.head_events = []
        self.track_events = []
        self.backward_events = []
        self.optimizer_events = []

    def callbacks(self):
        return ArmCallbacks(
            checkpoint_state=self.checkpoint_state,
            feature_forward=self.feature_forward,
            aux_forward=self.aux_forward,
            head_forward=self.head_forward,
            track_loss=self.track_loss,
            backward=self.backward,
            optimizer_step=self.optimizer_step,
            audit_counters=(
                None if self.audit_counts is None else self.audit_counters
            ),
        )

    def audit_counters(self):
        self.audit_counter_reads += 1
        return dict(self.audit_counts)

    def feature_forward(self, observation, event):
        assert set(observation) == {"visible_token"}
        result = FeatureForwardResult(
            value={"feature_id": event.row_position},
            reference_tensor=torch.ones(1, 4),
        )
        self.feature_events.append((observation, event, id(result.value)))
        return result

    def aux_forward(self, features, aux_targets, event):
        self.aux_events.append((features, aux_targets, event))
        return AuxForwardResult(loss=torch.tensor(2.0))

    def head_forward(self, features, prev_fy, event):
        self.head_events.append(
            (features, prev_fy.detach().clone(), event)
        )
        nonfinite = (
            event.branch == "branch2"
            and event.row_position == self.nonfinite_at
        )
        strafe = (
            event.branch == "branch2" and event.row_position == self.strafe_at
        )
        return HeadForwardResult(
            prediction=_prediction(
                prev_fy,
                nonfinite=nonfinite,
                strafe=strafe,
            ),
            g7_telemetry=_g7(omit_r_prev=self.omit_r_prev),
        )

    def track_loss(self, prediction, target, event):
        assert target.shape == (1, AP2_HORIZON, 3)
        self.track_events.append((prediction, target, event))
        return torch.tensor(1.0 if event.branch == "branch1" else 3.0)

    def backward(self, event):
        self.backward_events.append(event)

    def optimizer_step(self, event):
        self.optimizer_events.append(event)


class HookRecorder:
    def __init__(self):
        self.g6_events = []
        self.g7_events = []
        self.g9_events = []

    def hooks(self):
        return RunnerTelemetryHooks(
            g6_update=self.g6_update,
            on_g7_update=self.on_g7_update,
            on_g9_transition=self.on_g9_transition,
        )

    def g6_update(self, event):
        self.g6_events.append(event)
        in_window = event.u_pre >= 8
        return G6Update(
            u_pre=event.u_pre,
            aux_reachable=True,
            track_reachable=in_window,
            cosine_total_track=0.7 if in_window else None,
            signed_projection=1.0 if in_window else None,
            aux_track_ratio=1.0 if in_window else None,
        )

    def on_g7_update(self, arm, update):
        self.g7_events.append((arm, update))

    def on_g9_transition(self, receipt):
        self.g9_events.append(receipt)


def _run(
    *,
    ctrl=None,
    self_arm=None,
    rows=None,
    strafe_resets=frozenset({1010, 1011}),
    expected_resets=frozenset({1000, 1010, 1011}),
):
    ctrl = FakeArm(S_CTRL) if ctrl is None else ctrl
    self_arm = FakeArm(S_SELF) if self_arm is None else self_arm
    recorder = HookRecorder()
    result = run_paired_smoke(
        _rows() if rows is None else rows,
        callbacks={S_CTRL: ctrl.callbacks(), S_SELF: self_arm.callbacks()},
        hooks=recorder.hooks(),
        strafe_reset_original_indices=strafe_resets,
        expected_static_reset_original_indices=expected_resets,
    )
    return result, ctrl, self_arm, recorder


def test_paired_runner_exact_counts_loss_warmup_and_telemetry_contracts():
    result, ctrl, self_arm, recorder = _run()

    assert len(result.checkpoint_init_sha256) == 64
    assert result.count_receipt.passed
    assert result.count_receipt.to_dict()["loss"] == "L_aux+0.5*L1+0.5*L2"
    assert result.static_reset_original_indices == (1000, 1010, 1011)

    ctrl_counts = result.arms[S_CTRL].counts
    self_counts = result.arms[S_SELF].counts
    for counts in (ctrl_counts, self_counts):
        assert counts.rows == 256
        assert counts.feature_forwards == 256
        assert counts.aux_forwards == 256
        assert counts.head_forwards == 512
        assert counts.track_loss_calls == 512
        assert counts.backward_calls == 256
        assert counts.optimizer_steps == 128
        assert counts.controller_steps == 256
        assert counts.static_resets == 3
        assert counts.nonfinite_resets == 0
        assert counts.g7_updates == 128
        assert counts.g9_transitions == 256
        assert counts.expert_future_leak_count == 0
        assert counts.self_state_expert_overwrite_count == 0
    assert ctrl_counts.branch2_logged_rows == 256
    assert ctrl_counts.branch2_self_rows == 0
    assert ctrl_counts.g6_updates == 128
    assert self_counts.branch2_logged_rows == 32
    assert self_counts.branch2_self_rows == 224
    assert self_counts.g6_updates == 0

    assert result.arms[S_CTRL].row_losses == pytest.approx([4.0] * 256)
    assert result.arms[S_SELF].row_losses == pytest.approx([4.0] * 256)
    assert len(ctrl.backward_events) == 256
    assert all(event.unscaled_loss.item() == pytest.approx(4.0) for event in ctrl.backward_events)
    assert all(event.scaled_loss.item() == pytest.approx(2.0) for event in ctrl.backward_events)
    assert all(event.grad_accum == GRAD_ACCUM for event in ctrl.backward_events)
    assert len(ctrl.optimizer_events) == SMOKE_UPDATES
    assert ctrl.optimizer_events[0].row_positions == (0, 1)
    assert ctrl.optimizer_events[-1].u_pre == 127
    assert all(event.mean_loss == pytest.approx(4.0) for event in ctrl.optimizer_events)

    assert len(recorder.g6_events) == 128
    assert len(recorder.g7_events) == 256
    assert len(recorder.g9_events) == 512
    assert [update.u_pre for update in result.arms[S_CTRL].g7_updates] == list(range(128))
    assert all(update.head_observations == 4 for update in result.arms[S_CTRL].g7_updates)
    assert all(len(update.r_prev) == 4 for update in result.arms[S_CTRL].g7_updates)
    assert evaluate_g7(
        [update.gate_update() for update in result.arms[S_CTRL].g7_updates]
    ).passed

    first_transition = result.arms[S_SELF].g9.transitions[0]
    assert first_transition.raw_k0_fy == pytest.approx((0.001, 0.0))
    assert first_transition.post_safety_clamp == pytest.approx((0.001, 0.0, 0.0))
    assert first_transition.self_prev_after_fy == pytest.approx((0.0005, 0.0))
    assert first_transition.to_dict()["post_safety_clamp"] == pytest.approx(
        [0.001, 0.0, 0.0]
    )
    assert result.arms[S_SELF].g9.range_observation_count == 256 * 8
    assert result.arms[S_SELF].g9.range_violation_count == 0
    assert max(result.arms[S_SELF].g9.reconstruction_errors) == pytest.approx(0.0)
    assert evaluate_g9(**result.arms[S_SELF].g9.gate_kwargs()).passed

    self_branch2 = {
        event.row_position: prev
        for _features, prev, event in self_arm.head_events
        if event.branch == "branch2"
    }
    ctrl_branch2 = {
        event.row_position: prev
        for _features, prev, event in ctrl.head_events
        if event.branch == "branch2"
    }
    assert torch.equal(self_branch2[31], torch.zeros(1, 2))
    assert self_branch2[32][0, 0] > 0.0
    assert torch.equal(ctrl_branch2[32], torch.zeros(1, 2))
    assert result.arms[S_SELF].branch2_sources[31] == "logged"
    assert result.arms[S_SELF].branch2_sources[32] == "self"

    for arm_adapter in (ctrl, self_arm):
        for row_position in range(256):
            feature_ids = [
                id(features)
                for features, _prev, event in arm_adapter.head_events
                if event.row_position == row_position
            ]
            assert len(feature_ids) == 2
            assert feature_ids[0] == feature_ids[1]


def test_reset_plan_covers_sequence_mirror_frame_gap_and_strafe_receipt():
    rows = _rows(discontinuities=True)
    expected = frozenset({1000, 1010, 1011, 1100, 1150, 1200})
    result, _ctrl, _self, _recorder = _run(
        rows=rows,
        expected_resets=expected,
    )
    assert result.static_reset_original_indices == tuple(sorted(expected))
    assert result.arms[S_CTRL].counts.static_resets == len(expected)
    reasons = {
        receipt.original_row_index: receipt.reset_reasons
        for receipt in result.arms[S_CTRL].g9.transitions
        if receipt.reset
    }
    assert "sequence_discontinuity" in reasons[1100]
    assert "sequence_discontinuity" in reasons[1150]
    assert "sequence_discontinuity" in reasons[1200]
    assert reasons[1010] == ("strafe_reset",)


def test_checkpoint_init_sha_is_order_independent_and_mismatch_stops_preflight():
    state_a = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    state_b = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert checkpoint_init_sha256(state_a) == checkpoint_init_sha256(state_b)

    ctrl = FakeArm(S_CTRL, checkpoint_value=1.0)
    self_arm = FakeArm(S_SELF, checkpoint_value=9.0)
    with pytest.raises(RunnerContractError, match="checkpoint init SHA"):
        _run(ctrl=ctrl, self_arm=self_arm)
    assert ctrl.feature_events == []
    assert self_arm.feature_events == []


def test_g7_r_prev_is_mandatory_before_first_optimizer_step():
    ctrl = FakeArm(S_CTRL, omit_r_prev=True)
    with pytest.raises(RunnerContractError, match="r_prev"):
        _run(ctrl=ctrl)
    assert len(ctrl.backward_events) == 2
    assert ctrl.optimizer_events == []


def test_g7_gate_update_projects_abs_tanh_s_prev_to_the_evaluator():
    result, _ctrl, _self, _recorder = _run()
    for arm in (S_CTRL, S_SELF):
        for update in result.arms[arm].g7_updates:
            projected = update.gate_update()
            assert projected.abs_tanh_s_prev == update.abs_tanh_s_prev
            assert projected.abs_tanh_s_prev == pytest.approx(0.1)
    receipt = evaluate_g7(
        [update.gate_update() for update in result.arms[S_SELF].g7_updates]
    )
    assert receipt.passed
    assert receipt.checks["prev_scale_saturation_rate"]["passed"]
    assert receipt.metrics["prev_scale_saturation_rate"] == 0.0
    assert receipt.metrics["abs_tanh_s_prev_max"] == pytest.approx(0.1)


def test_adapter_audit_counters_overwrite_placeholders_and_fail_closed():
    clean = {
        "expert_future_leak_count": 0,
        "self_state_expert_overwrite_count": 0,
    }
    ctrl = FakeArm(S_CTRL, audit_counts=dict(clean))
    self_arm = FakeArm(S_SELF, audit_counts=dict(clean))
    result, ctrl, self_arm, _recorder = _run(ctrl=ctrl, self_arm=self_arm)
    assert result.count_receipt.passed
    assert ctrl.audit_counter_reads == 1
    assert self_arm.audit_counter_reads == 1
    for arm in (S_CTRL, S_SELF):
        counts = result.count_receipt.to_dict()["arms"][arm]
        assert counts["expert_future_leak_count"] == 0
        assert counts["self_state_expert_overwrite_count"] == 0

    dirty = FakeArm(
        S_SELF,
        audit_counts={
            "expert_future_leak_count": 1,
            "self_state_expert_overwrite_count": 0,
        },
    )
    with pytest.raises(RunnerContractError, match="count receipt failed"):
        _run(self_arm=dirty)

    overwrite = FakeArm(
        S_SELF,
        audit_counts={
            "expert_future_leak_count": 0,
            "self_state_expert_overwrite_count": 3,
        },
    )
    with pytest.raises(RunnerContractError, match="count receipt failed"):
        _run(self_arm=overwrite)


def test_adapter_audit_counters_malformed_returns_fail_closed():
    missing_key = FakeArm(
        S_CTRL, audit_counts={"expert_future_leak_count": 0}
    )
    with pytest.raises(RunnerContractError, match="missing"):
        _run(ctrl=missing_key)

    negative = FakeArm(
        S_CTRL,
        audit_counts={
            "expert_future_leak_count": -1,
            "self_state_expert_overwrite_count": 0,
        },
    )
    with pytest.raises(RunnerContractError, match="nonnegative"):
        _run(ctrl=negative)

    boolean = FakeArm(
        S_CTRL,
        audit_counts={
            "expert_future_leak_count": False,
            "self_state_expert_overwrite_count": 0,
        },
    )
    with pytest.raises(RunnerContractError, match="integer"):
        _run(ctrl=boolean)


def test_nonfinite_branch2_k0_synchronously_resets_and_fails_closed():
    self_arm = FakeArm(S_SELF, nonfinite_at=40)
    recorder = HookRecorder()
    with pytest.raises(RunnerNonfiniteActionError) as captured:
        run_paired_smoke(
            _rows(),
            callbacks={
                S_CTRL: FakeArm(S_CTRL).callbacks(),
                S_SELF: self_arm.callbacks(),
            },
            hooks=recorder.hooks(),
            strafe_reset_original_indices={1010, 1011},
            expected_static_reset_original_indices={1000, 1010, 1011},
        )
    receipt = captured.value.receipt
    assert receipt.arm == S_SELF
    assert receipt.row_position == 40
    assert receipt.u_pre == 20
    assert receipt.synchronized
    assert receipt.nonfinite_reset_count == 1
    assert receipt.controller_state_after_reset.prev_cmd == (0.0, 0.0, 0.0)
    assert receipt.controller_state_after_reset.ticks == 0
    assert torch.equal(receipt.self_prev_after_reset, torch.zeros(1, 2))
    assert recorder.g9_events[-1].synchronized_nonfinite_reset
    assert recorder.g9_events[-1].post_safety_clamp is None


def test_nonzero_predicted_strafe_and_reset_receipt_mismatch_fail_closed():
    with pytest.raises(RunnerContractError, match="predicted nonzero strafe"):
        _run(self_arm=FakeArm(S_SELF, strafe_at=3))

    with pytest.raises(RunnerContractError, match="reset plan"):
        _run(expected_resets=frozenset({1000}))


def test_runner_rejects_wrong_row_budget_before_callbacks():
    ctrl = FakeArm(S_CTRL)
    self_arm = FakeArm(S_SELF)
    recorder = HookRecorder()
    with pytest.raises(RunnerContractError, match="exactly 256 rows"):
        run_paired_smoke(
            _rows()[:-1],
            callbacks={S_CTRL: ctrl.callbacks(), S_SELF: self_arm.callbacks()},
            hooks=recorder.hooks(),
            strafe_reset_original_indices={1010, 1011},
            expected_static_reset_original_indices={1000, 1010, 1011},
        )
    assert ctrl.feature_events == []
    assert self_arm.feature_events == []


def test_checkpoint_sha_rejects_nonfinite_state():
    with pytest.raises(RunnerContractError, match="nonfinite"):
        checkpoint_init_sha256({"weight": torch.tensor([math.nan])})
