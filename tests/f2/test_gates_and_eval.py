import math

import pytest

from f2_experiment.evaluation import (
    G6Update,
    G7Update,
    G8_TAU,
    GateContractError,
    StratifiedLossSummary,
    aggregate_row_losses,
    aggregate_stratified_losses,
    build_smoke_gate_receipt,
    evaluate_g6,
    evaluate_g7,
    evaluate_g8,
    evaluate_g9,
    strata_masks_from_rows,
)


def _summary(overall, change=None, turn=None, other=None, count=4):
    return StratifiedLossSummary(
        means={
            "overall": overall,
            "change": overall if change is None else change,
            "turn": overall if turn is None else turn,
            "other": overall if other is None else other,
        },
        counts={"overall": count, "change": 1, "turn": 1, "other": 1},
    )


def _g6_updates(*, ratio=1.0, cosine=0.7, positive=120):
    updates = []
    for u_pre in range(128):
        in_window = u_pre >= 8
        projection = 1.0 if u_pre - 8 < positive else -1.0
        updates.append(
            G6Update(
                u_pre=u_pre,
                aux_reachable=u_pre != 0,
                track_reachable=u_pre >= 8,
                cosine_total_track=cosine if in_window else None,
                signed_projection=projection if in_window else None,
                aux_track_ratio=ratio if in_window else None,
            )
        )
    return updates


def _g7_updates(*, stream_ratio=0.4, total_ratio=0.8, saturated=0, prev_saturated=None):
    updates = []
    for u_pre in range(128):
        scale = 0.995 if u_pre < saturated else 0.2
        if prev_saturated is None:
            prev_scale = None
        else:
            prev_scale = 0.995 if u_pre < prev_saturated else 0.1
        updates.append(
            G7Update(
                u_pre=u_pre,
                per_method_over_base={"polar": [stream_ratio, stream_ratio - 0.1]},
                total_method_over_base=[total_ratio],
                abs_tanh_method_scales={"polar": scale},
                r_prev=[0.3],
                abs_tanh_s_prev=prev_scale,
            )
        )
    return updates


def _passing_g8():
    return evaluate_g8(
        s_self_update0={
            "logged": _summary(0.50),
            "self": _summary(0.80, 0.82, 0.78, 0.81),
        },
        s_self_update128={
            "logged": _summary(0.45),
            "self": _summary(0.60, 0.61, 0.59, 0.60),
        },
        s_ctrl_update128={
            "logged": _summary(0.50),
            "self": _summary(0.75),
        },
    )


def _passing_g9():
    return evaluate_g9(
        expected_static_resets=12,
        observed_static_resets=12,
        nonfinite_reset_count=0,
        range_violation_count=4,
        range_observation_count=100,
        reconstruction_errors=[0.0, 1e-6],
        first_quartile_self_errors=[1.0, 1.0],
        last_quartile_self_errors=[2.0, 2.0],
    )


def test_strata_use_controlled_axis_change_and_exact_transition_names():
    rows = [
        {
            "prev_action": [0.0, 0.9, 0.0],
            "step_actions": [[0.21, -0.9, 0.0]],
            "transition_type": "turn_onset",
        },
        {
            "prev_action": [0.0, 0.0, 0.0],
            "step_actions": [[0.0, 1.0, 0.0]],
            "transition_type": "other",
        },
        {
            "prev_action": [0.0, 0.0, 0.0],
            "step_actions": [[0.0, 0.0, 0.2]],
            "transition_type": "turn",
        },
    ]
    masks = strata_masks_from_rows(rows)
    assert masks == {
        "change": (True, False, False),
        "turn": (True, False, False),
        "other": (False, True, False),
    }
    summary = aggregate_row_losses([0.3, 0.2, 0.1], rows)
    assert summary.counts == {"overall": 3, "change": 1, "turn": 1, "other": 1}


def test_stratified_loss_uses_binary64_fsum_and_reports_all_strata():
    losses = [1e16, 1.0, 1.0]
    summary = aggregate_stratified_losses(
        losses,
        change_mask=[True, False, False],
        turn_mask=[False, True, False],
        other_mask=[False, False, True],
    )
    assert summary.means["overall"] == math.fsum(losses) / 3
    assert summary.to_dict()["accumulator"] == "IEEE-754 binary64 math.fsum"


def test_stratified_loss_zero_support_and_nonfinite_fail_closed():
    with pytest.raises(GateContractError, match="G8_ZERO_SUPPORT"):
        aggregate_stratified_losses(
            [0.1, 0.2],
            change_mask=[False, False],
            turn_mask=[True, False],
            other_mask=[False, True],
        )
    with pytest.raises(GateContractError, match="nonfinite"):
        aggregate_stratified_losses(
            [0.1, math.nan],
            change_mask=[True, False],
            turn_mask=[False, True],
            other_mask=[True, False],
        )


def test_g6_uses_pre_step_clock_8_through_127_and_108_of_120():
    receipt = evaluate_g6(_g6_updates(positive=108), block_mode="bstar")
    assert receipt.passed
    payload = receipt.to_dict()
    assert payload["metrics"]["clock_first"] == 0
    assert payload["metrics"]["clock_last"] == 127
    assert payload["metrics"]["gradient_window"] == [8, 127]
    assert payload["metrics"]["gradient_window_points"] == 120
    assert payload["metrics"]["positive_projection_count"] == 108
    assert payload["metrics"]["aux_reachable_count"] == 127
    assert payload["metrics"]["track_reachable_count"] == 120


def test_g6_rejects_old_updates_8_through_128_clock():
    updates = _g6_updates()
    updates = [
        G6Update(
            u_pre=record.u_pre + 1,
            aux_reachable=record.aux_reachable,
            track_reachable=record.track_reachable,
            cosine_total_track=record.cosine_total_track,
            signed_projection=record.signed_projection,
            aux_track_ratio=record.aux_track_ratio,
        )
        for record in updates
    ]
    with pytest.raises(GateContractError, match="0..127"):
        evaluate_g6(updates, block_mode="bstar")


def test_g6_threshold_failures_produce_stop_receipt():
    receipt = evaluate_g6(
        _g6_updates(ratio=1.50001, cosine=0.59, positive=107),
        block_mode="bstar",
    )
    assert not receipt.passed
    assert receipt.to_dict()["decision"] == "STOP"
    assert not receipt.checks["aux_track_ratio_median"]["passed"]
    assert not receipt.checks["cosine_median"]["passed"]
    assert not receipt.checks["positive_projection"]["passed"]


def test_g6_per_aux_fallback_uses_point_seven_five_per_aux():
    updates = []
    for u_pre in range(128):
        in_window = u_pre >= 8
        updates.append(
            G6Update(
                u_pre=u_pre,
                aux_reachable=True,
                track_reachable=True,
                cosine_total_track=0.7 if in_window else None,
                signed_projection=1.0 if in_window else None,
                per_aux_ratios=(
                    {"cot": 0.75, "future": 0.74, "verify": 0.73} if in_window else None
                ),
            )
        )
    receipt = evaluate_g6(updates, block_mode="per_aux")
    assert receipt.passed
    assert receipt.metrics["ratio_median"] == {
        "cot": 0.75,
        "future": 0.74,
        "verify": 0.73,
    }


def test_g6_nonfinite_is_a_hard_stop_not_a_metric_failure():
    updates = _g6_updates()
    updates[8] = G6Update(
        u_pre=8,
        aux_reachable=True,
        track_reachable=True,
        cosine_total_track=math.nan,
        signed_projection=1.0,
        aux_track_ratio=1.0,
    )
    with pytest.raises(GateContractError, match="nonfinite"):
        evaluate_g6(updates, block_mode="bstar")


def test_g7_exact_per_stream_total_median_and_saturation_gates():
    receipt = evaluate_g7(_g7_updates(saturated=6))
    assert receipt.passed
    assert receipt.metrics["method_scale_saturation_denominator"] == 128
    assert receipt.metrics["method_scale_saturation_rate"] == pytest.approx(6 / 128)
    assert receipt.metrics["per_stream_max"] == {"polar": 0.4}


def test_g7_bound_and_strict_five_percent_failures_stop():
    bound_failure = evaluate_g7(_g7_updates(stream_ratio=0.5001001))
    assert not bound_failure.passed
    assert not bound_failure.checks["per_stream_bound.polar"]["passed"]

    saturation_failure = evaluate_g7(_g7_updates(saturated=7))
    assert not saturation_failure.passed
    assert saturation_failure.metrics["method_scale_saturation_rate"] > 0.05


def test_g7_prev_saturation_six_of_128_passes_and_seven_fails():
    passing = evaluate_g7(_g7_updates(prev_saturated=6))
    assert passing.passed
    assert passing.checks["prev_scale_saturation_rate"]["passed"]
    assert passing.metrics["prev_scale_saturation_rate"] == pytest.approx(6 / 128)
    assert passing.metrics["abs_tanh_s_prev_max"] == pytest.approx(0.995)
    assert passing.to_dict()["contract"]["prev_saturation_denominator"] == "updates"

    failing = evaluate_g7(_g7_updates(prev_saturated=7))
    assert not failing.passed
    assert not failing.checks["prev_scale_saturation_rate"]["passed"]
    assert failing.metrics["prev_scale_saturation_rate"] == pytest.approx(7 / 128)
    assert failing.metrics["prev_scale_saturation_rate"] > 0.05


def test_g7_prev_scale_absent_everywhere_keeps_receipt_backward_compatible():
    receipt = evaluate_g7(_g7_updates())
    assert receipt.passed
    assert "prev_scale_saturation_rate" not in receipt.checks
    assert receipt.metrics["prev_scale_saturation_rate"] is None
    assert receipt.metrics["abs_tanh_s_prev_max"] is None


def test_g7_prev_scale_mixed_presence_out_of_range_and_nonscalar_fail_closed():
    mixed = _g7_updates(prev_saturated=0)
    mixed[5] = G7Update(
        u_pre=5,
        per_method_over_base={"polar": [0.4, 0.3]},
        total_method_over_base=[0.8],
        abs_tanh_method_scales={"polar": 0.2},
        r_prev=[0.3],
    )
    with pytest.raises(GateContractError, match="only some updates"):
        evaluate_g7(mixed)

    out_of_range = _g7_updates(prev_saturated=0)
    out_of_range[0] = G7Update(
        u_pre=0,
        per_method_over_base={"polar": [0.4, 0.3]},
        total_method_over_base=[0.8],
        abs_tanh_method_scales={"polar": 0.2},
        r_prev=[0.3],
        abs_tanh_s_prev=1.5,
    )
    with pytest.raises(GateContractError, match=r"must be in \[0,1\]"):
        evaluate_g7(out_of_range)

    nonscalar = _g7_updates(prev_saturated=0)
    nonscalar[0] = G7Update(
        u_pre=0,
        per_method_over_base={"polar": [0.4, 0.3]},
        total_method_over_base=[0.8],
        abs_tanh_method_scales={"polar": 0.2},
        r_prev=[0.3],
        abs_tanh_s_prev=[0.1, 0.2],
    )
    with pytest.raises(GateContractError, match="exactly one scalar"):
        evaluate_g7(nonscalar)


def test_g7_mapping_coercion_reads_abs_tanh_s_prev():
    payloads = [
        {
            "u_pre": u_pre,
            "per_method_over_base": {"polar": [0.4, 0.3]},
            "total_method_over_base": [0.8],
            "abs_tanh_method_scales": {"polar": 0.2},
            "r_prev": [0.3],
            "abs_tanh_s_prev": 0.25,
        }
        for u_pre in range(128)
    ]
    receipt = evaluate_g7(payloads)
    assert receipt.passed
    assert receipt.metrics["prev_scale_saturation_rate"] == 0.0
    assert receipt.metrics["abs_tanh_s_prev_max"] == pytest.approx(0.25)


def test_g7_requires_updates_times_streams_denominator_and_finite_values():
    updates = _g7_updates()
    updates[0] = G7Update(
        u_pre=0,
        per_method_over_base={"polar": [0.2]},
        total_method_over_base=[0.2],
        abs_tanh_method_scales={"polar": [0.1, 0.2]},
    )
    with pytest.raises(GateContractError, match="exactly one scalar"):
        evaluate_g7(updates)

    updates = _g7_updates()
    updates[0] = G7Update(
        u_pre=0,
        per_method_over_base={"polar": [math.inf]},
        total_method_over_base=[0.2],
        abs_tanh_method_scales={"polar": 0.1},
    )
    with pytest.raises(GateContractError, match="nonfinite"):
        evaluate_g7(updates)


def test_g8_all_strata_improve_gap_contracts_and_logged_ceiling_passes():
    receipt = _passing_g8()
    assert receipt.passed
    assert set(receipt.metrics["self_mode_improvement_delta"]) == {
        "overall",
        "change",
        "turn",
        "other",
    }
    assert receipt.metrics["overall_gaps_self_minus_logged"] == pytest.approx(
        {
            "s_self_update0": 0.30,
            "s_self_update128": 0.15,
            "s_ctrl_update128": 0.25,
        }
    )


def test_g8_improvement_requires_at_least_one_e_minus_six():
    update0 = _summary(1.0)
    exact_pass = _summary(1.0 - G8_TAU)
    receipt = evaluate_g8(
        s_self_update0={"logged": _summary(0.8), "self": update0},
        s_self_update128={"logged": _summary(0.85), "self": exact_pass},
        s_ctrl_update128={"logged": _summary(0.8), "self": _summary(1.1)},
    )
    assert receipt.passed

    too_small = _summary(1.0 - 0.5e-6)
    receipt = evaluate_g8(
        s_self_update0={"logged": _summary(0.8), "self": update0},
        s_self_update128={"logged": _summary(0.85), "self": too_small},
        s_ctrl_update128={"logged": _summary(0.8), "self": _summary(1.1)},
    )
    assert not receipt.passed
    assert not receipt.checks["self_improvement.overall"]["passed"]


def test_g8_gap_and_logged_ceiling_failures_are_explicit():
    no_gap_contraction = evaluate_g8(
        s_self_update0={"logged": _summary(0.5), "self": _summary(0.8)},
        s_self_update128={"logged": _summary(0.3), "self": _summary(0.6)},
        s_ctrl_update128={"logged": _summary(0.5), "self": _summary(0.7)},
    )
    assert not no_gap_contraction.checks["gap_contraction"]["passed"]
    assert not no_gap_contraction.checks["gap_below_s_ctrl_update128"]["passed"]

    ceiling_failure = evaluate_g8(
        s_self_update0={"logged": _summary(0.5), "self": _summary(0.8)},
        s_self_update128={"logged": _summary(0.56), "self": _summary(0.6)},
        s_ctrl_update128={"logged": _summary(0.5), "self": _summary(0.75)},
    )
    assert not ceiling_failure.checks["logged_ceiling"]["passed"]


def test_g8_zero_support_and_fixed_support_count_mismatch_hard_stop():
    with pytest.raises(GateContractError, match="G8_ZERO_SUPPORT"):
        StratifiedLossSummary(
            means={name: 0.5 for name in ("overall", "change", "turn", "other")},
            counts={"overall": 4, "change": 0, "turn": 1, "other": 1},
        )

    with pytest.raises(GateContractError, match="fixed-support counts differ"):
        evaluate_g8(
            s_self_update0={"logged": _summary(0.5), "self": _summary(0.8)},
            s_self_update128={
                "logged": _summary(0.45, count=5),
                "self": _summary(0.6),
            },
            s_ctrl_update128={"logged": _summary(0.5), "self": _summary(0.75)},
        )


def test_g9_exact_reset_strict_range_reconstruction_and_drift_pass():
    receipt = _passing_g9()
    assert receipt.passed
    assert receipt.metrics["range_violation_rate"] == pytest.approx(0.04)
    assert receipt.metrics["reconstruction_error_max"] == 1e-6
    assert receipt.metrics["self_drift_ratio"] == 2.0


def test_g9_each_threshold_failure_stops():
    receipt = evaluate_g9(
        expected_static_resets=12,
        observed_static_resets=11,
        nonfinite_reset_count=1,
        range_violation_count=5,
        range_observation_count=100,
        reconstruction_errors=[1.000001e-6],
        first_quartile_self_errors=[1.0],
        last_quartile_self_errors=[2.01],
    )
    assert not receipt.passed
    assert receipt.to_dict()["decision"] == "STOP"
    assert all(not check["passed"] for check in receipt.checks.values())


def test_g9_invalid_counts_and_nonfinite_telemetry_hard_stop():
    with pytest.raises(GateContractError, match="exceed observations"):
        evaluate_g9(
            expected_static_resets=1,
            observed_static_resets=1,
            nonfinite_reset_count=0,
            range_violation_count=2,
            range_observation_count=1,
            reconstruction_errors=[0.0],
            first_quartile_self_errors=[1.0],
            last_quartile_self_errors=[1.0],
        )
    with pytest.raises(GateContractError, match="nonfinite"):
        evaluate_g9(
            expected_static_resets=1,
            observed_static_resets=1,
            nonfinite_reset_count=0,
            range_violation_count=0,
            range_observation_count=1,
            reconstruction_errors=[math.nan],
            first_quartile_self_errors=[1.0],
            last_quartile_self_errors=[1.0],
        )


def test_combined_receipt_requires_g6_through_g9_and_propagates_stop():
    g6 = evaluate_g6(_g6_updates(), block_mode="bstar")
    g7 = evaluate_g7(_g7_updates())
    g8 = _passing_g8()
    g9 = _passing_g9()
    combined = build_smoke_gate_receipt(g6, g7, g8, g9)
    assert combined["passed"]
    assert combined["formal_training_authorized"]
    assert combined["gate_order"] == ["G6", "G7", "G8", "G9"]

    failed_g9 = evaluate_g9(
        expected_static_resets=12,
        observed_static_resets=11,
        nonfinite_reset_count=0,
        range_violation_count=0,
        range_observation_count=100,
        reconstruction_errors=[0.0],
        first_quartile_self_errors=[1.0],
        last_quartile_self_errors=[1.0],
    )
    combined = build_smoke_gate_receipt(g6, g7, g8, failed_g9)
    assert not combined["passed"]
    assert combined["decision"] == "STOP"
    assert not combined["formal_training_authorized"]

    with pytest.raises(GateContractError, match="requires"):
        build_smoke_gate_receipt(g6, g7, g8)
