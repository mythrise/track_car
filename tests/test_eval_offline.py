import numpy as np
import pytest

from data_pipeline.kinematics import integrate_actions
from scripts.eval_offline import (
    compute_metrics,
    evaluate_predictions,
    render_comparison_table,
    transition_event_mask,
    waypoints_to_step_actions,
)


def test_absolute_waypoints_are_inverted_to_step_actions():
    actions = np.asarray([[1.0, 0.0, 1.0], [0.5, 0.2, -0.5]], dtype=np.float32)
    waypoints = integrate_actions(actions, 0.1)
    recovered = waypoints_to_step_actions(waypoints, 0.1)
    np.testing.assert_allclose(recovered, actions, atol=1e-5)


def test_offline_metrics_cover_axis_sign_transition_and_saturation():
    gt = np.asarray([[[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, 1.0]]])
    pred = gt.copy()
    metrics = compute_metrics(pred, gt, np.zeros((1, 3)), np.ones((1, 3), dtype=bool))
    assert metrics["smooth_l1"] == {"forward": 0.0, "strafe": 0.0, "yaw": 0.0}
    assert metrics["turn_sign_accuracy"] == 1.0
    assert metrics["transition"]["f1"] == 1.0
    assert metrics["saturation_rate"]["forward"] == pytest.approx(2.0 / 3.0)


def test_transition_events_follow_turn_activity_and_sign_not_yaw_difference():
    actions = np.asarray([[[0.0, 0.0, 0.25], [0.0, 0.0, 0.7], [0.0, 0.0, -0.4], [0.0, 0.0, 0.0]]])
    previous = np.asarray([[0.0, 0.0, 0.1]])
    np.testing.assert_array_equal(
        transition_event_mask(actions, previous, threshold=0.2),
        [[True, False, True, True]],
    )


def test_offline_evaluation_groups_transition_types_and_renders_multiple_runs():
    records = [
        {
            "step_actions": [[1.0, 0.0, 0.0]],
            "prev_action": [1.0, 0.0, 0.0],
            "transition_type": "steady_forward",
        },
        {
            "step_actions": [[0.8, 0.0, -1.0]],
            "prev_action": [1.0, 0.0, 0.0],
            "transition_type": "turn_onset",
        },
    ]
    pred = np.asarray([record["step_actions"] for record in records])
    metrics = evaluate_predictions(pred, records)
    assert set(metrics["by_transition_type"]) == {"steady_forward", "turn_onset"}
    results = {
        "absolute": {"label_mode": "absolute", "metrics": metrics},
        "step": {"label_mode": "step_action", "metrics": metrics},
    }
    table = render_comparison_table(results)
    assert "absolute" in table and "step_action" in table
