import numpy as np
import pytest

from third_party.OpenTrackVLA.validation_metrics import (
    BalancedControlAccumulator,
    waypoints_to_step_actions,
)
from data_pipeline.kinematics import integrate_actions


def test_shared_waypoint_inversion_matches_kinematics():
    actions = np.asarray([[1.0, 0.0, 1.0], [0.5, 0.2, -0.5]], dtype=np.float32)
    recovered = waypoints_to_step_actions(integrate_actions(actions, 0.1), 0.1)
    np.testing.assert_allclose(recovered, actions, atol=1e-5)


def test_balanced_control_accumulator_is_episode_and_command_macro():
    metric = BalancedControlAccumulator()
    target = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.5, 0.0, 1.0]],
            [[1.0, 0.0, 0.0]],
        ]
    )
    prediction = np.asarray(
        [
            [[1.0, 0.0, 0.0]],
            [[0.5, 0.0, 0.0]],
            [[0.0, 0.0, 0.0]],
        ]
    )
    metric.add(
        prediction,
        target,
        np.ones((3, 1), dtype=bool),
        ["forward", "turn_right", "forward"],
        ["ep1", "ep1", "ep2"],
    )
    result = metric.compute()
    assert result["value"] == pytest.approx(1.0 / 3.0)
    assert result["support"]["ep1"] == {"forward": 1, "turn_right": 1}
