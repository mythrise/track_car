import math

import numpy as np

from data_pipeline.kinematics import integrate_actions


def test_pure_yaw_uses_every_action():
    trajectory = integrate_actions([[0, 0, 1], [0, 0, 1]], dt=0.5)
    np.testing.assert_allclose(trajectory[:, :2], 0.0, atol=1e-7)
    np.testing.assert_allclose(trajectory[:, 2], [0.5, 1.0], atol=1e-7)


def test_yaw_then_forward_uses_new_heading():
    trajectory = integrate_actions([[0, 0, math.pi / 2], [1, 0, 0]], dt=1.0)
    np.testing.assert_allclose(trajectory[-1], [0.0, 1.0, math.pi / 2], atol=1e-6)


def test_forward_then_yaw_keeps_forward_translation_in_start_frame():
    trajectory = integrate_actions([[1, 0, 0], [0, 0, math.pi / 2]], dt=1.0)
    np.testing.assert_allclose(trajectory[-1], [1.0, 0.0, math.pi / 2], atol=1e-6)


def test_mirror_round_trip():
    actions = np.asarray([[0.5, 0.2, -0.3], [0.8, -0.1, 0.4]], dtype=np.float32)
    mirrored_actions = actions.copy()
    mirrored_actions[:, 1:] *= -1
    mirrored_trajectory = integrate_actions(mirrored_actions, dt=0.1)
    expected = integrate_actions(actions, dt=0.1).copy()
    expected[:, 1:] *= -1
    np.testing.assert_allclose(mirrored_trajectory, expected, atol=1e-6)

    round_trip = mirrored_actions.copy()
    round_trip[:, 1:] *= -1
    np.testing.assert_allclose(round_trip, actions, atol=0.0)
