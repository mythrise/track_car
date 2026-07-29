import numpy as np
import pytest

from car_runtime.camera_source import apply_frame_rotation


def test_apply_frame_rotation_preserves_or_rotates_180_degrees():
    frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    assert apply_frame_rotation(frame, 0) is frame
    np.testing.assert_array_equal(apply_frame_rotation(frame, 180), frame[::-1, ::-1])


def test_apply_frame_rotation_rejects_unknown_angle():
    with pytest.raises(ValueError, match="Unsupported frame rotation"):
        apply_frame_rotation(np.zeros((1, 1, 3), dtype=np.uint8), 90)
