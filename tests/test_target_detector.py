import numpy as np

from data_pipeline.target_detector import PersonTargetDetector


def test_omdet_unavailable_falls_back_to_haar_and_warns_once():
    warnings = []

    def unavailable(_device):
        raise RuntimeError("offline")

    expected = (0.5, 0.4, 0.2, 0.3)
    detector = PersonTargetDetector(
        omdet_loader=unavailable,
        haar_detector=lambda _frame: expected,
        warning_sink=warnings.append,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    assert detector.detect(frame) == (expected, "haar")
    assert detector.detect(frame) == (expected, "haar")
    assert len(warnings) == 1
    assert "FALLING BACK TO HAAR" in warnings[0]
