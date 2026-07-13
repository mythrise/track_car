import sys
from types import ModuleType

import numpy as np

from data_pipeline import target_detector


PersonTargetDetector = target_detector.PersonTargetDetector


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


def test_fallback_warning_flattens_multiline_or_empty_exception_reason():
    warnings = []

    def unavailable(_device):
        raise RuntimeError("\nmissing torchvision\n  operator unavailable")

    detector = PersonTargetDetector(
        omdet_loader=unavailable,
        haar_detector=lambda _frame: None,
        warning_sink=warnings.append,
    )
    detector.detect(np.zeros((8, 8, 3), dtype=np.uint8))
    assert len(warnings) == 1
    assert "\n" not in warnings[0]
    assert "reason=missing torchvision operator unavailable" in warnings[0]


def test_omdet_loader_instantiates_from_config_and_loads_weights_without_meta_tensors(
    monkeypatch,
):
    calls = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["processor"] = (model_id, kwargs)
            return "processor"

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["config"] = (model_id, kwargs)
            return "config"

    class FakeModel:
        def __init__(self, config):
            calls["model_config"] = config

        def load_state_dict(self, state, strict):
            calls["state"] = (state, strict)
            return [], []

        def to(self, device):
            calls["device"] = device
            return self

        def eval(self):
            calls["eval"] = True
            return self

    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda model_id, filename, **kwargs: "/tmp/model.safetensors"
    fake_safetensors = ModuleType("safetensors")
    fake_safetensors_torch = ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda path: {"weight": path}
    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = FakeConfig
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.OmDetTurboForObjectDetection = FakeModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    processor, model = target_detector.load_omdet_components("cpu")
    assert processor == "processor"
    assert isinstance(model, FakeModel)
    assert calls["model_config"] == "config"
    assert calls["state"] == ({"weight": "/tmp/model.safetensors"}, False)
    assert calls["device"] == "cpu"
    assert calls["eval"] is True
