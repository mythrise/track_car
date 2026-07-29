import sys
from types import ModuleType, SimpleNamespace

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
    tmp_path,
):
    calls = {}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls["processor"] = (model_id, kwargs)
            return "processor"

    class FakeModel:
        def __init__(self, config):
            calls["model_config"] = config

        def state_dict(self):
            return {"weight": None}

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
    config_path = tmp_path / "config.json"
    config_path.write_text('{"model_type": "omdet-turbo"}', encoding="utf-8")

    def fake_download(model_id, filename, **kwargs):
        calls.setdefault("downloads", []).append((model_id, filename, kwargs))
        return str(config_path if filename == "config.json" else tmp_path / filename)

    fake_hub.hf_hub_download = fake_download
    fake_safetensors = ModuleType("safetensors")
    fake_safetensors_torch = ModuleType("safetensors.torch")
    fake_safetensors_torch.load_file = lambda path: {"weight": path}
    fake_transformers = ModuleType("transformers")
    fake_transformers.__version__ = "5.7.0"
    fake_transformers.AutoProcessor = FakeProcessor
    fake_transformers.OmDetTurboForObjectDetection = FakeModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    monkeypatch.setitem(sys.modules, "safetensors", fake_safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", fake_safetensors_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    def fake_explicit_config(raw):
        calls["raw_config"] = raw
        return "config"

    monkeypatch.setattr(target_detector, "explicit_omdet_config", fake_explicit_config)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    processor, model = target_detector.load_omdet_components("cpu")
    assert processor == "processor"
    assert isinstance(model, FakeModel)
    assert calls["model_config"] == "config"
    assert calls["raw_config"] == {"model_type": "omdet-turbo"}
    assert calls["state"] == ({"weight": str(tmp_path / "model.safetensors")}, False)
    assert calls["device"] == "cpu"
    assert calls["eval"] is True


def test_explicit_omdet_config_replaces_legacy_backbone_without_hub_lookup(monkeypatch):
    class FakeTimmBackboneConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return {"model_type": "timm_backbone", **self.kwargs}

    class FakeOmDetTurboConfig:
        @classmethod
        def from_dict(cls, raw):
            return SimpleNamespace(raw=raw)

    fake_transformers = ModuleType("transformers")
    fake_transformers.TimmBackboneConfig = FakeTimmBackboneConfig
    fake_transformers.OmDetTurboConfig = FakeOmDetTurboConfig
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    config = target_detector.explicit_omdet_config(
        {
            "model_type": "omdet-turbo",
            "backbone": "swin_tiny_patch4_window7_224",
            "backbone_config": None,
            "backbone_kwargs": {
                "out_indices": [1, 2, 3],
                "img_size": 640,
                "always_partition": True,
            },
            "use_timm_backbone": True,
            "use_pretrained_backbone": False,
        }
    )
    assert config.raw["backbone_config"] == {
        "model_type": "timm_backbone",
        "backbone": "swin_tiny_patch4_window7_224",
        "out_indices": [1, 2, 3],
        "features_only": True,
        "use_pretrained_backbone": False,
        "img_size": 640,
        "always_partition": True,
    }
    assert config.raw["backbone"] is None
    assert config.raw["use_timm_backbone"] is False
    assert config.raw["use_pretrained_backbone"] is False


def test_omdet_state_dict_normalizes_prefix_for_transformers_5_model():
    normalized = target_detector.normalize_omdet_state_dict(
        {
            "language_backbone.model.text_model.encoder.weight": "legacy",
            "vision_backbone.weight": "unchanged",
        },
        expected_keys={
            "language_backbone.model.encoder.weight",
            "vision_backbone.weight",
        },
    )
    assert normalized == {
        "language_backbone.model.encoder.weight": "legacy",
        "vision_backbone.weight": "unchanged",
    }


def test_omdet_state_dict_preserves_prefix_for_transformers_4_model():
    state = {
        "language_backbone.model.text_model.encoder.weight": "legacy",
        "vision_backbone.weight": "unchanged",
    }
    normalized = target_detector.normalize_omdet_state_dict(
        state,
        expected_keys={
            "language_backbone.model.text_model.encoder.weight",
            "vision_backbone.weight",
        },
    )
    assert normalized == state


def test_transformers_major_version_handles_release_and_dev_versions():
    assert target_detector.transformers_major_version("4.57.6") == 4
    assert target_detector.transformers_major_version("5.0.0.dev0") == 5
