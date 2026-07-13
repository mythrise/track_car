"""Shared OmDet-Turbo person detector with a visible Haar fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import cv2


OMDET_MODEL_ID = "omlab/omdet-turbo-swin-tiny-hf"


def _offline_mode() -> bool:
    return os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def load_omdet_components(device: str = "cpu"):
    """Load OmDet the same way for dataset building and model-side bbox use.

    Imports are intentionally lazy so integrity checks and unit tests never
    import torch or initialize model weights.
    """

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoProcessor, OmDetTurboForObjectDetection

    kwargs = {"local_files_only": True} if _offline_mode() else {}
    processor = AutoProcessor.from_pretrained(OMDET_MODEL_ID, **kwargs)
    # Instantiate from config, then load weights manually: from_pretrained()
    # leaves the timm swin backbone's non-persistent attn_mask buffers on the
    # meta device (they are not in the checkpoint), and .to(device) then fails
    # with "Cannot copy out of meta tensor".
    config = AutoConfig.from_pretrained(OMDET_MODEL_ID, **kwargs)
    model = OmDetTurboForObjectDetection(config)
    state_path = hf_hub_download(OMDET_MODEL_ID, "model.safetensors", **kwargs)
    missing, unexpected = model.load_state_dict(load_file(state_path), strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"OmDet-Turbo weights incomplete: {len(missing)} missing, {len(unexpected)} unexpected"
        )
    return processor, model.to(device).eval()


def detect_haar_face(frame) -> tuple[float, float, float, float] | None:
    """Return the largest frontal-face bbox as normalized ``cx, cy, w, h``."""

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    faces = cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) == 0:
        return None
    x, y, width, height = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    frame_h, frame_w = frame.shape[:2]
    return (
        (float(x) + float(width) / 2.0) / float(frame_w),
        (float(y) + float(height) / 2.0) / float(frame_h),
        float(width) / float(frame_w),
        float(height) / float(frame_h),
    )


def detect_omdet_person(frame, processor, model, threshold: float = 0.25):
    """Return the highest-scoring OmDet ``person`` bbox in normalized form."""

    import torch
    from PIL import Image

    if frame is None or getattr(frame, "size", 0) == 0:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")
    labels = ["person"]
    inputs = processor(image, text=labels, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=[(image.height, image.width)],
        text_labels=labels,
        threshold=float(threshold),
        nms_threshold=0.3,
    )
    if not results:
        return None
    result = results[0]
    boxes = result.get("boxes")
    scores = result.get("scores")
    text_labels = result.get("text_labels", [])
    if boxes is None or scores is None or len(boxes) == 0:
        return None

    candidates = []
    for index in range(len(boxes)):
        label = str(text_labels[index]).lower() if index < len(text_labels) else "person"
        if label in {"person", "human"}:
            candidates.append((float(scores[index].item()), index))
    if not candidates:
        return None
    _, best = max(candidates)
    x0, y0, x1, y1 = [float(value) for value in boxes[best].tolist()]
    width = max(1e-6, x1 - x0)
    height = max(1e-6, y1 - y0)
    return (
        (x0 + width / 2.0) / float(image.width),
        (y0 + height / 2.0) / float(image.height),
        width / float(image.width),
        height / float(image.height),
    )


class PersonTargetDetector:
    """Prefer OmDet-Turbo and fail over to the legacy Haar detector."""

    def __init__(
        self,
        device: str = "cpu",
        omdet_loader: Callable | None = None,
        haar_detector: Callable | None = None,
        warning_sink: Callable[[str], None] = print,
    ):
        self.device = str(device)
        self.omdet_loader = omdet_loader or load_omdet_components
        self.haar_detector = haar_detector or detect_haar_face
        self.warning_sink = warning_sink
        self._omdet_initialized = False
        self._processor = None
        self._model = None
        self._fallback_warned = False

    def _warn_fallback(self, reason: object) -> None:
        if self._fallback_warned:
            return
        self._fallback_warned = True
        reason_text = " ".join(str(reason).split())
        if not reason_text:
            reason_text = type(reason).__name__ if reason is not None else "unknown"
        self.warning_sink(
            "!!! [target_detector] OmDet-Turbo unavailable; FALLING BACK TO HAAR FRONTAL-FACE "
            f"pseudo-labels. Polar coverage/quality will be lower. reason={reason_text}"
        )

    def _ensure_omdet(self) -> None:
        if self._omdet_initialized:
            return
        self._omdet_initialized = True
        try:
            self._processor, self._model = self.omdet_loader(self.device)
        except Exception as exc:
            self._processor = None
            self._model = None
            self._warn_fallback(exc)

    @property
    def using_omdet(self) -> bool:
        self._ensure_omdet()
        return self._processor is not None and self._model is not None

    def detect_haar(self, frame):
        return self.haar_detector(frame)

    def detect(self, frame, haar_result=None):
        """Return ``(bbox, source)`` where source is ``omdet`` or ``haar``."""

        self._ensure_omdet()
        if self._processor is not None and self._model is not None:
            try:
                detected = detect_omdet_person(frame, self._processor, self._model)
                if detected is not None:
                    return detected, "omdet"
            except Exception as exc:
                self._processor = None
                self._model = None
                self._warn_fallback(exc)
        if self._processor is None or self._model is None:
            self._warn_fallback("model not initialized")
        fallback = haar_result if haar_result is not None else self.detect_haar(frame)
        return fallback, "haar"


def get_default_target_detector(device: str = "cpu") -> PersonTargetDetector:
    return PersonTargetDetector(device=device)
