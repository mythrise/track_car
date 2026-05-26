#!/usr/bin/env python3
"""Camera backends for Raspberry Pi and USB cameras."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2


BACKENDS = ("auto", "picamera2", "opencv", "v4l2")


def cv_backend(name: str):
    if name == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return 0


def _normalize_fourcc(fourcc: Optional[str]) -> Optional[str]:
    if fourcc is None:
        return None
    value = fourcc.strip().upper()
    if not value or value == "AUTO":
        return None
    if len(value) != 4:
        raise ValueError(f"Camera FourCC must be four characters, got: {fourcc}")
    return value


def _fourcc_to_string(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = [chr((code >> 8 * i) & 0xFF) for i in range(4)]
    return "".join(ch if ch.isprintable() else "?" for ch in chars)


@dataclass
class CameraSource:
    backend: str
    source: object

    def read(self):
        if self.backend == "picamera2":
            frame = self.source.capture_array()
            if frame is None:
                return False, None
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame
        return self.source.read()

    def release(self) -> None:
        if self.backend == "picamera2":
            self.source.stop()
            self.source.close()
        else:
            self.source.release()


def _open_picamera2(index: int, width: int, height: int) -> CameraSource:
    from picamera2 import Picamera2  # type: ignore

    try:
        camera = Picamera2(camera_num=index)
    except TypeError:
        camera = Picamera2()
    config = camera.create_video_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    return CameraSource("picamera2", camera)


def _open_opencv(index: int, backend: str, width: int, height: int, fourcc: Optional[str]) -> CameraSource:
    cap = cv2.VideoCapture(index, cv_backend(backend)) if backend == "v4l2" else cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"OpenCV backend {backend} could not open camera index {index}")
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(3, width)
    cap.set(4, height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    actual_fourcc = _fourcc_to_string(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"[camera] opencv actual_fourcc={actual_fourcc}", flush=True)
    return CameraSource(backend, cap)


def open_camera(
    index: int,
    backend: str,
    width: int,
    height: int,
    warmup: float,
    fourcc: Optional[str] = None,
) -> CameraSource:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported camera backend: {backend}")

    fourcc_value = _normalize_fourcc(fourcc)
    candidates = ["picamera2", "opencv", "v4l2"] if backend == "auto" else [backend]
    errors = []
    for candidate in candidates:
        t0 = time.time()
        fourcc_text = fourcc_value or "auto"
        print(
            f"[camera] trying backend={candidate} index={index} size={width}x{height} fourcc={fourcc_text}",
            flush=True,
        )
        try:
            if candidate == "picamera2":
                source = _open_picamera2(index, width, height)
            else:
                source = _open_opencv(index, candidate, width, height, fourcc_value)
            if warmup > 0:
                time.sleep(warmup)
            print(f"[camera] opened backend={source.backend} open_time={time.time() - t0:.3f}s", flush=True)
            return source
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            print(f"[camera] backend failed: {candidate}: {exc}", flush=True)

    raise RuntimeError(
        f"Could not open camera index {index}. Tried {', '.join(candidates)}. "
        f"Errors: {' | '.join(errors)}"
    )
