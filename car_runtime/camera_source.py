#!/usr/bin/env python3
"""Camera backends for Raspberry Pi and USB cameras."""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Optional

import cv2


BACKENDS = ("auto", "v4l2", "opencv", "picamera2")
FRAME_ROTATIONS = (0, 180)


def apply_frame_rotation(frame, degrees: int):
    """Return a frame in the configured camera orientation."""

    degrees = int(degrees)
    if degrees == 0:
        return frame
    if degrees == 180:
        return cv2.flip(frame, -1)
    raise ValueError(f"Unsupported frame rotation: {degrees}")


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


def _video_capture(index: int, backend: str):
    """Create VideoCapture while supporting older Raspberry Pi OpenCV builds."""
    if backend == "v4l2":
        try:
            return cv2.VideoCapture(index, cv_backend("v4l2"))
        except TypeError as exc:
            print(
                f"[camera] OpenCV does not support VideoCapture(index, apiPreference): {exc}; "
                "falling back to VideoCapture(index)",
                flush=True,
            )
    return cv2.VideoCapture(index)


class ThreadedOpenCVCamera:
    """OpenCV camera wrapper based on the proven vendor Camera.py behavior."""

    def __init__(
        self,
        index: int,
        backend: str,
        width: int,
        height: int,
        fourcc: Optional[str],
        fps: float,
        saturation: float,
        ready_timeout: float,
    ) -> None:
        self.index = int(index)
        self.backend = backend
        self.width = int(width)
        self.height = int(height)
        self.fourcc = fourcc
        self.fps = float(fps)
        self.saturation = float(saturation)
        self.ready_timeout = max(0.1, float(ready_timeout))
        self.cap = None
        self.opened = False
        self.frame = None
        self.frame_ok = False
        self._last_error = None
        self._lock = threading.Lock()
        self._thread = None

    def open(self) -> None:
        self.cap = _video_capture(self.index, self.backend)

        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(f"OpenCV backend {self.backend} could not open camera index {self.index}")

        if self.fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if self.saturation >= 0:
            self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_fourcc = _fourcc_to_string(self.cap.get(cv2.CAP_PROP_FOURCC))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"[camera] opencv actual_fourcc={actual_fourcc} fps={actual_fps:.1f} "
            f"size={actual_width}x{actual_height}",
            flush=True,
        )

        self.opened = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self.wait_until_ready()

    def _reader(self) -> None:
        while self.opened:
            try:
                ok, frame = self.cap.read()
            except Exception as exc:
                self._last_error = exc
                ok, frame = False, None

            with self._lock:
                self.frame_ok = bool(ok and frame is not None)
                if self.frame_ok:
                    self.frame = frame

            if not ok:
                time.sleep(0.01)

    def wait_until_ready(self) -> None:
        deadline = time.time() + self.ready_timeout
        while time.time() < deadline:
            with self._lock:
                ready = self.frame_ok and self.frame is not None
            if ready:
                print("[camera] first frame ready", flush=True)
                return
            time.sleep(0.01)
        raise RuntimeError(
            f"Camera opened but no frame arrived within {self.ready_timeout:.1f}s"
            + (f": {self._last_error}" if self._last_error else "")
        )

    def read(self):
        with self._lock:
            ok = self.frame_ok and self.frame is not None
            frame = None if self.frame is None else self.frame.copy()
        return ok, frame

    def release(self) -> None:
        self.opened = False
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("[camera] closed", flush=True)


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


def _open_opencv(
    index: int,
    backend: str,
    width: int,
    height: int,
    fourcc: Optional[str],
    fps: float,
    saturation: float,
    ready_timeout: float,
) -> CameraSource:
    camera = ThreadedOpenCVCamera(
        index=index,
        backend=backend,
        width=width,
        height=height,
        fourcc=fourcc,
        fps=fps,
        saturation=saturation,
        ready_timeout=ready_timeout,
    )
    try:
        camera.open()
    except Exception:
        camera.release()
        raise
    return CameraSource(backend, camera)


def open_camera(
    index: int,
    backend: str,
    width: int,
    height: int,
    warmup: float,
    fourcc: Optional[str] = None,
    fps: float = 30.0,
    saturation: float = 40.0,
    ready_timeout: float = 5.0,
) -> CameraSource:
    if backend not in BACKENDS:
        raise ValueError(f"Unsupported camera backend: {backend}")

    fourcc_value = _normalize_fourcc(fourcc)
    candidates = ["v4l2", "opencv", "picamera2"] if backend == "auto" else [backend]
    errors = []
    for candidate in candidates:
        t0 = time.time()
        candidate_fourcc = fourcc_value
        if candidate == "v4l2" and candidate_fourcc is None:
            candidate_fourcc = "MJPG"
        fourcc_text = candidate_fourcc or "auto"
        print(
            f"[camera] trying backend={candidate} index={index} size={width}x{height} "
            f"fourcc={fourcc_text} fps={fps:g}",
            flush=True,
        )
        try:
            if candidate == "picamera2":
                source = _open_picamera2(index, width, height)
            else:
                source = _open_opencv(
                    index,
                    candidate,
                    width,
                    height,
                    candidate_fourcc,
                    fps,
                    saturation,
                    ready_timeout,
                )
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
