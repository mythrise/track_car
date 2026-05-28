#!/usr/bin/env python3
"""Check Raspberry Pi camera startup and frame read latency."""

from __future__ import annotations

import argparse
import time

try:
    from camera_source import BACKENDS, open_camera
except ImportError:
    from car_runtime.camera_source import BACKENDS, open_camera


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--camera_backend", choices=BACKENDS, default="auto")
    ap.add_argument("--camera_fourcc", default="auto", help="OpenCV/V4L2 pixel format, for example MJPG or YUYV.")
    ap.add_argument("--camera_fps", type=float, default=30.0)
    ap.add_argument("--camera_saturation", type=float, default=40.0)
    ap.add_argument("--camera_ready_timeout", type=float, default=5.0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    cap = open_camera(
        args.camera_index,
        args.camera_backend,
        args.width,
        args.height,
        warmup=0.0,
        fourcc=args.camera_fourcc,
        fps=args.camera_fps,
        saturation=args.camera_saturation,
        ready_timeout=args.camera_ready_timeout,
    )

    for i in range(max(1, args.frames)):
        t1 = time.time()
        ok, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"[camera_check] frame={i} ok={ok} read_time={time.time() - t1:.3f}s shape={shape}", flush=True)
    cap.release()


if __name__ == "__main__":
    main()
