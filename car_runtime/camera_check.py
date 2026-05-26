#!/usr/bin/env python3
"""Check Raspberry Pi camera startup and frame read latency."""

from __future__ import annotations

import argparse
import time

import cv2


def cv_backend(name: str):
    if name == "v4l2" and hasattr(cv2, "CAP_V4L2"):
        return cv2.CAP_V4L2
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--camera_backend", choices=["auto", "v4l2"], default="auto")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--frames", type=int, default=5)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[camera_check] opening index={args.camera_index} backend={args.camera_backend}", flush=True)
    cap = (
        cv2.VideoCapture(args.camera_index, cv_backend(args.camera_backend))
        if args.camera_backend != "auto"
        else cv2.VideoCapture(args.camera_index)
    )
    print(f"[camera_check] open_time={time.time() - t0:.3f}s is_opened={cap.isOpened()}", flush=True)
    if not cap.isOpened():
        raise SystemExit(1)

    cap.set(3, args.width)
    cap.set(4, args.height)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    for i in range(max(1, args.frames)):
        t1 = time.time()
        ok, frame = cap.read()
        shape = None if frame is None else frame.shape
        print(f"[camera_check] frame={i} ok={ok} read_time={time.time() - t1:.3f}s shape={shape}", flush=True)
    cap.release()


if __name__ == "__main__":
    main()
