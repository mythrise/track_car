#!/usr/bin/env python3
"""Data collection script — records frames + motor commands while human teleops.

Run on Raspberry Pi while controlling the car via APP/keyboard:
    python3 data_pipeline/collect_data.py --episode_name ep001 --instruction "follow the person in red"

Output: data/collected/<episode_name>/frame_XXXXXX.jpg + meta_XXXXXX.json
"""

import argparse
import cv2
import json
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car_runtime"))

try:
    from car_hardware import MAX_SPEED, CarHardware, boosted_motors, command_from_key
    from camera_source import BACKENDS, open_camera
    from process_cleanup import cleanup_named_processes
except ImportError:
    from car_runtime.car_hardware import MAX_SPEED, CarHardware, boosted_motors, command_from_key
    from car_runtime.camera_source import BACKENDS, open_camera
    from car_runtime.process_cleanup import cleanup_named_processes


def read_key_nonblocking():
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    return " " if ch == " " else ch.lower()


def clamp_nonnegative(value: int, upper: int) -> int:
    return max(0, min(int(value), int(upper)))


def apply_keyboard_command(
    hardware,
    key,
    last_cmd,
    speed,
    kick_speed,
    kick_duration,
    kick_repeat,
    last_kick_time,
):
    """Apply one keyboard event using the same motor path as move_test.py."""
    if key is None:
        return last_cmd, last_kick_time, False, None

    now = time.time()
    next_cmd = command_from_key(key, speed)
    kick_speed = clamp_nonnegative(kick_speed, MAX_SPEED)
    kick_duration = max(0.0, min(float(kick_duration), 0.25))
    command_changed = next_cmd.name != last_cmd.name
    repeat_due = kick_repeat > 0 and (now - last_kick_time) >= kick_repeat
    should_kick = (
        kick_speed > 0
        and next_cmd.name != "stop"
        and kick_duration > 0
        and (command_changed or repeat_due)
    )

    if should_kick:
        kick_motors = boosted_motors(next_cmd.motors, kick_speed)
        hardware.run_motors_with_kick(next_cmd.motors, kick_motors, kick_duration)
        return next_cmd, now, True, kick_motors

    hardware.run_motors(next_cmd.motors)
    return next_cmd, last_kick_time, False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_name", required=True)
    ap.add_argument("--instruction", default="follow the person")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--camera_backend", choices=BACKENDS, default="auto")
    ap.add_argument("--camera_fourcc", default="auto", help="OpenCV/V4L2 pixel format, for example MJPG or YUYV.")
    ap.add_argument("--camera_fps", type=float, default=30.0)
    ap.add_argument("--camera_saturation", type=float, default=40.0)
    ap.add_argument("--camera_ready_timeout", type=float, default=5.0)
    ap.add_argument("--camera_warmup", type=float, default=1.0)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out_root", default="data/collected")
    ap.add_argument("--teleop", choices=["none", "keyboard"], default="none")
    ap.add_argument("--speed", type=int, default=300)
    ap.add_argument("--kick_speed", type=int, default=0,
                    help="Optional short startup kick wheel speed for keyboard teleop. Use 0 to disable.")
    ap.add_argument("--kick_duration", type=float, default=0.06,
                    help="Kick duration in seconds, clamped to 0.25.")
    ap.add_argument("--kick_repeat", type=float, default=0.75,
                    help="Minimum seconds between repeated kicks while holding the same key.")
    ap.add_argument("--max_frames", type=int, default=0,
                    help="Stop after this many saved frames. 0 means run until Ctrl+C.")
    ap.add_argument("--max_seconds", type=float, default=0.0,
                    help="Stop after this many seconds. 0 means run until Ctrl+C.")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--no_cleanup_processes", action="store_true",
                    help="Do not kill vendor camera/main processes before collection.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                    help="Print cleanup targets without killing them.")
    args = ap.parse_args()

    if not args.no_cleanup_processes:
        print("[startup] cleaning stale vendor processes", flush=True)
        cleanup_named_processes(["mjpg", "z_main"], dry_run=args.cleanup_dry_run)
        if args.cleanup_dry_run:
            return

    save_dir = os.path.join(args.out_root, args.episode_name)
    os.makedirs(save_dir, exist_ok=True)

    speed = clamp_nonnegative(args.speed, MAX_SPEED)
    kick_speed = clamp_nonnegative(args.kick_speed, MAX_SPEED)
    kick_duration = max(0.0, min(args.kick_duration, 0.25))
    kick_repeat = max(0.0, float(args.kick_repeat))
    fps = max(1, min(int(args.fps), 60))
    max_frames = max(0, int(args.max_frames))
    max_seconds = max(0.0, float(args.max_seconds))
    frame_idx = 0
    last_cmd = command_from_key(" ", speed)
    last_kick_time = 0.0
    start_time = time.time()
    cap = None
    hardware = None
    old_term = None
    interval = 1.0 / fps

    try:
        if args.teleop == "keyboard":
            print("[startup] opening car hardware", flush=True)
            hardware = CarHardware(reset_servos=False, dry_run=args.dry_run, uart_port=args.uart_port)
            hardware.stop()
            print("[startup] car hardware ready", flush=True)

        cap = open_camera(
            args.camera_index,
            args.camera_backend,
            args.width,
            args.height,
            max(0.0, min(args.camera_warmup, 5.0)),
            fourcc=args.camera_fourcc,
            fps=args.camera_fps,
            saturation=args.camera_saturation,
            ready_timeout=args.camera_ready_timeout,
        )

        print(f"[collect] saving to {save_dir}, press Ctrl+C to stop")
        print(f"[collect] speed={speed} fps={fps} dry_run={args.dry_run}")
        if args.teleop == "keyboard":
            print("[collect] keyboard teleop: w/s forward/back, a/d turn, q/e strafe, space/x stop")
            if sys.stdin.isatty():
                old_term = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())

        while True:
            if max_frames and frame_idx >= max_frames:
                break
            if max_seconds and (time.time() - start_time) >= max_seconds:
                break

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, -1)
            frame_kick_applied = False
            frame_kick_motors = None

            if args.teleop == "keyboard":
                key = read_key_nonblocking()
                last_cmd, last_kick_time, frame_kick_applied, frame_kick_motors = apply_keyboard_command(
                    hardware,
                    key,
                    last_cmd,
                    speed,
                    kick_speed,
                    kick_duration,
                    kick_repeat,
                    last_kick_time,
                )

            fname = f"frame_{frame_idx:06d}.jpg"
            frame_path = os.path.join(save_dir, fname)
            if not cv2.imwrite(frame_path, frame):
                raise RuntimeError(f"Failed to write frame: {frame_path}")

            meta = {
                "frame": fname,
                "timestamp": time.time(),
                "frame_idx": frame_idx,
                "instruction": args.instruction,
                "episode": args.episode_name,
                "teleop": args.teleop,
                "command": last_cmd.name,
                "motors": last_cmd.motors,
                "action": last_cmd.action,
                "speed": speed,
                "kick_applied": frame_kick_applied,
                "kick_motors": frame_kick_motors,
            }
            with open(os.path.join(save_dir, f"meta_{frame_idx:06d}.json"), "w") as f:
                json.dump(meta, f)

            frame_idx += 1
            elapsed = time.time() - t0
            if elapsed < interval:
                time.sleep(interval - elapsed)

            if frame_idx % 100 == 0:
                print(f"  collected {frame_idx} frames")

    except KeyboardInterrupt:
        pass
    finally:
        if old_term is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_term)
        if hardware is not None:
            hardware.close()
        if cap is not None:
            cap.release()
        # Write episode summary
        summary = {
            "episode": args.episode_name,
            "instruction": args.instruction,
            "n_frames": frame_idx,
            "width": args.width,
            "height": args.height,
            "fps": fps,
            "camera_backend": args.camera_backend,
            "camera_fourcc": args.camera_fourcc,
            "camera_fps": args.camera_fps,
            "camera_saturation": args.camera_saturation,
            "camera_ready_timeout": args.camera_ready_timeout,
            "teleop": args.teleop,
            "speed": speed,
            "kick_speed": kick_speed,
            "kick_duration": kick_duration,
            "kick_repeat": kick_repeat,
            "dry_run": args.dry_run,
        }
        with open(os.path.join(save_dir, "episode.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[collect] done. {frame_idx} frames saved to {save_dir}")


if __name__ == "__main__":
    main()
