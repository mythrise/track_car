#!/usr/bin/env python3
"""Data collection script — records frames + motor commands while human teleops.

Run on Raspberry Pi while controlling the car via APP/keyboard:
    python3 collect_data.py --episode_name ep001 --instruction "follow the person in red"

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

try:
    from car_hardware import CarHardware, command_from_key
except ImportError:
    from scripts.car.car_hardware import CarHardware, command_from_key


def read_key_nonblocking():
    if not sys.stdin.isatty():
        return None
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    ch = sys.stdin.read(1)
    return " " if ch == " " else ch.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode_name", required=True)
    ap.add_argument("--instruction", default="follow the person")
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out_root", default="data/collected")
    ap.add_argument("--teleop", choices=["none", "keyboard"], default="none")
    ap.add_argument("--speed", type=int, default=300)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    save_dir = os.path.join(args.out_root, args.episode_name)
    os.makedirs(save_dir, exist_ok=True)
    hardware = CarHardware(reset_servos=True, dry_run=args.dry_run) if args.teleop == "keyboard" else None

    cap = cv2.VideoCapture(0)
    cap.set(3, args.width)
    cap.set(4, args.height)
    time.sleep(1.0)

    frame_idx = 0
    interval = 1.0 / args.fps
    old_term = None
    print(f"[collect] saving to {save_dir}, press Ctrl+C to stop")
    if args.teleop == "keyboard":
        print("[collect] keyboard teleop: w/s forward/back, a/d turn, q/e strafe, space stop")
        if sys.stdin.isatty():
            old_term = termios.tcgetattr(sys.stdin.fileno())
            tty.setcbreak(sys.stdin.fileno())

    try:
        last_cmd = command_from_key(" ", args.speed)
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, -1)

            if args.teleop == "keyboard":
                key = read_key_nonblocking()
                if key is not None:
                    last_cmd = command_from_key(key, args.speed)
                    hardware.run_motors(last_cmd.motors)

            fname = f"frame_{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(save_dir, fname), frame)

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
        cap.release()
        # Write episode summary
        summary = {
            "episode": args.episode_name,
            "instruction": args.instruction,
            "n_frames": frame_idx,
            "width": args.width,
            "height": args.height,
            "teleop": args.teleop,
        }
        with open(os.path.join(save_dir, "episode.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[collect] done. {frame_idx} frames saved to {save_dir}")


if __name__ == "__main__":
    main()
