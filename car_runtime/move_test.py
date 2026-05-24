#!/usr/bin/env python3
"""Small, bounded movement test for the Raspberry Pi car.

Dry-run is the default. Add `--execute` only after the car is lifted or in a
clear low-speed test area.
"""

from __future__ import annotations

import argparse
import time

try:
    from car_hardware import CarHardware, command_from_key
except ImportError:
    from car_runtime.car_hardware import CarHardware, command_from_key


KEY_BY_MOVE = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "strafe_left": "q",
    "strafe_right": "e",
    "stop": " ",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--move",
        choices=sorted(KEY_BY_MOVE.keys()),
        default="forward",
        help="Movement primitive to test.",
    )
    ap.add_argument("--speed", type=int, default=200, help="Pulse delta from neutral 1500.")
    ap.add_argument("--duration", type=float, default=0.3, help="Movement duration in seconds.")
    ap.add_argument("--execute", action="store_true", help="Actually send UART commands.")
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--reset_servos", action="store_true", help="Reset pan/tilt servos before motor test.")
    args = ap.parse_args()

    duration = max(0.05, min(args.duration, 3.0))
    speed = max(0, min(args.speed, 600))
    cmd = command_from_key(KEY_BY_MOVE[args.move], speed)
    hardware = CarHardware(
        uart_port=args.uart_port,
        reset_servos=args.reset_servos,
        dry_run=not args.execute,
    )

    print(f"[move_test] move={cmd.name} speed={speed} duration={duration:.2f}s motors={cmd.motors}")
    try:
        hardware.run_motors(cmd.motors)
        time.sleep(duration)
    finally:
        hardware.close()
        print("[move_test] stopped")


if __name__ == "__main__":
    main()
