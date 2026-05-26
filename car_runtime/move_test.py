#!/usr/bin/env python3
"""Small, bounded movement test for the Raspberry Pi car.

Dry-run is the default. Add `--execute` only after the car is lifted or in a
clear low-speed test area.
"""

from __future__ import annotations

import argparse
import time

try:
    from car_hardware import CarHardware, boosted_motors, command_from_key
except ImportError:
    from car_runtime.car_hardware import CarHardware, boosted_motors, command_from_key


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
    ap.add_argument("--kick_speed", type=int, default=0,
                    help="Optional short startup kick pulse delta. Use 0 to disable.")
    ap.add_argument("--kick_duration", type=float, default=0.06,
                    help="Kick duration in seconds, clamped to 0.25.")
    args = ap.parse_args()

    duration = max(0.05, min(args.duration, 3.0))
    speed = max(0, min(args.speed, 600))
    kick_speed = max(0, min(args.kick_speed, 900))
    cmd = command_from_key(KEY_BY_MOVE[args.move], speed)
    kick_motors = boosted_motors(cmd.motors, kick_speed) if kick_speed else cmd.motors
    hardware = CarHardware(
        uart_port=args.uart_port,
        reset_servos=args.reset_servos,
        dry_run=not args.execute,
    )

    print(f"[move_test] move={cmd.name} speed={speed} duration={duration:.2f}s motors={cmd.motors}")
    if kick_speed and cmd.name != "stop":
        print(f"[move_test] kick_speed={kick_speed} kick_duration={args.kick_duration:.3f}s "
              f"kick_motors={kick_motors}")
    try:
        hardware.run_motors_with_kick(cmd.motors, kick_motors, args.kick_duration)
        time.sleep(duration)
    finally:
        hardware.close()
        print("[move_test] stopped")


if __name__ == "__main__":
    main()
