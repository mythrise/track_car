#!/usr/bin/env python3
"""Freely test motor speed thresholds on the Raspberry Pi car."""

from __future__ import annotations

import argparse
import time

try:
    from car_hardware import MAX_SPEED, CarHardware, boosted_motors, command_from_key
    from process_cleanup import cleanup_named_processes
except ImportError:
    from car_runtime.car_hardware import MAX_SPEED, CarHardware, boosted_motors, command_from_key
    from car_runtime.process_cleanup import cleanup_named_processes


KEY_BY_MOVE = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "strafe_left": "q",
    "strafe_right": "e",
}


def parse_speeds(text: str | None, start: int, stop: int, step: int) -> list[int]:
    if text:
        speeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    else:
        if step <= 0:
            raise ValueError("--step must be positive")
        speeds = list(range(start, stop + 1, step))
    return [max(0, min(speed, MAX_SPEED)) for speed in speeds]


def confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--move", choices=sorted(KEY_BY_MOVE), default="forward")
    ap.add_argument("--speeds", default=None, help="Comma-separated speed list, for example 120,160,220,300.")
    ap.add_argument("--start", type=int, default=100, help="Sweep start speed if --speeds is omitted.")
    ap.add_argument("--stop", type=int, default=500, help="Sweep stop speed if --speeds is omitted.")
    ap.add_argument("--step", type=int, default=50, help="Sweep step if --speeds is omitted.")
    ap.add_argument("--duration", type=float, default=0.25, help="Run duration per speed, seconds.")
    ap.add_argument("--pause", type=float, default=0.5, help="Pause between tests, seconds.")
    ap.add_argument("--kick_speed", type=int, default=0, help="Optional short startup kick speed.")
    ap.add_argument("--kick_duration", type=float, default=0.06, help="Kick duration, seconds.")
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--execute", action="store_true", help="Actually send UART commands.")
    ap.add_argument("--confirm_each", action="store_true", help="Ask before every speed step.")
    ap.add_argument("--no_cleanup_processes", action="store_true",
                    help="Do not kill vendor camera/main processes before speed test.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                    help="Print cleanup targets without killing them.")
    args = ap.parse_args()

    if not args.no_cleanup_processes:
        cleanup_named_processes(["mjpg", "z_main"], dry_run=args.cleanup_dry_run)
        if args.cleanup_dry_run:
            return

    speeds = parse_speeds(args.speeds, args.start, args.stop, args.step)
    duration = max(0.05, min(args.duration, 2.0))
    pause = max(0.0, min(args.pause, 5.0))
    kick_speed = max(0, min(args.kick_speed, MAX_SPEED))
    kick_duration = max(0.0, min(args.kick_duration, 0.25))

    print(f"[speed_sweep] move={args.move} speeds={speeds} duration={duration:.2f}s "
          f"kick_speed={kick_speed} execute={args.execute}")
    if args.execute and not confirm("Lift the car or clear the floor. Continue?"):
        print("[speed_sweep] aborted")
        return

    hardware = CarHardware(uart_port=args.uart_port, dry_run=not args.execute)
    try:
        for speed in speeds:
            cmd = command_from_key(KEY_BY_MOVE[args.move], speed)
            kick_motors = boosted_motors(cmd.motors, kick_speed) if kick_speed else cmd.motors
            print(f"[speed_sweep] speed={speed} motors={cmd.motors}", end="")
            if kick_speed:
                print(f" kick_motors={kick_motors}")
            else:
                print()

            if args.confirm_each and not confirm(f"Run speed {speed}?"):
                continue

            sent_motors = hardware.run_motors_with_kick(cmd.motors, kick_motors, kick_duration)
            if sent_motors != cmd.motors:
                print(f"[speed_sweep] wheel trim active: motors actually sent={sent_motors}")
            time.sleep(duration)
            hardware.stop()
            time.sleep(pause)
    finally:
        hardware.close()
        print("[speed_sweep] stopped")


if __name__ == "__main__":
    main()
