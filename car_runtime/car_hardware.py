#!/usr/bin/env python3
"""Hardware adapter for the Raspberry Pi smart car.

This module wraps the vendor examples in `示例代码.zip` into a small, reusable
API used by data collection and deployment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:
    import pigpio  # type: ignore
except ImportError:  # pragma: no cover - only available on Raspberry Pi
    pigpio = None

try:
    import z_uart as myUart  # type: ignore
except ImportError:  # pragma: no cover - only available on Raspberry Pi
    myUart = None


PIN_YUNTAI = 26
PIN_CAMERA = 12
NEUTRAL = 1500
MIN_PULSE = 500
MAX_PULSE = 2500


def clamp_pulse(value: float) -> int:
    return max(MIN_PULSE, min(MAX_PULSE, int(round(value))))


def angle_to_pulse(angle: float) -> int:
    """Convert a 0-180 degree servo angle to a 500-2500us pulse."""
    angle = max(0.0, min(180.0, float(angle)))
    return clamp_pulse(500 + (angle / 180.0) * 2000)


def format_motor_command(l1: int, r1: int, l2: int, r2: int) -> str:
    """Format the vendor serial command for four motor channels."""
    return "#006P{:0>4d}T0000!#007P{:0>4d}T0000!#008P{:0>4d}T0000!#009P{:0>4d}T0000!".format(
        clamp_pulse(l1),
        clamp_pulse(r1),
        clamp_pulse(l2),
        clamp_pulse(r2),
    )


def stop_command() -> str:
    return "#255P1500T1000!"


def waypoint_to_motor(wp: Sequence[float], base: int = NEUTRAL, scale: float = 300.0) -> List[int]:
    """Map an action-like (forward, strafe, yaw) vector to vendor motor pulses.

    The vendor examples use these primitive patterns:
      forward/back: [+, -, +, -]
      strafe:       [+, +, -, -]
      right turn:   [+, +*2/3, +, +*2/3]
      left turn:    [-*2/3, -, -*2/3, -]
    """
    x, y, th = float(wp[0]), float(wp[1]), float(wp[2])
    forward = x * scale
    strafe = y * scale
    yaw = th * scale
    if yaw >= 0:
        yaw_l = yaw
        yaw_r = yaw * 2.0 / 3.0
    else:
        yaw_l = yaw * 2.0 / 3.0
        yaw_r = yaw
    l1 = base + forward + strafe + yaw_l
    r1 = base - forward + strafe + yaw_r
    l2 = base + forward - strafe + yaw_l
    r2 = base - forward - strafe + yaw_r
    return [clamp_pulse(l1), clamp_pulse(r1), clamp_pulse(l2), clamp_pulse(r2)]


def motor_to_action(motors: Sequence[int], base: int = NEUTRAL, scale: float = 300.0) -> List[float]:
    """Approximate inverse of `waypoint_to_motor` for logged teleop commands."""
    l1, r1, l2, r2 = [float(v) for v in motors]
    a = l1 - base
    b = r1 - base
    c = l2 - base
    d = r2 - base
    vx = (a - b + c - d) / 4.0
    vy = (a + b - c - d) / 4.0
    wz = (a + b + c + d) / 4.0
    return [vx / scale, vy / scale, wz / scale]


@dataclass
class TeleopCommand:
    name: str
    motors: List[int]
    action: List[float]


def command_from_key(key: str, speed: int = 300) -> TeleopCommand:
    """Map simple keyboard commands to motor pulses and normalized actions.

    Keys:
      w/s: forward/back
      a/d: turn left/right
      q/e: strafe left/right
      space/x: stop
    """
    if key == "w":
        action = [1.0, 0.0, 0.0]
        name = "forward"
    elif key == "s":
        action = [-1.0, 0.0, 0.0]
        name = "backward"
    elif key == "a":
        action = [0.0, 0.0, -1.0]
        name = "turn_left"
    elif key == "d":
        action = [0.0, 0.0, 1.0]
        name = "turn_right"
    elif key == "q":
        action = [0.0, -1.0, 0.0]
        name = "strafe_left"
    elif key == "e":
        action = [0.0, 1.0, 0.0]
        name = "strafe_right"
    else:
        action = [0.0, 0.0, 0.0]
        name = "stop"
    motors = waypoint_to_motor(action, scale=speed)
    motors = [clamp_pulse(v) for v in motors]
    return TeleopCommand(name=name, motors=motors, action=action)


class CarHardware:
    """Safe hardware wrapper with automatic dry-run fallback."""

    def __init__(self, baud: int = 115200, reset_servos: bool = False, dry_run: bool | None = None):
        self.dry_run = (pigpio is None or myUart is None) if dry_run is None else dry_run
        self.pi = None
        if self.dry_run:
            print("[car_hardware] dry-run mode: pigpio/z_uart unavailable or disabled")
        else:
            missing = []
            if pigpio is None:
                missing.append("pigpio")
            if myUart is None:
                missing.append("z_uart.py")
            elif not hasattr(myUart, "setup_uart") or not hasattr(myUart, "uart_send_str"):
                missing.append("z_uart.py setup_uart/uart_send_str")
            if missing:
                raise RuntimeError(
                    "Real motor control requested, but hardware dependencies are missing: "
                    + ", ".join(missing)
                    + ". For data-pipeline testing, add --dry_run. For real movement, install "
                    "pigpio and put the vendor z_uart.py next to car_runtime/ or in PYTHONPATH."
                )
            self.pi = pigpio.pi()
            if not getattr(self.pi, "connected", True):
                raise RuntimeError(
                    "pigpio daemon is not reachable. Start it with: "
                    "sudo systemctl enable pigpiod && sudo systemctl start pigpiod"
                )
            myUart.setup_uart(baud)
            if reset_servos:
                self.set_pan_angle(90)
                self.set_tilt_angle(90)

    def send_uart(self, command: str) -> None:
        if self.dry_run:
            print(f"  [dry-run] uart: {command}")
            return
        myUart.uart_send_str(command)

    def run_raw(self, l1: int, r1: int, l2: int, r2: int) -> None:
        self.send_uart(format_motor_command(l1, r1, l2, r2))

    def run_motors(self, motors: Sequence[int]) -> None:
        self.run_raw(*[int(v) for v in motors])

    def run_action(self, action: Sequence[float], scale: float = 300.0) -> List[int]:
        motors = waypoint_to_motor(action, scale=scale)
        self.run_motors(motors)
        return motors

    def stop(self) -> None:
        self.send_uart(stop_command())

    def set_pan_pulse(self, pulse: int) -> None:
        if self.dry_run:
            print(f"  [dry-run] pan={clamp_pulse(pulse)}")
            return
        self.pi.set_servo_pulsewidth(PIN_YUNTAI, clamp_pulse(pulse))

    def set_tilt_pulse(self, pulse: int) -> None:
        if self.dry_run:
            print(f"  [dry-run] tilt={clamp_pulse(pulse)}")
            return
        self.pi.set_servo_pulsewidth(PIN_CAMERA, clamp_pulse(pulse))

    def set_pan_angle(self, angle: float) -> None:
        self.set_pan_pulse(angle_to_pulse(angle))

    def set_tilt_angle(self, angle: float) -> None:
        self.set_tilt_pulse(angle_to_pulse(angle))

    def enter_tracking_pose(self) -> None:
        """Vendor tracing-line pose: slightly left and downward."""
        self.set_pan_angle(95)
        self.set_tilt_angle(115)

    def close(self) -> None:
        self.stop()
        if self.pi is not None:
            self.pi.stop()
