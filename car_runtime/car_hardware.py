#!/usr/bin/env python3
"""Hardware adapter for the Raspberry Pi smart car.

This module wraps the vendor examples in `示例代码.zip` into a small, reusable
API used by data collection and deployment scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Sequence

try:
    from uart_transport import UartTransport
except ImportError:
    from car_runtime.uart_transport import UartTransport

try:
    import pigpio  # type: ignore
except ImportError:  # pragma: no cover - only available on Raspberry Pi
    pigpio = None


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


def motor_delta(motors: Sequence[int], base: int = NEUTRAL) -> int:
    return max(abs(int(v) - base) for v in motors)


def boosted_motors(motors: Sequence[int], kick_speed: int, base: int = NEUTRAL) -> List[int]:
    """Scale a motor command outward from neutral for a short startup kick."""
    current_delta = motor_delta(motors, base=base)
    if current_delta <= 0 or kick_speed <= current_delta:
        return [clamp_pulse(v) for v in motors]

    factor = float(kick_speed) / float(current_delta)
    return [clamp_pulse(base + (int(v) - base) * factor) for v in motors]


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

    def __init__(
        self,
        baud: int = 115200,
        uart_port: str | None = None,
        reset_servos: bool = False,
        dry_run: bool | None = None,
    ):
        self.dry_run = False if dry_run is None else dry_run
        self.uart = UartTransport(baud=baud, port=uart_port, dry_run=self.dry_run)
        self.pi = None
        self._servo_warning_printed = False
        if self.dry_run:
            print("[car_hardware] dry-run mode: hardware output disabled")
        else:
            if pigpio is not None:
                self.pi = pigpio.pi()
                if not getattr(self.pi, "connected", True):
                    self.pi = None
            if reset_servos and self.pi is None:
                raise RuntimeError(
                    "Servo reset requested, but pigpio is unavailable or pigpiod is not reachable. "
                    "Install/start it with: sudo apt install -y pigpio python3-pigpio && "
                    "sudo systemctl enable pigpiod && sudo systemctl start pigpiod"
                )
            if reset_servos:
                self.set_pan_angle(90)
                self.set_tilt_angle(90)

    def send_uart(self, command: str) -> None:
        self.uart.send_str(command)

    def run_raw(self, l1: int, r1: int, l2: int, r2: int) -> None:
        self.send_uart(format_motor_command(l1, r1, l2, r2))

    def run_motors(self, motors: Sequence[int]) -> None:
        self.run_raw(*[int(v) for v in motors])

    def run_motors_with_kick(
        self,
        steady_motors: Sequence[int],
        kick_motors: Sequence[int],
        kick_duration: float,
    ) -> None:
        kick_duration = max(0.0, min(float(kick_duration), 0.25))
        if kick_duration > 0 and motor_delta(steady_motors) > 0:
            self.run_motors(kick_motors)
            time.sleep(kick_duration)
        self.run_motors(steady_motors)

    def run_action(self, action: Sequence[float], scale: float = 300.0) -> List[int]:
        motors = waypoint_to_motor(action, scale=scale)
        self.run_motors(motors)
        return motors

    def stop(self) -> None:
        self.send_uart(stop_command())

    def _warn_no_servo(self) -> None:
        if not self._servo_warning_printed:
            print("[car_hardware] pigpio unavailable; skipping pan/tilt servo command")
            self._servo_warning_printed = True

    def set_pan_pulse(self, pulse: int) -> None:
        if self.dry_run:
            print(f"  [dry-run] pan={clamp_pulse(pulse)}")
            return
        if self.pi is None:
            self._warn_no_servo()
            return
        self.pi.set_servo_pulsewidth(PIN_YUNTAI, clamp_pulse(pulse))

    def set_tilt_pulse(self, pulse: int) -> None:
        if self.dry_run:
            print(f"  [dry-run] tilt={clamp_pulse(pulse)}")
            return
        if self.pi is None:
            self._warn_no_servo()
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
        try:
            self.stop()
        finally:
            self.uart.close()
            if self.pi is not None:
                self.pi.stop()
