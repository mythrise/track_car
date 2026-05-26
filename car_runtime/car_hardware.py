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
MIN_SPEED = -1000
MAX_SPEED = 1000
DEFAULT_BASE_SPEED = 400
DEFAULT_MAX_SPEED = 900

MOTOR_L1_CHANNEL = 6
MOTOR_R1_CHANNEL = 7
MOTOR_L2_CHANNEL = 8
MOTOR_R2_CHANNEL = 9


def clamp_pulse(value: float) -> int:
    return max(MIN_PULSE, min(MAX_PULSE, int(round(value))))


def clamp_speed(value: float) -> int:
    return max(MIN_SPEED, min(MAX_SPEED, int(round(value))))


def angle_to_pulse(angle: float) -> int:
    """Convert a 0-180 degree servo angle to a 500-2500us pulse."""
    angle = max(0.0, min(180.0, float(angle)))
    return clamp_pulse(500 + (angle / 180.0) * 2000)


def format_motor_command(l1: int, r1: int, l2: int, r2: int, time_ms: int = 0) -> str:
    """Format the vendor serial command for four motor channels."""
    time_ms = max(0, int(time_ms))
    return (
        f"#{MOTOR_L1_CHANNEL:03d}P{clamp_pulse(l1):04d}T{time_ms:04d}!"
        f"#{MOTOR_R1_CHANNEL:03d}P{clamp_pulse(r1):04d}T{time_ms:04d}!"
        f"#{MOTOR_L2_CHANNEL:03d}P{clamp_pulse(l2):04d}T{time_ms:04d}!"
        f"#{MOTOR_R2_CHANNEL:03d}P{clamp_pulse(r2):04d}T{time_ms:04d}!"
    )


def speed_to_pwm(speed_l1: int, speed_r1: int, speed_l2: int, speed_r2: int) -> List[int]:
    """Convert verified MotionControl wheel speeds to controller PWM pulses.

    The proven car wiring uses:
      left wheels:  PWM = 1500 - speed
      right wheels: PWM = 1500 + speed
    Positive speed means physical forward for that wheel.
    """
    return [
        clamp_pulse(NEUTRAL - clamp_speed(speed_l1)),
        clamp_pulse(NEUTRAL + clamp_speed(speed_r1)),
        clamp_pulse(NEUTRAL - clamp_speed(speed_l2)),
        clamp_pulse(NEUTRAL + clamp_speed(speed_r2)),
    ]


def build_motor_command(speed_l1: int, speed_r1: int, speed_l2: int, speed_r2: int, time_ms: int = 0) -> str:
    """Build the verified MotionControl serial command from wheel speeds."""
    return format_motor_command(
        *speed_to_pwm(speed_l1, speed_r1, speed_l2, speed_r2),
        time_ms=time_ms,
    )


def action_to_wheel_speeds(action: Sequence[float], scale: float = 300.0) -> List[int]:
    """Map normalized (forward, strafe_right, yaw_clockwise) action to wheel speeds."""
    forward, strafe, yaw = float(action[0]), float(action[1]), float(action[2])
    speed_l1 = (forward + strafe + yaw) * scale
    speed_r1 = (forward - strafe - yaw) * scale
    speed_l2 = (forward - strafe + yaw) * scale
    speed_r2 = (forward + strafe - yaw) * scale
    return [
        clamp_speed(speed_l1),
        clamp_speed(speed_r1),
        clamp_speed(speed_l2),
        clamp_speed(speed_r2),
    ]


def wheel_speeds_to_action(speeds: Sequence[float], scale: float = 300.0) -> List[float]:
    """Approximate inverse of `action_to_wheel_speeds`."""
    l1, r1, l2, r2 = [float(v) for v in speeds]
    forward = (l1 + r1 + l2 + r2) / 4.0
    strafe = (l1 - r1 - l2 + r2) / 4.0
    yaw = (l1 - r1 + l2 - r2) / 4.0
    return [forward / scale, strafe / scale, yaw / scale]


def motor_pulses_to_speeds(motors: Sequence[int], base: int = NEUTRAL) -> List[int]:
    """Convert logged motor PWM pulses back to MotionControl wheel speeds."""
    l1, r1, l2, r2 = [int(v) for v in motors]
    return [
        clamp_speed(base - l1),
        clamp_speed(r1 - base),
        clamp_speed(base - l2),
        clamp_speed(r2 - base),
    ]


def waypoint_to_motor(wp: Sequence[float], base: int = NEUTRAL, scale: float = 300.0) -> List[int]:
    """Map an action-like (forward, strafe_right, yaw_clockwise) vector to motor pulses."""
    if base != NEUTRAL:
        offset = int(base) - NEUTRAL
        return [
            clamp_pulse(pulse + offset)
            for pulse in speed_to_pwm(*action_to_wheel_speeds(wp, scale=scale))
        ]
    return speed_to_pwm(*action_to_wheel_speeds(wp, scale=scale))


def motor_to_action(motors: Sequence[int], base: int = NEUTRAL, scale: float = 300.0) -> List[float]:
    """Approximate inverse of `waypoint_to_motor` for logged teleop commands."""
    return wheel_speeds_to_action(motor_pulses_to_speeds(motors, base=base), scale=scale)


def stop_command() -> str:
    return build_motor_command(0, 0, 0, 0, time_ms=1000)


def calculate_follow_speeds(
    x_target: int,
    image_width: int = 640,
    base_speed: int = DEFAULT_BASE_SPEED,
    kp: float = 1.1,
    kd: float = 0.8,
    last_error: int = 0,
    max_speed: int = DEFAULT_MAX_SPEED,
) -> tuple[List[int], int]:
    """MotionControl follow controller: target x error to four wheel speeds."""
    center_x = image_width // 2
    error = int(x_target - center_x)
    derivative = error - int(last_error)
    turn_output = clamp_speed(kp * error + kd * derivative)
    turn_output = max(-abs(base_speed), min(abs(base_speed), turn_output))

    speed_l1 = base_speed - turn_output
    speed_r1 = base_speed + turn_output
    speed_l2 = base_speed - turn_output
    speed_r2 = base_speed + turn_output
    max_abs = abs(max_speed)
    speeds = [
        max(-max_abs, min(max_abs, int(speed_l1))),
        max(-max_abs, min(max_abs, int(speed_r1))),
        max(-max_abs, min(max_abs, int(speed_l2))),
        max(-max_abs, min(max_abs, int(speed_r2))),
    ]
    return speeds, error


def follow_target_to_motor(
    x_target: int,
    image_width: int = 640,
    base_speed: int = DEFAULT_BASE_SPEED,
    kp: float = 1.1,
    kd: float = 0.8,
    last_error: int = 0,
    max_speed: int = DEFAULT_MAX_SPEED,
) -> tuple[List[int], int]:
    """Convert a visual target x coordinate to motor pulses."""
    speeds, error = calculate_follow_speeds(
        x_target=x_target,
        image_width=image_width,
        base_speed=base_speed,
        kp=kp,
        kd=kd,
        last_error=last_error,
        max_speed=max_speed,
    )
    return speed_to_pwm(*speeds), error


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
                self.reset_pan_tilt()

    def send_uart(self, command: str) -> None:
        self.uart.send_str(command)

    def run_raw(self, l1: int, r1: int, l2: int, r2: int) -> None:
        """Run raw PWM pulses on the four motor channels."""
        self.send_uart(format_motor_command(l1, r1, l2, r2))

    def run_speeds(
        self,
        speed_l1: int,
        speed_r1: int,
        speed_l2: int,
        speed_r2: int,
        time_ms: int = 0,
    ) -> List[int]:
        """Run verified MotionControl wheel speeds and return the PWM pulses sent."""
        motors = speed_to_pwm(speed_l1, speed_r1, speed_l2, speed_r2)
        self.send_uart(build_motor_command(speed_l1, speed_r1, speed_l2, speed_r2, time_ms=time_ms))
        return motors

    def run_motors(self, motors: Sequence[int]) -> None:
        self.run_raw(*[int(v) for v in motors])

    def run_motors_with_kick(
        self,
        steady_motors: Sequence[int],
        kick_motors: Sequence[int],
        kick_duration: float,
    ) -> None:
        kick_duration = max(0.0, min(float(kick_duration), 0.25))
        steady = [clamp_pulse(v) for v in steady_motors]
        kick = [clamp_pulse(v) for v in kick_motors]
        if kick_duration > 0 and motor_delta(steady) > 0 and kick != steady:
            self.run_motors(kick)
            time.sleep(kick_duration)
        self.run_motors(steady)

    def run_action(self, action: Sequence[float], scale: float = 300.0) -> List[int]:
        motors = waypoint_to_motor(action, scale=scale)
        self.run_motors(motors)
        return motors

    def move_forward(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(speed, speed, speed, speed, time_ms=time_ms)

    def move_backward(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(-speed, -speed, -speed, -speed, time_ms=time_ms)

    def rotate_clockwise(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(speed, -speed, speed, -speed, time_ms=time_ms)

    def rotate_counter_clockwise(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(-speed, speed, -speed, speed, time_ms=time_ms)

    def strafe_left(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(-speed, speed, speed, -speed, time_ms=time_ms)

    def strafe_right(self, speed: int = DEFAULT_BASE_SPEED, time_ms: int = 0) -> List[int]:
        speed = abs(clamp_speed(speed))
        return self.run_speeds(speed, -speed, -speed, speed, time_ms=time_ms)

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

    def set_yuntai_angle(self, angle: float = 90) -> None:
        self.set_pan_angle(angle)

    def set_camera_angle(self, angle: float = 90) -> None:
        self.set_tilt_angle(angle)

    def reset_pan_tilt(self) -> None:
        self.set_yuntai_angle(90)
        self.set_camera_angle(90)
        time.sleep(2)

    def enter_tracking_pose(self) -> None:
        """Vendor tracing-line pose: slightly left and downward."""
        self.set_yuntai_angle(95)
        self.set_camera_angle(115)

    def enter_tracing_line_mode(self) -> None:
        self.enter_tracking_pose()

    def close(self) -> None:
        try:
            self.stop()
        finally:
            self.uart.close()
            if self.pi is not None:
                self.pi.stop()
