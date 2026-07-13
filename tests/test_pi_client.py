from car_runtime.pi_client import execute_motor_command, motor_direction


class FakeHardware:
    def __init__(self):
        self.calls = []

    def stop(self):
        self.calls.append(("stop",))

    def run_motors(self, motors):
        self.calls.append(("run", motors))

    def run_motors_with_kick(self, motors, boosted, duration):
        self.calls.append(("kick", motors, boosted, duration))


def test_stop_is_checked_before_motors_and_skips_motor_execution():
    hardware = FakeHardware()
    direction, last_kick = execute_motor_command(
        hardware,
        {"stop": True, "motors": "must-not-be-read"},
        kick_speed=900,
        kick_duration=0.2,
        kick_repeat=0.1,
        last_direction=(1, 1, 1, 1),
        last_kick_time=12.0,
        now=20.0,
    )
    assert hardware.calls == [("stop",)]
    assert direction == motor_direction([1500, 1500, 1500, 1500])
    assert last_kick == 12.0
