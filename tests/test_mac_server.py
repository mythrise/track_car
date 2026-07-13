from types import SimpleNamespace

import pytest

from inference_pipeline import mac_server


def safety_args(**overrides):
    values = {
        "stop_confidence": 0.3,
        "invalid_stop_frames": 5,
        "max_waypoint_abs": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_coarse_history_excludes_current_frame_at_inference():
    history = mac_server.CoarseHistoryBuffer(history=3, warmup_frames=3)
    for frame_id in range(3):
        assert not history.ready_for_inference()
        history.append_after_inference(frame_id)

    assert history.ready_for_inference()
    assert history.frames_for_inference() == [0, 1, 2]
    current_frame = 3
    assert current_frame not in history.frames_for_inference()
    history.append_after_inference(current_frame)
    assert history.frames_for_inference() == [1, 2, 3]


def test_warmup_result_is_neutral_stop():
    result = mac_server.make_stop_result("warmup", warmup_seen=2, warmup_frames=31)
    assert result["stop"] is True
    assert result["motors"] == [1500, 1500, 1500, 1500]
    assert result["debug"]["stop_reason"] == "warmup"


def test_ws1_cli_defaults_match_plan():
    parser = mac_server.build_parser()
    args = parser.parse_args([])
    mac_server.apply_checkpoint_metadata(args, None, [])
    mac_server.validate_args(parser, args)
    assert args.history == 31
    assert args.warmup_frames == 31
    assert args.state_mode == "stateless"
    assert args.stop_confidence == 0.3
    assert args.invalid_stop_frames == 5
    assert args.max_waypoint_abs == 2.0
    assert args.pan_recenter_per_s == 30.0


def test_low_confidence_triggers_stop_reason():
    reasons = mac_server.prediction_safety_reasons(
        0.29, 0, [[0.1, 0.0, 0.0]], [0.5, 0.0, 0.0], safety_args()
    )
    assert "low_confidence" in reasons


def test_invalid_streak_triggers_on_configured_frame():
    assert "invalid_streak" not in mac_server.prediction_safety_reasons(
        1.0, 4, [[0, 0, 0]], [0, 0, 0], safety_args()
    )
    assert "invalid_streak" in mac_server.prediction_safety_reasons(
        1.0, 5, [[0, 0, 0]], [0, 0, 0], safety_args()
    )


@pytest.mark.parametrize(
    ("waypoints", "action", "expected"),
    [
        ([[float("nan"), 0, 0]], [0, 0, 0], "waypoint_nonfinite"),
        ([[2.01, 0, 0]], [0, 0, 0], "waypoint_out_of_bounds"),
        ([[0, 0, 0]], [float("inf"), 0, 0], "action_nonfinite"),
        ([[0, 0, 0]], [2.01, 0, 0], "action_out_of_bounds"),
    ],
)
def test_invalid_geometry_triggers_stop(waypoints, action, expected):
    reasons = mac_server.prediction_safety_reasons(1.0, 0, waypoints, action, safety_args())
    assert expected in reasons


def test_mock_protocol_regression_forward():
    args = SimpleNamespace(mock_action="forward", mock_speed=200)
    result = mac_server.make_mock_result(args)
    assert result == {
        "motors": [1700, 1300, 1700, 1300],
        "confidence": 1.0,
        "mode": 0,
        "stop": False,
        "debug": {
            "mock_control": True,
            "mock_action": "forward",
            "mock_speed": 200,
            "action": [1.0, 0.0, 0.0],
            "motor_delta": 200,
        },
    }


def test_mock_wire_payload_regression(monkeypatch):
    class FakeConnection:
        def settimeout(self, value):
            self.timeout = value

    frames = iter([object(), None])
    sent = []
    monkeypatch.setattr(mac_server, "recv_json", lambda _conn: {"instruction": "follow"})
    monkeypatch.setattr(mac_server, "recv_jpeg_frame", lambda _conn: next(frames))
    monkeypatch.setattr(mac_server, "send_json", lambda _conn, payload: sent.append(payload))
    args = SimpleNamespace(
        timeout=2.0,
        mock_control=True,
        mock_action="forward",
        mock_speed=200,
        history=31,
        warmup_frames=31,
    )
    processed = mac_server.handle_connection(FakeConnection(), ("local", 1), args, None, None, None)
    assert processed == 1
    assert sent == [
        {
            "type": "command",
            "seq": 1,
            "motors": [1700, 1300, 1700, 1300],
            "pan": 1500,
            "tilt": 1500,
            "fps": sent[0]["fps"],
            "confidence": 1.0,
            "stop": False,
            "debug": {
                "mock_control": True,
                "mock_action": "forward",
                "mock_speed": 200,
                "action": [1.0, 0.0, 0.0],
                "motor_delta": 200,
            },
            "mode": 0,
        }
    ]


def test_stateless_reinitializes_while_rolling_reuses_state():
    class StubModel:
        def __init__(self):
            self.calls = 0

        def init_state(self, batch_size, device):
            self.calls += 1
            return {"call": self.calls, "batch": batch_size, "device": device}

    model = StubModel()
    stateless = SimpleNamespace(state_mode="stateless")
    first = mac_server._initial_state_for_frame(model, stateless, {"old": True}, 1, "cpu")
    second = mac_server._initial_state_for_frame(model, stateless, first, 1, "cpu")
    assert first != second

    rolling = SimpleNamespace(state_mode="rolling")
    old = {"old": True}
    assert mac_server._initial_state_for_frame(model, rolling, old, 1, "cpu") is old


def test_checkpoint_metadata_defaults_and_cli_override_warning():
    args = SimpleNamespace(history=31, control_dt=0.1, label_mode=None, force=False)
    mac_server.apply_checkpoint_metadata(
        args,
        {"history": 20, "dt": 0.2, "label_mode": "absolute", "n_waypoints": 8, "fps": 5},
        [],
    )
    assert (args.history, args.control_dt, args.label_mode) == (20, 0.2, "absolute")

    warnings = []
    args = SimpleNamespace(history=31, control_dt=0.1, label_mode=None, force=False)
    mac_server.apply_checkpoint_metadata(
        args,
        {"history": 20, "dt": 0.2, "label_mode": "absolute"},
        ["--history", "31"],
        warning_sink=warnings.append,
    )
    assert args.history == 31
    assert args.control_dt == 0.2
    assert "explicit CLI wins" in warnings[0]


def test_checkpoint_fail_closed_and_shadow_random_override():
    with pytest.raises(FileNotFoundError):
        mac_server.enforce_checkpoint_policy(None, "absolute", False, False)
    with pytest.raises(RuntimeError, match="context_proj"):
        mac_server.enforce_checkpoint_policy({"model_state": {}}, "absolute", False, False)
    mac_server.enforce_checkpoint_policy(None, "absolute", True, True)


def test_checkpoint_requires_all_control_critical_groups():
    state = {
        "context_proj.weight": object(),
        "cot.theta_head.weight": object(),
        "base.proj.net.0.weight": object(),
        "base.planner.net.0.weight": object(),
        "verifier.delta_head.weight": object(),
    }
    assert mac_server.missing_critical_checkpoint_keys(state, "absolute") == []
    del state["cot.theta_head.weight"]
    assert mac_server.missing_critical_checkpoint_keys(state, "absolute") == ["cot."]


def test_pan_deadzone_and_invalid_recenter_are_rate_limited_per_second():
    pan, _ = mac_server.waypoint_to_pan_tilt(
        1.9,
        current_pan=1600,
        confidence=1.0,
        elapsed=0.5,
        pan_recenter_per_s=30.0,
    )
    assert pan == 1585
    pan, _ = mac_server.waypoint_to_pan_tilt(
        20,
        current_pan=1600,
        confidence=0.2,
        elapsed=1.0,
        pan_recenter_per_s=30.0,
    )
    assert pan == 1570


def test_shadow_mode_never_sends_intended_motors():
    motors, stop, debug = mac_server.apply_shadow_output(
        [1700, 1300, 1700, 1300],
        {"action": [1.0, 0.0, 0.0]},
    )
    assert motors == [1500, 1500, 1500, 1500]
    assert stop is True
    assert debug["shadow_intended_motors"] == [1700, 1300, 1700, 1300]
    assert debug["shadow_intended_action"] == [1.0, 0.0, 0.0]


def test_safety_state_reset_clears_previous_action():
    state = {"last_action": [1.0, 0.0, 0.5], "extra": 1}
    mac_server.reset_safety_state(state)
    assert state == {"last_action": [0.0, 0.0, 0.0]}
