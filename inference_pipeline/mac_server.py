#!/usr/bin/env python3
"""Mac inference server — receives Pi frames and returns safe commands."""

from __future__ import annotations

import argparse
import math
import os
import socket
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_OPENTRACKVLA_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"
NEUTRAL_MOTORS = [1500, 1500, 1500, 1500]

sys.path.insert(0, str(PROJECT_ROOT / "car_runtime"))
try:
    from car_hardware import boosted_motors, command_from_key, motor_delta, waypoint_to_motor
    from car_protocol import recv_jpeg_frame, recv_json, send_json
    from process_cleanup import cleanup_port
except ImportError:
    from car_runtime.car_hardware import boosted_motors, command_from_key, motor_delta, waypoint_to_motor
    from car_runtime.car_protocol import recv_jpeg_frame, recv_json, send_json
    from car_runtime.process_cleanup import cleanup_port


MOCK_KEYS = {
    "stop": " ",
    "forward": "w",
    "backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "strafe_left": "q",
    "strafe_right": "e",
}


class CoarseHistoryBuffer:
    """Keep prior-frame coarse features without training-distribution padding."""

    def __init__(self, history: int, warmup_frames: int):
        self.history = int(history)
        self.warmup_frames = int(warmup_frames)
        if self.history <= 0:
            raise ValueError("history must be > 0")
        if self.warmup_frames < self.history:
            raise ValueError("warmup_frames must be >= history to avoid repeated-frame padding")
        self._frames = []
        self.seen_frames = 0

    def ready_for_inference(self) -> bool:
        return self.seen_frames >= self.warmup_frames and len(self._frames) == self.history

    def frames_for_inference(self):
        if not self.ready_for_inference():
            raise RuntimeError("coarse history is not warmed up")
        return list(self._frames)

    def append_after_inference(self, coarse_frame) -> None:
        self.seen_frames += 1
        self._frames.append(coarse_frame)
        if len(self._frames) > self.history:
            self._frames = self._frames[-self.history :]


def resolve_opentrackvla_root(root_arg):
    root = root_arg or os.environ.get("OPENTRACKVLA_ROOT")
    if root is None and (BUNDLED_OPENTRACKVLA_ROOT / "model.py").exists():
        root = BUNDLED_OPENTRACKVLA_ROOT
    if root is None:
        raise RuntimeError(
            "Full model mode requires bundled third_party/OpenTrackVLA, "
            "--opentrackvla_root, or OPENTRACKVLA_ROOT. "
            "Use --mock_control for protocol testing without OpenTrackVLA."
        )
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "model.py").exists():
        raise FileNotFoundError(f"OpenTrackVLA root not found or missing model.py: {root_path}")
    return root_path


def default_existing_path(*paths):
    for path in paths:
        if path is None:
            continue
        candidate = Path(path).expanduser()
        if candidate.exists():
            return str(candidate.resolve())
    return None


def configure_default_weight_paths(args, opentrackvla_root):
    from local_weights import (
        default_dinov3_candidates,
        default_qwen_candidates,
        default_siglip_candidates,
        resolve_local_model_path,
    )

    os.environ["QWEN_MODEL_PATH"] = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=args.qwen_model_path,
        env_var="QWEN_MODEL_PATH",
        candidates=default_qwen_candidates(),
    )
    os.environ["SIGLIP_MODEL_PATH"] = resolve_local_model_path(
        label="SigLIP",
        repo_id="google/siglip-so400m-patch14-384",
        explicit=args.siglip_model_path,
        env_var="SIGLIP_MODEL_PATH",
        candidates=default_siglip_candidates(),
    )
    os.environ["DINOV3_MODEL_PATH"] = resolve_local_model_path(
        label="DINOv3",
        repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        explicit=args.dinov3_model_path,
        env_var="DINOV3_MODEL_PATH",
        candidates=default_dinov3_candidates(),
    )

    if args.base_hf_model_dir is None:
        args.base_hf_model_dir = default_existing_path(
            opentrackvla_root / "ckpts_hf" / "opentrackvla-qwen06b",
        )
    elif args.base_hf_model_dir:
        args.base_hf_model_dir = str(Path(args.base_hf_model_dir).expanduser().resolve())



def configure_checkpoint_path(args, opentrackvla_root):
    if args.ckpt is None:
        args.ckpt = default_existing_path(
            opentrackvla_root / "ckpts_pfem" / "car_official_dinov3" / "pfem_epoch0.pt",
            opentrackvla_root / "ckpts_pfem" / "pfem_epoch0.pt",
        )
    elif args.ckpt:
        args.ckpt = str(Path(args.ckpt).expanduser().resolve())


def load_official_base(base_hf_model_dir):
    from safetensors.torch import load_file as load_safetensors
    from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

    base_hf_dir = Path(base_hf_model_dir).expanduser().resolve()
    hf_config = OpenTrackVLAConfig.from_pretrained(str(base_hf_dir), local_files_only=True)
    qwen_model_path = os.environ.get("QWEN_MODEL_PATH", "").strip()
    if qwen_model_path:
        hf_config.llm_name = qwen_model_path
    hf_model = OpenTrackVLAForWaypoint(hf_config)
    state_path = base_hf_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing OpenTrackVLA HF weights: {state_path}")
    missing, unexpected = hf_model.load_state_dict(load_safetensors(str(state_path)), strict=False)
    print(f"[server] loaded official base: {len(missing)} missing, {len(unexpected)} unexpected")
    return hf_model.model


def critical_checkpoint_prefixes(label_mode: str, aux_delta_vel: bool = False):
    shared = ["context_proj.", "cot.", "base.proj."]
    if label_mode == "step_action":
        prefixes = shared + ["step_action_head."]
        if aux_delta_vel:
            prefixes.append("prev_action_embed.")
        return prefixes
    return shared + ["base.planner.", "verifier.delta_head."]


def missing_critical_checkpoint_keys(model_state: dict, label_mode: str, aux_delta_vel: bool = False):
    keys = tuple(model_state)
    return [
        prefix
        for prefix in critical_checkpoint_prefixes(label_mode, aux_delta_vel)
        if not any(key.startswith(prefix) for key in keys)
    ]


def enforce_checkpoint_policy(
    checkpoint,
    label_mode: str,
    allow_random_init: bool,
    shadow_mode: bool,
    aux_delta_vel: bool = False,
):
    if allow_random_init and not shadow_mode:
        raise RuntimeError("--allow_random_init is only permitted together with --shadow_mode")
    if checkpoint is None:
        if allow_random_init and shadow_mode:
            print("!!! [server] WARNING: no PFEM checkpoint; random init allowed only because shadow mode is active")
            return
        raise FileNotFoundError("PFEM checkpoint is required in non-mock mode")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, dict):
        missing = ["model_state"] + critical_checkpoint_prefixes(label_mode, aux_delta_vel)
    else:
        missing = missing_critical_checkpoint_keys(model_state, label_mode, aux_delta_vel)
    if missing:
        message = "checkpoint is missing control-critical keys: " + ", ".join(missing)
        if allow_random_init and shadow_mode:
            print(f"!!! [server] WARNING: {message}; random values remain shadow-only")
            return
        raise RuntimeError(message)


def _option_is_explicit(argv, option: str) -> bool:
    return any(arg == option or arg.startswith(option + "=") for arg in argv)


def apply_checkpoint_metadata(args, meta: dict | None, argv, warning_sink=print):
    meta = meta if isinstance(meta, dict) else {}
    mappings = (
        ("--history", "history", "history", int),
        ("--control_dt", "control_dt", "dt", float),
        ("--label_mode", "label_mode", "label_mode", str),
    )
    for option, attr, meta_key, caster in mappings:
        if meta_key not in meta or meta[meta_key] is None:
            continue
        meta_value = caster(meta[meta_key])
        current = getattr(args, attr)
        if _option_is_explicit(argv, option):
            equal = math.isclose(float(current), float(meta_value)) if isinstance(meta_value, float) else current == meta_value
            if not equal:
                acknowledgement = " (--force acknowledged)" if getattr(args, "force", False) else ""
                warning_sink(
                    f"!!! [server] CHECKPOINT META CONFLICT: {option}={current!r}, "
                    f"checkpoint {meta_key}={meta_value!r}; explicit CLI wins{acknowledgement}"
                )
        else:
            setattr(args, attr, meta_value)
    if args.label_mode is None:
        args.label_mode = "absolute"
    train_args = meta.get("train_args") if isinstance(meta.get("train_args"), dict) else {}
    meta_aux = bool(meta.get("aux_delta_vel", train_args.get("aux_delta_vel", False)))
    if _option_is_explicit(argv, "--aux_delta_vel") or _option_is_explicit(argv, "--no-aux_delta_vel"):
        if getattr(args, "aux_delta_vel", None) != meta_aux:
            warning_sink(
                "!!! [server] CHECKPOINT META CONFLICT: auxiliary delta-velocity setting differs; "
                "explicit CLI wins"
            )
    else:
        args.aux_delta_vel = meta_aux
    if getattr(args, "aux_delta_vel", None) is None:
        args.aux_delta_vel = False
    args.checkpoint_meta = meta
    args.n_waypoints = int(meta.get("n_waypoints", 8))
    args.checkpoint_fps = meta.get("fps")
    return args


def load_model(
    checkpoint,
    device,
    opentrackvla_root,
    base_hf_model_dir=None,
    n_waypoints=8,
    label_mode="absolute",
    control_dt=0.1,
    aux_delta_vel=False,
    allow_random_init=False,
):
    sys.path.insert(0, str(opentrackvla_root))
    from model import OpenTrackVLA, ModelConfig
    from harness.harness_wrapper import PFEMHarness

    if base_hf_model_dir:
        base = load_official_base(base_hf_model_dir)
    else:
        mcfg = ModelConfig(
            llm_name=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-0.6B"),
            n_waypoints=int(n_waypoints),
            freeze_llm=True,
        )
        base = OpenTrackVLA(mcfg, vision_feat_dim=1536)
    base = base.to(device)
    model = PFEMHarness(
        base,
        label_mode=label_mode,
        dt=control_dt,
        aux_delta_vel=aux_delta_vel,
    ).to(device).eval()
    if checkpoint is not None:
        model_state = checkpoint.get("model_state", {})
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        missing = [key for key in missing if not key.startswith("base.llm.")]
        critical_missing = [
            key
            for key in missing
            if any(
                key.startswith(prefix)
                for prefix in critical_checkpoint_prefixes(label_mode, aux_delta_vel)
            )
        ]
        if critical_missing and not allow_random_init:
            raise RuntimeError(
                "checkpoint load left control-critical parameters uninitialized: "
                + ", ".join(critical_missing)
            )
        print(f"[server] loaded checkpoint: {len(missing)} missing, {len(unexpected)} unexpected")
    return model


def encode_frame(frame, encoder):
    import cv2
    import torch
    from PIL import Image
    from cache_gridpool import grid_pool_tokens

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")
    tokens_dino, height_patches, width_patches = encoder._encode_dino([image])
    tokens_siglip = encoder._encode_siglip([image], out_hw=(height_patches, width_patches))
    tokens = torch.cat([tokens_dino, tokens_siglip], dim=-1)
    fine = grid_pool_tokens(tokens, height_patches, width_patches, out_tokens=64)[0].float()
    coarse = grid_pool_tokens(tokens, height_patches, width_patches, out_tokens=4)[0].float()
    return coarse, fine


def build_tokens(coarse_history, fine_tokens, history):
    import torch

    frames = list(coarse_history)
    if len(frames) != int(history):
        raise RuntimeError(f"expected exactly {history} prior coarse frames, got {len(frames)}")
    coarse = torch.cat(frames, dim=0).unsqueeze(0)
    coarse_tidx = torch.arange(history).repeat_interleave(4).unsqueeze(0)
    fine = fine_tokens.unsqueeze(0)
    fine_tidx = torch.full((1, 64), fill_value=history, dtype=torch.long)
    return coarse, coarse_tidx, fine, fine_tidx


def _recenter_pan(current_pan: int, elapsed: float, rate_per_second: float) -> int:
    max_step = max(0.0, float(rate_per_second)) * max(0.0, float(elapsed))
    if max_step <= 0:
        return int(current_pan)
    delta = 1500.0 - float(current_pan)
    if abs(delta) <= max_step:
        return 1500
    return int(round(float(current_pan) + math.copysign(max_step, delta)))


def waypoint_to_pan_tilt(
    cot_theta_deg,
    current_pan=1500,
    current_tilt=1500,
    *,
    confidence=1.0,
    invalid_pred=False,
    stop_confidence=0.3,
    elapsed=0.0,
    pan_recenter_per_s=30.0,
):
    valid_target = (
        math.isfinite(float(cot_theta_deg))
        and math.isfinite(float(confidence))
        and not bool(invalid_pred)
        and float(confidence) >= float(stop_confidence)
    )
    if not valid_target:
        pan = _recenter_pan(current_pan, elapsed, pan_recenter_per_s)
        return max(500, min(2500, pan)), int(current_tilt)
    if abs(float(cot_theta_deg)) < 2.0:
        return int(current_pan), int(current_tilt)
    pan_delta = -float(cot_theta_deg) * 3.0
    return max(500, min(2500, int(float(current_pan) + pan_delta))), int(current_tilt)


def clamp_float(value, min_value, max_value):
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(min_value, min(max_value, value))


def waypoint_to_action_command(waypoints, args):
    n_waypoints = int(waypoints.shape[0])
    if n_waypoints <= 0:
        raise RuntimeError("model returned no waypoints")
    index = max(0, min(int(args.control_waypoint_index), n_waypoints - 1))
    horizon = (index + 1) * float(args.control_dt)
    if horizon <= 0:
        raise ValueError("--control_dt must be > 0")

    selected_waypoint = waypoints[index].detach().float().cpu().tolist()
    raw_action = [float(value) / horizon for value in selected_waypoint[:3]]
    max_abs = float(args.max_action_abs)
    action = [clamp_float(value, -max_abs, max_abs) for value in raw_action]
    motors = waypoint_to_motor(action, scale=float(args.motor_scale))
    if args.min_motor_delta > 0:
        motors = boosted_motors(motors, args.min_motor_delta)

    debug = {
        "control_waypoint_index": index,
        "control_dt": float(args.control_dt),
        "control_horizon": horizon,
        "raw_waypoint": selected_waypoint,
        "raw_action": raw_action,
        "action": action,
        "motor_scale": float(args.motor_scale),
        "min_motor_delta": int(args.min_motor_delta),
        "motor_delta": motor_delta(motors),
    }
    return motors, debug


def filter_step_action(raw_action, previous_action, elapsed, max_action_rate, action_ema, max_abs=1.0):
    raw = [clamp_float(value, -float(max_abs), float(max_abs)) for value in raw_action[:3]]
    previous = [clamp_float(value, -float(max_abs), float(max_abs)) for value in previous_action[:3]]
    max_delta = float(max_action_rate) * max(0.0, float(elapsed))
    rate_limited = [
        max(old - max_delta, min(old + max_delta, new))
        for old, new in zip(previous, raw)
    ]
    smoothing = float(action_ema)
    filtered = [
        smoothing * old + (1.0 - smoothing) * new
        for old, new in zip(previous, rate_limited)
    ]
    return filtered, rate_limited


def step_action_to_action_command(step_actions, args, safety_state, elapsed):
    if int(step_actions.shape[0]) <= 0:
        raise RuntimeError("model returned no step actions")
    raw_action = step_actions[0].detach().float().cpu().tolist()[:3]
    previous_action = list(safety_state.get("last_action", [0.0, 0.0, 0.0]))
    action, rate_limited = filter_step_action(
        raw_action,
        previous_action,
        elapsed,
        args.max_action_rate,
        args.action_ema,
        args.max_action_abs,
    )
    motors = waypoint_to_motor(action, scale=float(args.motor_scale))
    if args.min_motor_delta > 0:
        motors = boosted_motors(motors, args.min_motor_delta)
    debug = {
        "control_source": "step_action[0]",
        "raw_action": raw_action,
        "previous_action": previous_action,
        "rate_limited_action": rate_limited,
        "action": action,
        "elapsed": float(elapsed),
        "max_action_rate": float(args.max_action_rate),
        "action_ema": float(args.action_ema),
        "motor_scale": float(args.motor_scale),
        "min_motor_delta": int(args.min_motor_delta),
        "motor_delta": motor_delta(motors),
    }
    return motors, debug


def _numeric_values(value):
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().tolist()
    if isinstance(value, (list, tuple)):
        output = []
        for item in value:
            output.extend(_numeric_values(item))
        return output
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return []


def prediction_safety_reasons(confidence, invalid_streak, waypoints, raw_action, args):
    reasons = []
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = float("nan")
    if not math.isfinite(confidence_value):
        reasons.append("confidence_nonfinite")
    elif confidence_value < float(args.stop_confidence):
        reasons.append("low_confidence")
    if int(invalid_streak) >= int(args.invalid_stop_frames):
        reasons.append("invalid_streak")

    limit = float(args.max_waypoint_abs)
    for label, values in (("waypoint", _numeric_values(waypoints)), ("action", _numeric_values(raw_action))):
        if not values:
            reasons.append(f"{label}_empty")
            continue
        if any(not math.isfinite(value) for value in values):
            reasons.append(f"{label}_nonfinite")
        elif any(abs(value) > limit for value in values):
            reasons.append(f"{label}_out_of_bounds")
    return reasons


def make_mock_result(args):
    mock = command_from_key(MOCK_KEYS[args.mock_action], args.mock_speed)
    motors = mock.motors
    return {
        "motors": motors,
        "confidence": 1.0,
        "mode": 0,
        "stop": args.mock_action == "stop",
        "debug": {
            "mock_control": True,
            "mock_action": args.mock_action,
            "mock_speed": args.mock_speed,
            "action": mock.action,
            "motor_delta": motor_delta(motors),
        },
    }


def make_stop_result(reason: str, **debug_fields):
    debug = {"stop_reason": reason, "motor_delta": 0}
    debug.update(debug_fields)
    return {
        "motors": list(NEUTRAL_MOTORS),
        "confidence": 0.0,
        "stop": True,
        "debug": debug,
    }


def apply_shadow_output(intended_motors, debug):
    shadow_debug = dict(debug)
    shadow_debug["shadow_mode"] = True
    shadow_debug["shadow_intended_motors"] = list(intended_motors)
    shadow_debug["shadow_intended_action"] = list(debug.get("action", []))
    return list(NEUTRAL_MOTORS), True, shadow_debug


def reset_safety_state(state):
    state.clear()
    state["last_action"] = [0.0, 0.0, 0.0]


def commit_safety_state(state, action, *, invalid_pred=False):
    if invalid_pred:
        reset_safety_state(state)
        return False
    state["last_action"] = list(action)
    return True


def default_device():
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _scalar_bool(value) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _initial_state_for_frame(model, args, rolling_state, batch_size, device):
    if args.state_mode == "stateless" or rolling_state is None:
        return model.init_state(batch_size, device)
    return rolling_state


def handle_connection(conn, addr, args, model, encoder, device):
    conn.settimeout(args.timeout)
    print(f"[server] connected: {addr}")
    hello = recv_json(conn)
    if hello is None:
        print("[server] client disconnected before hello; waiting for next client")
        return 0
    instruction = hello.get("instruction", "follow the person")
    print(f"[server] instruction: {instruction}")

    batch_size = 1
    rolling_state = None
    history_buffer = CoarseHistoryBuffer(args.history, args.warmup_frames)
    pan, tilt = 1500, 1500
    invalid_streak = 0
    safety_state = {"last_action": [0.0, 0.0, 0.0]}
    model_previous_action = [0.0, 0.0, 0.0]
    frame_count = 0
    last_frame_time = time.monotonic()

    try:
        while True:
            frame = recv_jpeg_frame(conn)
            if frame is None:
                break
            frame_started = time.monotonic()
            elapsed = max(0.0, frame_started - last_frame_time)
            last_frame_time = frame_started
            started = time.time()

            if args.mock_control:
                result = make_mock_result(args)
                mode = result.pop("mode")
                payload_mode = mode
            else:
                import torch

                coarse_frame, fine_frame = encode_frame(frame, encoder)
                if not history_buffer.ready_for_inference():
                    history_buffer.append_after_inference(coarse_frame)
                    result = make_stop_result(
                        "warmup",
                        warmup_seen=history_buffer.seen_frames,
                        warmup_frames=args.warmup_frames,
                    )
                    mode = None
                    payload_mode = None
                else:
                    coarse_frames = history_buffer.frames_for_inference()
                    coarse, coarse_tidx, fine, fine_tidx = build_tokens(
                        coarse_frames,
                        fine_frame,
                        args.history,
                    )
                    state = _initial_state_for_frame(model, args, rolling_state, batch_size, device)
                    with torch.inference_mode():
                        previous_action_values = (
                            model_previous_action
                            if args.aux_delta_vel
                            else safety_state["last_action"]
                        )
                        previous_action = torch.tensor(
                            [previous_action_values],
                            dtype=torch.float32,
                            device=device,
                        )
                        output = model.forward_step(
                            coarse_tokens=coarse.to(device),
                            coarse_tidx=coarse_tidx.to(device),
                            fine_tokens=fine.to(device),
                            fine_tidx=fine_tidx.to(device),
                            instructions=[instruction],
                            prev_state=state,
                            prev_action=previous_action,
                        )
                    history_buffer.append_after_inference(coarse_frame)
                    rolling_state = output["new_state"] if args.state_mode == "rolling" else None
                    waypoints = output["waypoints"][0]
                    if args.label_mode == "step_action":
                        intended_motors, debug = step_action_to_action_command(
                            output["step_actions"][0], args, safety_state, elapsed
                        )
                    else:
                        intended_motors, debug = waypoint_to_action_command(waypoints, args)
                    confidence = float(output["C"][0].item())
                    invalid_pred = _scalar_bool(output["cot_decoded"]["invalid_pred"][0])
                    invalid_streak = invalid_streak + 1 if invalid_pred else 0
                    mode = int(output["orch"]["mode"][0].item())
                    debug["orchestrator_mode"] = mode
                    reasons = prediction_safety_reasons(
                        confidence,
                        invalid_streak,
                        waypoints,
                        debug["raw_action"],
                        args,
                    )
                    safety_stop = bool(reasons)
                    cot_theta = float(output["cot_decoded"]["theta_deg"][0].item())
                    pan, tilt = waypoint_to_pan_tilt(
                        cot_theta,
                        pan,
                        tilt,
                        confidence=confidence,
                        invalid_pred=(invalid_pred or safety_stop),
                        stop_confidence=args.stop_confidence,
                        elapsed=elapsed,
                        pan_recenter_per_s=args.pan_recenter_per_s,
                    )

                    sent_motors = intended_motors
                    stop = False
                    if safety_stop:
                        sent_motors = list(NEUTRAL_MOTORS)
                        stop = True
                        reset_safety_state(safety_state)
                        model_previous_action = [0.0, 0.0, 0.0]
                        debug["safety_stop_reasons"] = reasons
                        debug["safety_state_reset"] = True
                        if args.state_mode == "rolling":
                            rolling_state = None
                    else:
                        state_committed = commit_safety_state(
                            safety_state,
                            debug["action"],
                            invalid_pred=invalid_pred,
                        )
                        if not state_committed:
                            model_previous_action = [0.0, 0.0, 0.0]
                            debug["safety_state_reset"] = True
                            debug["safety_state_reset_reason"] = "invalid_pred"
                        elif args.label_mode == "step_action":
                            model_previous_action = list(debug["raw_action"])

                    if args.shadow_mode:
                        sent_motors, stop, debug = apply_shadow_output(intended_motors, debug)
                        print(
                            f"[shadow] seq={frame_count + 1} intended_motors={intended_motors} "
                            f"action={debug['action']} safety_reasons={reasons}"
                        )

                    result = {
                        "motors": sent_motors,
                        "confidence": confidence,
                        "stop": stop,
                        "debug": debug,
                    }
                    payload_mode = None

            duration = time.time() - started
            frame_count += 1
            payload = {
                "type": "command",
                "seq": frame_count,
                "motors": result["motors"],
                "pan": pan,
                "tilt": tilt,
                "fps": 1.0 / max(duration, 0.001),
                "confidence": result["confidence"],
                "stop": result["stop"],
                "debug": result["debug"],
            }
            # Preserve the existing mock protocol exactly.  Real-model mode is
            # diagnostic-only for orchestrator mode, per the WS1 safety plan.
            if payload_mode is not None:
                payload["mode"] = payload_mode
            send_json(conn, payload)

            if frame_count % 30 == 0:
                print(
                    f"  frame={frame_count} dt={duration * 1000:.0f}ms "
                    f"fps={1 / max(duration, 0.001):.1f} C={result['confidence']:.2f} "
                    f"mode={mode} motor_delta={result['debug']['motor_delta']}"
                )

    except socket.timeout:
        print("[server] socket timeout")
    except (ConnectionResetError, ConnectionError, OSError) as exc:
        print(f"[server] connection closed: {exc}")
    finally:
        reset_safety_state(safety_state)
        print(f"[server] client done. processed {frame_count} frames.")
    return frame_count


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--device", default=None)
    parser.add_argument("--opentrackvla_root", default=None, help="Path to OpenTrackVLA source.")
    parser.add_argument("--base_hf_model_dir", default=None, help="Official OpenTrackVLA HF checkpoint directory.")
    parser.add_argument("--qwen_model_path", default=None, help="Optional local Qwen directory.")
    parser.add_argument("--dinov3_model_path", default=None, help="Optional local DINOv3 directory.")
    parser.add_argument("--siglip_model_path", default=None, help="Optional local SigLIP directory.")
    parser.add_argument("--history", type=int, default=31)
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=None,
        help="Stop-only startup frames; defaults to the resolved history value.",
    )
    parser.add_argument("--state_mode", choices=("stateless", "rolling"), default="stateless")
    parser.add_argument("--mock_control", action="store_true", help="Skip model loading for protocol tests.")
    parser.add_argument("--mock_action", choices=sorted(MOCK_KEYS), default="stop")
    parser.add_argument("--mock_speed", type=int, default=200)
    parser.add_argument("--control_dt", type=float, default=0.1)
    parser.add_argument("--control_waypoint_index", type=int, default=1)
    parser.add_argument("--motor_scale", type=float, default=400.0)
    parser.add_argument("--min_motor_delta", type=int, default=0)
    parser.add_argument("--max_action_abs", type=float, default=1.0)
    parser.add_argument(
        "--max_action_rate",
        type=float,
        default=4.0,
        help="Maximum per-axis action change per second in step_action mode.",
    )
    parser.add_argument(
        "--action_ema",
        type=float,
        default=0.5,
        help="Previous-command weight for step_action EMA smoothing.",
    )
    parser.add_argument("--stop_confidence", type=float, default=0.3)
    parser.add_argument("--invalid_stop_frames", type=int, default=5)
    parser.add_argument("--max_waypoint_abs", type=float, default=2.0)
    parser.add_argument("--pan_recenter_per_s", type=float, default=30.0)
    parser.add_argument("--label_mode", choices=("absolute", "step_action"), default=None)
    parser.add_argument(
        "--aux_delta_vel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override checkpoint auxiliary prev-action conditioning metadata.",
    )
    parser.add_argument("--force", action="store_true", help="Acknowledge explicit CLI overrides of checkpoint metadata.")
    parser.add_argument("--allow_random_init", action="store_true")
    parser.add_argument("--shadow_mode", action="store_true")
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--no_cleanup_port", action="store_true")
    parser.add_argument("--cleanup_dry_run", action="store_true")
    return parser


def validate_args(parser, args):
    if args.control_dt <= 0:
        parser.error("--control_dt must be > 0")
    if args.control_waypoint_index < 0:
        parser.error("--control_waypoint_index must be >= 0")
    if args.motor_scale <= 0:
        parser.error("--motor_scale must be > 0")
    if args.min_motor_delta < 0:
        parser.error("--min_motor_delta must be >= 0")
    if args.max_action_abs <= 0:
        parser.error("--max_action_abs must be > 0")
    if args.max_action_rate <= 0:
        parser.error("--max_action_rate must be > 0")
    if not 0.0 <= args.action_ema <= 1.0:
        parser.error("--action_ema must be in [0, 1]")
    if not 0.0 <= args.stop_confidence <= 1.0:
        parser.error("--stop_confidence must be in [0, 1]")
    if args.invalid_stop_frames <= 0:
        parser.error("--invalid_stop_frames must be > 0")
    if args.max_waypoint_abs <= 0:
        parser.error("--max_waypoint_abs must be > 0")
    if args.pan_recenter_per_s < 0:
        parser.error("--pan_recenter_per_s must be >= 0")
    if args.allow_random_init and not args.shadow_mode:
        parser.error("--allow_random_init requires --shadow_mode")
    if args.warmup_frames is None:
        args.warmup_frames = args.history
    if args.warmup_frames < args.history:
        parser.error("--warmup_frames must be >= --history")


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)

    if not args.no_cleanup_port and args.cleanup_dry_run:
        cleanup_port(args.port, dry_run=args.cleanup_dry_run)
        return

    if args.mock_control:
        apply_checkpoint_metadata(args, None, raw_argv)
        validate_args(parser, args)
        if not args.no_cleanup_port:
            cleanup_port(args.port, dry_run=False)
        device = None
        model = None
        encoder = None
    else:
        import torch

        opentrackvla_root = resolve_opentrackvla_root(args.opentrackvla_root)
        sys.path.insert(0, str(opentrackvla_root))
        configure_checkpoint_path(args, opentrackvla_root)
        checkpoint = None
        if args.ckpt and Path(args.ckpt).is_file():
            checkpoint = torch.load(args.ckpt, map_location="cpu")
        apply_checkpoint_metadata(args, checkpoint.get("meta") if checkpoint else None, raw_argv)
        validate_args(parser, args)
        enforce_checkpoint_policy(
            checkpoint,
            args.label_mode,
            args.allow_random_init,
            args.shadow_mode,
            args.aux_delta_vel,
        )
        if not args.no_cleanup_port:
            cleanup_port(args.port, dry_run=False)
        configure_default_weight_paths(args, opentrackvla_root)
        if args.state_mode == "rolling":
            print("!!! [server] WARNING: rolling PFEM state is experimental and differs from the training distribution")

        from cache_gridpool import VisionCacheConfig, VisionFeatureCacher

        device = torch.device(args.device or default_device())
        print(f"[server] OpenTrackVLA root: {opentrackvla_root}")
        print(f"[server] base_hf_model_dir: {args.base_hf_model_dir or '(not set)'}")
        print(f"[server] pfem_ckpt: {args.ckpt or '(not set)'}")
        print(
            f"[server] checkpoint meta: history={args.history} dt={args.control_dt} "
            f"label_mode={args.label_mode} n_waypoints={args.n_waypoints}"
        )
        print(f"[server] DINOV3_MODEL_PATH: {os.environ.get('DINOV3_MODEL_PATH', '(not set)')}")
        print(f"[server] QWEN_MODEL_PATH: {os.environ.get('QWEN_MODEL_PATH', '(not set)')}")
        print(f"[server] SIGLIP_MODEL_PATH: {os.environ.get('SIGLIP_MODEL_PATH', '(not set)')}")
        model = load_model(
            checkpoint,
            device,
            opentrackvla_root,
            base_hf_model_dir=args.base_hf_model_dir,
            n_waypoints=args.n_waypoints,
            label_mode=args.label_mode,
            control_dt=args.control_dt,
            aux_delta_vel=args.aux_delta_vel,
            allow_random_init=args.allow_random_init,
        )
        encoder_device = "cuda" if device.type == "cuda" else "cpu"
        encoder = VisionFeatureCacher(VisionCacheConfig(image_size=384, batch_size=1, device=encoder_device))
        encoder.eval()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)
    print(f"[server] listening on port {args.port}...")

    total_frames = 0
    try:
        while True:
            conn, addr = server.accept()
            try:
                total_frames += handle_connection(conn, addr, args, model, encoder, device)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("[server] interrupted")
    finally:
        server.close()
        print(f"[server] shutdown. processed {total_frames} total frames.")


if __name__ == "__main__":
    main()
