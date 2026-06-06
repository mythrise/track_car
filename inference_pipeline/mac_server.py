#!/usr/bin/env python3
"""Mac inference server — receives frames from Pi, runs PFEM-Harness, sends commands.

Run on your Mac/PC:
    python inference_pipeline/mac_server.py --port 9999
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car_runtime"))

try:
    from car_hardware import boosted_motors, command_from_key, motor_delta, waypoint_to_motor
    from car_protocol import recv_jpeg_frame, recv_json, send_json
    from process_cleanup import cleanup_port
except ImportError:
    from car_runtime.car_hardware import boosted_motors, command_from_key, motor_delta, waypoint_to_motor
    from car_runtime.car_protocol import recv_jpeg_frame, recv_json, send_json
    from car_runtime.process_cleanup import cleanup_port


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_OPENTRACKVLA_ROOT = PROJECT_ROOT / "third_party" / "OpenTrackVLA"


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
        p = Path(path).expanduser()
        if p.exists():
            return str(p.resolve())
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


def load_model(ckpt_path, device, opentrackvla_root, base_hf_model_dir=None):
    import torch

    sys.path.insert(0, str(opentrackvla_root))
    from model import OpenTrackVLA, ModelConfig
    from harness.harness_wrapper import PFEMHarness

    if base_hf_model_dir:
        base = load_official_base(base_hf_model_dir)
    else:
        mcfg = ModelConfig(
            llm_name=os.environ.get("QWEN_MODEL_PATH", "Qwen/Qwen3-0.6B"),
            n_waypoints=8,
            freeze_llm=True,
        )
        base = OpenTrackVLA(mcfg, vision_feat_dim=1536)
    base = base.to(device)
    model = PFEMHarness(base).to(device).eval()
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        msd = ckpt.get("model_state", {})
        missing, unexpected = model.load_state_dict(msd, strict=False)
        missing = [k for k in missing if not k.startswith("base.llm.")]
        print(f"[server] loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected")
    return model


def encode_frame(frame, encoder):
    import torch
    import cv2
    from PIL import Image
    from cache_gridpool import grid_pool_tokens

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb).convert("RGB")
    tok_dino, hp, wp = encoder._encode_dino([pil])
    tok_sigl = encoder._encode_siglip([pil], out_hw=(hp, wp))
    tokens = torch.cat([tok_dino, tok_sigl], dim=-1)
    fine = grid_pool_tokens(tokens, hp, wp, out_tokens=64)[0].float()
    coarse = grid_pool_tokens(tokens, hp, wp, out_tokens=4)[0].float()
    return coarse, fine


def build_tokens(coarse_history, fine_tokens, history):
    import torch

    frames = list(coarse_history[-history:])
    if not frames:
        raise RuntimeError("empty coarse history")
    while len(frames) < history:
        frames.insert(0, frames[0])
    coarse = torch.cat(frames, dim=0).unsqueeze(0)
    coarse_tidx = torch.arange(history).repeat_interleave(4).unsqueeze(0)
    fine = fine_tokens.unsqueeze(0)
    fine_tidx = torch.full((1, 64), fill_value=history, dtype=torch.long)
    return coarse, coarse_tidx, fine, fine_tidx


def waypoint_to_pan_tilt(cot_theta_deg, current_pan=1500, current_tilt=1500):
    pan_delta = -cot_theta_deg * 3
    return (max(500, min(2500, int(current_pan + pan_delta))),
            current_tilt)


MOCK_KEYS = {
    "stop": " ",
    "forward": "w",
    "backward": "s",
    "turn_left": "a",
    "turn_right": "d",
    "strafe_left": "q",
    "strafe_right": "e",
}


def clamp_float(value, min_value, max_value):
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        return 0.0
    return max(min_value, min(max_value, value))


def waypoint_to_action_command(waypoints, args):
    n_waypoints = int(waypoints.shape[0])
    if n_waypoints <= 0:
        raise RuntimeError("model returned no waypoints")
    idx = max(0, min(int(args.control_waypoint_index), n_waypoints - 1))
    horizon = (idx + 1) * float(args.control_dt)
    if horizon <= 0:
        raise ValueError("--control_dt must be > 0")

    selected_wp = waypoints[idx].detach().float().cpu().tolist()
    raw_action = [float(value) / horizon for value in selected_wp[:3]]
    max_abs = float(args.max_action_abs)
    action = [clamp_float(value, -max_abs, max_abs) for value in raw_action]

    motors = waypoint_to_motor(action, scale=float(args.motor_scale))
    if args.min_motor_delta > 0:
        motors = boosted_motors(motors, args.min_motor_delta)

    debug = {
        "control_waypoint_index": idx,
        "control_dt": float(args.control_dt),
        "control_horizon": horizon,
        "raw_waypoint": selected_wp,
        "raw_action": raw_action,
        "action": action,
        "motor_scale": float(args.motor_scale),
        "min_motor_delta": int(args.min_motor_delta),
        "motor_delta": motor_delta(motors),
    }
    return motors, debug


def default_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def handle_connection(conn, addr, args, model, encoder, device):
    conn.settimeout(args.timeout)
    print(f"[server] connected: {addr}")

    hello = recv_json(conn)
    if hello is None:
        print("[server] client disconnected before hello; waiting for next client")
        return 0
    instruction = hello.get("instruction", "follow the person")
    print(f"[server] instruction: {instruction}")

    B = 1
    state = model.init_state(B, device) if model is not None else None
    coarse_history = []
    pan, tilt = 1500, 1500
    frame_count = 0

    try:
        while True:
            frame = recv_jpeg_frame(conn)
            if frame is None:
                break

            t0 = time.time()
            if args.mock_control:
                mock = command_from_key(MOCK_KEYS[args.mock_action], args.mock_speed)
                motors = mock.motors
                confidence = 1.0
                mode = 0
                debug = {
                    "mock_control": True,
                    "mock_action": args.mock_action,
                    "mock_speed": args.mock_speed,
                    "action": mock.action,
                    "motor_delta": motor_delta(motors),
                }
            else:
                import torch

                coarse_frame, fine_frame = encode_frame(frame, encoder)
                coarse_history.append(coarse_frame)
                if len(coarse_history) > args.history:
                    coarse_history = coarse_history[-args.history:]
                coarse, c_tidx, fine, f_tidx = build_tokens(coarse_history, fine_frame, args.history)

                with torch.inference_mode():
                    out = model.forward_step(
                        coarse_tokens=coarse.to(device),
                        coarse_tidx=c_tidx.to(device),
                        fine_tokens=fine.to(device),
                        fine_tidx=f_tidx.to(device),
                        instructions=[instruction],
                        prev_state=state,
                    )
                state = out["new_state"]
                wp = out["waypoints"][0]  # (8, 3)

                motors, debug = waypoint_to_action_command(wp, args)

                # Use Polar-CoT for pan-tilt
                cot_theta = out["cot_decoded"]["theta_deg"][0].item()
                pan, tilt = waypoint_to_pan_tilt(cot_theta, pan, tilt)
                confidence = float(out["C"][0].item())
                mode = int(out["orch"]["mode"][0].item())

            dt = time.time() - t0
            frame_count += 1

            send_json(conn, {
                "type": "command",
                "seq": frame_count,
                "motors": motors,
                "pan": pan,
                "tilt": tilt,
                "fps": 1.0 / max(dt, 0.001),
                "confidence": confidence,
                "mode": mode,
                "stop": args.mock_control and args.mock_action == "stop",
                "debug": debug,
            })

            if frame_count % 30 == 0:
                print(f"  frame={frame_count} dt={dt*1000:.0f}ms fps={1/max(dt,0.001):.1f} "
                      f"C={confidence:.2f} mode={mode} motor_delta={debug['motor_delta']}")

    except socket.timeout:
        print("[server] socket timeout")
    except (ConnectionResetError, ConnectionError, OSError) as exc:
        print(f"[server] connection closed: {exc}")
    finally:
        print(f"[server] client done. processed {frame_count} frames.")
    return frame_count


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--device", default=None)
    ap.add_argument("--opentrackvla_root", default=None,
                    help="Path to OpenTrackVLA source. Defaults to bundled third_party/OpenTrackVLA.")
    ap.add_argument("--base_hf_model_dir", default=None,
                    help="Official OpenTrackVLA HuggingFace checkpoint directory.")
    ap.add_argument("--qwen_model_path", default=None,
                    help="Optional local Qwen/Qwen3-0.6B directory for offline inference.")
    ap.add_argument("--dinov3_model_path", default=None,
                    help="Local DINOv3 model directory downloaded from ModelScope or HuggingFace.")
    ap.add_argument("--siglip_model_path", default=None,
                    help="Optional local SigLIP directory for offline inference.")
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--mock_control", action="store_true",
                    help="Do not load the model; return a fixed safe command for protocol testing.")
    ap.add_argument("--mock_action", choices=sorted(MOCK_KEYS), default="stop")
    ap.add_argument("--mock_speed", type=int, default=200)
    ap.add_argument("--control_dt", type=float, default=0.1,
                    help="Seconds between trained waypoints; used to convert waypoint displacement to action.")
    ap.add_argument("--control_waypoint_index", type=int, default=1,
                    help="Future waypoint index used for motor control. action = waypoint / ((index + 1) * dt).")
    ap.add_argument("--motor_scale", type=float, default=400.0,
                    help="PWM speed scale passed to waypoint_to_motor after waypoint-to-action conversion.")
    ap.add_argument("--min_motor_delta", type=int, default=0,
                    help="Optional minimum PWM delta from neutral for nonzero motor commands.")
    ap.add_argument("--max_action_abs", type=float, default=1.0,
                    help="Clamp each normalized action component to +/- this value before motor conversion.")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--no_cleanup_port", action="store_true",
                    help="Do not kill an existing process listening on --port before startup.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                    help="Print cleanup targets without killing them.")
    args = ap.parse_args()

    if args.control_dt <= 0:
        ap.error("--control_dt must be > 0")
    if args.control_waypoint_index < 0:
        ap.error("--control_waypoint_index must be >= 0")
    if args.motor_scale <= 0:
        ap.error("--motor_scale must be > 0")
    if args.min_motor_delta < 0:
        ap.error("--min_motor_delta must be >= 0")
    if args.max_action_abs <= 0:
        ap.error("--max_action_abs must be > 0")

    if not args.no_cleanup_port:
        cleanup_port(args.port, dry_run=args.cleanup_dry_run)
        if args.cleanup_dry_run:
            return

    if args.mock_control:
        device = None
        model = None
        encoder = None
    else:
        import torch

        opentrackvla_root = resolve_opentrackvla_root(args.opentrackvla_root)
        sys.path.insert(0, str(opentrackvla_root))
        configure_default_weight_paths(args, opentrackvla_root)
        from cache_gridpool import VisionCacheConfig, VisionFeatureCacher

        device = torch.device(args.device or default_device())
        print(f"[server] OpenTrackVLA root: {opentrackvla_root}")
        print(f"[server] base_hf_model_dir: {args.base_hf_model_dir or '(not set)'}")
        print(f"[server] pfem_ckpt: {args.ckpt or '(not set)'}")
        print(f"[server] DINOV3_MODEL_PATH: {os.environ.get('DINOV3_MODEL_PATH', '(not set)')}")
        print(f"[server] QWEN_MODEL_PATH: {os.environ.get('QWEN_MODEL_PATH', '(not set)')}")
        print(f"[server] SIGLIP_MODEL_PATH: {os.environ.get('SIGLIP_MODEL_PATH', '(not set)')}")
        model = load_model(args.ckpt, device, opentrackvla_root, base_hf_model_dir=args.base_hf_model_dir)
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
