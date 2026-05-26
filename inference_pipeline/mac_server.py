#!/usr/bin/env python3
"""Mac inference server — receives frames from Pi, runs PFEM-Harness, sends commands.

Run on your Mac/PC:
    python inference_pipeline/mac_server.py --ckpt ckpts_pfem/pfem_epoch0.pt --port 9999
"""

import argparse
import os
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "car_runtime"))

try:
    from car_hardware import command_from_key, waypoint_to_motor
    from car_protocol import recv_jpeg_frame, recv_json, send_json
    from process_cleanup import cleanup_port
except ImportError:
    from car_runtime.car_hardware import command_from_key, waypoint_to_motor
    from car_runtime.car_protocol import recv_jpeg_frame, recv_json, send_json
    from car_runtime.process_cleanup import cleanup_port


def resolve_opentrackvla_root(root_arg):
    root = root_arg or os.environ.get("OPENTRACKVLA_ROOT")
    if root is None:
        raise RuntimeError(
            "Full model mode requires --opentrackvla_root or OPENTRACKVLA_ROOT. "
            "Use --mock_control for protocol testing without OpenTrackVLA."
        )
    root_path = Path(root).expanduser().resolve()
    if not (root_path / "model.py").exists():
        raise FileNotFoundError(f"OpenTrackVLA root not found or missing model.py: {root_path}")
    return root_path


def load_model(ckpt_path, device, opentrackvla_root):
    import torch

    sys.path.insert(0, str(opentrackvla_root))
    from model import OpenTrackVLA, ModelConfig
    from harness.harness_wrapper import PFEMHarness

    mcfg = ModelConfig(n_waypoints=8, freeze_llm=True)
    base = OpenTrackVLA(mcfg, vision_feat_dim=1536).to(device)
    model = PFEMHarness(base).to(device).eval()
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        msd = ckpt.get("model_state", {})
        model.load_state_dict(msd, strict=False)
        print(f"[server] loaded {ckpt_path}")
    return model


def frame_to_tokens(frame, encoder=None):
    """Convert a raw BGR frame to (coarse_tokens, fine_tokens) + tidx.

    In full deployment, this uses SigLIP+DINOv2. For quick testing without
    those models, we use a random projection placeholder.
    """
    import torch

    h, w = frame.shape[:2]
    # Flatten + normalize to a 1536-d feature
    small = cv2.resize(frame, (64, 48))
    flat = small.astype(np.float32).flatten() / 255.0
    # Pad/truncate to 1536
    if len(flat) > 1536:
        feat = flat[:1536]
    else:
        feat = np.pad(flat, (0, 1536 - len(flat)))
    feat = torch.from_numpy(feat).float()

    # Create fine tokens (64 tokens) and coarse tokens (31*4 tokens)
    fine = feat.unsqueeze(0).unsqueeze(0).expand(1, 64, -1)
    coarse = feat.unsqueeze(0).unsqueeze(0).expand(1, 124, -1)
    fine_tidx = torch.full((1, 64), fill_value=31, dtype=torch.long)
    coarse_tidx = torch.arange(31).repeat_interleave(4).unsqueeze(0)
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


def handle_connection(conn, addr, args, model, device):
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
            else:
                import torch

                coarse, c_tidx, fine, f_tidx = frame_to_tokens(frame)

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

                # Use waypoint[1] for motor command (same as trained_agent.py)
                motors = waypoint_to_motor(wp[1])

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
            })

            if frame_count % 30 == 0:
                print(f"  frame={frame_count} dt={dt*1000:.0f}ms fps={1/max(dt,0.001):.1f} "
                      f"C={confidence:.2f} mode={mode}")

    except socket.timeout:
        print("[server] socket timeout")
    except (ConnectionResetError, ConnectionError, OSError) as exc:
        print(f"[server] connection closed: {exc}")
    finally:
        print(f"[server] client done. processed {frame_count} frames.")
    return frame_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--device", default=None)
    ap.add_argument("--opentrackvla_root", default=None,
                    help="Path to the full OpenTrackVLA repo for non-mock model inference.")
    ap.add_argument("--mock_control", action="store_true",
                    help="Do not load the model; return a fixed safe command for protocol testing.")
    ap.add_argument("--mock_action", choices=sorted(MOCK_KEYS), default="stop")
    ap.add_argument("--mock_speed", type=int, default=200)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--no_cleanup_port", action="store_true",
                    help="Do not kill an existing process listening on --port before startup.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                    help="Print cleanup targets without killing them.")
    args = ap.parse_args()

    if not args.no_cleanup_port:
        cleanup_port(args.port, dry_run=args.cleanup_dry_run)
        if args.cleanup_dry_run:
            return

    if args.mock_control:
        device = None
        model = None
    else:
        import torch
        opentrackvla_root = resolve_opentrackvla_root(args.opentrackvla_root)
        device = torch.device(args.device or default_device())
        model = load_model(args.ckpt, device, opentrackvla_root)

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
                total_frames += handle_connection(conn, addr, args, model, device)
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("[server] interrupted")
    finally:
        server.close()
        print(f"[server] shutdown. processed {total_frames} total frames.")


if __name__ == "__main__":
    main()
