#!/usr/bin/env python3
"""
inference_pipeline/f2_server.py  --  SA-Hstar (F2 / Harness) online inference server.

Wire protocol: identical to mac_server.py (recv_jpeg_frame / send_json).
Loads a Harness checkpoint via build_eval_row_predictor_from_checkpoint.
PFEM checkpoints are NOT supported here; use mac_server.py for those.

Usage:
    python inference_pipeline/f2_server.py \
        --ckpt  path/to/Harness_seed0_OFFICIAL.pt \
        --receipt experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json \
        --port 9999
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Wire-protocol helpers  (same dual-path as mac_server.py)
# ---------------------------------------------------------------------------
try:
    from car_protocol import recv_jpeg_frame, recv_json, send_json
    from process_cleanup import cleanup_port
except ModuleNotFoundError:
    from car_runtime.car_protocol import recv_jpeg_frame, recv_json, send_json
    from car_runtime.process_cleanup import cleanup_port

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEUTRAL_MOTORS = [1500, 1500, 1500, 1500]
HISTORY = 31        # coarse-frame history length (matches receipt / training)
MOTOR_SCALE = 400.0  # ±1 action unit → ±400 PWM around 1500 neutral
MAX_ABS = 1.0       # hard clamp on raw action values
MAX_ACTION_RATE = 2.0  # max change per second (rate-limit filter)
ACTION_EMA = 0.0    # EMA smoothing factor (0 = disabled)


# ---------------------------------------------------------------------------
# Motor helpers
# ---------------------------------------------------------------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def action_to_motors(fwd: float, yaw: float, scale: float = MOTOR_SCALE) -> list[int]:
    """Convert continuous (forward, yaw) action → 4-channel PWM list."""
    m0 = int(round(1500.0 + _clamp(fwd, -MAX_ABS, MAX_ABS) * scale))  # throttle
    m3 = int(round(1500.0 + _clamp(yaw, -MAX_ABS, MAX_ABS) * scale))  # steering
    return [m0, 1500, 1500, m3]


def filter_action(
    raw: tuple[float, float],
    prev: tuple[float, float],
    elapsed: float,
    max_rate: float,
    ema: float,
) -> tuple[float, float]:
    """Rate-limit then optionally EMA-smooth (forward, yaw)."""
    f, y = raw
    if elapsed > 0 and max_rate > 0:
        d = max_rate * elapsed
        f = _clamp(f, prev[0] - d, prev[0] + d)
        y = _clamp(y, prev[1] - d, prev[1] + d)
    if ema > 0:
        f = ema * prev[0] + (1.0 - ema) * f
        y = ema * prev[1] + (1.0 - ema) * y
    return (f, y)


# ---------------------------------------------------------------------------
# OpenTrackVLA root resolution  (mirrors mac_server.py)
# ---------------------------------------------------------------------------
def _opentrackvla_root(hint: str | None) -> Path:
    if hint:
        p = Path(hint)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"--opentrackvla_root not found: {hint}")
    candidates = [
        Path(__file__).parent.parent / "third_party" / "OpenTrackVLA",
        Path("third_party") / "OpenTrackVLA",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "Cannot locate OpenTrackVLA root. Pass --opentrackvla_root explicitly."
    )


# ---------------------------------------------------------------------------
# Vision-token encoding  (identical logic to mac_server.py encode_frame)
# ---------------------------------------------------------------------------
def encode_frame(frame, encoder):
    """BGR ndarray → (coarse_tokens, fine_tokens) CPU float32 tensors.

    coarse shape: (4, 1536)   fine shape: (64, 1536)
    """
    import cv2
    import torch
    from PIL import Image
    from cache_gridpool import grid_pool_tokens

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb).convert("RGB")
    tokens_dino, h_p, w_p = encoder._encode_dino([image])
    tokens_siglip = encoder._encode_siglip([image], out_hw=(h_p, w_p))
    tokens = torch.cat([tokens_dino, tokens_siglip], dim=-1)
    fine   = grid_pool_tokens(tokens, h_p, w_p, out_tokens=64)[0].float()
    coarse = grid_pool_tokens(tokens, h_p, w_p, out_tokens=4)[0].float()
    return coarse, fine


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_predictor(ckpt_path: str, receipt_path: str, device_str: str):
    """Return (EvalRowPredictor, arm_name) from a Harness checkpoint."""
    import torch
    from f2_experiment.assembly_model import build_eval_row_predictor_from_checkpoint

    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    print(f"[f2_server] receipt: {receipt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    arm_name = payload.get("arm", "S-SELF")
    u_pre    = payload.get("u_pre", "?")
    print(f"[f2_server] checkpoint arm={arm_name!r}  u_pre={u_pre}  ({ckpt_path})")

    predictor = build_eval_row_predictor_from_checkpoint(
        project_root=".",
        receipt_document=receipt,
        arm=arm_name,
        payload=payload,
        device=torch.device(device_str),
    )
    print("[f2_server] EvalRowPredictor ready.")
    return predictor, arm_name


# ---------------------------------------------------------------------------
# Connection handler
# ---------------------------------------------------------------------------
def handle_connection(conn, addr, args, predictor, encoder, device):
    import torch
    from f2_experiment.assembly_data import ObservationPacket
    from f2_experiment.runner import RunnerRow

    conn.settimeout(args.timeout)
    print(f"[f2_server] connected: {addr}")

    hello = recv_json(conn)
    if hello is None:
        print("[f2_server] client disconnected before hello")
        return 0

    instruction = hello.get("instruction", "follow the person")
    print(f"[f2_server] instruction: {instruction!r}")

    coarse_history: deque = deque(maxlen=HISTORY)
    prev_fy: tuple[float, float] = (0.0, 0.0)   # (forward, yaw) from last accepted step
    position: int = 0     # predictor call counter; resets to 0 after safety stop
    frame_count: int = 0
    last_t = time.monotonic()
    pan, tilt = 1500, 1500

    try:
        while True:
            frame = recv_jpeg_frame(conn)
            if frame is None:
                break

            t_now = time.monotonic()
            elapsed = max(0.001, t_now - last_t)
            last_t = t_now
            t0 = time.time()

            coarse, fine = encode_frame(frame, encoder)
            coarse_history.append(coarse)

            # ── warmup: need HISTORY coarse frames before first inference ──
            if len(coarse_history) < HISTORY:
                payload = {
                    "type": "command", "seq": frame_count + 1,
                    "motors": list(NEUTRAL_MOTORS), "pan": pan, "tilt": tilt,
                    "fps": 0.0, "confidence": 0.0, "stop": True,
                    "debug": {"phase": "warmup",
                              "warmup_seen": len(coarse_history),
                              "warmup_required": HISTORY},
                }
                send_json(conn, payload)
                frame_count += 1
                continue

            # ── inference ──────────────────────────────────────────────────
            with torch.no_grad():
                coarse_stack = torch.cat(list(coarse_history), dim=0)           # (H*4, 1536)
                coarse_tidx  = torch.arange(HISTORY, dtype=torch.long).repeat_interleave(4)
                fine_tidx    = torch.full((64,), fill_value=HISTORY, dtype=torch.long)

                obs = ObservationPacket(
                    coarse_tokens=coarse_stack,
                    coarse_tidx=coarse_tidx,
                    fine_tokens=fine,
                    fine_tidx=fine_tidx,
                    instruction=instruction,
                )
                dummy_targets = torch.zeros(8, 3, dtype=torch.float32)
                reset = (position == 0)
                row = RunnerRow(
                    original_row_index=frame_count,
                    sequence_id="live",
                    frame_idx=frame_count,
                    mirrored=False,
                    logged_prev_action=(prev_fy[0], 0.0, prev_fy[1]),
                    target_actions=dummy_targets,
                    observation=obs,
                )
                prev_tensor = torch.tensor([[prev_fy[0], prev_fy[1]]], dtype=torch.float32)
                pred = predictor(row, prev_tensor, mode="self", reset=reset, position=position)

            k0 = pred.raw_actions[0, 0].detach().float().cpu()   # shape (3,)
            raw_fwd = float(k0[0].item())
            raw_yaw = float(k0[2].item())

            reasons: list[str] = []
            if not (math.isfinite(raw_fwd) and math.isfinite(raw_yaw)):
                reasons.append("nonfinite_action")
            if abs(raw_fwd) > 2.5 or abs(raw_yaw) > 2.5:
                reasons.append("action_magnitude_exceeded")

            if reasons:
                motors = list(NEUTRAL_MOTORS)
                stop   = True
                prev_fy  = (0.0, 0.0)
                position = 0
            else:
                filt = filter_action((raw_fwd, raw_yaw), prev_fy, elapsed,
                                     args.max_action_rate, args.action_ema)
                fwd = _clamp(filt[0], -MAX_ABS, MAX_ABS)
                yaw = _clamp(filt[1], -MAX_ABS, MAX_ABS)
                motors   = action_to_motors(fwd, yaw, scale=args.motor_scale)
                stop     = False
                prev_fy  = (fwd, yaw)
                position += 1

            frame_count += 1
            payload = {
                "type": "command", "seq": frame_count,
                "motors": motors, "pan": pan, "tilt": tilt,
                "fps": 1.0 / max(time.time() - t0, 0.001),
                "confidence": 1.0, "stop": stop,
                "debug": {
                    "raw_forward": raw_fwd, "raw_yaw": raw_yaw,
                    "forward": prev_fy[0], "yaw": prev_fy[1],
                    "elapsed": elapsed, "position": position,
                    "reset": reset, "safety_reasons": reasons,
                },
            }
            send_json(conn, payload)

    except (OSError, ConnectionResetError) as exc:
        print(f"[f2_server] connection error: {exc}")

    print(f"[f2_server] disconnected: {addr}  frames={frame_count}")
    return frame_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SA-Hstar (F2/Harness) online inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Harness S-SELF .pt checkpoint")
    p.add_argument(
        "--receipt",
        default="experiments/windows_cuda_f2/assembly_receipt_cuda_final_v1.json",
        help="Assembly receipt JSON (must match checkpoint)",
    )
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--device", default=None, help="cuda / mps / cpu (auto if omitted)")
    p.add_argument("--opentrackvla_root", default=None,
                   help="Path to third_party/OpenTrackVLA (auto-detected if omitted)")
    p.add_argument("--motor_scale", type=float, default=MOTOR_SCALE)
    p.add_argument("--max_action_rate", type=float, default=MAX_ACTION_RATE,
                   help="Max action change per second (rate limiter)")
    p.add_argument("--action_ema", type=float, default=ACTION_EMA,
                   help="EMA smoothing factor 0–1 (0=disabled)")
    p.add_argument("--timeout", type=float, default=2.0,
                   help="Per-frame socket read timeout in seconds")
    p.add_argument("--no_cleanup_port", action="store_true")
    return p


def _default_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def main(argv=None):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    parser = build_parser()
    args = parser.parse_args(argv)

    # Add OpenTrackVLA to sys.path so cache_gridpool and model code are importable
    ovla_root = _opentrackvla_root(args.opentrackvla_root)
    if str(ovla_root) not in sys.path:
        sys.path.insert(0, str(ovla_root))
    print(f"[f2_server] OpenTrackVLA root: {ovla_root}")

    device = args.device or _default_device()
    print(f"[f2_server] device={device}")

    if not args.no_cleanup_port:
        cleanup_port(args.port, dry_run=False)

    # Load SA-Hstar predictor
    predictor, arm_name = load_predictor(args.ckpt, args.receipt, device)

    # Vision encoder (identical to mac_server.py)
    import torch
    from cache_gridpool import VisionCacheConfig, VisionFeatureCacher

    enc_dev = "cuda" if torch.device(device).type == "cuda" else "cpu"
    encoder = VisionFeatureCacher(
        VisionCacheConfig(image_size=384, batch_size=1, device=enc_dev)
    )
    encoder.eval()
    print(f"[f2_server] Vision encoder ready (device={enc_dev})")
    print(f"[f2_server] Listening on :{args.port}  arm={arm_name!r}  "
          f"motor_scale={args.motor_scale}  history={HISTORY}")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", args.port))
    server.listen(1)

    total = 0
    try:
        while True:
            conn, addr = server.accept()
            try:
                total += handle_connection(
                    conn, addr, args, predictor, encoder, torch.device(device)
                )
            finally:
                conn.close()
    except KeyboardInterrupt:
        print("[f2_server] interrupted")
    finally:
        server.close()
        print(f"[f2_server] shutdown.  total_frames={total}")


if __name__ == "__main__":
    main()
