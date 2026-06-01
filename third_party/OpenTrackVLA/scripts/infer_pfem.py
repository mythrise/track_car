#!/usr/bin/env python3
"""PFEM-Harness inference on a directory of frames or a single image.

Usage:
    python scripts/infer_pfem.py --input data/test_frames/ --ckpt ckpts_pfem/pfem_epoch0.pt
    python scripts/infer_pfem.py --input frame.jpg --ckpt ckpts_pfem/pfem_epoch0.pt
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import OpenTrackVLA, ModelConfig
from harness.harness_wrapper import PFEMHarness
from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens
from local_weights import (
    default_dinov3_candidates,
    default_qwen_candidates,
    default_siglip_candidates,
    resolve_local_model_path,
)


def get_default_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_official_base(base_hf_model_dir):
    from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint

    base_hf_dir = Path(base_hf_model_dir)
    hf_config = OpenTrackVLAConfig.from_pretrained(str(base_hf_dir), local_files_only=True)
    hf_config.llm_name = os.environ["QWEN_MODEL_PATH"]
    hf_model = OpenTrackVLAForWaypoint(hf_config)
    state_path = base_hf_dir / "model.safetensors"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing OpenTrackVLA HF weights: {state_path}")
    missing, unexpected = hf_model.load_state_dict(load_safetensors(str(state_path)), strict=False)
    print(f"[infer] loaded official base: {len(missing)} missing, {len(unexpected)} unexpected")
    return hf_model.model


def load_model(ckpt_path, device, base_hf_model_dir=None):
    if base_hf_model_dir:
        base = load_official_base(base_hf_model_dir)
    else:
        mcfg = ModelConfig(
            llm_name=os.environ["QWEN_MODEL_PATH"],
            n_waypoints=8,
            freeze_llm=True,
        )
        base = OpenTrackVLA(mcfg, vision_feat_dim=1536)
    base = base.to(device)
    model = PFEMHarness(base).to(device).eval()
    if ckpt_path and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt.get("model_state", {}), strict=False)
        missing = [k for k in missing if not k.startswith("base.llm.")]
        print(f"[infer] loaded ckpt: {len(missing)} missing, {len(unexpected)} unexpected")
    return model


def encode_frame(encoder, img_pil):
    tok_dino, Hp, Wp = encoder._encode_dino([img_pil])
    tok_sigl = encoder._encode_siglip([img_pil], out_hw=(Hp, Wp))
    Vt = torch.cat([tok_dino, tok_sigl], dim=-1)
    Vfine = grid_pool_tokens(Vt, Hp, Wp, out_tokens=64)[0].float()
    Vcoarse = grid_pool_tokens(Vt, Hp, Wp, out_tokens=4)[0].float()
    return Vcoarse, Vfine


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--instruction", default="follow the person")
    ap.add_argument("--history", type=int, default=31)
    ap.add_argument("--output", default="infer_pfem_out")
    ap.add_argument("--base_hf_model_dir", default=None)
    ap.add_argument("--qwen_model_path", default=None)
    ap.add_argument("--dinov3_model_path", default=None)
    ap.add_argument("--siglip_model_path", default=None)
    args = ap.parse_args()

    os.environ["QWEN_MODEL_PATH"] = resolve_local_model_path(
        label="Qwen/Qwen3-0.6B",
        repo_id="Qwen/Qwen3-0.6B",
        explicit=args.qwen_model_path,
        env_var="QWEN_MODEL_PATH",
        candidates=default_qwen_candidates(),
    )
    os.environ["DINOV3_MODEL_PATH"] = resolve_local_model_path(
        label="DINOv3",
        repo_id="facebook/dinov3-vits16-pretrain-lvd1689m",
        explicit=args.dinov3_model_path,
        env_var="DINOV3_MODEL_PATH",
        candidates=default_dinov3_candidates(),
    )
    os.environ["SIGLIP_MODEL_PATH"] = resolve_local_model_path(
        label="SigLIP",
        repo_id="google/siglip-so400m-patch14-384",
        explicit=args.siglip_model_path,
        env_var="SIGLIP_MODEL_PATH",
        candidates=default_siglip_candidates(),
    )

    device = torch.device(get_default_device())
    print(f"[infer] device={device}")

    model = load_model(args.ckpt, device, base_hf_model_dir=args.base_hf_model_dir)
    encoder_device = "cuda" if device.type == "cuda" else "cpu"
    encoder_cfg = VisionCacheConfig(image_size=384, batch_size=1, device=encoder_device)
    encoder = VisionFeatureCacher(encoder_cfg)
    encoder.eval()

    input_path = Path(args.input)
    if input_path.is_file():
        frames = [input_path]
    else:
        frames = sorted(input_path.glob("*.jpg")) + sorted(input_path.glob("*.png"))

    os.makedirs(args.output, exist_ok=True)

    B = 1
    state = model.init_state(B, device)
    coarse_history = []
    results = []

    from PIL import Image

    for i, fpath in enumerate(frames):
        pil = Image.open(str(fpath)).convert("RGB")
        Vc, Vf = encode_frame(encoder, pil)

        coarse_history.append(Vc)
        if len(coarse_history) > args.history:
            coarse_history = coarse_history[-args.history:]

        # Pad history if shorter
        padded = coarse_history + [coarse_history[0]] * (args.history - len(coarse_history))
        padded = padded[:args.history]

        coarse_tokens = torch.cat(padded, dim=0).unsqueeze(0)
        coarse_tidx = torch.arange(args.history).repeat_interleave(4).unsqueeze(0)
        fine_tokens = Vf.unsqueeze(0)
        fine_tidx = torch.full((1, 64), args.history, dtype=torch.long)

        with torch.inference_mode():
            out = model.forward_step(
                coarse_tokens=coarse_tokens.to(device),
                coarse_tidx=coarse_tidx.to(device),
                fine_tokens=fine_tokens.to(device),
                fine_tidx=fine_tidx.to(device),
                instructions=[args.instruction],
                prev_state=state,
            )
        state = out["new_state"]

        wp = out["waypoints"][0].cpu().numpy()
        cot = out["cot_decoded"]
        result = {
            "frame": str(fpath),
            "theta_deg": float(cot["theta_deg"][0]),
            "dist_m": float(cot["dist_m"][0]),
            "invalid": bool(cot["invalid_pred"][0]),
            "confidence": float(out["C"][0]),
            "mode": int(out["orch"]["mode"][0]),
            "waypoints": wp.tolist(),
        }
        results.append(result)

        if i % 10 == 0:
            print(f"  frame {i}/{len(frames)}: θ={result['theta_deg']:.1f}° "
                  f"d={result['dist_m']:.2f}m C={result['confidence']:.2f} "
                  f"mode={result['mode']} invalid={result['invalid']}")

    out_json = os.path.join(args.output, "predictions.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[infer] done. {len(results)} predictions → {out_json}")


if __name__ == "__main__":
    main()
