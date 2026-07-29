# track_car

Raspberry Pi smart-car project for local TrackVLA/PFEM inference.

The repo is organized by pipeline:

```text
car_runtime/              Raspberry Pi runtime, motor control, TCP protocol
data_pipeline/            image/action/state collection and JSONL conversion
inference_pipeline/       computer-side inference server
third_party/OpenTrackVLA/ PFEM-capable OpenTrackVLA source code, no weights
weights/                  model weight placement notes and manifest
docs/                     architecture and protocol docs
```

## Current State

The runnable pieces are now in one GitHub repo:

```text
Pi camera -> TCP/JPEG -> computer inference server
computer model -> command JSON -> Pi UART motor control
collected data -> JSONL -> PFEM training script
OpenTrackVLA source -> bundled under third_party/OpenTrackVLA
```

Large model weights and collected data are still not committed.

## Required Weights

Put weights in this layout after cloning:

```text
track_car/
  weights/modelscope/dinov3-vits16-pretrain-lvd1689m/

  third_party/OpenTrackVLA/
    ckpts_hf/opentrackvla-qwen06b/
    ckpts_hf/qwen3-0.6b/                         # optional if HF cache works
    ckpts_hf/siglip-so400m-patch14-384/          # optional if HF cache works
    ckpts_pfem/car_official_dinov3/pfem_epoch0.pt
```

Required assets:

| Asset | Role |
| --- | --- |
| `opentrackvla-qwen06b` | official OpenTrackVLA 0.6B base planner |
| `pfem_epoch0.pt` | trained car PFEM checkpoint |
| `dinov3-vits16-pretrain-lvd1689m` | frozen DINOv3 visual encoder |
| `Qwen/Qwen3-0.6B` | frozen Qwen backbone, local dir or Hugging Face cache |
| `google/siglip-so400m-patch14-384` | frozen SigLIP encoder, local dir or Hugging Face cache |

See `weights/README.md` for copy/download commands.

Runtime model loading is local-only. If Qwen or SigLIP is missing locally, the
server will fail with the checked paths instead of trying to download online.
Non-mock inference is also fail-closed: the PFEM checkpoint and its
`schema_version`, `label_mode`, `history`, and `dt` metadata must be present.
Random PFEM initialization is only available with the explicit
`--allow_random_init --shadow_mode` stop-only combination.

## Install

```bash
cd track_car
python -m pip install -r requirements.txt
```

For CUDA Windows machines, install the correct PyTorch build from the official
PyTorch selector before running real-time inference.

## Communication Test

Run on the computer:

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --mock_control \
  --mock_action stop
```

Run on the Raspberry Pi:

```bash
python3 car_runtime/pi_client.py \
  --server_ip <computer-ip> \
  --server_port 9999 \
  --instruction "follow the person in red shirt" \
  --camera_backend v4l2 \
  --camera_fourcc MJPG \
  --width 320 \
  --height 240 \
  --dry_run
```

## Real Model Server

If weights are placed in the default layout:

```bash
python inference_pipeline/mac_server.py --port 9999 --timeout 30
```

Explicit version:

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --timeout 30 \
  --opentrackvla_root third_party/OpenTrackVLA \
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --ckpt third_party/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt \
  --dinov3_model_path weights/modelscope/dinov3-vits16-pretrain-lvd1689m
```

For fully offline inference, also pass:

```bash
  --qwen_model_path third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b \
  --siglip_model_path third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384
```

Then start the Pi client without `--dry_run` only after the server prints model
path information and the dry-run command stream looks sane.

## Data Pipeline

Collect on Raspberry Pi:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name ep001 \
  --instruction "follow the person in red shirt" \
  --teleop keyboard \
  --speed 400 \
  --fps 5 \
  --camera_backend v4l2 \
  --camera_fourcc MJPG
```

Convert on the computer:

```bash
python data_pipeline/build_training_data.py \
  --input data/collected \
  --output data/car_train.jsonl
```

If the source is already a single episode directory, use the dedicated adapter
instead of pointing the multi-episode builder at it:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
conda run --no-capture-output -n pytorch \
python -m data_preprocess.prepare_collected_episode \
  --episode data/test006 \
  --output data/test006_train.jsonl \
  --processed_episode_dir data/processed/test006 \
  --rotate_180_all \
  --keep_orientation_frames 0 200 \
  --detector_device mps
```

Conversion uses OmDet-Turbo person boxes when available, with a warned Haar
fallback. The default `step_action` samples include `step_actions`,
`prev_action`, and `delta_vel`. The sidecar manifest binds the JSONL by SHA-256
and row count; training rejects edited, truncated, or field-incomplete data.

Train PFEM:

```bash
python third_party/OpenTrackVLA/scripts/train_pfem.py \
  --train_json data/car_train.jsonl \
  --epochs 1 \
  --batch_size 2 \
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --cache_root data/vision_cache_dinov3 \
  --out_dir third_party/OpenTrackVLA/ckpts_pfem/car_official_dinov3
```

## Safety

Start every new setup in this order:

```text
mock stop -> Pi dry-run -> lifted-car movement -> low-speed floor test -> model control
```

The Pi client stops motors on Ctrl+C, disconnect, or socket timeout.
