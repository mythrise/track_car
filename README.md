# track_car

Local TrackVLA/PFEM car project scaffold for a Raspberry Pi smart car.

This repository is now organized by pipeline:

```text
car_runtime/          Raspberry Pi runtime, motor control, TCP protocol
data_pipeline/        image/action/state collection and JSONL conversion
inference_pipeline/   Mac-side inference server and TrackVLA integration entrypoint
weights/              model weight manifest and download/placement notes
docs/                 architecture, protocol, and pipeline documentation
```

It intentionally does not include large model weights, collected images, videos,
or the full OpenTrackVLA repository.

## What Was Uploaded

The first upload was only the runnable car bridge:

```text
Pi camera -> TCP/JPEG -> Mac server
Mac command JSON -> TCP -> Pi motor UART / pan-tilt PWM
```

That was not the full data pipeline. This version adds the project structure
needed to make the data pipeline, inference pipeline, and weight management
explicit.

## Missing Weights And Local Files

This repository does not contain the large model/data artifacts needed for
real TrackVLA/PFEM inference. Mock control works without these files; real
model control does not.

### Download / Prepare Yourself

| Item | Required for | Suggested local path | Notes |
| --- | --- | --- | --- |
| Full OpenTrackVLA repo | full model inference source code | `/Users/mythrise/科研实习/OpenTrackVLA` | Not vendored into this repo. Pass it with `--opentrackvla_root` or `OPENTRACKVLA_ROOT`. |
| `omlab/opentrackvla-qwen06b` | official OpenTrackVLA 0.6B checkpoint / baseline planner | `/Users/mythrise/科研实习/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b` | Contains `model.safetensors`, `config.json`, and HF wrapper files. Around 1.2 GB locally. |
| `Qwen/Qwen3-0.6B` | LLM backbone used by native `model.py` and PFEM wrapper | Hugging Face cache or a local model dir | Transformers can auto-download it, but pre-download it if the Mac will run offline. |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | official DINOv3 visual tokens | local DINOv3 dir, then set `DINOV3_MODEL_PATH=/path/to/dinov3` | Gated Hugging Face model; request access officially. |
| `google/siglip-so400m-patch14-384` | SigLIP visual tokens | Hugging Face cache or local model dir | Used together with DINOv3 in OpenTrackVLA `VisionFeatureCacher`. |
| `ckpts_pfem/pfem_epoch*.pt` | PFEM-Harness car-control checkpoint | `/Users/mythrise/科研实习/OpenTrackVLA/ckpts_pfem/pfem_epoch3.pt` | Not downloaded from this repo; train it with OpenTrackVLA `scripts/train_pfem.py` or copy from your experiment machine. |
| `ckpts/model_epoch*.pt` | legacy/custom OpenTrackVLA training checkpoint | `/Users/mythrise/科研实习/OpenTrackVLA/ckpts/` | Optional alternative when evaluating custom checkpoints. |

Example downloads:

```bash
cd /Users/mythrise/科研实习/OpenTrackVLA

hf download omlab/opentrackvla-qwen06b \
  --local-dir ckpts_hf/opentrackvla-qwen06b

hf download Qwen/Qwen3-0.6B \
  --local-dir ckpts_hf/qwen3-0.6b

hf download google/siglip-so400m-patch14-384 \
  --local-dir ckpts_hf/siglip-so400m-patch14-384

# DINOv3 is gated. After Hugging Face access is approved:
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir ckpts_hf/dinov3-vits16-pretrain-lvd1689m
export DINOV3_MODEL_PATH=/Users/mythrise/科研实习/OpenTrackVLA/ckpts_hf/dinov3-vits16-pretrain-lvd1689m
```

Then create a local manifest:

```bash
cp weights/weights_manifest.example.json weights/weights_manifest.local.json
# edit weights/weights_manifest.local.json to match your machine
python weights/resolve_weights.py --manifest weights/weights_manifest.local.json
```

### Not Uploaded To GitHub

These are intentionally excluded:

```text
OpenTrackVLA/ full source tree
ckpts_hf/ downloaded Hugging Face model snapshots
ckpts/ and ckpts_pfem/ training checkpoints
*.pt, *.pth, *.safetensors model files
data/collected/ raw car episodes
data/*.jsonl training/evaluation data
sim_data/, world_rollouts/, Habitat/EVT-Bench outputs
*.jpg, *.jpeg, *.png, *.mp4, *.mov, *.avi media files
weights/*.local.json machine-specific paths
.env, venv/, .venv/, __pycache__/
```

For real Raspberry Pi motor execution, `car_runtime/uart_transport.py` sends
the same UART strings used in the vendor infrared remote example. Install
`python3-serial` on the Pi and verify the UART with
`python3 car_runtime/hardware_check.py --open_uart`. `pigpio`/`pigpiod` are
only required for pan/tilt servo control.

## Pipelines

### 1. Car Runtime

Purpose: run on Raspberry Pi and execute commands safely.

Key files:

```text
car_runtime/pi_client.py       sends frames to Mac and executes returned commands
car_runtime/car_hardware.py    vendor UART/PWM hardware adapter
car_runtime/car_protocol.py    length-prefixed TCP JSON/JPEG protocol
car_runtime/move_test.py       bounded motor smoke test
```

Safe communication test:

```bash
# Mac
python inference_pipeline/mac_server.py --port 9999 --mock_control --mock_action stop

# Raspberry Pi
python3 car_runtime/pi_client.py --server_ip <Mac_IP> --server_port 9999 --dry_run
```

Bounded movement test:

```bash
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3 --execute
```

### 2. Data Pipeline

Purpose: collect camera frames, control state, motor commands, and convert them
into training samples.

Key files:

```text
data_pipeline/collect_data.py
data_pipeline/build_training_data.py
data_pipeline/README.md
data_pipeline/schemas/
```

Collect on Raspberry Pi:

```bash
python3 data_pipeline/collect_data.py \
  --episode_name ep001 \
  --instruction "follow the person in red shirt" \
  --teleop keyboard \
  --speed 200 \
  --fps 10
```

Convert on Mac:

```bash
python data_pipeline/build_training_data.py \
  --input data/collected \
  --output data/car_train.jsonl
```

### 3. Inference Pipeline

Purpose: run the local Mac server that receives frames and returns motor
commands. It supports two modes:

```text
mock mode       no model, safe protocol/movement tests
model mode      loads OpenTrackVLA/PFEM from a separate OpenTrackVLA repo
```

Mock mode:

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --mock_control \
  --mock_action stop
```

Full model mode:

```bash
python inference_pipeline/mac_server.py \
  --opentrackvla_root /path/to/OpenTrackVLA \
  --ckpt /path/to/ckpts_pfem/pfem_epoch3.pt \
  --port 9999
```

`OPENTRACKVLA_ROOT=/path/to/OpenTrackVLA` can be used instead of
`--opentrackvla_root`.

### 4. Weights

Purpose: track what weights are needed without committing them.

See:

```text
weights/README.md
weights/weights_manifest.example.json
```

Large files such as `.pt`, `.safetensors`, datasets, videos, and collected
images are ignored by git.

## Recommended First Run

1. Put Mac and Raspberry Pi on the same hotspot/LAN.
2. Find the Mac IP from the Pi side.
3. Run Mac mock server with `--mock_action stop`.
4. Run Pi client with `--dry_run`.
5. Lift the car and test one low-speed mock movement.
6. Only after that, collect data or attach the full model path.

## Safety

Do not begin with full model control. Start with:

```text
mock stop -> dry-run client -> lifted-car movement -> low-speed floor test
```

The Pi client stops motors on Ctrl+C, disconnect, or socket timeout.
