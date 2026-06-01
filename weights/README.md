# Weights

Do not commit model weights to this repository.

The code needed for the car runtime, data pipeline, PFEM training, and model
inference is now in this repo. The large model files still need to be copied or
downloaded locally.

## Required Layout

Place weights in this layout inside a fresh `track_car` clone:

```text
track_car/
  weights/
    modelscope/
      dinov3-vits16-pretrain-lvd1689m/
        config.json
        model.safetensors
        preprocessor_config.json

  third_party/
    OpenTrackVLA/
      ckpts_hf/
        opentrackvla-qwen06b/
          config.json
          model.safetensors

        qwen3-0.6b/                         # optional if HF cache works
          config.json
          model.safetensors or shards

        siglip-so400m-patch14-384/          # optional if HF cache works
          config.json
          model.safetensors or shards

      ckpts_pfem/
        car_official_dinov3/
          pfem_epoch0.pt
```

## What Each Weight Does

| Asset | Required | Role |
| --- | --- | --- |
| `opentrackvla-qwen06b` | yes | Official OpenTrackVLA 0.6B base planner checkpoint. |
| `pfem_epoch0.pt` | yes | The car-specific PFEM checkpoint trained from our test005/test006 data. |
| `dinov3-vits16-pretrain-lvd1689m` | yes | Frozen DINOv3 visual encoder for current camera frames. |
| `Qwen/Qwen3-0.6B` | yes, via local dir or HF cache | Frozen LLM backbone used inside the OpenTrackVLA base. |
| `google/siglip-so400m-patch14-384` | yes, via local dir or HF cache | Frozen SigLIP visual encoder used together with DINOv3. |

Qwen, DINOv3, and SigLIP are frozen. The trained car checkpoint is the PFEM
adapter/checkpoint, not a Qwen or DINOv3 fine-tune.

## Current Experiment Files To Copy

From the original Mac workspace, the trained run used these local files:

```text
/Users/mythrise/科研实习/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt
/Users/mythrise/科研实习/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b/
/Users/mythrise/科研实习/track_car/weights/modelscope/dinov3-vits16-pretrain-lvd1689m/
```

If the target computer cannot access Hugging Face, also copy:

```text
/Users/mythrise/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
/Users/mythrise/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384
```

On Windows, the equivalent Hugging Face cache root is usually:

```text
C:\Users\<your-user>\.cache\huggingface\hub\
```

## Copy Commands From This Mac

Run from `/Users/mythrise/科研实习`:

```bash
tar -czf track_car_infer_weights_20260529.tar.gz \
  OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt \
  OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  track_car/weights/modelscope/dinov3-vits16-pretrain-lvd1689m
```

If the target machine is offline from Hugging Face:

```bash
tar -czf hf_cache_qwen_siglip_20260529.tar.gz \
  -C /Users/mythrise/.cache/huggingface/hub \
  models--Qwen--Qwen3-0.6B \
  models--google--siglip-so400m-patch14-384
```

After extracting `track_car_infer_weights_20260529.tar.gz`, copy the contents
into the required layout above.

## Download Commands

Official OpenTrackVLA base:

```bash
cd track_car
hf download omlab/opentrackvla-qwen06b \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b
```

Qwen and SigLIP must be available locally. The code runs in offline mode and
will not try to download from Hugging Face at runtime. Either put normal model
directories here, or copy Hugging Face cache folders with `snapshots/<hash>/`.

Normal local directories:

```bash
hf download Qwen/Qwen3-0.6B \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b

hf download google/siglip-so400m-patch14-384 \
  --local-dir third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384
```

Copied cache folders are also accepted:

```text
third_party/OpenTrackVLA/ckpts_hf/models--Qwen--Qwen3-0.6B/snapshots/<hash>/config.json
third_party/OpenTrackVLA/ckpts_hf/models--google--siglip-so400m-patch14-384/snapshots/<hash>/config.json
```

DINOv3 from ModelScope:

```bash
modelscope download \
  --model facebook/dinov3-vits16-pretrain-lvd1689m \
  --local_dir weights/modelscope/dinov3-vits16-pretrain-lvd1689m
```

## Inference Command

If weights are placed exactly as above, the server can use defaults:

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

For fully offline inference, add:

```bash
  --qwen_model_path third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b \
  --siglip_model_path third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384
```

## Not Uploaded

These paths and file types are intentionally ignored by git:

```text
weights/modelscope/
third_party/OpenTrackVLA/ckpts_hf/
third_party/OpenTrackVLA/ckpts/
third_party/OpenTrackVLA/ckpts_pfem/
*.pt
*.pth
*.safetensors
data/
sim_data/
world_rollouts/
*.jpg
*.jpeg
*.png
*.mp4
*.mov
*.avi
weights/*.local.json
```
