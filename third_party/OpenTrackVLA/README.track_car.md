# Bundled OpenTrackVLA Source

This directory is a source-only export of the PFEM-capable OpenTrackVLA code
used by `track_car`.

It is committed so a fresh clone of `track_car` has the model code needed for
training and inference:

```text
model.py                  OpenTrackVLA/Qwen planner wrapper
cache_gridpool.py         DINOv3 + SigLIP visual token encoder
open_trackvla_hf/         Hugging Face wrapper for opentrackvla-qwen06b
harness/                  PFEM modules and harness wrapper
scripts/train_pfem.py     PFEM training entrypoint
scripts/infer_pfem.py     offline image/directory inference entrypoint
```

Large assets are intentionally not committed here. Put them in these ignored
local paths:

```text
third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b/
third_party/OpenTrackVLA/ckpts_hf/qwen3-0.6b/
third_party/OpenTrackVLA/ckpts_hf/siglip-so400m-patch14-384/
third_party/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt
weights/modelscope/dinov3-vits16-pretrain-lvd1689m/
```

The server defaults to this bundled source path when `--opentrackvla_root` is
not provided:

```bash
python inference_pipeline/mac_server.py --port 9999
```

Use explicit paths when testing another OpenTrackVLA checkout:

```bash
python inference_pipeline/mac_server.py \
  --opentrackvla_root /path/to/OpenTrackVLA \
  --base_hf_model_dir /path/to/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --ckpt /path/to/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt
```
