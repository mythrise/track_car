# TrackVLA / OpenTrackVLA Integration

This repository vendors the PFEM-capable OpenTrackVLA source needed for car
inference and training under `third_party/OpenTrackVLA`. Full inference expects
that source plus local weights:

```text
track_car/third_party/OpenTrackVLA/
  model.py
  harness/
  cache_gridpool.py
  ckpts_hf/opentrackvla-qwen06b/
  ckpts_pfem/car_official_dinov3/pfem_epoch0.pt

track_car/weights/modelscope/dinov3-vits16-pretrain-lvd1689m/
```

## Recommended Model Stack

```text
Raspberry Pi camera
  -> JPEG over TCP
Mac inference server
  -> OpenTrackVLA / PFEM model
  -> waypoint/action
  -> motor pulse command
Raspberry Pi executor
```

## Weight Roles

| Weight | Role |
| --- | --- |
| `opentrackvla-qwen06b` | Base OpenTrackVLA Qwen0.6B planner checkpoint. |
| `Qwen/Qwen3-0.6B` | Native LLM backbone loaded by OpenTrackVLA `model.py`. |
| `pfem_epoch0.pt` | PFEM-Harness checkpoint trained on collected car data. |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | Official DINOv3 visual tower; gated on Hugging Face. |
| `google/siglip-so400m-patch14-384` | SigLIP visual tower used with DINOv3. |

See `weights/README.md` for the exact download commands and the list of large
files that are intentionally not uploaded to GitHub.

## Runtime Contract

The model server should output one command JSON per input frame:

```json
{
  "type": "command",
  "seq": 12,
  "motors": [1700, 1300, 1700, 1300],
  "pan": 1500,
  "tilt": 1500,
  "fps": 8.4,
  "confidence": 0.72,
  "mode": 0,
  "stop": false
}
```

The Raspberry Pi never runs the VLA model in the current design. It only sends
camera frames and executes returned commands.
