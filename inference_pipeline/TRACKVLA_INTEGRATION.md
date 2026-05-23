# TrackVLA / OpenTrackVLA Integration

This repository does not vendor the full OpenTrackVLA codebase. Full inference
expects the separate project to exist locally:

```text
/path/to/OpenTrackVLA/
  model.py
  harness/
  cache_gridpool.py
  ckpts_hf/opentrackvla-qwen06b/
  ckpts_pfem/
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
| `pfem_epoch*.pt` | PFEM-Harness checkpoint trained on collected/sim data. |
| vision encoder weights | SigLIP/DINOv2 weights used by OpenTrackVLA tokenization. |

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
