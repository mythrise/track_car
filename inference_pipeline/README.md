# Inference Pipeline

Mac-side server for local inference.

The server receives JPEG frames from the Raspberry Pi and returns JSON motor
commands.

## Modes

| Mode | Command | Purpose |
| --- | --- | --- |
| Mock stop | `--mock_control --mock_action stop` | Test TCP and camera without movement. |
| Mock movement | `--mock_control --mock_action forward --mock_speed 150` | Lifted-car bounded movement test. |
| Full model | `--opentrackvla_root third_party/OpenTrackVLA --ckpt ...` | Load OpenTrackVLA/PFEM and return model commands. |

## Mock Server

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --mock_control \
  --mock_action stop
```

Before binding, the server now clears stale processes already listening on
`--port`. Preview cleanup without killing anything:

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --mock_control \
  --cleanup_dry_run
```

Standalone cleanup command:

```bash
python car_runtime/kill_port.py --port 9999 --dry_run
python car_runtime/kill_port.py --port 9999
```

Disable port cleanup:

```bash
python inference_pipeline/mac_server.py \
  --port 9999 \
  --mock_control \
  --no_cleanup_port
```

The server keeps listening after a client disconnects, so `nc`/TCP probe tests
will not force you to restart it.

## Full Model Server

```bash
python inference_pipeline/mac_server.py \
  --opentrackvla_root third_party/OpenTrackVLA \
  --base_hf_model_dir third_party/OpenTrackVLA/ckpts_hf/opentrackvla-qwen06b \
  --ckpt third_party/OpenTrackVLA/ckpts_pfem/car_official_dinov3/pfem_epoch0.pt \
  --dinov3_model_path weights/modelscope/dinov3-vits16-pretrain-lvd1689m \
  --port 9999
```

Without `--ckpt`, the server loads the PFEM wrapper with randomly initialized
PFEM heads around the OpenTrackVLA base. That is useful only for plumbing tests.

## Important

`mac_server.py` now uses the bundled `third_party/OpenTrackVLA` source by
default and encodes frames with the same DINOv3 + SigLIP path used by PFEM
training. Put the required weights in the paths documented by `weights/README.md`.
