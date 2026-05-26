# Inference Pipeline

Mac-side server for local inference.

The server receives JPEG frames from the Raspberry Pi and returns JSON motor
commands.

## Modes

| Mode | Command | Purpose |
| --- | --- | --- |
| Mock stop | `--mock_control --mock_action stop` | Test TCP and camera without movement. |
| Mock movement | `--mock_control --mock_action forward --mock_speed 150` | Lifted-car bounded movement test. |
| Full model | `--opentrackvla_root /path/to/OpenTrackVLA --ckpt ...` | Load OpenTrackVLA/PFEM and return model commands. |

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
  --opentrackvla_root /Users/mythrise/科研实习/OpenTrackVLA \
  --ckpt /Users/mythrise/科研实习/OpenTrackVLA/ckpts_pfem/pfem_epoch3.pt \
  --port 9999
```

Without `--ckpt`, the server loads the PFEM wrapper with randomly initialized
PFEM heads around the OpenTrackVLA base. That is useful only for plumbing tests.

## Important

`mac_server.py` currently contains a lightweight placeholder `frame_to_tokens`
path for non-mock model mode. For serious model results, replace it with the
same SigLIP+DINOv2 `VisionFeatureCacher` path used in the full OpenTrackVLA
project.
