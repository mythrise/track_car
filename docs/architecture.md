# Project Architecture

The project is split into four practical pipelines.

## 1. Runtime Control Pipeline

```text
Mac command JSON
  -> Raspberry Pi pi_client.py
  -> car_hardware.py
  -> vendor UART command
  -> motor controller / pan-tilt servos
```

Files:

```text
car_runtime/pi_client.py
car_runtime/car_hardware.py
car_runtime/move_test.py
```

## 2. Data Pipeline

```text
camera frame + timestamp + keyboard/action + motor pulses
  -> data/collected/<episode>/
  -> data_pipeline/build_training_data.py
  -> JSONL training samples
```

Files:

```text
data_pipeline/collect_data.py
data_pipeline/build_training_data.py
data_pipeline/schemas/
```

## 3. Inference Pipeline

```text
Raspberry Pi frame
  -> car_protocol.py length-prefixed JPEG
  -> inference_pipeline/mac_server.py
  -> OpenTrackVLA/PFEM model
  -> waypoint/action
  -> motor pulse command JSON
```

Files:

```text
inference_pipeline/mac_server.py
inference_pipeline/TRACKVLA_INTEGRATION.md
third_party/OpenTrackVLA/
```

## 4. Weight Pipeline

```text
weights manifest
  -> bundled OpenTrackVLA source root
  -> base Qwen0.6B weights
  -> trained PFEM checkpoint
```

Files:

```text
weights/README.md
weights/weights_manifest.example.json
```

## Current Reality

The control and communication pipeline is runnable now. The full model pipeline
has source code in this repo under `third_party/OpenTrackVLA`, but still needs
local weights in the paths documented by `weights/README.md`. Always test in
mock mode before enabling real model control.
