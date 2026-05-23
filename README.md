# track_car

Raspberry Pi smart-car bridge for local Mac inference.

This repository contains the runnable communication and hardware-control layer
for a Raspberry Pi car:

```text
Raspberry Pi camera -> TCP/JPEG -> Mac server
Mac command JSON -> TCP -> Raspberry Pi motor UART / pan-tilt PWM
```

It intentionally does not include model weights, training data, or the full
OpenTrackVLA repository.

## Layout

```text
scripts/car/
  car_protocol.py       length-prefixed TCP JSON/JPEG protocol
  car_hardware.py       Raspberry Pi motor UART + pan/tilt wrapper
  mac_server.py         Mac-side server; supports safe mock-control mode
  pi_client.py          Raspberry Pi client; captures camera and executes commands
  move_test.py          bounded one-shot motor smoke test
  collect_data.py       camera + keyboard teleop data collection
  build_training_data.py collected episodes -> JSONL training samples
```

## Protocol

All TCP payloads use:

```text
4-byte big-endian payload length + payload bytes
```

Session flow:

```text
Pi -> Mac: JSON hello
loop:
  Pi -> Mac: JPEG frame bytes
  Mac -> Pi: JSON command
```

Command JSON example:

```json
{
  "type": "command",
  "seq": 1,
  "motors": [1500, 1500, 1500, 1500],
  "pan": 1500,
  "tilt": 1500,
  "fps": 10.0,
  "confidence": 1.0,
  "mode": 0,
  "stop": true
}
```

## First Network Test

On the Mac:

```bash
python scripts/car/mac_server.py --port 9999 --mock_control --mock_action stop
```

On the Raspberry Pi:

```bash
python3 scripts/car/pi_client.py --server_ip <Mac_IP> --server_port 9999 --dry_run
```

This verifies Wi-Fi, TCP framing, camera capture, JPEG upload, JSON command
return, and stop behavior without moving the car.

## First Movement Test

Keep the car lifted or in a clear low-speed area.

On the Mac:

```bash
python scripts/car/mac_server.py \
  --port 9999 \
  --mock_control \
  --mock_action forward \
  --mock_speed 150
```

On the Raspberry Pi:

```bash
python3 scripts/car/pi_client.py --server_ip <Mac_IP> --server_port 9999
```

Stop both processes after one or two seconds. The Pi client sends stop on
Ctrl+C, disconnect, or socket timeout.

## Direct Motor Smoke Test

On the Raspberry Pi:

```bash
python3 scripts/car/move_test.py --move forward --speed 200 --duration 0.3
python3 scripts/car/move_test.py --move forward --speed 200 --duration 0.3 --execute
```

The first command prints the UART command without moving hardware. The second
actually moves and then stops in a `finally` block.

## Raspberry Pi Requirements

The vendor Raspberry Pi image should provide:

- `pigpio`
- `z_uart`
- OpenCV / `cv2`
- camera access via `cv2.VideoCapture(0)`

Start pigpio if needed:

```bash
sudo systemctl enable --now pigpiod
```

## Full Model Mode

`mac_server.py` can be placed back inside the OpenTrackVLA project and run
without `--mock_control` to call the PFEM/OpenTrackVLA model path. The current
standalone repository is meant for reliable communication and car-control
testing first.
