# Raspberry Pi Car Integration

This directory contains the real-car bridge for PFEM-Harness.

## Files

| file | purpose |
| --- | --- |
| `car_hardware.py` | Shared hardware adapter for motor UART and pan/tilt servos. Based on `示例代码.zip`. |
| `collect_data.py` | Records camera frames and optional keyboard teleop actions. |
| `build_training_data.py` | Converts collected episodes into JSONL training data. |
| `pi_client.py` | Raspberry Pi runtime client: sends frames to Mac server and executes returned commands. |
| `mac_server.py` | Mac inference server: receives frames, predicts waypoints, returns motor/servo commands. |
| `car_protocol.py` | Length-prefixed TCP protocol shared by Mac and Raspberry Pi. |

## TCP Protocol

The Mac runs a TCP server and the Raspberry Pi connects as a client.

All messages use the same framing:

```text
4-byte big-endian payload length + payload bytes
```

Session flow:

```text
Pi -> Mac: JSON hello
Mac waits
loop:
  Pi -> Mac: JPEG frame bytes
  Mac -> Pi: JSON command
```

Hello JSON:

```json
{
  "type": "hello",
  "protocol": 1,
  "instruction": "follow the person",
  "width": 320,
  "height": 240
}
```

Command JSON:

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

For the first network test, run the Mac server in safe mock mode. This does
not load the model and returns stop commands:

```bash
python scripts/car/mac_server.py --port 9999 --mock_control --mock_action stop
```

Then run the Pi client without moving hardware:

```bash
python3 pi_client.py --server_ip <Mac_IP> --server_port 9999 --dry_run
```

After the protocol test passes, keep the car lifted and test one bounded mock
movement:

```bash
# Mac
python scripts/car/mac_server.py --port 9999 --mock_control --mock_action forward --mock_speed 150

# Pi
python3 pi_client.py --server_ip <Mac_IP> --server_port 9999
```

Stop both processes after a few seconds. The Pi client stops the motors on
Ctrl+C, disconnect, or socket timeout.

## Hardware Command Mapping

Keyboard teleop uses:

| key | action |
| --- | --- |
| `w` | forward |
| `s` | backward |
| `a` | turn left |
| `d` | turn right |
| `q` | strafe left |
| `e` | strafe right |
| `space` | stop |

The motor protocol follows the vendor sample:

```text
#006PxxxxT0000!#007PxxxxT0000!#008PxxxxT0000!#009PxxxxT0000!
```

Stop command:

```text
#255P1500T1000!
```

## First Smoke Test

Print the exact UART motor command without moving the car:

```bash
python3 scripts/car/move_test.py --move forward --speed 200 --duration 0.3
```

After the car is lifted or placed in a clear low-speed area, run one bounded
movement:

```bash
python3 scripts/car/move_test.py --move forward --speed 200 --duration 0.3 --execute
```

Run this on the Raspberry Pi without moving hardware:

```bash
python3 scripts/car/collect_data.py \
  --episode_name smoke_dry \
  --instruction "follow the person" \
  --teleop keyboard \
  --dry_run
```

Then run on hardware at low speed:

```bash
python3 scripts/car/collect_data.py \
  --episode_name ep001 \
  --instruction "follow the person in red shirt" \
  --teleop keyboard \
  --speed 200
```

Convert collected data:

```bash
python scripts/car/build_training_data.py \
  --input data/collected \
  --output data/car_train.jsonl
```

Generate PFEM pseudo-labels:

```bash
python harness/schedule/pseudo_labels/generate_all.py \
  --input data/car_train.jsonl \
  --output data/car_train_labeled.jsonl
```
