# Car Runtime

Runtime code that runs on the Raspberry Pi car or supports direct hardware
testing.

## Files

| File | Role |
| --- | --- |
| `pi_client.py` | Captures camera frames, sends them to the Mac server, receives command JSON, executes motors and pan/tilt. |
| `car_hardware.py` | Wraps vendor UART motor commands and pigpio pan/tilt PWM. |
| `car_protocol.py` | Shared length-prefixed TCP protocol. |
| `move_test.py` | Bounded single-action smoke test. |

## Protocol Smoke Test

```bash
python3 car_runtime/pi_client.py \
  --server_ip <Mac_IP> \
  --server_port 9999 \
  --dry_run
```

## Direct Motor Test

Dry run:

```bash
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3
```

Real movement:

```bash
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3 --execute
```

Keep the car lifted or in a clear low-speed area before using `--execute`.
