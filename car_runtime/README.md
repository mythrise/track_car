# Car Runtime

Runtime code that runs on the Raspberry Pi car or supports direct hardware
testing.

## Files

| File | Role |
| --- | --- |
| `pi_client.py` | Captures camera frames, sends them to the Mac server, receives command JSON, executes motors and pan/tilt. |
| `car_hardware.py` | Wraps vendor UART motor commands and pigpio pan/tilt PWM. |
| `uart_transport.py` | Opens the Raspberry Pi UART and sends vendor motor strings with pyserial. |
| `hardware_check.py` | Checks UART and pigpio availability before real movement. |
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

Check runtime dependencies:

```bash
python3 car_runtime/hardware_check.py
python3 car_runtime/hardware_check.py --open_uart
```

Dry run:

```bash
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3
```

Real movement:

```bash
python3 car_runtime/move_test.py --move forward --speed 200 --duration 0.3 --execute
```

Keep the car lifted or in a clear low-speed area before using `--execute`.

If your Raspberry Pi uses a different UART device:

```bash
python3 car_runtime/move_test.py \
  --move forward \
  --speed 120 \
  --duration 0.2 \
  --execute \
  --uart_port /dev/serial0
```

## Hardware Dependencies For Real Movement

Dry-run mode works without motor hardware dependencies. Real movement requires:

```text
python3-serial or pyserial
UART device such as /dev/ttyAMA0 or /dev/serial0
pigpio and pigpiod only if pan/tilt servos are used
```

Install runtime packages on Raspberry Pi:

```bash
sudo apt update
sudo apt install -y python3-serial pigpio python3-pigpio
sudo usermod -aG dialout $USER
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

After changing the `dialout` group, log out and back in, or reboot. If
`hardware_check.py --open_uart` still cannot open the port, enable the serial
interface with `sudo raspi-config`: disable serial login shell, enable serial
hardware.

The project now sends the same UART protocol shown in the vendor infrared
remote example directly through `uart_transport.py`; a separate vendor
`z_uart.py` file is no longer required for our runtime.
