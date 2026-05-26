# Car Runtime

Runtime code that runs on the Raspberry Pi car or supports direct hardware
testing.

## Files

| File | Role |
| --- | --- |
| `pi_client.py` | Captures camera frames, sends them to the Mac server, receives command JSON, executes motors and pan/tilt. |
| `car_hardware.py` | Wraps vendor UART motor commands and pigpio pan/tilt PWM. |
| `uart_transport.py` | Opens the Raspberry Pi UART and sends vendor motor strings with pyserial. |
| `camera_check.py` | Measures camera open/read latency. |
| `hardware_check.py` | Checks UART and pigpio availability before real movement. |
| `kill_port.py` | Standalone command to clear a TCP port before startup. |
| `process_cleanup.py` | Kills stale vendor camera/main processes and stale TCP port listeners. |
| `car_protocol.py` | Shared length-prefixed TCP protocol. |
| `move_test.py` | Bounded single-action smoke test. |
| `speed_sweep.py` | Free speed threshold testing for real motor tuning. |

## Protocol Smoke Test

Clear a stale server port before startup:

```bash
python3 car_runtime/kill_port.py --port 9999 --dry_run
python3 car_runtime/kill_port.py --port 9999
```

`inference_pipeline/mac_server.py` already runs the same port cleanup by
default before it binds `--port`.

```bash
python3 car_runtime/pi_client.py \
  --server_ip <Mac_IP> \
  --server_port 9999 \
  --dry_run
```

`pi_client.py` cleans stale vendor processes before opening camera/hardware:

```text
mjpg
z_main
```

Preview cleanup targets without killing anything:

```bash
python3 car_runtime/pi_client.py \
  --server_ip <Mac_IP> \
  --server_port 9999 \
  --dry_run \
  --cleanup_dry_run
```

Disable this cleanup if needed:

```bash
python3 car_runtime/pi_client.py \
  --server_ip <Mac_IP> \
  --server_port 9999 \
  --dry_run \
  --no_cleanup_processes
```

If cleanup says permission denied, stop the process manually with `sudo`, or
run the client with sufficient permissions for that test.

## Direct Motor Test

Check camera startup time:

```bash
python3 car_runtime/camera_check.py
python3 car_runtime/camera_check.py --camera_backend v4l2
```

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

Short startup kick for weak low-speed starts:

```bash
python3 car_runtime/move_test.py \
  --move forward \
  --speed 160 \
  --kick_speed 350 \
  --kick_duration 0.06 \
  --duration 0.3 \
  --execute
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

## Startup Kick

Some motors do not move at low steady pulse deltas because static friction is
higher than rolling friction. Use a short kick instead of raising the whole
run speed:

```text
steady speed: 120-220
kick speed:   300-450
duration:     0.04-0.08 seconds
```

Start conservative. `kick_duration` is clamped to at most `0.25` seconds in
code.

`pi_client.py` also supports the same idea for commands received from the
Windows server:

```bash
python3 car_runtime/pi_client.py \
  --server_ip <Windows_IP> \
  --server_port 9999 \
  --kick_speed 350 \
  --kick_duration 0.06
```

## Speed Sweep

Use this when the motor spins but the car still cannot move under load. Start
with the car lifted, then repeat on the floor with very short durations.

Dry-run a custom speed list:

```bash
python3 car_runtime/speed_sweep.py \
  --move forward \
  --speeds 120,160,220,300,380,460 \
  --kick_speed 400
```

Real sweep:

```bash
python3 car_runtime/speed_sweep.py \
  --move forward \
  --speeds 160,220,300,380,460,540 \
  --duration 0.25 \
  --pause 0.8 \
  --kick_speed 420 \
  --kick_duration 0.06 \
  --execute \
  --confirm_each
```

If the car only starts moving at very high values, check battery voltage,
wheel friction, load, and whether the motor direction mapping is fighting
itself.

Both `move_test.py` and `speed_sweep.py` clear stale `mjpg`/`z_main` vendor
processes before testing. Preview with `--cleanup_dry_run`, or disable with
`--no_cleanup_processes`.
