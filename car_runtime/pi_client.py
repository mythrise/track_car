#!/usr/bin/env python3
"""Raspberry Pi C3 car client — captures frames, sends to Mac server, executes commands.

Run on the Raspberry Pi:
    python3 car_runtime/pi_client.py --server_ip 192.168.12.100 --server_port 9999
"""

import argparse
import cv2
import socket
import sys
import time

try:
    from camera_source import BACKENDS, open_camera
    from car_hardware import MAX_SPEED, CarHardware, NEUTRAL, boosted_motors, motor_delta
    from car_protocol import recv_json, send_jpeg_frame, send_json
    from process_cleanup import cleanup_named_processes
except ImportError:
    from car_runtime.camera_source import BACKENDS, open_camera
    from car_runtime.car_hardware import MAX_SPEED, CarHardware, NEUTRAL, boosted_motors, motor_delta
    from car_runtime.car_protocol import recv_json, send_jpeg_frame, send_json
    from car_runtime.process_cleanup import cleanup_named_processes


def setup_hardware(dry_run=False, uart_port=None, reset_servos=False):
    return CarHardware(reset_servos=reset_servos, dry_run=dry_run, uart_port=uart_port)


def connect_server(server_ip, server_port, connect_timeout, frame_timeout, hello):
    try:
        sock = socket.create_connection((server_ip, server_port), timeout=connect_timeout)
    except socket.timeout as exc:
        raise RuntimeError(
            f"Timed out connecting to {server_ip}:{server_port}. "
            "Check Windows server, IPv4 address, same LAN/hotspot, and firewall."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not connect to {server_ip}:{server_port}: {exc}. "
            "Check Windows server, IPv4 address, same LAN/hotspot, and firewall."
        ) from exc

    sock.settimeout(frame_timeout)
    send_json(sock, hello)
    print(f"[pi_client] connected to {server_ip}:{server_port}")
    return sock


def motor_direction(motors):
    direction = []
    for value in motors:
        delta = int(value) - NEUTRAL
        if delta > 20:
            direction.append(1)
        elif delta < -20:
            direction.append(-1)
        else:
            direction.append(0)
    return tuple(direction)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_ip", default="192.168.12.100")
    ap.add_argument("--server_port", type=int, default=9999)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--camera_index", type=int, default=0)
    ap.add_argument("--camera_backend", choices=BACKENDS, default="auto")
    ap.add_argument("--camera_fourcc", default="auto", help="OpenCV/V4L2 pixel format, for example MJPG or YUYV.")
    ap.add_argument("--camera_warmup", type=float, default=1.0)
    ap.add_argument("--instruction", default="follow the person")
    ap.add_argument("--connect_timeout", type=float, default=5.0)
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--jpeg_quality", type=int, default=70)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--reset_servos", action="store_true", help="Reset pan/tilt servos on startup.")
    ap.add_argument("--kick_speed", type=int, default=0,
                    help="Optional short startup kick pulse delta for received motor commands. Use 0 to disable.")
    ap.add_argument("--kick_duration", type=float, default=0.06,
                    help="Kick duration in seconds, clamped to 0.25.")
    ap.add_argument("--kick_repeat", type=float, default=0.75,
                    help="Minimum seconds between repeated kicks while receiving the same direction.")
    ap.add_argument("--no_cleanup_processes", action="store_true",
                    help="Do not kill vendor camera/main processes before opening hardware.")
    ap.add_argument("--cleanup_dry_run", action="store_true",
                    help="Print cleanup targets without killing them.")
    args = ap.parse_args()

    hello = {
        "type": "hello",
        "protocol": 1,
        "instruction": args.instruction,
        "width": args.width,
        "height": args.height,
    }

    try:
        sock = connect_server(
            args.server_ip,
            args.server_port,
            args.connect_timeout,
            args.timeout,
            hello,
        )
    except RuntimeError as exc:
        print(f"[pi_client] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    hardware = None
    cap = None
    try:
        if not args.no_cleanup_processes:
            print("[startup] cleaning stale vendor processes", flush=True)
            cleanup_named_processes(["mjpg", "z_main"], dry_run=args.cleanup_dry_run)
            if args.cleanup_dry_run:
                return

        cap = open_camera(
            args.camera_index,
            args.camera_backend,
            args.width,
            args.height,
            max(0.0, min(args.camera_warmup, 5.0)),
            fourcc=args.camera_fourcc,
        )

        print("[startup] opening car hardware", flush=True)
        hardware = setup_hardware(
            dry_run=args.dry_run,
            uart_port=args.uart_port,
            reset_servos=args.reset_servos,
        )
        print("[startup] car hardware ready", flush=True)

        frame_idx = 0
        last_direction = motor_direction([NEUTRAL, NEUTRAL, NEUTRAL, NEUTRAL])
        last_kick_time = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, -1)
            send_jpeg_frame(sock, frame, quality=args.jpeg_quality)

            cmd = recv_json(sock)
            if cmd is None:
                break

            # Execute motor command
            motors = cmd.get("motors", [1500, 1500, 1500, 1500])
            now = time.time()
            direction = motor_direction(motors)
            kick_speed = max(0, min(args.kick_speed, MAX_SPEED))
            kick_duration = max(0.0, min(args.kick_duration, 0.25))
            direction_changed = direction != last_direction
            repeat_due = args.kick_repeat > 0 and (now - last_kick_time) >= args.kick_repeat
            should_kick = (
                kick_speed > 0
                and motor_delta(motors) > 0
                and kick_duration > 0
                and (direction_changed or repeat_due)
            )
            if should_kick:
                hardware.run_motors_with_kick(motors, boosted_motors(motors, kick_speed), kick_duration)
                last_kick_time = now
            else:
                hardware.run_motors(motors)
            last_direction = direction

            # Execute pan-tilt
            if "pan" in cmd:
                hardware.set_pan_pulse(int(cmd["pan"]))
            if "tilt" in cmd:
                hardware.set_tilt_pulse(int(cmd["tilt"]))

            # Safety: ultrasonic check (if available)
            if cmd.get("stop"):
                hardware.stop()

            frame_idx += 1
            if frame_idx % 30 == 0:
                print(f"[pi_client] frame={frame_idx} server_fps={cmd.get('fps')}")

    except socket.timeout:
        print("[pi_client] socket timeout; stopping hardware")
        if hardware is not None:
            hardware.stop()
    except (ConnectionError, OSError) as exc:
        print(f"[pi_client] connection error: {exc}; stopping hardware")
        if hardware is not None:
            hardware.stop()
    except KeyboardInterrupt:
        pass
    finally:
        if hardware is not None:
            hardware.close()
        if cap is not None:
            cap.release()
        sock.close()
        print("[pi_client] shutdown.")


if __name__ == "__main__":
    main()
