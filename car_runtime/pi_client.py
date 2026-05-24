#!/usr/bin/env python3
"""Raspberry Pi C3 car client — captures frames, sends to Mac server, executes commands.

Run on the Raspberry Pi:
    python3 car_runtime/pi_client.py --server_ip 192.168.12.100 --server_port 9999
"""

import argparse
import cv2
import socket
import time

try:
    from car_hardware import CarHardware
    from car_protocol import recv_json, send_jpeg_frame, send_json
except ImportError:
    from car_runtime.car_hardware import CarHardware
    from car_runtime.car_protocol import recv_json, send_jpeg_frame, send_json


def setup_hardware(dry_run=False, uart_port=None, reset_servos=False):
    return CarHardware(reset_servos=reset_servos, dry_run=dry_run, uart_port=uart_port)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server_ip", default="192.168.12.100")
    ap.add_argument("--server_port", type=int, default=9999)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--instruction", default="follow the person")
    ap.add_argument("--timeout", type=float, default=0.75)
    ap.add_argument("--jpeg_quality", type=int, default=70)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--reset_servos", action="store_true", help="Reset pan/tilt servos on startup.")
    args = ap.parse_args()

    hardware = setup_hardware(
        dry_run=args.dry_run,
        uart_port=args.uart_port,
        reset_servos=args.reset_servos,
    )
    cap = cv2.VideoCapture(0)
    cap.set(3, args.width)
    cap.set(4, args.height)
    time.sleep(1.0)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)
    sock.connect((args.server_ip, args.server_port))
    send_json(sock, {
        "type": "hello",
        "protocol": 1,
        "instruction": args.instruction,
        "width": args.width,
        "height": args.height,
    })
    print(f"[pi_client] connected to {args.server_ip}:{args.server_port}")

    frame_idx = 0
    try:
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
            hardware.run_motors(motors)

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
        hardware.stop()
    except (ConnectionError, OSError) as exc:
        print(f"[pi_client] connection error: {exc}; stopping hardware")
        hardware.stop()
    except KeyboardInterrupt:
        pass
    finally:
        hardware.close()
        cap.release()
        sock.close()
        print("[pi_client] shutdown.")


if __name__ == "__main__":
    main()
