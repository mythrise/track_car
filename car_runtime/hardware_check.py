#!/usr/bin/env python3
"""Check Raspberry Pi car runtime hardware dependencies."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pigpio  # type: ignore
except ImportError:  # pragma: no cover
    pigpio = None

try:
    from uart_transport import UartTransport, resolve_uart_port
except ImportError:
    from car_runtime.uart_transport import UartTransport, resolve_uart_port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uart_port", default=None, help="UART device, for example /dev/ttyAMA0 or /dev/serial0.")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--open_uart", action="store_true", help="Actually open the UART port, but do not send commands.")
    args = ap.parse_args()

    port = resolve_uart_port(args.uart_port)
    print(f"uart_port: {port} exists={Path(port).exists()}")

    if args.open_uart:
        uart = UartTransport(baud=args.baud, port=port, dry_run=False)
        uart.close()
        print("uart_open: ok")
    else:
        print("uart_open: skipped; pass --open_uart to test")

    if pigpio is None:
        print("pigpio: not installed")
    else:
        pi = pigpio.pi()
        print(f"pigpio: installed connected={getattr(pi, 'connected', False)}")
        if getattr(pi, "connected", False):
            pi.stop()


if __name__ == "__main__":
    main()
