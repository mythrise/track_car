#!/usr/bin/env python3
"""UART transport for the vendor motor controller protocol."""

from __future__ import annotations

import os
import time
from pathlib import Path

try:
    import serial  # type: ignore
except ImportError:  # pragma: no cover - only installed on Raspberry Pi/runtime
    serial = None


DEFAULT_UART_PORTS = (
    "/dev/ttyAMA0",
    "/dev/serial0",
    "/dev/ttyS0",
)


def resolve_uart_port(port: str | None = None) -> str:
    if port:
        return port

    env_port = os.environ.get("CAR_UART_PORT", "").strip()
    if env_port:
        return env_port

    for candidate in DEFAULT_UART_PORTS:
        if Path(candidate).exists():
            return candidate
    return DEFAULT_UART_PORTS[0]


class UartTransport:
    """Small pyserial wrapper matching the vendor `uart_send_str` behavior."""

    def __init__(self, baud: int = 115200, port: str | None = None, dry_run: bool = False):
        self.baud = baud
        self.port = resolve_uart_port(port)
        self.dry_run = dry_run
        self.ser = None

        if self.dry_run:
            print(f"[uart] dry-run mode port={self.port} baud={self.baud}")
            return

        if serial is None:
            raise RuntimeError(
                "pyserial is not installed. Install it on Raspberry Pi with "
                "`sudo apt install -y python3-serial` or `python3 -m pip install pyserial`."
            )

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.1)
            self._flush_input()
            print(f"[uart] opened {self.port} baud={self.baud}")
        except Exception as exc:
            raise RuntimeError(
                f"Could not open UART port {self.port} at {self.baud} baud. "
                "Set --uart_port or CAR_UART_PORT if your car uses another device."
            ) from exc

    def _flush_input(self) -> None:
        if self.ser is None:
            return
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()
        elif hasattr(self.ser, "flushInput"):
            self.ser.flushInput()

    def send_str(self, command: str) -> None:
        if self.dry_run:
            print(f"  [dry-run] uart: {command}")
            return
        if self.ser is None:
            raise RuntimeError("UART is not open")
        self.ser.write(command.encode("utf-8"))
        time.sleep(0.01)
        self._flush_input()

    def close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None
