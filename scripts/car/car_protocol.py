#!/usr/bin/env python3
"""Length-prefixed TCP protocol shared by Mac server and Raspberry Pi client."""

from __future__ import annotations

import json
import socket
import struct
from typing import Any


MAX_BLOB_BYTES = 32 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024


class ProtocolError(RuntimeError):
    pass


def recv_exact(sock: socket.socket, n_bytes: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < n_bytes:
        chunk = sock.recv(n_bytes - len(chunks))
        if not chunk:
            return None
        chunks.extend(chunk)
    return bytes(chunks)


def send_blob(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack(">I", len(data)) + data)


def recv_blob(sock: socket.socket, max_bytes: int = MAX_BLOB_BYTES) -> bytes | None:
    raw_len = recv_exact(sock, 4)
    if raw_len is None:
        return None
    length = struct.unpack(">I", raw_len)[0]
    if length > max_bytes:
        raise ProtocolError(f"message too large: {length} > {max_bytes}")
    return recv_exact(sock, length)


def send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    send_blob(sock, data)


def recv_json(sock: socket.socket) -> dict[str, Any] | None:
    data = recv_blob(sock, max_bytes=MAX_JSON_BYTES)
    if data is None:
        return None
    obj = json.loads(data.decode("utf-8"))
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload must be an object")
    return obj


def encode_jpeg(frame, quality: int = 70) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ProtocolError("failed to JPEG-encode frame")
    return buf.tobytes()


def send_jpeg_frame(sock: socket.socket, frame, quality: int = 70) -> None:
    send_blob(sock, encode_jpeg(frame, quality=quality))


def recv_jpeg_frame(sock: socket.socket):
    import cv2
    import numpy as np

    data = recv_blob(sock)
    if data is None:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ProtocolError("failed to JPEG-decode frame")
    return frame
