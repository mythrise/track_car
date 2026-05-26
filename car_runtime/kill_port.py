#!/usr/bin/env python3
"""Kill a process listening on a TCP port."""

from __future__ import annotations

import argparse

try:
    from process_cleanup import cleanup_port
except ImportError:
    from car_runtime.process_cleanup import cleanup_port


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9999, help="TCP port to clear before startup.")
    ap.add_argument("--dry_run", action="store_true", help="Print target processes without killing them.")
    args = ap.parse_args()

    cleanup_port(args.port, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
