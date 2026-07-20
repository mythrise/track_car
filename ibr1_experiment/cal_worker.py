"""Official isolated worker for one IBR1 CAL role.

The module stays stdlib-only at import time.  Its CLI establishes the frozen
cuBLAS and CUDA reproducibility policy before importing the calibration
lifecycle, then writes exactly one canonical JSON result to stdout.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import sys
from typing import Any

from .runtime_contract import (
    require_official_python,
    require_official_torch_cuda,
)


WORKER_RESULT_CLASS = "ibr1_cal_worker_result"


class IBR1CalWorkerError(RuntimeError):
    """Raised when the official CAL worker cannot start safely."""


def _challenge(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("parent challenge must be 64 lowercase hex characters")
    return value


def _positive_pid(value: str) -> int:
    try:
        pid = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("parent PID must be an integer") from exc
    if pid <= 0:
        raise argparse.ArgumentTypeError("parent PID must be positive")
    return pid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ibr1_experiment.cal_worker")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--role", required=True, choices=("main", "reproduction"))
    parser.add_argument("--bootstrap-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-challenge", required=True, type=_challenge)
    parser.add_argument("--parent-pid", required=True, type=_positive_pid)
    return parser


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    require_official_python()
    args = build_parser().parse_args(argv)
    if args.parent_pid == os.getpid():
        raise IBR1CalWorkerError("parent PID cannot equal the CAL worker PID")

    root = Path(args.project_root).expanduser().resolve()
    bootstrap = Path(args.bootstrap_receipt).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    os.environ["IBR1_CAL_PARENT_CHALLENGE"] = args.parent_challenge
    os.environ["IBR1_CAL_PARENT_PID"] = str(args.parent_pid)

    # Redirect all subordinate/model chatter to stderr so stdout contains one
    # and only one canonical machine result.
    with redirect_stdout(sys.stderr):
        from f2_experiment.reproducibility import (
            configure_cuda_reproducibility,
            prepare_cublas_workspace_config,
        )

        prepare_cublas_workspace_config()
        import torch

        runtime = require_official_torch_cuda(torch)
        configure_cuda_reproducibility(torch)

        # Import only after cuBLAS and CUDA policy establishment.
        from .calibration import run_ibr1_cal_audit_once

        result = run_ibr1_cal_audit_once(
            root,
            role=args.role,
            bootstrap_receipt_path=bootstrap,
            output_dir=output,
        )

    payload = {
        "schema_version": 1,
        "analysis_class": WORKER_RESULT_CLASS,
        "role": args.role,
        "parent_challenge": args.parent_challenge,
        "parent_pid": args.parent_pid,
        "child_pid": os.getpid(),
        "output_dir": str(output),
        "runtime": runtime,
        "calibration_result": result,
    }
    sys.stdout.write(_canonical_json(payload) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
