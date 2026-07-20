"""Native-Windows command line entry points for the IBR1 authority chain.

The module is intentionally standard-library-only at import time.  Every
command establishes the frozen cuBLAS environment before importing PyTorch
or any IBR1 module that imports PyTorch transitively.  Model execution stays
in the dedicated CAL workers and, later, the smoke lifecycle command.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib
import json
import os
from pathlib import Path
import sys
from typing import Any

from .runtime_contract import (
    IBR1RuntimeContractError,
    require_official_python,
    require_official_torch_cuda,
)


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


class IBR1CliError(RuntimeError):
    """Raised when the native-Windows IBR1 CLI cannot fail closed."""


def _prepare_pre_torch_environment() -> None:
    """Freeze the only process environment setting required pre-import."""

    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise IBR1CliError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen IBR1 "
            f"contract: {existing!r} != {CUBLAS_WORKSPACE_CONFIG!r}"
        )
    if "torch" in sys.modules and existing != CUBLAS_WORKSPACE_CONFIG:
        raise IBR1CliError(
            "PyTorch was imported before the CLI established the frozen "
            "cuBLAS workspace contract"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


def _import_module(name: str) -> Any:
    return importlib.import_module(name)


def _configure_runtime(*, require_cuda: bool) -> dict[str, Any]:
    """Configure deterministic PyTorch only after the pre-import guard."""

    _prepare_pre_torch_environment()
    reproducibility = _import_module("f2_experiment.reproducibility")
    torch = _import_module("torch")
    try:
        if require_cuda:
            require_official_python()
            require_official_torch_cuda(torch)
    except IBR1RuntimeContractError as exc:
        raise IBR1CliError(str(exc)) from exc
    receipt = reproducibility.configure_cuda_reproducibility(torch)
    cuda_available = bool(torch.cuda.is_available())
    if require_cuda and not cuda_available:
        raise IBR1CliError(
            "this IBR1 authority command requires CUDA; CPU fallback is "
            "forbidden"
        )
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": os.name,
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": cuda_available,
        "device": "cuda:0" if cuda_available else None,
        "device_name": device_name,
        "cuda_reproducibility": receipt,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ibr1_experiment.cli",
        description=(
            "Build and verify the fixed IBR1 Windows/CUDA authority chain. "
            "No command authorizes formal training."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--project-root", default=".")

    bootstrap = subparsers.add_parser("build-bootstrap")
    bootstrap.add_argument("--project-root", default=".")
    bootstrap.add_argument("--output", required=True)

    cal_pair = subparsers.add_parser("run-cal-pair")
    cal_pair.add_argument("--project-root", default=".")
    cal_pair.add_argument("--bootstrap-receipt", required=True)
    cal_pair.add_argument("--output-dir", required=True)
    cal_pair.add_argument("--freeze-output", required=True)
    cal_pair.add_argument("--final-output", required=True)

    verify = subparsers.add_parser("verify-assembly")
    verify.add_argument("--project-root", default=".")
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--phase", choices=("bootstrap", "final"), default=None)
    return parser


def _print_result(value: Any) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_cuda = args.command != "verify-assembly"
    runtime = _configure_runtime(require_cuda=require_cuda)
    root = Path(args.project_root).expanduser().resolve()

    if args.command == "run-cal-pair":
        cal_pair = _import_module("ibr1_experiment.cal_pair")
        result = cal_pair.run_live_cal_pair_and_freeze(
            root,
            bootstrap_receipt_path=args.bootstrap_receipt,
            output_dir=args.output_dir,
            freeze_output_path=args.freeze_output,
            final_output_path=args.final_output,
        )
        printable_result = {
            key: value
            for key, value in result.items()
            if key != "final_authority_capability"
        }
        _print_result(
            {
                **printable_result,
                "runtime": runtime,
                "formal_training_authorized": False,
            }
        )
        return 0

    authority = _import_module("ibr1_experiment.authority")
    if args.command == "preflight":
        chain = authority.verify_authority_chain(root)
        negative = authority.verify_f2_negative_evidence(root)
        _print_result(
            {
                "analysis_class": "ibr1_windows_cuda_preflight",
                "family_id": authority.IBR1_FAMILY_ID,
                "architecture_lock": authority.IBR1_ARCHITECTURE_LOCK,
                "authority_chain_payload_sha256": chain[
                    "receipt_payload_sha256"
                ],
                "f2_negative_seal": negative["negative_seal"],
                "runtime": runtime,
                "formal_training_authorized": False,
                "internal_test": "sealed",
                "internal_test_opened": False,
            }
        )
        return 0
    if args.command == "build-bootstrap":
        result = authority.freeze_assembly_receipt(
            root,
            args.output,
            phase=authority.ASSEMBLY_PHASE_BOOTSTRAP,
        )
        _print_result(
            {
                **result,
                "runtime": runtime,
                "formal_training_authorized": False,
            }
        )
        return 0
    document = authority.verify_assembly_receipt(
        root,
        args.receipt,
        required_phase=args.phase,
    )
    _print_result(
        {
            "path": str(Path(args.receipt).expanduser().resolve()),
            "analysis_class": document["analysis_class"],
            "phase": document["phase"],
            "receipt_payload_sha256": document["receipt_payload_sha256"],
            "runtime": runtime,
            "formal_training_authorized": False,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
