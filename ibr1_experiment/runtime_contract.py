"""Frozen production runtime identity for the official IBR1 lifecycle.

This module is intentionally standard-library-only so both the parent
orchestrator and each worker can validate the interpreter before importing
PyTorch-heavy lifecycle code.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


OFFICIAL_PYTHON_EXECUTABLE = Path(
    r"E:\anaconda\envs\pytorch\python.exe"
)
OFFICIAL_TORCH_VERSION = "2.6.0+cu124"
OFFICIAL_CUDA_RUNTIME = "12.4"
OFFICIAL_DEVICE = "cuda:0"


class IBR1RuntimeContractError(RuntimeError):
    """Raised when the official interpreter/CUDA identity has drifted."""


def _normalized_path(value: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(value)))


def require_official_python(
    executable: str | os.PathLike[str] | None = None,
) -> Path:
    """Require the one interpreter frozen for production IBR1 execution."""

    observed = sys.executable if executable is None else executable
    if _normalized_path(observed) != _normalized_path(OFFICIAL_PYTHON_EXECUTABLE):
        raise IBR1RuntimeContractError(
            "official IBR1 execution requires "
            f"{OFFICIAL_PYTHON_EXECUTABLE}, observed {Path(observed).resolve()}"
        )
    return OFFICIAL_PYTHON_EXECUTABLE


def require_official_torch_cuda(torch_module: Any) -> dict[str, Any]:
    """Require exact PyTorch/CUDA versions and select the frozen CUDA device."""

    torch_version = str(getattr(torch_module, "__version__", ""))
    version_namespace = getattr(torch_module, "version", None)
    cuda_runtime = str(getattr(version_namespace, "cuda", ""))
    if torch_version != OFFICIAL_TORCH_VERSION:
        raise IBR1RuntimeContractError(
            "official IBR1 execution requires torch "
            f"{OFFICIAL_TORCH_VERSION}, observed {torch_version!r}"
        )
    if cuda_runtime != OFFICIAL_CUDA_RUNTIME:
        raise IBR1RuntimeContractError(
            "official IBR1 execution requires CUDA runtime "
            f"{OFFICIAL_CUDA_RUNTIME}, observed {cuda_runtime!r}"
        )
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        raise IBR1RuntimeContractError(
            "official IBR1 execution requires cuda:0; CPU fallback is forbidden"
        )
    if int(cuda.device_count()) < 1:
        raise IBR1RuntimeContractError("official IBR1 cuda:0 is not visible")
    cuda.set_device(0)
    if int(cuda.current_device()) != 0:
        raise IBR1RuntimeContractError("official IBR1 device selection drifted from cuda:0")
    return {
        "python_executable": str(OFFICIAL_PYTHON_EXECUTABLE),
        "torch_version": torch_version,
        "cuda_runtime": cuda_runtime,
        "device": OFFICIAL_DEVICE,
    }


__all__ = [
    "IBR1RuntimeContractError",
    "OFFICIAL_CUDA_RUNTIME",
    "OFFICIAL_DEVICE",
    "OFFICIAL_PYTHON_EXECUTABLE",
    "OFFICIAL_TORCH_VERSION",
    "require_official_python",
    "require_official_torch_cuda",
]
