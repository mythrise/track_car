"""Fail-closed CUDA reproducibility contract for the Windows F2 lifecycle.

The production CLI prepares the cuBLAS workspace policy before importing
torch-heavy assembly modules.  Model-side integration seams call the same
configuration again before constructing a CUDA CAL/smoke model, making the
contract idempotent and safe for direct programmatic use.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from types import MappingProxyType
from typing import Any


class F2CudaReproducibilityError(RuntimeError):
    """Raised when the frozen CUDA execution policy cannot be established."""


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
CUDA_REPRODUCIBILITY_CONTRACT_ID = "windows_cuda_deterministic_v1"
CUDA_REPRODUCIBILITY_SETTINGS = MappingProxyType(
    {
        "contract_id": CUDA_REPRODUCIBILITY_CONTRACT_ID,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "sdpa_backend": "math_only",
        "sdpa_flash_enabled": False,
        "sdpa_mem_efficient_enabled": False,
        "sdpa_cudnn_enabled": False,
        "sdpa_math_enabled": True,
    }
)
_CONFIGURED_TORCH_MODULES: dict[int, Any] = {}


def prepare_cublas_workspace_config() -> None:
    """Set the frozen cuBLAS policy before CUDA/cuBLAS initialization."""

    existing = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if existing not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise F2CudaReproducibilityError(
            "CUBLAS_WORKSPACE_CONFIG conflicts with the frozen Windows F2 "
            f"contract: {existing!r} != {CUBLAS_WORKSPACE_CONFIG!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG


def _torch_module(torch_module: Any | None) -> Any:
    if torch_module is not None:
        return torch_module
    import torch

    return torch


def cuda_reproducibility_receipt(torch_module: Any | None = None) -> dict[str, Any]:
    """Read and validate the live deterministic CUDA policy."""

    torch = _torch_module(torch_module)
    cuda_backend = torch.backends.cuda
    cudnn_backend = torch.backends.cudnn
    receipt = {
        "contract_id": CUDA_REPRODUCIBILITY_CONTRACT_ID,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": bool(cudnn_backend.benchmark),
        "cudnn_deterministic": bool(cudnn_backend.deterministic),
        "matmul_allow_tf32": bool(cuda_backend.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(cudnn_backend.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
        "sdpa_backend": "math_only",
        "sdpa_flash_enabled": bool(cuda_backend.flash_sdp_enabled()),
        "sdpa_mem_efficient_enabled": bool(
            cuda_backend.mem_efficient_sdp_enabled()
        ),
        "sdpa_cudnn_enabled": bool(cuda_backend.cudnn_sdp_enabled()),
        "sdpa_math_enabled": bool(cuda_backend.math_sdp_enabled()),
        "torch_version": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
    }
    return validate_cuda_reproducibility_receipt(receipt)


def validate_cuda_reproducibility_receipt(value: Any) -> dict[str, Any]:
    """Validate receipt settings without importing or inspecting CUDA data."""

    if not isinstance(value, Mapping):
        raise F2CudaReproducibilityError(
            "CUDA reproducibility receipt must be a mapping"
        )
    normalized = dict(value)
    for name, expected in CUDA_REPRODUCIBILITY_SETTINGS.items():
        if normalized.get(name) != expected:
            raise F2CudaReproducibilityError(
                f"CUDA reproducibility setting {name!r} differs from the "
                f"frozen contract: {normalized.get(name)!r} != {expected!r}"
            )
    for name in ("torch_version", "cuda_runtime"):
        if not isinstance(normalized.get(name), str) or not normalized[name]:
            raise F2CudaReproducibilityError(
                f"CUDA reproducibility receipt {name!r} must be nonempty"
            )
    expected_keys = set(CUDA_REPRODUCIBILITY_SETTINGS) | {
        "torch_version",
        "cuda_runtime",
    }
    if set(normalized) != expected_keys:
        raise F2CudaReproducibilityError(
            "CUDA reproducibility receipt fields differ from the frozen contract"
        )
    return {name: normalized[name] for name in sorted(normalized)}


def configure_cuda_reproducibility(
    torch_module: Any | None = None,
) -> dict[str, Any]:
    """Apply the deterministic CUDA policy and return its receipt."""

    torch = _torch_module(torch_module)
    module_id = id(torch)
    configured_here = _CONFIGURED_TORCH_MODULES.get(module_id) is torch
    if (
        bool(torch.cuda.is_initialized())
        and not configured_here
    ):
        raise F2CudaReproducibilityError(
            "CUDA was initialized before the frozen Windows F2 "
            "reproducibility contract; deterministic policy is too late"
        )
    prepare_cublas_workspace_config()
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    receipt = cuda_reproducibility_receipt(torch)
    _CONFIGURED_TORCH_MODULES[module_id] = torch
    return receipt


__all__ = [
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_REPRODUCIBILITY_CONTRACT_ID",
    "CUDA_REPRODUCIBILITY_SETTINGS",
    "F2CudaReproducibilityError",
    "configure_cuda_reproducibility",
    "cuda_reproducibility_receipt",
    "prepare_cublas_workspace_config",
    "validate_cuda_reproducibility_receipt",
]
