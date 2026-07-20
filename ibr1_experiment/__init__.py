"""IBR1 successor experiment without mutating the sealed F2 namespace.

The package root intentionally stays free of ``torch`` imports.  Native
Windows command-line entry points must establish the frozen cuBLAS/CUDA
reproducibility policy before any model module imports PyTorch.  Public
symbols therefore use PEP 562 lazy attribute loading.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "ENGINE_TO_FAMILY_ARM": (".assembly_model", "ENGINE_TO_FAMILY_ARM"),
    "FAMILY_TO_ENGINE_ARM": (".assembly_model", "FAMILY_TO_ENGINE_ARM"),
    "F2SealedInitEvidence": (".assembly_model", "F2SealedInitEvidence"),
    "GeometryCollector": (".diagnostics", "GeometryCollector"),
    "GradientDiagnosticsCollector": (
        ".diagnostics",
        "GradientDiagnosticsCollector",
    ),
    "IBR1_ARCHITECTURE_LOCK": (".model", "IBR1_ARCHITECTURE_LOCK"),
    "IBR1AssemblyContractError": (
        ".assembly_model",
        "IBR1AssemblyContractError",
    ),
    "IBR1DiagnosticsContractError": (
        ".diagnostics",
        "IBR1DiagnosticsContractError",
    ),
    "IBR1_FAMILY_ID": (".model", "IBR1_FAMILY_ID"),
    "IBR1G6Instrument": (".diagnostics", "IBR1G6Instrument"),
    "IBR1PairedArms": (".assembly_model", "IBR1PairedArms"),
    "IBR1AP2Model": (".model", "IBR1AP2Model"),
    "IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS": (
        ".assembly_model",
        "IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS",
    ),
    "IBR1_CTRL": (".assembly_model", "IBR1_CTRL"),
    "IBR1_FROZEN_AUX_COEFFICIENTS": (
        ".assembly_model",
        "IBR1_FROZEN_AUX_COEFFICIENTS",
    ),
    "IBR1ModelContractError": (".model", "IBR1ModelContractError"),
    "IBR1NormalizedBoundedHead": (".model", "IBR1NormalizedBoundedHead"),
    "IBR1Prediction": (".model", "IBR1Prediction"),
    "IBR1_SELF": (".assembly_model", "IBR1_SELF"),
    "OptimizerDiagnosticsHandle": (
        ".diagnostics",
        "OptimizerDiagnosticsHandle",
    ),
    "build_ibr1_package": (".assembly_model", "build_ibr1_package"),
    "build_ibr1_paired_arms": (
        ".assembly_model",
        "build_ibr1_paired_arms",
    ),
    "normalized_cumulative_decode": (".model", "normalized_cumulative_decode"),
    "read_sealed_f2_init_evidence": (
        ".assembly_model",
        "read_sealed_f2_init_evidence",
    ),
    "run_live_cal_pair_and_freeze": (
        ".cal_pair",
        "run_live_cal_pair_and_freeze",
    ),
    "run_authoritative_smoke": (
        ".lifecycle",
        "run_authoritative_smoke",
    ),
    "wrap_eval_predictor": (".diagnostics", "wrap_eval_predictor"),
    "wrap_training_head_forward": (
        ".diagnostics",
        "wrap_training_head_forward",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load one public model/assembly symbol only when it is requested."""

    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
