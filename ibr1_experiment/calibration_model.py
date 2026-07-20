"""Real model-side IBR1 CAL row auditor over the frozen F2 observation path.

The lifecycle runner in :mod:`ibr1_experiment.calibration` owns receipt
emission.  This module owns the model-side callback passed to that runner.  It
uses a frozen F2 :class:`CalRowAuditor` for the subordinate evidence and a
separately initialized IBR1 arm for the authoritative FP32 geometry checks.
Neither arm owns an optimizer and no update method is exposed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

from f2_experiment.assembly import CalRowAudit
from f2_experiment.assembly_model import (
    CalRowAuditor,
    F2ArmModules,
    _assert_row_targets_not_aliased,
    _extract_observation,
    _observation_storage_ptrs,
    build_cal_audit_context,
    default_device,
    load_base_checkpoint,
)
from f2_experiment.model import (
    AP2_HORIZON,
    F2ModelContractError,
    assert_prev_free_tensors,
)
from f2_experiment.reproducibility import configure_cuda_reproducibility
from f2_experiment.runner import RunnerRow, checkpoint_init_sha256

from .assembly_model import (
    F2SealedInitEvidence,
    IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    IBR1_PACKAGE,
    build_ibr1_package,
    read_sealed_f2_init_evidence,
)
from . import calibration as _calibration_lifecycle
from .calibration import IBR1CalRowAudit
from .model import IBR1Prediction


IBR1_CAL_SEED = 0
IBR1_CAL_PROBE_SURFACE = "base.proj"
IBR1_CAL_GEOMETRY_DTYPE = torch.float32
IBR1_CAL_CONTROLLED_SHAPE = (AP2_HORIZON, 2)
IBR1_CAL_CONTROLLED_CELLS = AP2_HORIZON * 2
IBR1_CAL_RECONSTRUCTION_THRESHOLD = 1e-6


class IBR1CalibrationModelContractError(RuntimeError):
    """Raised when the executable IBR1 CAL model contract must stop."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IBR1CalibrationModelContractError(message)


def _finite_max_abs(value: torch.Tensor, label: str) -> float:
    _require(isinstance(value, torch.Tensor), f"{label} must be a tensor")
    _require(value.numel() > 0, f"{label} must be nonempty")
    detached = value.detach()
    _require(
        bool(torch.isfinite(detached).all().to(device="cpu").item()),
        f"{label} contains nonfinite values",
    )
    result = float(torch.max(torch.abs(detached)).to("cpu", torch.float64).item())
    _require(math.isfinite(result), f"{label} maximum is nonfinite")
    return result


class IBR1ModelCalRowAuditor:
    """Stateful real-row callback for ``run_ibr1_cal_audit_once``.

    The subordinate F2 callback and the IBR1 geometry arm are disjoint but
    initialized from byte-identical copies of the same base.  The returned
    ``subordinate_audit`` is exactly the object emitted by the frozen F2
    :class:`CalRowAuditor`; all IBR1 fields are measured from an actual adapter
    encode and model forward on the same :class:`RunnerRow`.
    """

    def __init__(
        self,
        *,
        project_root: Path,
        subordinate_auditor: CalRowAuditor,
        ibr1_arm: F2ArmModules,
        sealed_evidence: F2SealedInitEvidence,
    ) -> None:
        _require(
            isinstance(subordinate_auditor, CalRowAuditor),
            "subordinate auditor must be the frozen F2 CalRowAuditor",
        )
        _require(isinstance(ibr1_arm, F2ArmModules), "IBR1 arm is malformed")
        _require(ibr1_arm.package == IBR1_PACKAGE, "IBR1 CAL package drifted")
        _require(
            isinstance(sealed_evidence, F2SealedInitEvidence),
            "sealed F2 init evidence is malformed",
        )
        self.project_root = project_root
        self.subordinate_auditor = subordinate_auditor
        self.ibr1_arm = ibr1_arm
        self.sealed_evidence = sealed_evidence
        self.optimizer_objects = 0
        self.optimizer_updates = 0
        self._device = next(ibr1_arm.model.parameters()).device
        self._state: Mapping[str, Any] | None = None
        self._position = -1
        self._f2_init_sha256 = subordinate_auditor.initial_state_sha256
        self._ibr1_init_sha256 = checkpoint_init_sha256(
            ibr1_arm.full_state_dict()
        )
        self._assert_init_binding()

    def _assert_init_binding(self) -> None:
        live = read_sealed_f2_init_evidence(self.project_root)
        _require(
            live == self.sealed_evidence,
            "live sealed F2 init evidence drifted after CAL construction",
        )
        _require(
            live.seed == IBR1_CAL_SEED,
            "sealed F2 comparator seed is not zero",
        )
        _require(
            self.subordinate_auditor.seed == IBR1_CAL_SEED,
            "subordinate F2 CAL seed is not zero",
        )
        current_f2 = checkpoint_init_sha256(
            self.subordinate_auditor.arm.full_state_dict()
        )
        current_ibr1 = checkpoint_init_sha256(self.ibr1_arm.full_state_dict())
        _require(
            current_f2
            == self._f2_init_sha256
            == self._ibr1_init_sha256
            == current_ibr1
            == live.checkpoint_init_sha256,
            "F2/IBR1 CAL initialization bytes differ from sealed update-0",
        )

    def context_receipt(self) -> dict[str, Any]:
        """Return the standard F2-compatible seed/device/init receipt."""

        self._assert_init_binding()
        context = self.subordinate_auditor.context_receipt()
        _require(
            context.get("seed") == IBR1_CAL_SEED
            and context.get("device") == str(self._device)
            and context.get("package") == IBR1_PACKAGE
            and context.get("probe_surface") == IBR1_CAL_PROBE_SURFACE
            and context.get("checkpoint_init_sha256")
            == self.sealed_evidence.checkpoint_init_sha256,
            "IBR1 CAL context receipt drifted",
        )
        return context

    def _ibr1_geometry(
        self,
        row: RunnerRow,
        reasons: Sequence[str],
        position: int,
    ) -> tuple[str, bool, float, int, float, bool]:
        reset = bool(tuple(reasons))
        if position == 0:
            _require(reset, "IBR1 CAL position 0 must carry a reset reason")
            self._assert_init_binding()
            self._state = self.ibr1_arm.adapter.init_state(1, self._device)
            self._position = 0
        else:
            _require(
                self._state is not None and position == self._position + 1,
                "IBR1 CAL geometry position discontinuity",
            )
            self._position = position

        prev_leaf = torch.tensor(
            [[row.logged_prev_action[0], row.logged_prev_action[2]]],
            dtype=IBR1_CAL_GEOMETRY_DTYPE,
            device=self._device,
            requires_grad=True,
        )
        extracted = _extract_observation(row.observation)
        _assert_row_targets_not_aliased(
            row,
            _observation_storage_ptrs(extracted),
            self.ibr1_arm.adapter,
            f"IBR1 CAL position {position}",
        )
        output = self.ibr1_arm.adapter.encode_step(
            extracted["coarse_tokens"],
            extracted["coarse_tidx"],
            extracted["fine_tokens"],
            extracted["fine_tidx"],
            [extracted["instruction"]],
            self._state,
            reset_mask=reset,
            yaw_hist=extracted.get("yaw_hist"),
            yaw_curr=extracted.get("yaw_curr"),
        )
        self._state = output["new_state"]
        reference = output["h_act"]
        _require(
            reference.dtype == IBR1_CAL_GEOMETRY_DTYPE,
            "authoritative IBR1 CAL feature geometry is not FP32",
        )
        _require(reference.requires_grad, "IBR1 CAL observation graph is dead")
        prev_leaf = prev_leaf.to(dtype=reference.dtype)
        _require(prev_leaf.requires_grad, "IBR1 CAL previous-action leaf lost grad")

        audited = {"base_h_act": reference}
        for name, tensor in output["method_features"].items():
            if tensor.requires_grad:
                audited[f"method_{name}"] = tensor
        prev_free = True
        try:
            assert_prev_free_tensors(audited, prev_leaf)
        except F2ModelContractError:
            prev_free = False

        model_output = self.ibr1_arm.model(
            output["base_features"],
            prev_leaf,
            method_features=output["method_features"],
            method_alphas=output["method_alphas"],
        )
        prediction = model_output.prediction
        _require(
            isinstance(prediction, IBR1Prediction),
            "IBR1 CAL model returned a non-IBR1 prediction",
        )
        geometry_tensors = (
            prediction.raw_fy,
            prediction.delta_fy,
            prediction.latent_delta_fy,
            prediction.cumulative_latent_fy,
            prediction.additive_prebound_fy,
        )
        _require(
            all(tensor.dtype == IBR1_CAL_GEOMETRY_DTYPE for tensor in geometry_tensors),
            "authoritative IBR1 CAL geometry is not FP32",
        )
        raw_fy = prediction.raw_fy
        _require(
            raw_fy.shape == (1, *IBR1_CAL_CONTROLLED_SHAPE),
            "IBR1 CAL controlled tensor is not exactly one (8,2) row",
        )
        controlled_cells = int(raw_fy[0].numel())
        _require(
            controlled_cells == IBR1_CAL_CONTROLLED_CELLS,
            "IBR1 CAL controlled-cell count is not 16",
        )
        expected_persistence = prev_leaf.unsqueeze(-2).expand_as(raw_fy)
        zero_init_persistence = torch.equal(raw_fy, expected_persistence)
        post_decode_abs_max = _finite_max_abs(raw_fy, "IBR1 post-decode action")
        _require(
            post_decode_abs_max <= 1.0,
            "IBR1 post-decode action exceeded the frozen range",
        )
        reconstruction = prev_leaf.unsqueeze(-2) + torch.cumsum(
            prediction.delta_fy, dim=-2
        )
        reconstruction_error = _finite_max_abs(
            reconstruction - raw_fy,
            "IBR1 realized-delta reconstruction error",
        )
        _require(
            reconstruction_error <= IBR1_CAL_RECONSTRUCTION_THRESHOLD,
            "IBR1 realized-delta reconstruction exceeded 1e-6",
        )
        return (
            str(raw_fy.dtype),
            zero_init_persistence,
            post_decode_abs_max,
            controlled_cells,
            reconstruction_error,
            prev_free,
        )

    def __call__(
        self,
        row: RunnerRow,
        reasons: Sequence[str],
        position: int,
    ) -> IBR1CalRowAudit:
        _require(isinstance(row, RunnerRow), "IBR1 CAL auditor requires a RunnerRow")
        _require(
            isinstance(position, int) and not isinstance(position, bool) and position >= 0,
            "IBR1 CAL position must be a nonnegative integer",
        )
        subordinate = self.subordinate_auditor(row, reasons, position)
        _require(
            isinstance(subordinate, CalRowAudit),
            "frozen F2 CalRowAuditor returned the wrong evidence type",
        )
        (
            geometry_dtype,
            zero_init_persistence,
            post_decode_abs_max,
            controlled_cells,
            reconstruction_error,
            prev_free,
        ) = self._ibr1_geometry(row, reasons, position)
        return IBR1CalRowAudit(
            subordinate_audit=subordinate,
            geometry_dtype=geometry_dtype,
            zero_init_persistence=zero_init_persistence,
            post_decode_abs_max=post_decode_abs_max,
            controlled_tensor_shape=IBR1_CAL_CONTROLLED_SHAPE,
            controlled_cells=controlled_cells,
            realized_delta_reconstruction_error=reconstruction_error,
            prev_free_observation_graph=prev_free,
        )


def build_ibr1_cal_row_auditor(
    project_root: str | Path,
    *,
    base: nn.Module | None = None,
    device: torch.device | str | None = None,
    seed: int = IBR1_CAL_SEED,
) -> IBR1ModelCalRowAuditor:
    """Build the single zero-update IBR1 CAL model callback.

    With no injected ``base`` this is the production factory: it selects the
    default device, requires exactly ``cuda:0``, configures deterministic CUDA,
    and loads the frozen local HF base.  CPU is accepted only when a test or
    audit harness explicitly injects a base and an explicit device.
    """

    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed == 0,
        "IBR1 CAL seed is frozen at zero",
    )
    root = Path(project_root).expanduser().resolve()
    injected_base = base is not None
    if not injected_base:
        target_device = default_device() if device is None else torch.device(device)
        _require(
            str(target_device) == "cuda:0",
            "production IBR1 CAL requires cuda:0 and forbids CPU fallback",
        )
        configure_cuda_reproducibility()
        base, _load_report = load_base_checkpoint()
    else:
        _require(
            isinstance(base, nn.Module),
            "injected IBR1 CAL base must be an nn.Module",
        )
        _require(
            device is not None,
            "an injected IBR1 CAL base requires an explicit device",
        )
        target_device = torch.device(device)
        if target_device.type == "cuda":
            _require(
                str(target_device) == "cuda:0",
                "IBR1 CAL CUDA injection must use cuda:0",
            )
            configure_cuda_reproducibility()

    sealed = read_sealed_f2_init_evidence(root)
    _require(
        isinstance(sealed, F2SealedInitEvidence) and sealed.seed == seed,
        "live sealed F2 init evidence does not bind seed zero",
    )
    assert base is not None
    try:
        f2_base = copy.deepcopy(base)
        ibr1_base = copy.deepcopy(base)
    except Exception as exc:
        raise IBR1CalibrationModelContractError(
            "IBR1 CAL base could not be copied into disjoint F2/IBR1 arms"
        ) from exc

    subordinate = build_cal_audit_context(
        f2_base,
        device=target_device,
        seed=seed,
    )
    torch.manual_seed(seed)
    ibr1_arm = build_ibr1_package(
        ibr1_base,
        device=target_device,
        aux_coefficients=IBR1_CAL_PLACEHOLDER_AUX_COEFFICIENTS,
    )
    return IBR1ModelCalRowAuditor(
        project_root=root,
        subordinate_auditor=subordinate,
        ibr1_arm=ibr1_arm,
        sealed_evidence=sealed,
    )


__all__ = [
    "IBR1_CAL_CONTROLLED_CELLS",
    "IBR1_CAL_CONTROLLED_SHAPE",
    "IBR1_CAL_GEOMETRY_DTYPE",
    "IBR1_CAL_RECONSTRUCTION_THRESHOLD",
    "IBR1_CAL_SEED",
    "IBR1CalibrationModelContractError",
    "IBR1ModelCalRowAuditor",
    "build_ibr1_cal_row_auditor",
]


# Register the real factory/class objects only after this module has completed
# its circular import of ``calibration.IBR1CalRowAudit``.  The lifecycle keeps
# these object/code references so mutable ``__module__``/``__qualname__``
# metadata cannot spoof a production component later.
_calibration_lifecycle._bind_production_calibration_model_components(
    sys.modules[__name__],
    build_ibr1_cal_row_auditor,
    IBR1ModelCalRowAuditor,
)
