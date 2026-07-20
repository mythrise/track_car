"""IBR1 normalized cumulative bounded-residual action parameterization.

This module deliberately reuses the frozen F2 fusion and AP2 tensor contract
without editing :mod:`f2_experiment`.  The only behavioral change is the
action decode:

``z = cumsum(latent_delta)``
``a = (prev + z) / (1 + abs(z))``

For ``abs(prev) <= 1``, ``abs(prev + z) <= 1 + abs(z)`` proves that every
controlled action lies in ``[-1, 1]``.  At the zero-initialized head,
``z == 0`` and the output is numerically exact persistence under
``torch.equal`` on the same dtype/device (signed-zero bit identity is not a
claim; see PRIMARY amendment 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from f2_experiment.model import (
    ACTION_MAX_ABS,
    AP2_HORIZON,
    AP2DeltaHead,
    AP2Prediction,
    BoundedContextFusion,
    F2AP2Model,
    F2ModelContractError,
    F2ModelOutput,
    _validate_float_tensor,
)


IBR1_FAMILY_ID = "IBR1"
IBR1_ARCHITECTURE_LOCK = "L1+D2+IBR1-NCBR+F2-SMOKE-CONTRACT"


class IBR1ModelContractError(F2ModelContractError):
    """Raised when the IBR1 action geometry must fail closed."""


def _any_true(value: torch.Tensor) -> bool:
    return bool(value.detach().any().to(device="cpu").item())


def _assert_finite(value: torch.Tensor, label: str) -> None:
    if _any_true(~torch.isfinite(value)):
        raise IBR1ModelContractError(f"IBR1_RUNTIME_NONFINITE: {label}")


def normalized_cumulative_decode(
    prev_fy: torch.Tensor,
    latent_delta_fy: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode latent deltas into bounded actions and realized increments.

    Returns ``(raw_fy, realized_delta_fy, cumulative_latent_fy,
    additive_prebound_fy)``.  ``realized_delta_fy`` is defined so the frozen
    runner's ``prev + cumsum(delta_fy)`` reconstruction telescopes to
    ``raw_fy`` even though the trainable head emits a separate latent delta.
    """

    prev = _validate_float_tensor(prev_fy, "prev_fy")
    latent = _validate_float_tensor(latent_delta_fy, "latent_delta_fy")
    if latent.ndim < 2 or latent.shape[-2:] != (AP2_HORIZON, 2):
        raise IBR1ModelContractError(
            "latent_delta_fy must end in shape (8, 2)"
        )
    if prev.shape != latent.shape[:-2] + (2,):
        raise IBR1ModelContractError(
            "prev_fy leading shape must match latent_delta_fy"
        )
    if prev.device != latent.device:
        raise IBR1ModelContractError(
            "prev_fy and latent_delta_fy must use the same device"
        )
    if prev.dtype != latent.dtype:
        raise IBR1ModelContractError(
            "prev_fy and latent_delta_fy must use the same dtype"
        )
    if latent.dtype not in (torch.float32, torch.float64):
        raise IBR1ModelContractError(
            "IBR1 geometry permits float32 production or float64 audit only"
        )
    _assert_finite(prev, "IBR1 prev_fy")
    _assert_finite(latent, "IBR1 latent_delta_fy")
    if _any_true(torch.abs(prev) > ACTION_MAX_ABS):
        raise IBR1ModelContractError("PREV_ACTION_OUTSIDE_FROZEN_DOMAIN")

    cumulative = torch.cumsum(latent, dim=-2)
    prebound = prev.unsqueeze(-2) + cumulative
    normalizer = 1.0 + torch.abs(cumulative)
    raw_fy = prebound / normalizer
    _assert_finite(raw_fy, "IBR1 a_raw")
    if _any_true(torch.abs(raw_fy) > ACTION_MAX_ABS):
        raise IBR1ModelContractError("IBR1_RANGE_PROOF_VIOLATION")

    previous = torch.cat(
        (prev.unsqueeze(-2), raw_fy[..., :-1, :]), dim=-2
    )
    realized_delta = raw_fy - previous
    return raw_fy, realized_delta, cumulative, prebound


@dataclass(frozen=True)
class IBR1Prediction(AP2Prediction):
    """Runner-compatible actions plus IBR1 latent-geometry telemetry."""

    latent_delta_fy: torch.Tensor
    cumulative_latent_fy: torch.Tensor
    additive_prebound_fy: torch.Tensor
    normalizer_fy: torch.Tensor
    prebound_violation_mask: torch.Tensor

    @property
    def prebound_overshoot_fy(self) -> torch.Tensor:
        return torch.relu(torch.abs(self.additive_prebound_fy) - ACTION_MAX_ABS)

    @property
    def boundary_margin_fy(self) -> torch.Tensor:
        return ACTION_MAX_ABS - torch.abs(self.raw_fy)


class IBR1NormalizedBoundedHead(AP2DeltaHead):
    """AP2-shaped head with a normalized cumulative bounded decode."""

    def forward(
        self,
        head_input: torch.Tensor,
        prev_fy: torch.Tensor,
    ) -> IBR1Prediction:
        context = _validate_float_tensor(head_input, "head_input")
        if context.ndim < 1 or context.shape[-1] != self.d_model:
            raise IBR1ModelContractError(
                f"head_input must end in dimension {self.d_model}"
            )
        _assert_finite(context, "IBR1 head_input")
        prev = _validate_float_tensor(prev_fy, "prev_fy")
        expected_prev_shape = context.shape[:-1] + (2,)
        if prev.shape != expected_prev_shape:
            raise IBR1ModelContractError(
                f"prev_fy must have shape {tuple(expected_prev_shape)}"
            )

        hidden = self.trunk(context)
        latent_delta = torch.stack(
            (self.forward_branch(hidden), self.yaw_branch(hidden)), dim=-1
        )
        raw_fy, realized_delta, cumulative, prebound = (
            normalized_cumulative_decode(prev, latent_delta)
        )
        raw_actions = torch.stack(
            (
                raw_fy[..., 0],
                torch.zeros_like(raw_fy[..., 0]),
                raw_fy[..., 1],
            ),
            dim=-1,
        )
        bounded_future = raw_actions[..., 1:, :].detach()
        normalizer = 1.0 + torch.abs(cumulative)
        return IBR1Prediction(
            delta_fy=realized_delta,
            raw_actions=raw_actions,
            bounded_future_actions=bounded_future,
            latent_delta_fy=latent_delta,
            cumulative_latent_fy=cumulative,
            additive_prebound_fy=prebound,
            normalizer_fy=normalizer,
            prebound_violation_mask=torch.abs(prebound) > ACTION_MAX_ABS,
        )


class IBR1AP2Model(F2AP2Model):
    """F2 fusion with an init-order-compatible IBR1 action head."""

    architecture_lock = IBR1_ARCHITECTURE_LOCK

    def __init__(
        self,
        d_model: int = 1024,
        method_dims: Mapping[str, int] | None = None,
    ) -> None:
        # Recreate the exact F2 construction order (fusion, then action head)
        # so a reset manual seed yields identical parameter bytes and names.
        nn.Module.__init__(self)
        self.fusion = BoundedContextFusion(
            d_model=d_model,
            method_dims=method_dims,
        )
        self.action_head = IBR1NormalizedBoundedHead(d_model=d_model)

    def forward(
        self,
        base_features: torch.Tensor,
        prev_fy: torch.Tensor,
        method_features: Mapping[str, torch.Tensor] | None = None,
        method_alphas: Mapping[str, float | torch.Tensor] | None = None,
    ) -> F2ModelOutput:
        return super().forward(
            base_features,
            prev_fy,
            method_features=method_features,
            method_alphas=method_alphas,
        )


__all__ = [
    "IBR1_ARCHITECTURE_LOCK",
    "IBR1_FAMILY_ID",
    "IBR1AP2Model",
    "IBR1ModelContractError",
    "IBR1NormalizedBoundedHead",
    "IBR1Prediction",
    "normalized_cumulative_decode",
]
