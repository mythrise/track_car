"""Isolated torch implementation of the approved F2 model contract.

The module is intentionally independent from the frozen OpenTrackVLA tree.  It
implements the L1+D2+AP2+F2 context geometry, bounded previous-action stream,
and raw-action AP2 head selected by the Fable corrigenda.  Controller-facing
future-horizon telemetry delegates to the shared controller implementation;
it is never used as the training prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from . import controller as controller_core
from .support import ARCHITECTURE_LOCK, F2ContractError


UNIT_EPS = 1e-6
BASE_NORM_MIN = 1e-3
METHOD_STREAM_BOUND = 0.5
TOTAL_METHOD_BOUND = 1.0
PREV_STREAM_BOUND = 0.5
RATIO_TOLERANCE = 1e-4
LAYERSCALE_SATURATION = 0.99
ACTION_MAX_ABS = 1.0
AP2_HORIZON = 8
AP2_HIDDEN_DIM = 256
PREV_HIDDEN_DIM = 128
CONTROLLED_AXES = (0, 2)


class F2ModelContractError(F2ContractError):
    """Raised when the isolated model must fail closed."""


def _validate_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise F2ModelContractError(f"{label} must be a positive integer")
    return value


def _validate_float_tensor(value: Any, label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise F2ModelContractError(f"{label} must be a torch.Tensor")
    if not value.is_floating_point():
        raise F2ModelContractError(f"{label} must use a floating dtype")
    return value


def _any_true(value: torch.Tensor) -> bool:
    return bool(value.detach().any().to(device="cpu").item())


def _assert_finite(value: torch.Tensor, label: str) -> None:
    if _any_true(~torch.isfinite(value)):
        raise F2ModelContractError(f"HS7_FUSION_RUNTIME_NONFINITE: {label}")


def unit_l2(value: torch.Tensor, eps: float = UNIT_EPS) -> torch.Tensor:
    """Apply the frozen U(v)=v/max(l2(v), eps) operation on the last axis."""

    tensor = _validate_float_tensor(value, "unit_l2.value")
    if isinstance(eps, bool) or not isinstance(eps, (int, float)):
        raise F2ModelContractError("unit_l2 eps must be numeric")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise F2ModelContractError("unit_l2 eps must be finite and positive")
    norm = torch.linalg.vector_norm(tensor, dim=-1, keepdim=True)
    floor = norm.new_full((), eps_value)
    return tensor / torch.maximum(norm, floor)


def _coerce_alpha(
    value: float | torch.Tensor,
    *,
    leading_shape: torch.Size,
    reference: torch.Tensor,
    label: str,
) -> torch.Tensor:
    if isinstance(value, bool):
        raise F2ModelContractError(f"{label} must be numeric")
    if isinstance(value, torch.Tensor):
        alpha = value.to(device=reference.device, dtype=reference.dtype)
    elif isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise F2ModelContractError(f"{label} is nonfinite")
        alpha = reference.new_tensor(float(value))
    else:
        raise F2ModelContractError(f"{label} must be numeric or a tensor")
    if alpha.ndim == len(leading_shape) + 1 and alpha.shape[-1] == 1:
        alpha = alpha.squeeze(-1)
    try:
        alpha = torch.broadcast_to(alpha, leading_shape)
    except RuntimeError as exc:
        raise F2ModelContractError(
            f"{label} cannot broadcast to {tuple(leading_shape)}"
        ) from exc
    _assert_finite(alpha, label)
    if _any_true((alpha < 0.0) | (alpha > 1.0)):
        raise F2ModelContractError(f"{label} must lie in [0,1]")
    return alpha


@dataclass(frozen=True)
class ContextComposition:
    """Previous-action-free context and its constructive-bound evidence."""

    base_stream: torch.Tensor
    method_streams: Mapping[str, torch.Tensor]
    method_alphas: Mapping[str, torch.Tensor]
    method_total: torch.Tensor
    x0: torch.Tensor
    x: torch.Tensor
    base_input_norm: torch.Tensor
    base_stream_norm: torch.Tensor
    x0_norm: torch.Tensor
    x_norm: torch.Tensor
    per_method_over_base: Mapping[str, torch.Tensor]
    total_method_over_base: torch.Tensor


@dataclass(frozen=True)
class G7Telemetry:
    """Per-forward tensors needed to aggregate the frozen G7 gates."""

    r_prev: torch.Tensor
    per_method_over_base: Mapping[str, torch.Tensor]
    total_method_over_base: torch.Tensor
    abs_tanh_method_scales: Mapping[str, torch.Tensor]
    abs_tanh_s_prev: torch.Tensor
    method_saturation_fraction: torch.Tensor
    prev_saturation_indicator: torch.Tensor
    base_input_norm: torch.Tensor
    base_stream_norm: torch.Tensor
    x0_norm: torch.Tensor
    x_norm: torch.Tensor
    prev_stream_norm: torch.Tensor

    @property
    def layerscale_saturation_rate_both(self) -> Mapping[str, torch.Tensor]:
        return {
            "method": self.method_saturation_fraction,
            "prev": self.prev_saturation_indicator,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "r_prev": self.r_prev,
            "per_method_over_base": dict(self.per_method_over_base),
            "total_method_over_base": self.total_method_over_base,
            "abs_tanh_method_scales": dict(self.abs_tanh_method_scales),
            "abs_tanh_s_prev": self.abs_tanh_s_prev,
            "layerscale_saturation_rate_both": dict(
                self.layerscale_saturation_rate_both
            ),
            "base_input_norm": self.base_input_norm,
            "base_stream_norm": self.base_stream_norm,
            "x0_norm": self.x0_norm,
            "x_norm": self.x_norm,
            "prev_stream_norm": self.prev_stream_norm,
        }

    def detached(self) -> G7Telemetry:
        return G7Telemetry(
            r_prev=self.r_prev.detach(),
            per_method_over_base={
                name: value.detach()
                for name, value in self.per_method_over_base.items()
            },
            total_method_over_base=self.total_method_over_base.detach(),
            abs_tanh_method_scales={
                name: value.detach()
                for name, value in self.abs_tanh_method_scales.items()
            },
            abs_tanh_s_prev=self.abs_tanh_s_prev.detach(),
            method_saturation_fraction=self.method_saturation_fraction.detach(),
            prev_saturation_indicator=self.prev_saturation_indicator.detach(),
            base_input_norm=self.base_input_norm.detach(),
            base_stream_norm=self.base_stream_norm.detach(),
            x0_norm=self.x0_norm.detach(),
            x_norm=self.x_norm.detach(),
            prev_stream_norm=self.prev_stream_norm.detach(),
        )


@dataclass(frozen=True)
class FusedContext:
    """Head input after bounded previous-action conditioning."""

    head_input: torch.Tensor
    prev_stream: torch.Tensor
    composition: ContextComposition
    telemetry: G7Telemetry


class BoundedContextFusion(nn.Module):
    """Constructively bounded method and previous-action context fusion."""

    def __init__(
        self,
        d_model: int = 1024,
        method_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = _validate_positive_int(d_model, "d_model")
        dims = {} if method_dims is None else dict(method_dims)
        for name, input_dim in dims.items():
            if not isinstance(name, str) or not name or "." in name:
                raise F2ModelContractError(
                    "method stream names must be nonempty module-safe strings"
                )
            _validate_positive_int(input_dim, f"method_dims[{name!r}]")
        self.method_dims = dims
        self.method_projections = nn.ModuleDict(
            {
                name: nn.Linear(input_dim, self.d_model)
                for name, input_dim in dims.items()
            }
        )
        self.method_scales = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in dims}
        )
        self.prev_projection = nn.Sequential(
            nn.Linear(2, PREV_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(PREV_HIDDEN_DIM, self.d_model),
        )
        self.s_prev = nn.Parameter(torch.zeros(()))
        self.head_norm = nn.LayerNorm(self.d_model, eps=1e-5, elementwise_affine=True)
        self._sqrt_d = math.sqrt(self.d_model)

    def compose_context(
        self,
        base_features: torch.Tensor,
        method_features: Mapping[str, torch.Tensor] | None = None,
        method_alphas: Mapping[str, float | torch.Tensor] | None = None,
    ) -> ContextComposition:
        """Build x from base and method features without accepting prev_action."""

        base = _validate_float_tensor(base_features, "base_features")
        if base.ndim < 1 or base.shape[-1] != self.d_model:
            raise F2ModelContractError(
                f"base_features must end in dimension {self.d_model}"
            )
        _assert_finite(base, "base_features")
        base_input_norm = torch.linalg.vector_norm(base, dim=-1)
        _assert_finite(base_input_norm, "base_input_norm")
        if _any_true(base_input_norm < BASE_NORM_MIN):
            raise F2ModelContractError(
                "HS7_FUSION_RUNTIME_BASE_NORM_LT_1E_3"
            )

        supplied_features = {} if method_features is None else dict(method_features)
        supplied_alphas = {} if method_alphas is None else dict(method_alphas)
        expected = set(self.method_dims)
        if set(supplied_features) != expected:
            raise F2ModelContractError(
                "method_features keys must exactly match configured method streams"
            )
        unknown_alphas = set(supplied_alphas) - expected
        if unknown_alphas:
            raise F2ModelContractError(
                f"unknown method alpha keys: {sorted(unknown_alphas)!r}"
            )

        base_stream = self._sqrt_d * unit_l2(base)
        base_stream_norm = torch.linalg.vector_norm(base_stream, dim=-1)
        method_streams: dict[str, torch.Tensor] = {}
        alpha_tensors: dict[str, torch.Tensor] = {}
        per_method_over_base: dict[str, torch.Tensor] = {}
        method_total = torch.zeros_like(base_stream)
        alpha_total = torch.zeros_like(base_stream_norm)

        for name in self.method_dims:
            feature = _validate_float_tensor(
                supplied_features[name], f"method_features[{name!r}]"
            )
            expected_shape = base.shape[:-1] + (self.method_dims[name],)
            if feature.shape != expected_shape:
                raise F2ModelContractError(
                    f"method_features[{name!r}] must have shape "
                    f"{tuple(expected_shape)}"
                )
            _assert_finite(feature, f"method_features[{name!r}]")
            projected = self.method_projections[name](feature)
            _assert_finite(projected, f"projected_method[{name!r}]")
            alpha = _coerce_alpha(
                supplied_alphas.get(name, 1.0),
                leading_shape=base.shape[:-1],
                reference=base,
                label=f"method_alphas[{name!r}]",
            )
            scale = METHOD_STREAM_BOUND * torch.tanh(self.method_scales[name])
            stream = (
                scale
                * alpha.unsqueeze(-1)
                * self._sqrt_d
                * unit_l2(projected)
            )
            _assert_finite(stream, f"method_stream[{name!r}]")
            ratio = torch.linalg.vector_norm(stream, dim=-1) / base_stream_norm
            if _any_true(ratio > METHOD_STREAM_BOUND + RATIO_TOLERANCE):
                raise F2ModelContractError(
                    f"HS8_G7_METHOD_RATIO_EXCEEDED: {name}"
                )
            method_streams[name] = stream
            alpha_tensors[name] = alpha
            per_method_over_base[name] = ratio
            method_total = method_total + stream
            alpha_total = alpha_total + alpha

        if _any_true(alpha_total > 2.0 + RATIO_TOLERANCE):
            raise F2ModelContractError("HS8_G7_METHOD_ALPHA_SUM_EXCEEDED")
        total_method_over_base = (
            torch.linalg.vector_norm(method_total, dim=-1) / base_stream_norm
        )
        if _any_true(
            total_method_over_base > TOTAL_METHOD_BOUND + RATIO_TOLERANCE
        ):
            raise F2ModelContractError("HS8_G7_TOTAL_METHOD_RATIO_EXCEEDED")

        x0 = base_stream + method_total
        _assert_finite(x0, "x0")
        x0_norm = torch.linalg.vector_norm(x0, dim=-1)
        _assert_finite(x0_norm, "x0_norm")
        if _any_true(x0_norm < BASE_NORM_MIN):
            raise F2ModelContractError("HS7_FUSION_RUNTIME_X0_NORM_LT_1E_3")
        x = self._sqrt_d * unit_l2(x0)
        _assert_finite(x, "x")
        x_norm = torch.linalg.vector_norm(x, dim=-1)

        return ContextComposition(
            base_stream=base_stream,
            method_streams=method_streams,
            method_alphas=alpha_tensors,
            method_total=method_total,
            x0=x0,
            x=x,
            base_input_norm=base_input_norm,
            base_stream_norm=base_stream_norm,
            x0_norm=x0_norm,
            x_norm=x_norm,
            per_method_over_base=per_method_over_base,
            total_method_over_base=total_method_over_base,
        )

    def condition_on_prev(
        self,
        composition: ContextComposition,
        prev_fy: torch.Tensor,
    ) -> FusedContext:
        """Add the bounded two-axis previous-action stream and apply LN_head."""

        if not isinstance(composition, ContextComposition):
            raise F2ModelContractError("composition must be a ContextComposition")
        prev = _validate_float_tensor(prev_fy, "prev_fy")
        expected_shape = composition.x.shape[:-1] + (2,)
        if prev.shape != expected_shape:
            raise F2ModelContractError(
                f"prev_fy must have shape {tuple(expected_shape)}"
            )
        _assert_finite(prev, "prev_fy")
        if _any_true(torch.abs(prev) > ACTION_MAX_ABS):
            raise F2ModelContractError("PREV_ACTION_OUTSIDE_FROZEN_DOMAIN")

        projected_prev = self.prev_projection(prev)
        _assert_finite(projected_prev, "P_prev(prev_fy)")
        prev_scale = PREV_STREAM_BOUND * torch.tanh(self.s_prev)
        prev_stream = (
            prev_scale * self._sqrt_d * unit_l2(projected_prev)
        )
        _assert_finite(prev_stream, "p_prev")
        prev_stream_norm = torch.linalg.vector_norm(prev_stream, dim=-1)
        r_prev = prev_stream_norm / composition.x_norm
        if _any_true(r_prev > PREV_STREAM_BOUND + RATIO_TOLERANCE):
            raise F2ModelContractError("HS8_G7_R_PREV_EXCEEDED")

        pre_norm = composition.x + prev_stream
        _assert_finite(pre_norm, "x_plus_p_prev")
        head_input = self.head_norm(pre_norm)
        _assert_finite(head_input, "head_in")

        abs_method_scales = {
            name: torch.abs(torch.tanh(scale))
            for name, scale in self.method_scales.items()
        }
        if abs_method_scales:
            method_saturation_fraction = torch.stack(
                [
                    (value >= LAYERSCALE_SATURATION).to(head_input.dtype)
                    for value in abs_method_scales.values()
                ]
            ).mean()
        else:
            method_saturation_fraction = head_input.new_zeros(())
        abs_tanh_s_prev = torch.abs(torch.tanh(self.s_prev))
        prev_saturation_indicator = (
            abs_tanh_s_prev >= LAYERSCALE_SATURATION
        ).to(head_input.dtype)
        telemetry = G7Telemetry(
            r_prev=r_prev,
            per_method_over_base=composition.per_method_over_base,
            total_method_over_base=composition.total_method_over_base,
            abs_tanh_method_scales=abs_method_scales,
            abs_tanh_s_prev=abs_tanh_s_prev,
            method_saturation_fraction=method_saturation_fraction,
            prev_saturation_indicator=prev_saturation_indicator,
            base_input_norm=composition.base_input_norm,
            base_stream_norm=composition.base_stream_norm,
            x0_norm=composition.x0_norm,
            x_norm=composition.x_norm,
            prev_stream_norm=prev_stream_norm,
        )
        return FusedContext(
            head_input=head_input,
            prev_stream=prev_stream,
            composition=composition,
            telemetry=telemetry,
        )

    def forward(
        self,
        base_features: torch.Tensor,
        prev_fy: torch.Tensor,
        method_features: Mapping[str, torch.Tensor] | None = None,
        method_alphas: Mapping[str, float | torch.Tensor] | None = None,
    ) -> FusedContext:
        composition = self.compose_context(
            base_features,
            method_features=method_features,
            method_alphas=method_alphas,
        )
        return self.condition_on_prev(composition, prev_fy)


@dataclass(frozen=True)
class AP2Prediction:
    """AP2 raw action reconstruction and detached k1..k7 telemetry."""

    delta_fy: torch.Tensor
    raw_actions: torch.Tensor
    bounded_future_actions: torch.Tensor

    @property
    def raw_fy(self) -> torch.Tensor:
        return self.raw_actions[..., CONTROLLED_AXES]


class AP2DeltaHead(nn.Module):
    """Eight-step persistence-residual head with two zero-init branches."""

    def __init__(self, d_model: int = 1024) -> None:
        super().__init__()
        self.d_model = _validate_positive_int(d_model, "d_model")
        self.horizon = AP2_HORIZON
        self.trunk = nn.Sequential(
            nn.Linear(self.d_model, AP2_HIDDEN_DIM),
            nn.GELU(),
        )
        self.forward_branch = nn.Linear(AP2_HIDDEN_DIM, AP2_HORIZON)
        self.yaw_branch = nn.Linear(AP2_HIDDEN_DIM, AP2_HORIZON)
        self.reset_delta_parameters()

    def reset_delta_parameters(self) -> None:
        nn.init.zeros_(self.forward_branch.weight)
        nn.init.zeros_(self.forward_branch.bias)
        nn.init.zeros_(self.yaw_branch.weight)
        nn.init.zeros_(self.yaw_branch.bias)

    def _bounded_future_telemetry(
        self, raw_actions: torch.Tensor
    ) -> torch.Tensor:
        detached = raw_actions.detach().reshape(-1, AP2_HORIZON, 3).to("cpu")
        bounded_rows: list[list[tuple[float, float, float]]] = []
        for sample in detached:
            bounded_steps: list[tuple[float, float, float]] = []
            for step_index in range(1, AP2_HORIZON):
                forward = float(sample[step_index, 0].item())
                yaw = float(sample[step_index, 2].item())
                bounded_steps.append(
                    controller_core.clamp_stage(
                        (forward, yaw), max_abs=ACTION_MAX_ABS
                    )
                )
            bounded_rows.append(bounded_steps)
        result = raw_actions.new_tensor(bounded_rows)
        return result.reshape(raw_actions.shape[:-2] + (AP2_HORIZON - 1, 3))

    def forward(
        self,
        head_input: torch.Tensor,
        prev_fy: torch.Tensor,
    ) -> AP2Prediction:
        context = _validate_float_tensor(head_input, "head_input")
        if context.ndim < 1 or context.shape[-1] != self.d_model:
            raise F2ModelContractError(
                f"head_input must end in dimension {self.d_model}"
            )
        _assert_finite(context, "head_input")
        prev = _validate_float_tensor(prev_fy, "prev_fy")
        expected_prev_shape = context.shape[:-1] + (2,)
        if prev.shape != expected_prev_shape:
            raise F2ModelContractError(
                f"prev_fy must have shape {tuple(expected_prev_shape)}"
            )
        _assert_finite(prev, "prev_fy")
        if _any_true(torch.abs(prev) > ACTION_MAX_ABS):
            raise F2ModelContractError("PREV_ACTION_OUTSIDE_FROZEN_DOMAIN")

        hidden = self.trunk(context)
        delta_forward = self.forward_branch(hidden)
        delta_yaw = self.yaw_branch(hidden)
        delta_fy = torch.stack((delta_forward, delta_yaw), dim=-1)
        cumulative_delta = torch.cumsum(delta_fy, dim=-2)
        raw_fy = prev.unsqueeze(-2) + cumulative_delta
        raw_actions = torch.stack(
            (
                raw_fy[..., 0],
                torch.zeros_like(raw_fy[..., 0]),
                raw_fy[..., 1],
            ),
            dim=-1,
        )
        _assert_finite(raw_actions, "a_raw")
        bounded_future = self._bounded_future_telemetry(raw_actions)
        return AP2Prediction(
            delta_fy=delta_fy,
            raw_actions=raw_actions,
            bounded_future_actions=bounded_future,
        )


@dataclass(frozen=True)
class AP2TrackLoss:
    """Controlled-axis Smooth-L1 components computed only from a_raw."""

    total: torch.Tensor
    forward: torch.Tensor
    yaw: torch.Tensor


def _masked_mean(error: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return error.mean()
    if not isinstance(valid_mask, torch.Tensor):
        raise F2ModelContractError("valid_mask must be a torch.Tensor")
    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    try:
        mask = torch.broadcast_to(mask, error.shape)
    except RuntimeError as exc:
        raise F2ModelContractError(
            f"valid_mask cannot broadcast to {tuple(error.shape)}"
        ) from exc
    _assert_finite(mask, "valid_mask")
    if _any_true(mask < 0.0):
        raise F2ModelContractError("valid_mask cannot contain negative values")
    denominator = mask.sum()
    if not math.isfinite(float(denominator.detach().to("cpu").item())):
        raise F2ModelContractError("valid_mask denominator is nonfinite")
    if float(denominator.detach().to("cpu").item()) <= 0.0:
        raise F2ModelContractError("valid_mask has zero support")
    return (error * mask).sum() / denominator


def ap2_track_loss(
    prediction: AP2Prediction,
    target_actions: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    yaw_weight: float = 2.0,
) -> AP2TrackLoss:
    """Compute the primary track loss from raw, never bounded, AP2 actions."""

    if not isinstance(prediction, AP2Prediction):
        raise F2ModelContractError("prediction must be an AP2Prediction")
    target = _validate_float_tensor(target_actions, "target_actions")
    raw = prediction.raw_actions
    if target.shape != raw.shape:
        raise F2ModelContractError(
            f"target_actions must have shape {tuple(raw.shape)}"
        )
    _assert_finite(target, "target_actions")
    if isinstance(yaw_weight, bool) or not isinstance(yaw_weight, (int, float)):
        raise F2ModelContractError("yaw_weight must be numeric")
    yaw_scale = float(yaw_weight)
    if not math.isfinite(yaw_scale) or yaw_scale < 0.0:
        raise F2ModelContractError("yaw_weight must be finite and nonnegative")

    forward_error = F.smooth_l1_loss(
        raw[..., 0], target[..., 0], reduction="none"
    )
    yaw_error = F.smooth_l1_loss(
        raw[..., 2], target[..., 2], reduction="none"
    )
    forward_loss = _masked_mean(forward_error, valid_mask)
    yaw_loss = _masked_mean(yaw_error, valid_mask)
    total = forward_loss + yaw_scale * yaw_loss
    return AP2TrackLoss(total=total, forward=forward_loss, yaw=yaw_loss)


@dataclass(frozen=True)
class F2ModelOutput:
    fused_context: FusedContext
    prediction: AP2Prediction


class F2AP2Model(nn.Module):
    """Small composition wrapper used by the isolated smoke runner."""

    architecture_lock = ARCHITECTURE_LOCK

    def __init__(
        self,
        d_model: int = 1024,
        method_dims: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.fusion = BoundedContextFusion(
            d_model=d_model,
            method_dims=method_dims,
        )
        self.action_head = AP2DeltaHead(d_model=d_model)

    def forward(
        self,
        base_features: torch.Tensor,
        prev_fy: torch.Tensor,
        method_features: Mapping[str, torch.Tensor] | None = None,
        method_alphas: Mapping[str, float | torch.Tensor] | None = None,
    ) -> F2ModelOutput:
        fused = self.fusion(
            base_features,
            prev_fy,
            method_features=method_features,
            method_alphas=method_alphas,
        )
        prediction = self.action_head(fused.head_input, prev_fy)
        return F2ModelOutput(fused_context=fused, prediction=prediction)

    def optimizer_parameter_groups(
        self,
        *,
        head_lr: float = 3e-4,
        head_weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Return disjoint ordinary-head, method-scale, and s_prev groups."""

        if not math.isfinite(head_lr) or head_lr <= 0.0:
            raise F2ModelContractError("head_lr must be finite and positive")
        if not math.isfinite(head_weight_decay) or head_weight_decay < 0.0:
            raise F2ModelContractError(
                "head_weight_decay must be finite and nonnegative"
            )
        method_scale_parameters = list(self.fusion.method_scales.parameters())
        excluded = {id(self.fusion.s_prev)} | {
            id(parameter) for parameter in method_scale_parameters
        }
        ordinary_parameters = [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in excluded
        ]
        groups: list[dict[str, Any]] = [
            {
                "name": "ordinary_head",
                "params": ordinary_parameters,
                "lr": float(head_lr),
                "weight_decay": float(head_weight_decay),
            }
        ]
        if method_scale_parameters:
            groups.append(
                {
                    "name": "method_layerscales",
                    "params": method_scale_parameters,
                    "lr": 3e-4,
                    "weight_decay": 0.0,
                }
            )
        groups.append(
            {
                "name": "prev_layerscale",
                "params": [self.fusion.s_prev],
                "lr": 3e-4,
                "weight_decay": 0.0,
            }
        )

        grouped_ids = [
            id(parameter)
            for group in groups
            for parameter in group["params"]
        ]
        model_ids = [id(parameter) for parameter in self.parameters()]
        if len(grouped_ids) != len(set(grouped_ids)) or set(grouped_ids) != set(
            model_ids
        ):
            raise F2ModelContractError("optimizer parameter groups are not exact")
        return groups


def assert_prev_free_tensor(
    tensor: torch.Tensor,
    prev_action: torch.Tensor,
    *,
    label: str = "tensor",
) -> None:
    """Fail when a differentiable output graph contains prev_action.

    The audit input must require gradients.  A zero numerical derivative still
    counts as a graph dependency and therefore fails.
    """

    output = _validate_float_tensor(tensor, label)
    previous = _validate_float_tensor(prev_action, "prev_action")
    if not previous.requires_grad:
        raise F2ModelContractError(
            "prev_action must require grad for a prev-free graph audit"
        )
    if not output.requires_grad:
        raise F2ModelContractError(
            f"{label} must require grad for a prev-free graph audit"
        )
    dependency = torch.autograd.grad(
        output.sum(),
        previous,
        allow_unused=True,
        retain_graph=True,
        create_graph=False,
    )[0]
    if dependency is not None:
        raise F2ModelContractError(f"PREV_GRAPH_LEAK: {label}")


def assert_prev_free_tensors(
    tensors: Mapping[str, torch.Tensor],
    prev_action: torch.Tensor,
) -> None:
    if not isinstance(tensors, Mapping) or not tensors:
        raise F2ModelContractError("tensors must be a nonempty mapping")
    for label, tensor in tensors.items():
        assert_prev_free_tensor(tensor, prev_action, label=label)


def assert_step0_controlled_axis_persistence(
    prediction: AP2Prediction,
    prev_fy: torch.Tensor,
) -> None:
    """Enforce exact raw two-axis persistence before the first optimizer step."""

    if not isinstance(prediction, AP2Prediction):
        raise F2ModelContractError("prediction must be an AP2Prediction")
    prev = _validate_float_tensor(prev_fy, "prev_fy")
    expected = prev.unsqueeze(-2).expand(prev.shape[:-1] + (AP2_HORIZON, 2))
    if not torch.equal(prediction.raw_fy, expected):
        raise F2ModelContractError("HS6_STEP0_CONTROLLED_AXIS_PARITY")
    if torch.count_nonzero(prediction.raw_actions[..., 1]).item() != 0:
        raise F2ModelContractError("HS6_STEP0_STRAFE_NONZERO")


def controlled_axis_targets(actions_xyz: torch.Tensor) -> torch.Tensor:
    """Return [forward,yaw] without ever claiming full-vector identity."""

    actions = _validate_float_tensor(actions_xyz, "actions_xyz")
    if actions.ndim < 1 or actions.shape[-1] != 3:
        raise F2ModelContractError("actions_xyz must end in dimension 3")
    return actions[..., CONTROLLED_AXES]


__all__ = [
    "ACTION_MAX_ABS",
    "AP2_HIDDEN_DIM",
    "AP2_HORIZON",
    "BASE_NORM_MIN",
    "CONTROLLED_AXES",
    "LAYERSCALE_SATURATION",
    "METHOD_STREAM_BOUND",
    "PREV_HIDDEN_DIM",
    "PREV_STREAM_BOUND",
    "RATIO_TOLERANCE",
    "TOTAL_METHOD_BOUND",
    "UNIT_EPS",
    "AP2DeltaHead",
    "AP2Prediction",
    "AP2TrackLoss",
    "BoundedContextFusion",
    "ContextComposition",
    "F2AP2Model",
    "F2ModelContractError",
    "F2ModelOutput",
    "FusedContext",
    "G7Telemetry",
    "ap2_track_loss",
    "assert_prev_free_tensor",
    "assert_prev_free_tensors",
    "assert_step0_controlled_axis_persistence",
    "controlled_axis_targets",
    "unit_l2",
]
