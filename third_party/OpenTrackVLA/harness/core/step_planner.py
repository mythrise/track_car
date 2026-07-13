"""Decoupled heads and losses for per-step car actions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StepActionHead(nn.Module):
    """Predict forward and yaw independently while keeping strafe at zero."""

    def __init__(self, d_model: int, n_steps: int = 8):
        super().__init__()
        self.n_steps = int(n_steps)
        self.trunk = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
        )
        self.forward_branch = nn.Linear(256, self.n_steps)
        self.yaw_branch = nn.Linear(256, self.n_steps)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(ctx)
        forward = torch.tanh(self.forward_branch(hidden))
        yaw = torch.tanh(self.yaw_branch(hidden))
        strafe = torch.zeros_like(forward)
        return torch.stack((forward, strafe, yaw), dim=-1)


class DeltaVelocityHead(nn.Module):
    """Auxiliary delta-velocity prediction with the required [-2, 2] range."""

    def __init__(self, d_model: int, n_steps: int = 8):
        super().__init__()
        self.n_steps = int(n_steps)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, self.n_steps * 3),
        )

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return 2.0 * torch.tanh(self.net(ctx)).view(-1, self.n_steps, 3)


def masked_smooth_l1(pred, target, valid_mask=None):
    error = F.smooth_l1_loss(pred, target, reduction="none")
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(device=error.device, dtype=error.dtype)
    while mask.dim() < error.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(error)
    denominator = mask.sum()
    if denominator.item() <= 0:
        return error.sum() * 0.0
    return (error * mask).sum() / denominator


def step_action_track_loss(
    pred_actions,
    target_actions,
    valid_mask=None,
    *,
    lambda_yaw: float = 2.0,
    pred_delta_vel=None,
    target_delta_vel=None,
    delta_vel_weight: float = 0.2,
):
    forward_loss = masked_smooth_l1(
        pred_actions[..., 0], target_actions[..., 0], valid_mask
    )
    yaw_loss = masked_smooth_l1(
        pred_actions[..., 2], target_actions[..., 2], valid_mask
    )
    total = forward_loss + float(lambda_yaw) * yaw_loss
    delta_vel_loss = total.new_zeros(())
    if pred_delta_vel is not None:
        if target_delta_vel is None:
            raise ValueError("target_delta_vel is required when the auxiliary head is enabled")
        delta_vel_loss = masked_smooth_l1(pred_delta_vel, target_delta_vel, valid_mask)
        total = total + float(delta_vel_weight) * delta_vel_loss
    return total, forward_loss, yaw_loss, delta_vel_loss
