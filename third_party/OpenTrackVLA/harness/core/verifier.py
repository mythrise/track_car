"""Verifier — dual-head: q^write (TIM gate) + δ (action residual)."""

from __future__ import annotations
import torch
import torch.nn as nn


class Verifier(nn.Module):
    def __init__(self, d_model: int, n_waypoints: int = 8, action_dims: int = 3):
        super().__init__()
        self.nw = n_waypoints
        self.ad = action_dims
        in_dim = d_model * 4
        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )
        self.q_head = nn.Linear(256, 1)
        self.delta_head = nn.Linear(256, n_waypoints * action_dims)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def forward(self, f_t, tim_mean, h_t, fut_mean):
        z = self.trunk(torch.cat([f_t, tim_mean, h_t, fut_mean], dim=-1))
        q = torch.sigmoid(self.q_head(z).squeeze(-1))
        delta = torch.tanh(self.delta_head(z).view(-1, self.nw, self.ad)) * 0.15
        return q, delta

    @staticmethod
    def runtime_gate(q_write, mode_idx):
        cautious = (mode_idx >= 1).float()
        return (cautious + (q_write < 0.4).float()).clamp(0.0, 1.0)
