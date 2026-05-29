"""iter02: FLARE-style mid-layer alignment with future target embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FLAREAlignHead(nn.Module):
    def __init__(self, d_model: int, sig_dim: int = 8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, sig_dim),
        )

    def forward(self, h_mid):
        return F.normalize(self.proj(h_mid.float()), dim=-1)
