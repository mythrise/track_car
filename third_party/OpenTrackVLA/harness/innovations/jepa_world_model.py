"""iter03: V-JEPA-AC action-conditioned latent world model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPASourceEncoder(nn.Module):
    def __init__(self, feat_dim: int, latent_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        return F.normalize(self.enc(x), dim=-1)


class JEPATargetEncoder(nn.Module):
    def __init__(self, feat_dim: int, latent_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_ema(self, source, rho=0.995):
        sd = source.enc.state_dict()
        own = self.enc.state_dict()
        for k in own:
            own[k] = rho * own[k] + (1 - rho) * sd[k]
        self.enc.load_state_dict(own)

    @torch.no_grad()
    def forward(self, x):
        return F.normalize(self.enc(x), dim=-1)


class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim: int = 64, action_dim: int = 3):
        super().__init__()
        self.act_proj = nn.Linear(action_dim, latent_dim)
        self.predictor = nn.Sequential(
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim), nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, z_t, a_t):
        return F.normalize(self.predictor(torch.cat([z_t, self.act_proj(a_t)], dim=-1)), dim=-1)
