"""Hierarchical Multi-Horizon Future Module (Δ1=4, Δ2=8, Δ3=16).

Stop-grad conditioning: Δ2 conditions on detached Δ1 output, etc.
Predicts: future target embedding, future polar, future visibility.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureModule(nn.Module):
    def __init__(self, d_model: int, sig_dim: int = 8, n_theta: int = 60, n_dist: int = 30,
                 horizons=(4, 8, 16), n_fut_per: int = 4, action_dim: int = 3):
        super().__init__()
        self.horizons = horizons
        self.n_fut_per = n_fut_per
        self.d_model = d_model
        self.act_proj = nn.Linear(action_dim, d_model)
        self.shared = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model), nn.GELU(),
        )
        self.h_proj = nn.ModuleList()
        for i in range(len(horizons)):
            in_dim = d_model + (d_model if i > 0 else 0)
            self.h_proj.append(nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, n_fut_per * d_model),
            ))
        self.head_emb = nn.ModuleList([nn.Linear(d_model, sig_dim) for _ in horizons])
        self.head_theta = nn.ModuleList([nn.Linear(d_model, n_theta) for _ in horizons])
        self.head_dist = nn.ModuleList([nn.Linear(d_model, n_dist) for _ in horizons])
        self.head_vis = nn.ModuleList([nn.Linear(d_model, 1) for _ in horizons])

    def forward(self, h_t, tim_mean, evt_tok, last_action):
        cond = torch.cat([h_t, tim_mean, evt_tok, self.act_proj(last_action)], dim=-1)
        s = self.shared(cond)
        out = {}
        prev_rep = None
        all_tokens = []
        for i, h in enumerate(self.horizons):
            B = s.size(0)
            inp = s if prev_rep is None else torch.cat([s, prev_rep.detach()], dim=-1)
            tok = self.h_proj[i](inp).view(B, self.n_fut_per, self.d_model)
            rep = tok.mean(dim=1)
            out[h] = {
                "tokens": tok,
                "emb": F.normalize(self.head_emb[i](rep), dim=-1),
                "theta_logits": self.head_theta[i](rep),
                "dist_logits": self.head_dist[i](rep),
                "vis_logit": self.head_vis[i](rep).squeeze(-1),
            }
            all_tokens.append(tok)
            prev_rep = rep
        out["all_tokens"] = torch.cat(all_tokens, dim=1)
        return out
