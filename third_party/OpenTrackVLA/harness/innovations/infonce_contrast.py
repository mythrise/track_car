"""iter06+20: InfoNCE contrastive + hard-negative mining for TIM distractor robustness."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReIDProjector(nn.Module):
    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, proj_dim))

    def forward(self, x):
        return F.normalize(self.proj(x), dim=-1)


def infonce_hard_negative(tim_proj, target_proj, temperature=0.1):
    """InfoNCE with hard-negative mining."""
    B = tim_proj.size(0)
    sim = tim_proj @ target_proj.T
    device = sim.device
    mask_diag = torch.eye(B, device=device).bool()
    sim_neg = sim.masked_fill(mask_diag, -1e9)
    hard_neg_idx = sim_neg.argmax(dim=-1)
    pos_logit = sim[torch.arange(B, device=device), torch.arange(B, device=device)]
    neg_logit = sim[torch.arange(B, device=device), hard_neg_idx]
    logits = torch.stack([pos_logit, neg_logit], dim=-1) / temperature
    return F.cross_entropy(logits, torch.zeros(B, dtype=torch.long, device=device))
