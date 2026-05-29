"""Polar-CoT head for OpenTrackVLA — joint (60θ + 30d + invalid) classification.

Taps into the LLM's last hidden state h_act and produces:
- theta_logits (B, 60): angular bin classification
- dist_logits (B, 30): distance bin classification
- invalid_logit (B,): target not visible / occluded flag
- confidence (B,): normalized entropy-based confidence score
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PolarCoTHead(nn.Module):
    def __init__(self, d_model: int, n_theta: int = 60, n_dist: int = 30,
                 dist_min: float = 0.6, dist_max: float = 5.0):
        super().__init__()
        self.n_theta = n_theta
        self.n_dist = n_dist
        self.dist_min = dist_min
        self.dist_max = dist_max
        self.norm = nn.LayerNorm(d_model)
        self.theta_head = nn.Linear(d_model, n_theta)
        self.dist_head = nn.Linear(d_model, n_dist)
        self.invalid_head = nn.Linear(d_model, 1)
        self._init()

    def _init(self):
        for m in (self.theta_head, self.dist_head, self.invalid_head):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> dict:
        h = self.norm(h.float())
        return {
            "theta_logits": self.theta_head(h),
            "dist_logits": self.dist_head(h),
            "invalid_logit": self.invalid_head(h).squeeze(-1),
        }

    @torch.no_grad()
    def decode(self, out: dict) -> dict:
        invalid_p = torch.sigmoid(out["invalid_logit"])
        theta_idx = out["theta_logits"].argmax(dim=-1)
        dist_idx = out["dist_logits"].argmax(dim=-1)
        theta_deg = (theta_idx.float() + 0.5) * (360.0 / self.n_theta) - 180.0
        dist_m = self.dist_min + (dist_idx.float() + 0.5) * (
            (self.dist_max - self.dist_min) / self.n_dist
        )
        p_t = F.softmax(out["theta_logits"], dim=-1)
        p_d = F.softmax(out["dist_logits"], dim=-1)
        H_t = -(p_t * p_t.clamp_min(1e-9).log()).sum(-1)
        H_d = -(p_d * p_d.clamp_min(1e-9).log()).sum(-1)
        conf = 0.5 * ((1.0 - H_t / math.log(self.n_theta)) +
                       (1.0 - H_d / math.log(self.n_dist))) * (1.0 - invalid_p)
        return {
            "theta_idx": theta_idx, "dist_idx": dist_idx,
            "invalid_pred": invalid_p > 0.5,
            "theta_deg": theta_deg, "dist_m": dist_m,
            "confidence": conf.clamp(0, 1),
        }


def polar_cot_loss(out, theta_gt, dist_gt, invalid_gt, label_smoothing=0.02):
    valid = (invalid_gt < 0.5).float()
    L_t = F.cross_entropy(out["theta_logits"], theta_gt.clamp_min(0),
                          reduction="none", label_smoothing=label_smoothing)
    L_d = F.cross_entropy(out["dist_logits"], dist_gt.clamp_min(0),
                          reduction="none", label_smoothing=label_smoothing)
    L_t = (L_t * valid).sum() / valid.sum().clamp_min(1.0)
    L_d = (L_d * valid).sum() / valid.sum().clamp_min(1.0)
    L_inv = F.binary_cross_entropy_with_logits(out["invalid_logit"], invalid_gt)
    return L_t + L_d + L_inv
