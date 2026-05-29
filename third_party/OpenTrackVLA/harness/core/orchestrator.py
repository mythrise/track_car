"""Orchestrator — soft mode (no CE supervision) + metadata tokens + alpha weights."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("NORMAL", "CAUTIOUS", "SEARCH_LEFT", "SEARCH_RIGHT")
META_FIELDS = ("identity_conf", "occlusion_risk", "crowd_level",
               "aggressiveness", "reacq_patience")


class Orchestrator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mode_table = nn.Embedding(len(MODES), d_model)
        _bin_sizes = {"identity_conf": 5, "occlusion_risk": 5, "crowd_level": 3,
                      "aggressiveness": 3, "reacq_patience": 3}
        self.meta_embs = nn.ModuleDict({f: nn.Embedding(_bin_sizes[f], d_model) for f in META_FIELDS})
        self.mode_mlp = nn.Sequential(
            nn.LayerNorm(d_model * len(META_FIELDS)),
            nn.Linear(d_model * len(META_FIELDS), 64), nn.GELU(),
            nn.Linear(64, len(MODES)),
        )
        self.alpha_mlp = nn.Sequential(
            nn.LayerNorm(d_model * len(META_FIELDS)),
            nn.Linear(d_model * len(META_FIELDS), 32), nn.GELU(),
            nn.Linear(32, 4),
        )

    @staticmethod
    def _bin(x, n):
        return (x.clamp(0, 1) * (n - 1)).long()

    def compose_metadata(self, C, q_write, invalid_streak, distractor_rate):
        return {
            "identity_conf": self._bin(C, 5),
            "occlusion_risk": self._bin(invalid_streak, 5),
            "crowd_level": self._bin(distractor_rate, 3),
            "aggressiveness": torch.ones_like(C, dtype=torch.long),
            "reacq_patience": torch.ones_like(C, dtype=torch.long),
        }

    def forward(self, meta_bins, drop_meta=False):
        toks = []
        for f in META_FIELDS:
            e = self.meta_embs[f](meta_bins[f])
            if drop_meta:
                e = torch.zeros_like(e)
            toks.append(e)
        stack = torch.stack(toks, dim=1)
        flat = stack.flatten(1)
        mode_logits = self.mode_mlp(flat)
        mode_p = torch.softmax(mode_logits, dim=-1)
        mode_emb = mode_p @ self.mode_table.weight
        alpha = torch.softmax(self.alpha_mlp(flat), dim=-1)
        return {
            "mode": mode_p.argmax(dim=-1),
            "mode_p": mode_p,
            "mode_emb": mode_emb,
            "meta_tokens": stack,
            "alpha_tim": alpha[:, 0],
            "alpha_evt": alpha[:, 1],
            "alpha_fut": alpha[:, 2],
            "alpha_verify": alpha[:, 3],
        }
