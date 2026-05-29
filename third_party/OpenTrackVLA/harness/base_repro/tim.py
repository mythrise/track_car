"""Target Identification Memory (TIM) — 4-token long-horizon identity memory.

TrackVLA++ multiplicative gate: g = w * q_write, where
w = C_{t-1} / (C_avg + C_{t-1}).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TIM(nn.Module):
    def __init__(self, d_model: int, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.slot_emb = nn.Parameter(torch.zeros(n_tokens, d_model))
        nn.init.normal_(self.slot_emb, std=0.02)

    def init_state(self, B: int, device, dtype=torch.float32):
        return {
            "mem": torch.zeros(B, self.n_tokens, self.d_model, device=device, dtype=dtype),
            "C_avg": torch.zeros(B, device=device, dtype=dtype),
            "C_cnt": torch.zeros(B, device=device, dtype=torch.long),
            "initialized": torch.zeros(B, device=device, dtype=torch.bool),
        }

    def compute_gate(self, C: torch.Tensor, q_write: torch.Tensor, state: dict) -> torch.Tensor:
        w = C / (state["C_avg"] + C).clamp_min(1e-6)
        return w * q_write

    def update(self, state: dict, candidate: torch.Tensor,
               C: torch.Tensor, q_write: torch.Tensor,
               invalid_mask=None) -> dict:
        B = candidate.size(0)
        g = self.compute_gate(C, q_write, state)
        if invalid_mask is not None:
            g = g.masked_fill(invalid_mask, 0.0)
        g3 = g.view(B, 1, 1)
        new_mem = (1.0 - g3) * state["mem"] + g3 * (candidate + self.slot_emb.unsqueeze(0))
        first_valid = ~state["initialized"]
        if invalid_mask is not None:
            first_valid = first_valid & (~invalid_mask)
        if first_valid.any():
            fv = first_valid.view(B, 1, 1)
            new_mem = torch.where(fv, candidate + self.slot_emb.unsqueeze(0), new_mem)
        valid_inc = (~invalid_mask).long() if invalid_mask is not None else torch.ones(B, dtype=torch.long, device=C.device)
        C_valid = C.masked_fill(invalid_mask, 0.0) if invalid_mask is not None else C
        new_cnt = state["C_cnt"] + valid_inc
        new_avg = (state["C_avg"] * state["C_cnt"].float() + C_valid) / new_cnt.clamp_min(1).float()
        return {
            "mem": new_mem,
            "C_avg": new_avg,
            "C_cnt": new_cnt,
            "initialized": state["initialized"] | (valid_inc > 0),
            "last_gate": g,
        }


def roi_pool_candidate(v_fine: torch.Tensor, theta_idx: torch.Tensor,
                       n_theta: int, n_tokens: int = 4) -> torch.Tensor:
    """Pool V_fine (B, N, D) into n_tokens candidate tokens guided by theta sector."""
    B, N, D = v_fine.shape
    sectors = torch.linspace(0, n_theta - 1, N, device=v_fine.device)
    targets = theta_idx.float().unsqueeze(-1)
    weight = torch.softmax(-((sectors.unsqueeze(0) - targets) ** 2) / 4.0, dim=-1)
    weighted = (weight.unsqueeze(-1) * v_fine).sum(dim=1)
    return weighted.unsqueeze(1).expand(-1, n_tokens, -1).contiguous()
