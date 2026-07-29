"""Target Identification Memory (TIM) — 4-token long-horizon identity memory.

TrackVLA++ confidence gate: w = C_{t-1} / (C_avg + C_{t-1}).
The optional q_write multiplier is a PFEM extension; TrackVLA++-Lite passes 1.
"""

from __future__ import annotations
import math
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
               invalid_mask=None, *, count_invalid_in_average: bool = False) -> dict:
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
        initialized_inc = (
            (~invalid_mask)
            if invalid_mask is not None
            else torch.ones(B, dtype=torch.bool, device=C.device)
        )
        if count_invalid_in_average:
            # TrackVLA++ Eq. (4)-(5): invalid implies C=0 but still occupies a
            # timestep in the historical confidence average.
            valid_inc = torch.ones(B, dtype=torch.long, device=C.device)
        else:
            valid_inc = (~invalid_mask).long() if invalid_mask is not None else torch.ones(B, dtype=torch.long, device=C.device)
        C_valid = C.masked_fill(invalid_mask, 0.0) if invalid_mask is not None else C
        new_cnt = state["C_cnt"] + valid_inc
        new_avg = (state["C_avg"] * state["C_cnt"].float() + C_valid) / new_cnt.clamp_min(1).float()
        return {
            "mem": new_mem,
            "C_avg": new_avg,
            "C_cnt": new_cnt,
            "initialized": state["initialized"] | initialized_inc,
            "last_gate": g,
        }


def roi_pool_candidate(v_fine: torch.Tensor, theta_idx: torch.Tensor,
                       n_theta: int, n_tokens: int = 4,
                       horizontal_fov_deg: float = 60.0) -> torch.Tensor:
    """Select spatial 8x8-grid tokens from the Polar-CoT horizontal sector.

    Grid-pooled fine tokens are row-major image patches.  Polar angle controls
    the image column; it must not be compared with the flattened token index.
    Without a vertical target coordinate, ``n_tokens`` rows are sampled down
    that column to retain top-to-bottom appearance cues for identity memory.
    Angles outside the single-camera FoV clamp to the nearest edge column.
    """
    B, N, D = v_fine.shape
    side = math.isqrt(N)
    if side * side != N:
        raise ValueError(f"fine token count must form a square grid, got N={N}")
    if n_tokens <= 0:
        raise ValueError("n_tokens must be positive")
    theta_deg = (theta_idx.float() + 0.5) * (360.0 / float(n_theta)) - 180.0
    normalized_x = (theta_deg + horizontal_fov_deg / 2.0) / horizontal_fov_deg
    columns = torch.round(normalized_x.clamp(0.0, 1.0) * (side - 1)).long()
    rows = torch.linspace(0, side - 1, steps=n_tokens, device=v_fine.device)
    rows = torch.round(rows).long()
    grid = v_fine.view(B, side, side, D)
    batch_index = torch.arange(B, device=v_fine.device).unsqueeze(1)
    return grid[
        batch_index,
        rows.unsqueeze(0).expand(B, -1),
        columns.unsqueeze(1).expand(-1, n_tokens),
    ].contiguous()
