"""Cognitive Event Bank (CEB) — 6 types, type-preserving merge, cross-attn read."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

EVENT_TYPES = ("OCC_START", "INVALID_STREAK", "DISTRACTOR_ALERT",
               "WEAK_UPDATE", "RECOVERY_LEFT", "RECOVERY_RIGHT")


class CognitiveEventBank(nn.Module):
    def __init__(self, d_model: int, n_types: int = 6, L: int = 6):
        super().__init__()
        self.d_model = d_model
        self.n_types = n_types
        self.L = L
        self.type_emb = nn.Embedding(n_types, d_model)
        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def init_state(self, B, device):
        return {
            "tokens": torch.zeros(B, self.L, self.d_model, device=device),
            "types": torch.full((B, self.L), -1, dtype=torch.long, device=device),
            "age": torch.zeros(B, self.L, dtype=torch.long, device=device),
        }

    def write(self, state, h_t, triggers, conf):
        if not triggers:
            state["age"] = state["age"] + 1
            return state
        new = {k: v.clone() for k, v in state.items()}
        new["age"] = new["age"] + 1
        for b, type_id in triggers:
            tok = self.encoder(torch.cat([h_t[b], self.type_emb.weight[type_id]], dim=-1))
            slot = -1
            for s in range(self.L):
                if new["types"][b, s].item() == -1:
                    slot = s; break
            if slot < 0:
                same = (new["types"][b] == type_id)
                if same.any():
                    idxs = same.nonzero().squeeze(-1)
                    slot = int(idxs[new["age"][b][same].argmax()].item())
                else:
                    slot = int(new["age"][b].argmax().item())
            new["tokens"][b, slot] = tok.detach()
            new["types"][b, slot] = type_id
            new["age"][b, slot] = 0
        return new

    def read(self, state, h_t):
        B = h_t.size(0)
        K = self.k_proj(state["tokens"])
        V = self.v_proj(state["tokens"])
        Q = self.q_proj(h_t).unsqueeze(1)
        att = (Q @ K.transpose(1, 2)) * self.scale
        mask = (state["types"] < 0).unsqueeze(1)
        att = att.masked_fill(mask, -1e9)
        w = torch.softmax(att, dim=-1)
        out = (w @ V).squeeze(1)
        all_empty = (state["types"] < 0).all(dim=-1, keepdim=True)
        out = out.masked_fill(all_empty, 0.0)
        return out
