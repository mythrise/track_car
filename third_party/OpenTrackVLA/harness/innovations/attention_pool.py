"""iter25: Learned attention pooling over visual tokens."""

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, tokens):
        B = tokens.size(0)
        q = self.query.expand(B, -1).unsqueeze(1)
        K = self.k(tokens)
        V = self.v(tokens)
        att = (q @ K.transpose(1, 2)) * self.scale
        return (torch.softmax(att, dim=-1) @ V).squeeze(1)
