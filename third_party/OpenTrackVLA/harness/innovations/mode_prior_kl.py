"""iter26: Mode-prior KL regularization against collapse."""

import torch
import torch.nn.functional as F


def mode_prior_kl(mode_p: torch.Tensor, weight: float = 0.05):
    """KL(mode_p || uniform) regularizer to prevent mode collapse."""
    uniform = torch.ones_like(mode_p) / mode_p.size(-1)
    # F.kl_div(input=log Q, target=P) computes KL(P || Q).
    # We want KL(mode_p || uniform), so: input=log(uniform), target=mode_p.
    return weight * F.kl_div(uniform.log(), mode_p, reduction="batchmean")
