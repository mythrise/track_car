"""iter04: RoVer-style test-time anchor rescoring.

At inference, rescore AnchorDiT candidates using Future Δ=4 polar prediction
and Verifier q_write. Zero training cost — inference only.
"""

import math
import torch


def rover_rescore(scores, denoised, q_write, fut_theta_logits, n_waypoints=8, action_dims=3):
    """Rescore anchor candidates using future polar prediction.

    Args:
        scores: (B, K) raw anchor scores
        denoised: (B, K, nw*ad) denoised candidate trajectories
        q_write: (B,) verifier confidence
        fut_theta_logits: (B, n_theta) future angle prediction

    Returns:
        (B, nw, ad) selected best candidate trajectory
    """
    B, K, _ = denoised.shape
    cand = denoised.view(B, K, n_waypoints, action_dims)
    wp0_dx = cand[..., 0, 0]
    wp0_dy = cand[..., 0, 1]
    cand_theta = torch.atan2(wp0_dy, wp0_dx)

    n_theta = fut_theta_logits.size(-1)
    fut_idx = fut_theta_logits.argmax(dim=-1)
    fut_rad = (fut_idx.float() + 0.5) * (2 * math.pi / n_theta) - math.pi
    angle_diff = (cand_theta - fut_rad.unsqueeze(-1)).cos()

    rover_score = scores + 2.0 * angle_diff * q_write.unsqueeze(-1)
    chosen = rover_score.argmax(dim=-1)
    B_idx = torch.arange(B, device=scores.device)
    return torch.tanh(cand[B_idx, chosen])
