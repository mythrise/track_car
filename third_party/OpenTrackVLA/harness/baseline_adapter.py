"""Uniform ``forward_step`` adapter for the no-Harness OpenTrackVLA baseline."""

from __future__ import annotations

import torch.nn as nn


class OpenTrackVLABaselineAdapter(nn.Module):
    """Expose native OpenTrackVLA through the stateful evaluator interface.

    The adapter intentionally adds no trainable modules and no persistent state.
    It exists only so the exact same offline evaluation loop can compare the
    native waypoint planner with TrackVLA++-Lite and PFEM checkpoints.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def init_state(self, batch_size: int, device):
        del batch_size, device
        return {}

    def forward_step(
        self,
        coarse_tokens,
        coarse_tidx,
        fine_tokens,
        fine_tidx,
        instructions,
        prev_state=None,
        distractor_rate=None,
        yaw_hist=None,
        yaw_curr=None,
        prev_action=None,
    ):
        del prev_state, distractor_rate, prev_action
        waypoints = self.base(
            coarse_tokens,
            coarse_tidx,
            fine_tokens,
            fine_tidx,
            instructions,
            yaw_hist=yaw_hist,
            yaw_curr=yaw_curr,
        )
        return {"waypoints": waypoints, "new_state": {}}
