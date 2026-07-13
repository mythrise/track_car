"""Shared planar kinematics for car action sequences.

Actions use the repository convention ``[forward, strafe_right, yaw_clockwise]``
and are expressed in the vehicle body frame.  Returned waypoints are poses in
the local frame at the beginning of the sequence.
"""

from __future__ import annotations

import numpy as np


def integrate_actions(actions, dt: float) -> np.ndarray:
    """Compose every action into one waypoint per input step.

    ``waypoint[k]`` includes actions ``0..k``.  Translation for a step is
    rotated by the yaw accumulated *before* that step, matching discrete pose
    composition (so a yaw step followed by forward motion moves along the new
    heading).  No action is dropped.
    """

    step_dt = float(dt)
    if not np.isfinite(step_dt) or step_dt <= 0:
        raise ValueError("dt must be a finite positive number")

    # Keep the shared implementation differentiable when it is used by the
    # step-action training head.  The numpy path remains the public data
    # pipeline behavior used by the builder and existing callers.
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a training dependency
        torch = None
    if torch is not None and isinstance(actions, torch.Tensor):
        return _integrate_actions_torch(actions, step_dt)

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] < 1:
        raise ValueError("actions must have shape (T, D) with D >= 1")
    if not np.isfinite(arr).all():
        raise ValueError("actions contain NaN or Inf")
    if arr.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = 0.0
    y = 0.0
    yaw = 0.0
    waypoints = np.zeros((arr.shape[0], 3), dtype=np.float32)
    for index, action in enumerate(arr):
        forward = float(action[0])
        strafe = float(action[1]) if arr.shape[1] > 1 else 0.0
        yaw_rate = float(action[2]) if arr.shape[1] > 2 else 0.0

        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))
        x += (cos_yaw * forward - sin_yaw * strafe) * step_dt
        y += (sin_yaw * forward + cos_yaw * strafe) * step_dt
        yaw += yaw_rate * step_dt
        waypoints[index] = (x, y, yaw)

    return waypoints


def _integrate_actions_torch(actions, dt: float):
    import torch

    arr = actions
    squeeze_batch = False
    if arr.dim() == 2:
        arr = arr.unsqueeze(0)
        squeeze_batch = True
    if arr.dim() != 3 or arr.size(-1) < 1:
        raise ValueError("actions must have shape (T, D) or (B, T, D) with D >= 1")
    if arr.size(1) == 0:
        result = arr.new_zeros((arr.size(0), 0, 3))
        return result[0] if squeeze_batch else result
    forward = arr[..., 0]
    strafe = arr[..., 1] if arr.size(-1) > 1 else torch.zeros_like(forward)
    yaw_rate = arr[..., 2] if arr.size(-1) > 2 else torch.zeros_like(forward)
    yaw_delta = yaw_rate * dt
    yaw = torch.cumsum(yaw_delta, dim=1)
    yaw_before = yaw - yaw_delta
    cos_yaw = torch.cos(yaw_before)
    sin_yaw = torch.sin(yaw_before)
    dx = (cos_yaw * forward - sin_yaw * strafe) * dt
    dy = (sin_yaw * forward + cos_yaw * strafe) * dt
    result = torch.stack(
        (torch.cumsum(dx, dim=1), torch.cumsum(dy, dim=1), yaw),
        dim=-1,
    )
    return result[0] if squeeze_batch else result
