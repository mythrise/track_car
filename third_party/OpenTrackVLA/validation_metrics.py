"""Shared validation-selection metrics for matched tracking experiments."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


PRIMARY_COMMANDS = (
    "forward",
    "turn_left",
    "turn_right",
    "backward",
    "stop",
)


def waypoints_to_step_actions(waypoints, dt: float):
    """Invert the shared discrete planar pose composition."""

    poses = np.asarray(waypoints, dtype=np.float64)
    if poses.ndim == 2:
        poses = poses[None, ...]
        squeeze = True
    elif poses.ndim == 3:
        squeeze = False
    else:
        raise ValueError("waypoints must have shape (T, 3) or (B, T, 3)")
    if poses.shape[-1] != 3 or float(dt) <= 0:
        raise ValueError("waypoints must have 3 axes and dt must be > 0")
    previous = np.concatenate((np.zeros_like(poses[:, :1]), poses[:, :-1]), axis=1)
    world_delta = poses[..., :2] - previous[..., :2]
    yaw_before = previous[..., 2]
    cos_yaw = np.cos(yaw_before)
    sin_yaw = np.sin(yaw_before)
    forward = (
        cos_yaw * world_delta[..., 0] + sin_yaw * world_delta[..., 1]
    ) / float(dt)
    strafe = (
        -sin_yaw * world_delta[..., 0] + cos_yaw * world_delta[..., 1]
    ) / float(dt)
    yaw = (poses[..., 2] - previous[..., 2]) / float(dt)
    actions = np.stack((forward, strafe, yaw), axis=-1)
    return actions[0] if squeeze else actions


class BalancedControlAccumulator:
    """Accumulate episode-macro, command-balanced first-action error."""

    def __init__(self):
        self._errors = defaultdict(lambda: defaultdict(list))
        self._support = defaultdict(lambda: defaultdict(int))

    def add(
        self,
        pred_actions,
        target_actions,
        valid_mask,
        commands,
        episodes,
    ) -> None:
        prediction = np.asarray(pred_actions, dtype=np.float64)
        target = np.asarray(target_actions, dtype=np.float64)
        valid = np.asarray(valid_mask, dtype=bool)
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("predicted and target actions must share shape (B,T,3)")
        if valid.shape != prediction.shape[:2]:
            raise ValueError("valid_mask must have shape (B,T)")
        if len(commands) != prediction.shape[0] or len(episodes) != prediction.shape[0]:
            raise ValueError("commands/episodes must match batch size")
        for index, (command, episode) in enumerate(zip(commands, episodes)):
            command = str(command)
            episode = str(episode or "unknown")
            if command not in PRIMARY_COMMANDS or not bool(valid[index, 0]):
                continue
            error = (
                abs(float(prediction[index, 0, 0] - target[index, 0, 0]))
                + 2.0
                * abs(float(prediction[index, 0, 2] - target[index, 0, 2]))
            ) / 3.0
            self._errors[episode][command].append(error)
            self._support[episode][command] += 1

    def compute(self) -> dict:
        by_episode = {}
        for episode, commands in self._errors.items():
            command_means = [
                float(np.mean(values)) for values in commands.values() if values
            ]
            if command_means:
                by_episode[episode] = float(np.mean(command_means))
        return {
            "value": (
                float(np.mean(list(by_episode.values()))) if by_episode else None
            ),
            "by_episode": dict(sorted(by_episode.items())),
            "support": {
                episode: dict(sorted(commands.items()))
                for episode, commands in sorted(self._support.items())
            },
        }
