"""Helpers for rolling model state across ordered collected-data clips."""

from __future__ import annotations

import torch


def sample_sequence_key(batch: dict, index: int = 0):
    """Return the stable identity and frame index used for reset decisions."""

    sequence_ids = (
        batch.get("sequence_id")
        or batch.get("chunk_id")
        or batch.get("clip_id")
        or batch.get("episode")
        or [""]
    )
    sequence_id = str(sequence_ids[index])
    frame_idx = int(batch["frame_idx"][index].item())
    mirrored = bool(batch.get("mirrored", torch.zeros(1, dtype=torch.bool))[index].item())
    return sequence_id, frame_idx, mirrored


def continues_sequence(previous_key, current_key) -> bool:
    """True only for consecutive, non-mirrored samples in the same clip."""

    if previous_key is None:
        return False
    previous_sequence, previous_frame, previous_mirrored = previous_key
    current_sequence, current_frame, current_mirrored = current_key
    return (
        not previous_mirrored
        and not current_mirrored
        and current_sequence == previous_sequence
        and current_frame == previous_frame + 1
    )


def detach_state(value):
    """Detach a nested rolling state for one-step truncated BPTT."""

    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_state(item) for key, item in value.items()}
    if isinstance(value, list):
        return [detach_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(detach_state(item) for item in value)
    return value
