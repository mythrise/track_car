"""iter27: Stochastic frame augmentation for training robustness."""

import torch


def augment_features(feats: torch.Tensor, noise_std: float = 0.03, channel_drop: float = 0.1):
    """Apply Gaussian noise + channel dropout to feature tensors during training."""
    if not feats.requires_grad:
        feats = feats.clone()
    feats = feats + noise_std * torch.randn_like(feats)
    mask = (torch.rand(feats.shape[-1], device=feats.device) > channel_drop).float()
    return feats * mask
