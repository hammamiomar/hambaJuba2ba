"""SLERP interpolation for tensors."""

import torch
from typing import List


def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """Spherical linear interpolation between two tensors.

    Args:
        v0: Start tensor
        v1: End tensor
        t: Interpolation factor in [0, 1]
        eps: Small value for numerical stability

    Returns:
        Interpolated tensor
    """
    # normalize
    v0_norm = v0 / (torch.norm(v0, dim=-1, keepdim=True) + eps)
    v1_norm = v1 / (torch.norm(v1, dim=-1, keepdim=True) + eps)

    # dot product
    dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    # angle
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # handle small angles (fall back to lerp)
    if (sin_theta.abs() < eps).any():
        return (1 - t) * v0 + t * v1

    # slerp
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta

    return s0 * v0 + s1 * v1


def slerp_batch(
    v0: torch.Tensor,
    v1: torch.Tensor,
    t_values: List[float]
) -> torch.Tensor:
    """Batch SLERP interpolation at multiple t values.

    Args:
        v0: Start tensor (single)
        v1: End tensor (single)
        t_values: List of interpolation factors

    Returns:
        Batch of interpolated tensors, shape (len(t_values), *v0.shape)
    """
    results = []
    for t in t_values:
        results.append(slerp(v0, v1, t))
    return torch.cat(results, dim=0)


def generate_latent_from_seed(
    seed: int,
    latent_shape: tuple,
    device: str,
    dtype: torch.dtype
) -> torch.Tensor:
    """Generate latent noise from seed.

    Args:
        seed: Random seed
        latent_shape: (batch, channels, height, width)
        device: Device to place tensor on
        dtype: Data type

    Returns:
        Random latent tensor
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(latent_shape, generator=generator, device=device, dtype=dtype)
