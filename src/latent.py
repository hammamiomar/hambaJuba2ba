"""Latent space walk generation."""

import torch
from typing import Tuple


class LatentWalk:
    """Generates circular walks in latent space."""

    def __init__(self, latent_shape: Tuple[int, int, int, int], seed: int = 42):
        """Initialize with fixed basis vectors for circular motion.

        Args:
            latent_shape: (batch, channels, height, width) in latent space
            seed: Random seed for reproducible noise
        """
        self.latent_shape = latent_shape
        self.seed = seed

        # Create two orthogonal basis vectors in latent space
        torch.manual_seed(seed)
        self.noise_x = torch.randn(latent_shape)

        torch.manual_seed(seed + 1)
        self.noise_y = torch.randn(latent_shape)

    def get_latent(self, angle: float, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get latent at given angle in the circular walk.

        Args:
            angle: Angle in radians (0 to 2π is one full rotation)
            device: Device to place tensor on ("cpu", "cuda", "mps")

        Returns:
            Latent tensor of shape (batch, channels, height, width)
        """
        # Simple circular interpolation
        latent = (
            torch.cos(torch.tensor(angle)) * self.noise_x
            + torch.sin(torch.tensor(angle)) * self.noise_y
        )

        return latent.to(device, dtype)

    def reset(self, seed: int):
        """Reset with new random basis vectors.

        Args:
            seed: New random seed
        """
        self.seed = seed
        torch.manual_seed(seed)
        self.noise_x = torch.randn(self.latent_shape)

        torch.manual_seed(seed + 1)
        self.noise_y = torch.randn(self.latent_shape)
