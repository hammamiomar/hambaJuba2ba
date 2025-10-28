"""4-corner random walk for latents and prompt embeddings."""

import torch
from typing import Tuple, List


class FourCornerWalk:
    """Random walk in 4-corner space (works for latents or embeddings)."""

    def __init__(self, corners: List[torch.Tensor], seed: int = 42):
        """Initialize with 4 corner tensors.

        Args:
            corners: List of 4 tensors defining the corners
            seed: Random seed
        """
        assert len(corners) == 4, "Need exactly 4 corners"
        self.corners = corners
        self.angle = torch.tensor(0.0)
        self.generator = torch.Generator().manual_seed(seed)

    @classmethod
    def from_latent_noise(
        cls,
        latent_shape: Tuple[int, int, int, int],
        seed: int = 42,
    ):
        """Create walk from 4 random latent noise tensors."""
        torch.manual_seed(seed)
        corners = [torch.randn(latent_shape) for _ in range(4)]
        return cls(corners, seed)

    @classmethod
    def from_prompt_embeddings(
        cls,
        embed_src: torch.Tensor,
        embed_tgt: torch.Tensor,
        seed: int = 42,
    ):
        """Create walk from source and target prompt embeddings."""
        mid = (embed_src + embed_tgt) / 2
        direction = embed_tgt - embed_src
        distance = torch.norm(direction) / 2

        # Orthogonal offset
        ortho = torch.zeros_like(direction)
        ortho[..., 0] = -direction[..., 1]
        ortho[..., 1] = direction[..., 0]
        ortho = ortho / (torch.norm(ortho) + 1e-8) * distance

        corners = [embed_src, mid + ortho, embed_tgt, mid - ortho]
        return cls(corners, seed)

    def step(self, delta_std: float = 0.05):
        """Take random step."""
        delta = torch.randn(1, generator=self.generator).item() * delta_std
        self.angle += delta
        self.angle = self.angle % (2 * torch.pi)

    def step_directed(self, dx: float, dy: float, magnitude: float = 1.0):
        """Step in user-specified direction.

        Args:
            dx: X direction in [-1, 1]
            dy: Y direction in [-1, 1]
            magnitude: Speed multiplier
        """
        delta_angle = torch.atan2(torch.tensor(dy), torch.tensor(dx)).item()
        step_size = magnitude * 0.1
        self.angle = (self.angle + delta_angle * step_size) % (2 * torch.pi)

    def get_position(self) -> tuple[float, float]:
        """Get current position in 2D space [0, 1] x [0, 1]."""
        x = (torch.cos(self.angle) + 1) / 2
        y = (torch.sin(self.angle) + 1) / 2
        return (x.item(), y.item())

    def get(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get interpolated tensor at current angle."""
        x = (torch.cos(self.angle) + 1) / 2
        y = (torch.sin(self.angle) + 1) / 2

        # Bilinear interpolation
        c0 = self.corners[0] * (1 - x) * (1 - y)
        c1 = self.corners[1] * (1 - x) * y
        c2 = self.corners[2] * x * y
        c3 = self.corners[3] * x * (1 - y)

        return (c0 + c1 + c2 + c3).to(device, dtype)
