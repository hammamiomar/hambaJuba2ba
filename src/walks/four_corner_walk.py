"""2D SLERP grid walk for matrix-based exploration."""

import torch
from typing import List
from .base import BaseWalk
from ..interpolation import slerp


class FourCornerWalk(BaseWalk):
    """2D SLERP grid walk with intuitive matrix control."""

    def __init__(self, corners: List[torch.Tensor]):
        """Initialize with 4 corner tensors [NW, NE, SE, SW].

        Args:
            corners: List of 4 tensors (top-left, top-right, bottom-right, bottom-left)
        """
        assert len(corners) == 4, "Need exactly 4 corners"
        self.corner_nw = corners[0]  # top-left
        self.corner_ne = corners[1]  # top-right
        self.corner_se = corners[2]  # bottom-right
        self.corner_sw = corners[3]  # bottom-left

        self.x = 0.5  # horizontal position [0, 1]
        self.y = 0.5  # vertical position [0, 1]

    @classmethod
    def from_latent_noise(cls, latent_shape: tuple, seed: int = 42):
        """Create walk from 4 random latent noise tensors."""
        torch.manual_seed(seed)
        corners = [torch.randn(latent_shape) for _ in range(4)]
        return cls(corners)

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

        # orthogonal offset
        ortho = torch.zeros_like(direction)
        ortho[..., 0] = -direction[..., 1]
        ortho[..., 1] = direction[..., 0]
        ortho = ortho / (torch.norm(ortho) + 1e-8) * distance

        # NW, NE, SE, SW
        corners = [
            embed_src,           # top-left
            mid + ortho,         # top-right
            embed_tgt,           # bottom-right
            mid - ortho,         # bottom-left
        ]
        return cls(corners)

    def step(self):
        """Random walk step (for exploration mode)."""
        # small random movement
        self.x = max(0.0, min(1.0, self.x + torch.randn(1).item() * 0.02))
        self.y = max(0.0, min(1.0, self.y + torch.randn(1).item() * 0.02))

    def step_directed(self, dx: float, dy: float, magnitude: float = 1.0):
        """Move in user-specified direction.

        Args:
            dx: X direction in [-1, 1]
            dy: Y direction in [-1, 1]
            magnitude: Speed multiplier
        """
        self.x = max(0.0, min(1.0, self.x + dx * magnitude * 0.05))
        self.y = max(0.0, min(1.0, self.y + dy * magnitude * 0.05))

    def set_position(self, x: float, y: float):
        """Set position directly from matrix control."""
        self.x = max(0.0, min(1.0, x))
        self.y = max(0.0, min(1.0, y))

    def get_position(self) -> tuple[float, float]:
        """Get current position in 2D space [0, 1] x [0, 1]."""
        return (self.x, self.y)

    def get(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get interpolated tensor at current position via 2D SLERP."""
        # horizontal SLERP at top and bottom
        top = slerp(self.corner_nw, self.corner_ne, self.x)
        bottom = slerp(self.corner_sw, self.corner_se, self.x)

        # vertical SLERP between results
        result = slerp(top, bottom, self.y)

        return result.to(device, dtype)
