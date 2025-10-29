"""8-corner 3D cube walk for latents and prompt embeddings."""

import torch
from typing import Tuple, List


class FourCornerWalk:
    """3D walk in 8-corner cube space (works for latents or embeddings)."""

    def __init__(self, corners: List[torch.Tensor], seed: int = 42):
        """Initialize with 8 corner tensors forming a cube.

        Args:
            corners: List of 8 tensors defining cube vertices
            seed: Random seed
        """
        assert len(corners) == 8, "Need exactly 8 corners for cube"
        self.corners = corners
        self.x = torch.tensor(0.5)
        self.y = torch.tensor(0.5)
        self.z = torch.tensor(0.5)
        self.generator = torch.Generator().manual_seed(seed)

    @classmethod
    def from_latent_noise(
        cls,
        latent_shape: Tuple[int, int, int, int],
        seed: int = 42,
    ):
        """Create walk from 8 random latent noise tensors (cube vertices)."""
        torch.manual_seed(seed)
        corners = [torch.randn(latent_shape) for _ in range(8)]
        return cls(corners, seed)

    @classmethod
    def from_prompt_embeddings(
        cls,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        embed_c: torch.Tensor,
        embed_d: torch.Tensor,
        seed: int = 42,
    ):
        """Create 8-corner cube from 4 prompt embeddings.

        Bottom square (z=0): A & B with orthogonal midpoint
        Top square (z=1): C & D with orthogonal midpoint
        """
        def make_square(e1: torch.Tensor, e2: torch.Tensor) -> list[torch.Tensor]:
            mid = (e1 + e2) / 2
            direction = e2 - e1
            distance = torch.norm(direction) / 2

            ortho = torch.zeros_like(direction)
            ortho[..., 0] = -direction[..., 1]
            ortho[..., 1] = direction[..., 0]
            ortho = ortho / (torch.norm(ortho) + 1e-8) * distance

            return [
                e1,
                mid + ortho,
                e2,
                mid - ortho,
            ]

        bottom = make_square(embed_a, embed_b)
        top = make_square(embed_c, embed_d)

        # 8 corners: bottom 4 then top 4
        corners = bottom + top
        return cls(corners, seed)

    def step(self, delta_std: float = 0.05):
        """Take random step in 3D space."""
        dx = torch.randn(1, generator=self.generator).item() * delta_std
        dy = torch.randn(1, generator=self.generator).item() * delta_std
        dz = torch.randn(1, generator=self.generator).item() * delta_std

        self.x = torch.clamp(self.x + dx, 0.0, 1.0)
        self.y = torch.clamp(self.y + dy, 0.0, 1.0)
        self.z = torch.clamp(self.z + dz, 0.0, 1.0)

    def step_directed(self, dx: float, dy: float, dz: float, magnitude: float = 1.0):
        """Step in user-specified 3D direction.

        Args:
            dx: X direction in [-1, 1]
            dy: Y direction in [-1, 1]
            dz: Z direction in [-1, 1]
            magnitude: Speed multiplier
        """
        step_size = magnitude * 0.05
        self.x = torch.clamp(self.x + dx * step_size, 0.0, 1.0)
        self.y = torch.clamp(self.y + dy * step_size, 0.0, 1.0)
        self.z = torch.clamp(self.z + dz * step_size, 0.0, 1.0)

    def get_position(self) -> tuple[float, float, float]:
        """Get current position in 3D space [0, 1]^3."""
        return (self.x.item(), self.y.item(), self.z.item())

    def get(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get interpolated tensor using trilinear interpolation."""
        x, y, z = self.x, self.y, self.z

        # Trilinear interpolation across 8 cube corners
        c000 = self.corners[0] * (1 - x) * (1 - y) * (1 - z)
        c100 = self.corners[1] * x * (1 - y) * (1 - z)
        c010 = self.corners[2] * (1 - x) * y * (1 - z)
        c110 = self.corners[3] * x * y * (1 - z)
        c001 = self.corners[4] * (1 - x) * (1 - y) * z
        c101 = self.corners[5] * x * (1 - y) * z
        c011 = self.corners[6] * (1 - x) * y * z
        c111 = self.corners[7] * x * y * z

        return (c000 + c100 + c010 + c110 + c001 + c101 + c011 + c111).to(device, dtype)
