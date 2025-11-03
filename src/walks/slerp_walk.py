"""Linear SLERP walk between source and target."""

import torch
from typing import List
from .base import BaseWalk
from ..interpolation import slerp


class SLERPWalk(BaseWalk):
    """Ping-pong walk via SLERP interpolation."""

    def __init__(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        step_size: float = 0.01,
    ):
        """Initialize SLERP walk.

        Args:
            source: Starting tensor
            target: Ending tensor
            step_size: How much to advance per step [0, 1]
        """
        self.source = source
        self.target = target
        self.step_size = step_size
        self.t = 0.0
        self.direction = 1  # 1 = forward, -1 = reverse

    def step(self):
        """Advance position, ping-pong at boundaries."""
        self.t += self.direction * self.step_size

        # bounce at edges
        if self.t >= 1.0:
            self.t = 1.0
            self.direction = -1
        elif self.t <= 0.0:
            self.t = 0.0
            self.direction = 1

    def set_position(self, t: float):
        """Manually set interpolation position."""
        self.t = max(0.0, min(1.0, t))

    def get(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get interpolated tensor at current position."""
        result = slerp(self.source, self.target, self.t)
        return result.to(device, dtype)

    def get_batch(
        self,
        batch_size: int,
        device: str,
        dtype: torch.dtype
    ) -> tuple[torch.Tensor, List[float]]:
        """Get batch of progressive interpolations.

        Returns:
            Tuple of (batch tensor, list of t positions)
        """
        t_values = []
        for i in range(batch_size):
            t_values.append(self.t)
            self.step()

        # generate batch
        results = []
        for t in t_values:
            results.append(slerp(self.source, self.target, t))

        batch = torch.cat(results, dim=0).to(device, dtype)
        return batch, t_values

    def get_position(self) -> tuple[float, float]:
        """Get current position (returns t and direction for viz)."""
        return (self.t, float(self.direction))
