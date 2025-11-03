"""Abstract base class for walk strategies."""

from abc import ABC, abstractmethod
import torch


class BaseWalk(ABC):
    """Base interface for latent/embedding walks."""

    @abstractmethod
    def step(self, **kwargs):
        """Advance the walk by one step."""
        pass

    @abstractmethod
    def get(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        """Get current interpolated tensor."""
        pass

    @abstractmethod
    def get_position(self) -> tuple[float, ...]:
        """Get current position in walk space (for visualization)."""
        pass
