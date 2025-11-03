"""Walk strategies for latent and embedding spaces."""

from .base import BaseWalk
from .slerp_walk import SLERPWalk
from .four_corner_walk import FourCornerWalk

__all__ = ["BaseWalk", "SLERPWalk", "FourCornerWalk"]
