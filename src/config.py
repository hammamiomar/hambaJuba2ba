"""Simple configuration for HyperSD15 streaming pipeline."""

from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class PipelineConfig:
    """
    Configuration for HyperSD15 streaming pipeline.
    """

    # Model settings
    model_id: str = "runwayml/stable-diffusion-v1-5"
    lora_id: str = "ByteDance/Hyper-SD"
    lora_filename: str = "Hyper-SD15-1step-lora.safetensors"

    # Device and dtype
    device: Literal["cuda", "mps", "cpu"] = "mps"
    dtype: Literal["float16", "float32", "bfloat16"] = "float16"

    # Generation settings
    height: int = 384  # 384? or 512
    width: int = 384
    num_inference_steps: int = 1  # HyperSD supports 1-step
    guidance_scale: float = 0.0  # No CFG for speed
    batch_size: int = 4  # Frames per batch for streambatch optimization

    # Optimizations
    use_tiny_vae: bool = True  # 3-5x faster decoding

    # Latent walk settings
    walk_speed: float = 0.02  # Radians per frame
    seed: int = 42

    # Streaming settings
    jpeg_quality: int = 85  # JPEG compression quality

    def get_torch_dtype(self) -> torch.dtype:
        """Convert string dtype to torch dtype."""
        return {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }[self.dtype]

    @property
    def latent_height(self) -> int:
        """Height in latent space (VAE downsamples by 8)."""
        return self.height // 8

    @property
    def latent_width(self) -> int:
        """Width in latent space (VAE downsamples by 8)."""
        return self.width // 8
