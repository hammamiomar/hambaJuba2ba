"""Shared pipeline utilities and initialization."""

import logging
import numpy as np
import cv2
import torch
from typing import Optional

from src.config import PipelineConfig
from src.pipeline import HyperSD15Pipeline

logger = logging.getLogger("uvicorn")

# Global pipeline (lazy-loaded)
_pipeline: Optional[HyperSD15Pipeline] = None
_config: Optional[PipelineConfig] = None


async def get_or_create_pipeline() -> tuple[HyperSD15Pipeline, PipelineConfig]:
    """Initialize pipeline on first request.

    Returns:
        Tuple of (pipeline, config)
    """
    global _pipeline, _config

    if _pipeline is None:
        logger.info("Initializing pipeline (takes ~1 minute first time)...")

        _config = PipelineConfig()
        _pipeline = HyperSD15Pipeline(_config)
        _pipeline.load()

        logger.info("Pipeline ready!")

    return _pipeline, _config


def tensor_to_jpeg(tensor: torch.Tensor, quality: int = 85) -> bytes:
    """Convert PyTorch tensor to JPEG bytes.

    Args:
        tensor: Image tensor (C, H, W) in range [0, 1]
        quality: JPEG quality (0-100)

    Returns:
        JPEG-encoded bytes
    """
    # to numpy (H, W, C) uint8
    arr = tensor.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # RGB → BGR
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # encode
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()
