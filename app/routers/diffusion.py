"""WebSocket endpoints for HyperSD15 streaming.

Clean and simple - load model, stream frames, done.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
import logging
import asyncio
import math
import numpy as np
import cv2
import torch
import time
from typing import Optional
from collections import deque

from src.config import PipelineConfig
from src.pipeline import HyperSD15Pipeline
from src.latent import LatentWalk

logger = logging.getLogger("uvicorn")
router = APIRouter()

# Global pipeline (lazy-loaded on first request)
# TODO: Replace with proper dependency injection for multi-client support
_pipeline: Optional[HyperSD15Pipeline] = None
_latent_walk: Optional[LatentWalk] = None
_config: Optional[PipelineConfig] = None


async def get_or_create_pipeline() -> tuple[
    HyperSD15Pipeline, LatentWalk, PipelineConfig
]:
    """Initialize pipeline on first request (slow, ~1 minute).

    Returns:
        Tuple of (pipeline, latent_walk, config)
    """
    global _pipeline, _latent_walk, _config

    if _pipeline is None:
        logger.info("Initializing pipeline (this takes ~1 minute first time)...")

        # Create config
        _config = PipelineConfig()

        # Create and load pipeline
        _pipeline = HyperSD15Pipeline(_config)
        _pipeline.load()

        # Create latent walk generator
        latent_shape = (1, 4, _config.latent_height, _config.latent_width)
        _latent_walk = LatentWalk(latent_shape, seed=_config.seed)

        logger.info("Pipeline ready!")

    return _pipeline, _latent_walk, _config


def tensor_to_jpeg(tensor: torch.Tensor, quality: int = 85) -> bytes:
    """Convert PyTorch tensor (C, H, W) to JPEG bytes.

    Args:
        tensor: Image tensor in range [0, 1]
        quality: JPEG quality (0-100)

    Returns:
        JPEG-encoded bytes
    """
    # Convert to numpy (H, W, C) uint8
    arr = tensor.cpu().permute(1, 2, 0).numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    # RGB → BGR (OpenCV convention)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # Encode JPEG
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


@router.websocket("/ws/generate/")
async def stream_latent_walk(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    try:
        pipeline, latent_walk, config = await get_or_create_pipeline()

        angle = 0.0
        frame_count = 0

        latent_times = deque(maxlen=30)
        inference_times = deque(maxlen=30)
        jpeg_times = deque(maxlen=30)
        send_times = deque(maxlen=30)
        loop_times = deque(maxlen=30)

        logger.info("Starting stream at MAX FPS")

        while True:
            loop_start = time.perf_counter()

            t0 = time.perf_counter()
            latent = latent_walk.get_latent(
                angle, device=config.device, dtype=config.get_torch_dtype()
            )
            latent_times.append(time.perf_counter() - t0)

            t1 = time.perf_counter()
            frame_tensor = pipeline.generate(latent)
            inference_times.append(time.perf_counter() - t1)

            t2 = time.perf_counter()
            frame_bytes = tensor_to_jpeg(frame_tensor, quality=config.jpeg_quality)
            jpeg_times.append(time.perf_counter() - t2)

            t3 = time.perf_counter()
            await websocket.send_bytes(frame_bytes)
            send_times.append(time.perf_counter() - t3)

            loop_times.append(time.perf_counter() - loop_start)

            angle = (angle + config.walk_speed) % (2 * math.pi)
            frame_count += 1

            if frame_count % 30 == 0 and len(loop_times) == 30:
                gen_time = sum(latent_times) + sum(inference_times) + sum(jpeg_times)
                gen_fps = 30.0 / gen_time
                delivery_fps = 30.0 / sum(loop_times)

                logger.info(
                    f"\nFrame {frame_count} | MAX FPS"
                    f"\nGeneration: {gen_fps:.1f} FPS"
                    f"\nDelivery:   {delivery_fps:.1f} FPS"
                    f"\nLatent:     {sum(latent_times) / 30 * 1000:.1f}ms"
                    f"\nInference:  {sum(inference_times) / 30 * 1000:.1f}ms"
                    f"\nJPEG:       {sum(jpeg_times) / 30 * 1000:.1f}ms"
                    f"\nWebSocket:  {sum(send_times) / 30 * 1000:.1f}ms"
                )

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in stream: {e}", exc_info=True)
    finally:
        if _pipeline:
            _pipeline.cleanup()


@router.get("/test/random")
async def test_random_image() -> Response:
    """Test endpoint: return random noise image."""
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

    # Encode to JPEG
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.websocket("/ws/test/")
async def test_stream_random(websocket: WebSocket):
    """Test endpoint: stream random noise at target FPS."""
    await websocket.accept()
    logger.info("Test client connected")

    try:
        target_fps = 30
        frame_time = 1.0 / target_fps

        while True:
            loop_start = time.perf_counter()

            # Generate random frame
            arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
            arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            _, buf = cv2.imencode(".jpg", arr_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Send to client
            await websocket.send_bytes(buf.tobytes())

            # Sleep for FPS
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0, frame_time - elapsed)
            await asyncio.sleep(sleep_time)

    except WebSocketDisconnect:
        logger.info("Test client disconnected")
    except Exception as e:
        logger.error(f"Error in test stream: {e}", exc_info=True)
