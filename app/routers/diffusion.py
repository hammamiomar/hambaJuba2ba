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
from src.walk import FourCornerWalk

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


def generate_batch_sync(
    pipeline,
    noise_walk,
    prompt_walk,
    config,
    batch_size,
    prompt_vec,
    latent_vec,
) -> tuple[list[bytes], dict]:
    """GPU-bound batch generation (runs in executor)."""
    t0 = time.perf_counter()

    # Step walks and collect batch
    latents_list = []
    embeds_list = []
    for i in range(batch_size):
        # Use directed step if user is controlling, else random
        if prompt_vec[3] > 0.01:
            prompt_walk.step_directed(
                prompt_vec[0], prompt_vec[1], prompt_vec[2], prompt_vec[3]
            )
        else:
            prompt_walk.step()

        if latent_vec[3] > 0.01:
            noise_walk.step_directed(
                latent_vec[0], latent_vec[1], latent_vec[2], latent_vec[3]
            )
        else:
            noise_walk.step()

        latents_list.append(noise_walk.get(config.device, config.get_torch_dtype()))
        embeds_list.append(prompt_walk.get(config.device, config.get_torch_dtype()))

    latents_batch = torch.cat(latents_list, dim=0)
    embeds_batch = torch.cat(embeds_list, dim=0)
    walk_time = time.perf_counter() - t0

    # Get positions for edge detection
    prompt_pos = prompt_walk.get_position()
    latent_pos = noise_walk.get_position()

    # Inference
    t1 = time.perf_counter()
    frames = pipeline.generate_batch(latents_batch, prompt_embeds=embeds_batch)
    inference_time = time.perf_counter() - t1

    # Encode to JPEG
    t2 = time.perf_counter()
    frame_bytes_list = [
        tensor_to_jpeg(frame, quality=config.jpeg_quality) for frame in frames
    ]
    jpeg_time = time.perf_counter() - t2

    metrics = {
        "walk_time": walk_time,
        "inference_time": inference_time,
        "jpeg_time": jpeg_time,
        "batch_time": time.perf_counter() - t0,
        "prompt_pos": prompt_pos,
        "latent_pos": latent_pos,
    }

    return frame_bytes_list, metrics


@router.websocket("/ws/generate/")
async def stream_latent_walk(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected")

    producer_task = None
    consumer_task = None
    stop_event = asyncio.Event()

    # Direction control state (dx, dy, dz, magnitude)
    direction_state = {
        "prompt_vec": [0.0, 0.0, 0.0, 0.0],
        "latent_vec": [0.0, 0.0, 0.0, 0.0],
    }

    try:
        pipeline, latent_walk, config = await get_or_create_pipeline()

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "start":
                source_prompt = msg.get(
                    "source_prompt", "Burger covered in sopping wet oil in a gutter"
                )
                target_prompt = msg.get("target_prompt", "steamy burger")
                prompt_c = msg.get("prompt_c", "crispy golden fries")
                prompt_d = msg.get("prompt_d", "fresh green salad")
                logger.info(
                    f"Starting: A={source_prompt}, B={target_prompt}, C={prompt_c}, D={prompt_d}"
                )

                # Encode all 4 prompts
                embed_a = pipeline.encode_prompt(source_prompt)
                embed_b = pipeline.encode_prompt(target_prompt)
                embed_c = pipeline.encode_prompt(prompt_c)
                embed_d = pipeline.encode_prompt(prompt_d)

                # Create 4-prompt semantic cube
                prompt_walk = FourCornerWalk.from_prompt_embeddings(
                    embed_a, embed_b, embed_c, embed_d, seed=config.seed
                )

                latent_shape = (1, 4, config.latent_height, config.latent_width)
                noise_walk = FourCornerWalk.from_latent_noise(
                    latent_shape, seed=config.seed + 1
                )

                frame_queue = asyncio.Queue(maxsize=16)
                stop_event.clear()

                # Producer: generates batches
                async def producer():
                    loop = asyncio.get_event_loop()
                    batch_times = deque(maxlen=10)
                    frame_count = 0

                    while not stop_event.is_set():
                        t_start = time.perf_counter()
                        frame_bytes_list, metrics = await loop.run_in_executor(
                            None,
                            generate_batch_sync,
                            pipeline,
                            noise_walk,
                            prompt_walk,
                            config,
                            config.batch_size,
                            direction_state["prompt_vec"],
                            direction_state["latent_vec"],
                        )

                        for frame_bytes in frame_bytes_list:
                            await frame_queue.put(frame_bytes)

                        batch_times.append(time.perf_counter() - t_start)
                        frame_count += config.batch_size

                        # Send position and edge proximity every 10 frames
                        if frame_count % 10 == 0:
                            prompt_x, prompt_y, prompt_z = metrics["prompt_pos"]
                            latent_x, latent_y, latent_z = metrics["latent_pos"]

                            prompt_proximity = max(
                                abs(prompt_x - 0.5), abs(prompt_y - 0.5), abs(prompt_z - 0.5)
                            ) * 2
                            latent_proximity = max(
                                abs(latent_x - 0.5), abs(latent_y - 0.5), abs(latent_z - 0.5)
                            ) * 2

                            await websocket.send_json(
                                {
                                    "type": "position_update",
                                    "prompt_pos": [prompt_x, prompt_y, prompt_z],
                                    "latent_pos": [latent_x, latent_y, latent_z],
                                    "prompt_proximity": prompt_proximity,
                                    "latent_proximity": latent_proximity,
                                }
                            )

                        if len(batch_times) == 10:
                            avg_batch_time = sum(batch_times) / 10
                            gen_fps = config.batch_size / avg_batch_time
                            logger.info(f"Producer: {gen_fps:.1f} FPS (generation)")

                # Consumer: sends at fixed rate
                async def consumer():
                    frame_count = 0
                    send_times = deque(maxlen=30)
                    last_log = time.perf_counter()
                    target_fps = 10.0  # Adjust as needed
                    frame_interval = 1.0 / target_fps

                    while not stop_event.is_set():
                        try:
                            frame_bytes = await asyncio.wait_for(
                                frame_queue.get(), timeout=0.5
                            )
                        except asyncio.TimeoutError:
                            continue

                        t_send_start = time.perf_counter()
                        await websocket.send_bytes(frame_bytes)
                        send_times.append(time.perf_counter() - t_send_start)
                        frame_count += 1

                        # Smooth pacing
                        elapsed = time.perf_counter() - t_send_start
                        sleep_time = max(0, frame_interval - elapsed)
                        await asyncio.sleep(sleep_time)

                        # Log delivery metrics
                        if time.perf_counter() - last_log >= 3.0:
                            delivery_fps = frame_count / (
                                time.perf_counter() - last_log
                            )
                            avg_send = sum(send_times) / len(send_times) * 1000
                            queue_depth = frame_queue.qsize()
                            logger.info(
                                f"Frame {frame_count} | Delivery: {delivery_fps:.1f} FPS "
                                f"| Send: {avg_send:.1f}ms | Queue: {queue_depth}"
                            )
                            frame_count = 0
                            last_log = time.perf_counter()

                producer_task = asyncio.create_task(producer())
                consumer_task = asyncio.create_task(consumer())

            elif action == "update_direction":
                direction_state["prompt_vec"] = msg.get("prompt_vec", [0.0, 0.0, 0.0, 0.0])
                direction_state["latent_vec"] = msg.get("latent_vec", [0.0, 0.0, 0.0, 0.0])

            elif action == "stop":
                logger.info("Stopping")
                stop_event.set()
                if producer_task:
                    producer_task.cancel()
                if consumer_task:
                    consumer_task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        stop_event.set()
        if producer_task:
            producer_task.cancel()
        if consumer_task:
            consumer_task.cancel()
        if _pipeline:
            _pipeline.cleanup()


@router.websocket("/ws/generate/single")
async def stream_latent_walk_single(websocket: WebSocket):
    await websocket.accept()
    logger.info("Client connected (SINGLE mode)")

    try:
        pipeline, latent_walk, config = await get_or_create_pipeline()

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "start":
                prompt = msg.get(
                    "prompt", "Burger covered in sopping wet oil in a gutter"
                )
                logger.info(f"Starting SINGLE generation with prompt: {prompt}")

                angle = 0.0
                frame_count = 0
                generating = True

                latent_times = deque(maxlen=30)
                inference_times = deque(maxlen=30)
                jpeg_times = deque(maxlen=30)
                send_times = deque(maxlen=30)
                loop_times = deque(maxlen=30)

                while generating:
                    loop_start = time.perf_counter()

                    t0 = time.perf_counter()
                    latent = latent_walk.get_latent(
                        angle, device=config.device, dtype=config.get_torch_dtype()
                    )
                    latent_times.append(time.perf_counter() - t0)

                    t1 = time.perf_counter()
                    frame_tensor = pipeline.generate(latent, prompt=prompt)
                    inference_times.append(time.perf_counter() - t1)

                    t2 = time.perf_counter()
                    frame_bytes = tensor_to_jpeg(
                        frame_tensor, quality=config.jpeg_quality
                    )
                    jpeg_times.append(time.perf_counter() - t2)

                    t3 = time.perf_counter()
                    await websocket.send_bytes(frame_bytes)
                    send_times.append(time.perf_counter() - t3)

                    loop_times.append(time.perf_counter() - loop_start)

                    angle = (angle + config.walk_speed) % (2 * math.pi)
                    frame_count += 1

                    if frame_count % 30 == 0 and len(loop_times) == 30:
                        gen_time = (
                            sum(latent_times) + sum(inference_times) + sum(jpeg_times)
                        )
                        gen_fps = 30.0 / gen_time
                        delivery_fps = 30.0 / sum(loop_times)

                        logger.info(
                            f"\nFrame {frame_count} | SINGLE"
                            f"\nGeneration: {gen_fps:.1f} FPS"
                            f"\nDelivery:   {delivery_fps:.1f} FPS"
                            f"\nLatent:     {sum(latent_times) / 30 * 1000:.1f}ms"
                            f"\nInference:  {sum(inference_times) / 30 * 1000:.1f}ms"
                            f"\nJPEG:       {sum(jpeg_times) / 30 * 1000:.1f}ms"
                            f"\nWebSocket:  {sum(send_times) / 30 * 1000:.1f}ms"
                        )

                    try:
                        check_msg = await asyncio.wait_for(
                            websocket.receive_json(), timeout=0.001
                        )
                        if check_msg.get("action") == "stop":
                            logger.info("Stopping generation")
                            generating = False
                    except asyncio.TimeoutError:
                        pass

            elif action == "stop":
                logger.info("Received stop (not generating)")

    except WebSocketDisconnect:
        logger.info("Client disconnected (SINGLE mode)")
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
