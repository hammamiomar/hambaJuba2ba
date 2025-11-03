"""Four-corner matrix exploration mode."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import asyncio
import time
import torch
from typing import Optional
from collections import deque

from .pipeline import get_or_create_pipeline, tensor_to_jpeg
from src.walks import FourCornerWalk
from src.interpolation import generate_latent_from_seed

logger = logging.getLogger("uvicorn")
router = APIRouter()


@router.websocket("/ws/fourcorner/")
async def matrix_exploration(websocket: WebSocket):
    """2D SLERP grid exploration with matrix controls."""
    await websocket.accept()
    logger.info("Client connected (four-corner mode)")

    producer_task = None
    consumer_task = None
    stop_event = asyncio.Event()

    # Direction control state
    direction_state = {
        "prompt_vec": [0.0, 0.0, 0.0],
        "latent_vec": [0.0, 0.0, 0.0],
    }

    # Walk state
    prompt_walk: Optional[FourCornerWalk] = None
    latent_walk: Optional[FourCornerWalk] = None

    try:
        pipeline, config = await get_or_create_pipeline()

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "start":
                source_prompt = msg.get("source_prompt", "moldy burger in sewer")
                target_prompt = msg.get("target_prompt", "steamy burger")
                logger.info(f"Starting four-corner: {source_prompt} → {target_prompt}")

                # setup prompt walk
                embed_src = pipeline.encode_prompt(source_prompt)
                embed_tgt = pipeline.encode_prompt(target_prompt)
                prompt_walk = FourCornerWalk.from_prompt_embeddings(
                    embed_src, embed_tgt, seed=config.seed
                )

                # setup latent walk
                latent_shape = (1, 4, config.latent_height, config.latent_width)
                latent_walk = FourCornerWalk.from_latent_noise(
                    latent_shape, seed=config.seed + 1
                )

                frame_queue = asyncio.Queue(maxsize=16)
                stop_event.clear()

                # producer
                async def producer():
                    loop = asyncio.get_event_loop()
                    batch_times = deque(maxlen=10)
                    frame_count = 0

                    while not stop_event.is_set():
                        t_start = time.perf_counter()
                        frame_bytes_list = await loop.run_in_executor(
                            None,
                            generate_batch_sync,
                            pipeline,
                            prompt_walk,
                            latent_walk,
                            config.batch_size,
                            direction_state["prompt_vec"],
                            direction_state["latent_vec"],
                            config,
                        )

                        for frame_bytes in frame_bytes_list:
                            await frame_queue.put(frame_bytes)

                        batch_times.append(time.perf_counter() - t_start)
                        frame_count += config.batch_size

                        # send edge proximity
                        if frame_count % 10 == 0:
                            prompt_x, prompt_y = prompt_walk.get_position()
                            latent_x, latent_y = latent_walk.get_position()

                            prompt_proximity = max(
                                abs(prompt_x - 0.5), abs(prompt_y - 0.5)
                            ) * 2
                            latent_proximity = max(
                                abs(latent_x - 0.5), abs(latent_y - 0.5)
                            ) * 2

                            await websocket.send_json({
                                "type": "edge_proximity",
                                "prompt": prompt_proximity,
                                "latent": latent_proximity,
                            })

                        if len(batch_times) == 10:
                            avg_batch = sum(batch_times) / 10
                            gen_fps = config.batch_size / avg_batch
                            logger.info(f"Four-corner: {gen_fps:.1f} FPS")

                # consumer
                async def consumer():
                    frame_count = 0
                    last_log = time.perf_counter()
                    target_fps = 10.0
                    frame_interval = 1.0 / target_fps

                    while not stop_event.is_set():
                        try:
                            frame_bytes = await asyncio.wait_for(
                                frame_queue.get(), timeout=0.5
                            )
                        except asyncio.TimeoutError:
                            continue

                        t_send = time.perf_counter()
                        await websocket.send_bytes(frame_bytes)
                        frame_count += 1

                        elapsed = time.perf_counter() - t_send
                        await asyncio.sleep(max(0, frame_interval - elapsed))

                        if time.perf_counter() - last_log >= 3.0:
                            delivery_fps = frame_count / (time.perf_counter() - last_log)
                            logger.info(f"Delivery: {delivery_fps:.1f} FPS | Queue: {frame_queue.qsize()}")
                            frame_count = 0
                            last_log = time.perf_counter()

                producer_task = asyncio.create_task(producer())
                consumer_task = asyncio.create_task(consumer())

            elif action == "update_direction":
                direction_state["prompt_vec"] = msg.get("prompt_vec", [0.0, 0.0, 0.0])
                direction_state["latent_vec"] = msg.get("latent_vec", [0.0, 0.0, 0.0])

            elif action == "stop":
                logger.info("Stopping four-corner")
                stop_event.set()
                if producer_task:
                    producer_task.cancel()
                if consumer_task:
                    consumer_task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected (four-corner)")
    except Exception as e:
        logger.error(f"Error in four-corner: {e}", exc_info=True)
    finally:
        stop_event.set()
        if producer_task:
            producer_task.cancel()
        if consumer_task:
            consumer_task.cancel()


def generate_batch_sync(
    pipeline,
    prompt_walk,
    latent_walk,
    batch_size,
    prompt_vec,
    latent_vec,
    config,
):
    """GPU-bound batch generation for four-corner mode."""
    # collect batch
    prompt_embeds_list = []
    latents_list = []

    for i in range(batch_size):
        # update walks based on user direction
        if prompt_vec[2] > 0.01:
            prompt_walk.step_directed(prompt_vec[0], prompt_vec[1], prompt_vec[2])
        else:
            prompt_walk.step()

        if latent_vec[2] > 0.01:
            latent_walk.step_directed(latent_vec[0], latent_vec[1], latent_vec[2])
        else:
            latent_walk.step()

        prompt_embeds_list.append(prompt_walk.get(config.device, config.get_torch_dtype()))
        latents_list.append(latent_walk.get(config.device, config.get_torch_dtype()))

    prompt_batch = torch.cat(prompt_embeds_list, dim=0)
    latent_batch = torch.cat(latents_list, dim=0)

    # generate
    frames = pipeline.generate_batch(latent_batch, prompt_embeds=prompt_batch)

    # encode
    return [tensor_to_jpeg(frame, quality=config.jpeg_quality) for frame in frames]
