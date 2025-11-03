"""Exploration modes: looping and audio-reactive."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import JSONResponse
import logging
import asyncio
import time
import torch
from typing import Optional
from collections import deque
from pathlib import Path
import tempfile
import uuid

from .pipeline import get_or_create_pipeline, tensor_to_jpeg
from src.walks import SLERPWalk
from src.interpolation import generate_latent_from_seed
from src.audio import load_audio, extract_rms, sample_rms_at_time, get_duration

logger = logging.getLogger("uvicorn")
router = APIRouter()

# Audio analysis cache {audio_id: (rms_values, timestamps)}
_audio_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


@router.post("/api/audio/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and analyze audio file.

    Returns:
        JSON with audio_id, rms_values, timestamps, duration
    """
    try:
        # save to temp file
        audio_id = str(uuid.uuid4())
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # load and analyze
        waveform, sample_rate = load_audio(tmp_path)
        rms_values, timestamps = extract_rms(waveform, sample_rate)
        duration = get_duration(tmp_path)

        # cache
        _audio_cache[audio_id] = (rms_values, timestamps)

        # cleanup
        Path(tmp_path).unlink()

        logger.info(f"Audio uploaded: {audio_id}, duration={duration:.2f}s")

        return JSONResponse({
            "audio_id": audio_id,
            "rms": rms_values.cpu().tolist(),
            "timestamps": timestamps.cpu().tolist(),
            "duration": duration,
        })

    except Exception as e:
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        return JSONResponse({"error": str(e)}, status_code=500)


@router.websocket("/ws/exploration/looping")
async def looping_exploration(websocket: WebSocket):
    """Looping SLERP interpolation between src and target."""
    await websocket.accept()
    logger.info("Client connected (looping mode)")

    producer_task = None
    consumer_task = None
    stop_event = asyncio.Event()

    # Walk state
    prompt_walk: Optional[SLERPWalk] = None
    latent_walk: Optional[SLERPWalk] = None

    # Step size state (dynamically updatable)
    step_state = {
        "prompt_step": 0.02,
        "latent_step": 0.02,
    }

    try:
        pipeline, config = await get_or_create_pipeline()

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "start":
                source_prompt = msg.get("source_prompt", "moldy burger in sewer")
                target_prompt = msg.get("target_prompt", "steamy burger")
                step_state["prompt_step"] = msg.get("prompt_step", 0.02)
                step_state["latent_step"] = msg.get("latent_step", 0.02)
                source_seed = msg.get("source_seed", 42)
                target_seed = msg.get("target_seed", 43)

                logger.info(f"Starting looping: {source_prompt} → {target_prompt}")

                # encode prompts
                embed_src = pipeline.encode_prompt(source_prompt)
                embed_tgt = pipeline.encode_prompt(target_prompt)
                prompt_walk = SLERPWalk(embed_src, embed_tgt, step_size=step_state["prompt_step"])

                # generate latent seeds
                latent_shape = (1, 4, config.latent_height, config.latent_width)
                latent_src = generate_latent_from_seed(
                    source_seed, latent_shape, config.device, config.get_torch_dtype()
                )
                latent_tgt = generate_latent_from_seed(
                    target_seed, latent_shape, config.device, config.get_torch_dtype()
                )
                latent_walk = SLERPWalk(latent_src, latent_tgt, step_size=step_state["latent_step"])

                frame_queue = asyncio.Queue(maxsize=16)
                stop_event.clear()

                # producer: generate batches
                async def producer():
                    loop = asyncio.get_event_loop()
                    batch_times = deque(maxlen=10)
                    frame_count = 0

                    while not stop_event.is_set():
                        t_start = time.perf_counter()

                        # update step sizes from shared state
                        prompt_walk.step_size = step_state["prompt_step"]
                        latent_walk.step_size = step_state["latent_step"]

                        # generate batch via executor
                        frame_bytes_list = await loop.run_in_executor(
                            None,
                            generate_batch_sync,
                            pipeline,
                            prompt_walk,
                            latent_walk,
                            config.batch_size,
                            config,
                        )

                        for frame_bytes in frame_bytes_list:
                            await frame_queue.put(frame_bytes)

                        batch_times.append(time.perf_counter() - t_start)
                        frame_count += config.batch_size

                        if len(batch_times) == 10:
                            avg_batch = sum(batch_times) / 10
                            gen_fps = config.batch_size / avg_batch
                            logger.info(f"Looping: {gen_fps:.1f} FPS")

                        # send position update
                        if frame_count % 10 == 0:
                            prompt_t, _ = prompt_walk.get_position()
                            latent_t, _ = latent_walk.get_position()
                            await websocket.send_json({
                                "type": "interp_pos",
                                "prompt_t": prompt_t,
                                "latent_t": latent_t,
                            })

                # consumer: send at fixed rate
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

                        # pace delivery
                        elapsed = time.perf_counter() - t_send
                        await asyncio.sleep(max(0, frame_interval - elapsed))

                        if time.perf_counter() - last_log >= 3.0:
                            delivery_fps = frame_count / (time.perf_counter() - last_log)
                            logger.info(f"Delivery: {delivery_fps:.1f} FPS | Queue: {frame_queue.qsize()}")
                            frame_count = 0
                            last_log = time.perf_counter()

                producer_task = asyncio.create_task(producer())
                consumer_task = asyncio.create_task(consumer())

            elif action == "update_steps":
                if "prompt_step" in msg:
                    step_state["prompt_step"] = msg["prompt_step"]
                if "latent_step" in msg:
                    step_state["latent_step"] = msg["latent_step"]
                logger.info(f"Updated steps: prompt={step_state['prompt_step']}, latent={step_state['latent_step']}")

            elif action == "stop":
                logger.info("Stopping looping")
                stop_event.set()
                if producer_task:
                    producer_task.cancel()
                if consumer_task:
                    consumer_task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected (looping)")
    except Exception as e:
        logger.error(f"Error in looping: {e}", exc_info=True)
    finally:
        stop_event.set()
        if producer_task:
            producer_task.cancel()
        if consumer_task:
            consumer_task.cancel()


def generate_batch_sync(pipeline, prompt_walk, latent_walk, batch_size, config):
    """GPU-bound batch generation."""
    # get batch of interpolations
    prompt_batch, _ = prompt_walk.get_batch(batch_size, config.device, config.get_torch_dtype())
    latent_batch, _ = latent_walk.get_batch(batch_size, config.device, config.get_torch_dtype())

    # generate
    frames = pipeline.generate_batch(latent_batch, prompt_embeds=prompt_batch)

    # encode to JPEG
    return [tensor_to_jpeg(frame, quality=config.jpeg_quality) for frame in frames]


@router.websocket("/ws/exploration/audio")
async def audio_exploration(websocket: WebSocket):
    """Audio-reactive SLERP interpolation."""
    await websocket.accept()
    logger.info("Client connected (audio mode)")

    producer_task = None
    consumer_task = None
    stop_event = asyncio.Event()

    # Walk state
    prompt_walk: Optional[SLERPWalk] = None
    latent_walk: Optional[SLERPWalk] = None
    audio_id: Optional[str] = None

    # Playback state (frontend-driven)
    playback_state = {
        "current_time": 0.0,
        "is_playing": False,
    }

    try:
        pipeline, config = await get_or_create_pipeline()

        while True:
            msg = await websocket.receive_json()
            action = msg.get("action")

            if action == "start":
                source_prompt = msg.get("source_prompt", "moldy burger in sewer")
                target_prompt = msg.get("target_prompt", "steamy burger")
                source_seed = msg.get("source_seed", 42)
                target_seed = msg.get("target_seed", 43)
                audio_id = msg.get("audio_id")

                if not audio_id or audio_id not in _audio_cache:
                    await websocket.send_json({"type": "error", "message": "Audio not loaded"})
                    continue

                logger.info(f"Starting audio mode: {source_prompt} → {target_prompt}")

                # encode prompts
                embed_src = pipeline.encode_prompt(source_prompt)
                embed_tgt = pipeline.encode_prompt(target_prompt)
                prompt_walk = SLERPWalk(embed_src, embed_tgt, step_size=0)  # manual control

                # generate latent seeds
                latent_shape = (1, 4, config.latent_height, config.latent_width)
                latent_src = generate_latent_from_seed(
                    source_seed, latent_shape, config.device, config.get_torch_dtype()
                )
                latent_tgt = generate_latent_from_seed(
                    target_seed, latent_shape, config.device, config.get_torch_dtype()
                )
                latent_walk = SLERPWalk(latent_src, latent_tgt, step_size=0)  # manual control

                frame_queue = asyncio.Queue(maxsize=16)
                stop_event.clear()

                rms_values, timestamps = _audio_cache[audio_id]

                # producer (follows frontend playback time)
                async def producer():
                    loop = asyncio.get_event_loop()

                    while not stop_event.is_set():
                        # wait for playback to be active
                        if not playback_state["is_playing"]:
                            await asyncio.sleep(0.1)
                            continue

                        # use time from frontend
                        current_time = playback_state["current_time"]

                        # sample RMS at frontend's current time
                        rms = sample_rms_at_time(rms_values, timestamps, current_time)

                        # set walk positions based on RMS
                        prompt_walk.set_position(rms)
                        latent_walk.set_position(rms)

                        # generate single frame
                        frame_bytes = await loop.run_in_executor(
                            None,
                            generate_single_frame_sync,
                            pipeline,
                            prompt_walk,
                            latent_walk,
                            config,
                        )

                        await frame_queue.put(frame_bytes)

                        # send position update
                        await websocket.send_json({
                            "type": "interp_pos",
                            "prompt_t": rms,
                            "latent_t": rms,
                            "audio_time": current_time,
                        })

                        # small sleep to avoid spinning
                        await asyncio.sleep(0.01)

                # consumer
                async def consumer():
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

                        elapsed = time.perf_counter() - t_send
                        await asyncio.sleep(max(0, frame_interval - elapsed))

                producer_task = asyncio.create_task(producer())
                consumer_task = asyncio.create_task(consumer())
                playback_state["is_playing"] = True

            elif action == "audio_timeupdate":
                playback_state["current_time"] = msg.get("time", 0.0)

            elif action == "audio_play":
                playback_state["is_playing"] = True
                logger.info("Audio playback resumed")

            elif action == "audio_pause":
                playback_state["is_playing"] = False
                logger.info("Audio playback paused")

            elif action == "audio_seek":
                playback_state["current_time"] = msg.get("time", 0.0)
                logger.info(f"Audio seeked to {playback_state['current_time']:.2f}s")

            elif action == "stop":
                logger.info("Stopping audio mode")
                stop_event.set()
                if producer_task:
                    producer_task.cancel()
                if consumer_task:
                    consumer_task.cancel()

    except WebSocketDisconnect:
        logger.info("Client disconnected (audio)")
    except Exception as e:
        logger.error(f"Error in audio mode: {e}", exc_info=True)
    finally:
        stop_event.set()
        if producer_task:
            producer_task.cancel()
        if consumer_task:
            consumer_task.cancel()


def generate_single_frame_sync(pipeline, prompt_walk, latent_walk, config):
    """Generate single frame."""
    prompt_embed = prompt_walk.get(config.device, config.get_torch_dtype())
    latent = latent_walk.get(config.device, config.get_torch_dtype())

    frames = pipeline.generate_batch(latent, prompt_embeds=prompt_embed)
    return tensor_to_jpeg(frames[0], quality=config.jpeg_quality)
