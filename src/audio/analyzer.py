"""Audio analysis utilities using torchaudio."""

import torch


def extract_rms(
    waveform: torch.Tensor,
    sample_rate: int,
    target_fps: float = 30.0,
    hop_length: int = 512
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract RMS amplitude envelope from audio.

    Args:
        waveform: Audio tensor (1, samples)
        sample_rate: Sample rate in Hz
        target_fps: Target frame rate for RMS extraction
        hop_length: Hop length for frame analysis

    Returns:
        Tuple of (rms_values, timestamps)
        rms_values: Tensor of RMS amplitudes [0, 1] normalized
        timestamps: Tensor of timestamps in seconds
    """
    audio = waveform.squeeze()
    frame_length = int(sample_rate / target_fps)

    # compute RMS for each frame
    rms_values = []
    timestamps = []

    for i in range(0, audio.shape[0] - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        rms = torch.sqrt(torch.mean(frame ** 2))
        rms_values.append(rms)
        timestamps.append(i / sample_rate)

    rms_tensor = torch.stack(rms_values)
    time_tensor = torch.tensor(timestamps)

    # normalize to [0, 1]
    if rms_tensor.max() > 0:
        rms_tensor = rms_tensor / rms_tensor.max()

    return rms_tensor, time_tensor


def sample_rms_at_time(
    rms_values: torch.Tensor,
    timestamps: torch.Tensor,
    target_time: float
) -> float:
    """Sample RMS value at specific time via linear interpolation.

    Args:
        rms_values: Tensor of RMS values
        timestamps: Tensor of timestamps
        target_time: Time in seconds to sample

    Returns:
        Interpolated RMS value at target_time
    """
    # clamp to valid range
    target_time = max(timestamps[0].item(), min(timestamps[-1].item(), target_time))

    # find nearest indices
    idx = torch.searchsorted(timestamps, torch.tensor(target_time)).item()

    if idx == 0:
        return rms_values[0].item()
    if idx >= len(timestamps):
        return rms_values[-1].item()

    # linear interpolation
    t0, t1 = timestamps[idx - 1].item(), timestamps[idx].item()
    v0, v1 = rms_values[idx - 1].item(), rms_values[idx].item()

    alpha = (target_time - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)
