"""Audio loading utilities using torchaudio."""

import torch
import torchaudio
from pathlib import Path


def load_audio(file_path: str, target_sr: int = 22050) -> tuple[torch.Tensor, int]:
    """Load audio file and resample to target sample rate.

    Args:
        file_path: Path to audio file (mp3, wav, etc)
        target_sr: Target sample rate in Hz

    Returns:
        Tuple of (waveform tensor, sample_rate)
        waveform shape: (channels, samples)
    """
    waveform, sample_rate = torchaudio.load(file_path)

    # resample if needed
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = resampler(waveform)
        sample_rate = target_sr

    # convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform, sample_rate


def get_duration(file_path: str) -> float:
    """Get audio file duration in seconds."""
    metadata = torchaudio.info(file_path)
    return metadata.num_frames / metadata.sample_rate
