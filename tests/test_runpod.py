import torch
import subprocess
import sys
import pytest


def test_cuda():
    """Test CUDA is available."""
    assert torch.cuda.is_available(), "CUDA not available"
    print(f"\nCUDA device: {torch.cuda.get_device_name(0)}")


def test_tensorrt():
    """Test TensorRT can be imported."""
    import tensorrt as trt
    print(f"\nTensorRT version: {trt.__version__}")


def test_streamdiffusion_tensorrt_benchmark():
    """Run StreamDiffusion TensorRT benchmark."""
    cmd = [
        "/opt/venv/bin/python",
        "/workspace/hambaJuba2ba/packages/streamdiffusion/examples/benchmark/single.py",
        "--acceleration", "tensorrt",
        "--iterations", "10",
        "--warmup", "3"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    print(f"\nBenchmark output:\n{result.stdout}")
    if result.stderr:
        print(f"Stderr:\n{result.stderr}")

    assert result.returncode == 0, "TensorRT benchmark failed"
