#!/bin/bash

# Simple setup for hambajuba2ba - gets you coding FAST
# Designed for RunPod pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

set -e

echo "Setting up hambajuba2ba for development..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    # Source it for current session
    source $HOME/.local/bin/env
fi

# Install dependencies
echo "Installing Python dependencies..."
uv sync --extra cuda

# Install StreamDiffusion in development mode
echo "Installing StreamDiffusion fork..."
uv pip install -e ./streamdiffusion[dev]

# Install TensorRT
echo "Installing TensorRT..."
uv run python -m hambajuba2ba.install_tensorrt

# Quick verification
echo "Verifying installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "Setup complete! Ready to code."
echo ""
echo "Try:"
echo "  uv run python main.py"
echo "  uv run python test_streamdiffusion.py"
echo "  uv run jupyter notebook  # For Jupyter development"
