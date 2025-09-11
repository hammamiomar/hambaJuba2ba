#!/bin/bash

# Simple setup for hambajuba2ba - gets you coding FAST
# Designed for RunPod pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

set -e

echo "Setting up hambajuba2ba for development..."

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to the PATH for the current session and for future sessions
    export PATH="$HOME/.cargo/bin:$PATH"
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
fi

# Create a virtual environment with uv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating uv virtual environment..."
    uv venv
fi

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
uv sync --extra cuda

# Install StreamDiffusion in development mode
echo "Installing StreamDiffusion fork..."
uv pip install -e ./streamdiffusion[dev]

# Install TensorRT
echo "Installing TensorRT..."
uv run python -m hambajuba2ba.install_tensorrt

# --- THIS IS THE CRITICAL FIX ---
echo "Configuring environment for TensorRT libraries..."
# Find the directory containing nvidia libraries installed by uv
NVIDIA_LIB_DIR=$(uv run python -c "import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')")

if [ -d "$NVIDIA_LIB_DIR" ]; then
    echo "Found NVIDIA libraries in: $NVIDIA_LIB_DIR"
    # Add this path to LD_LIBRARY_PATH for this session and future sessions
    export LD_LIBRARY_PATH=$NVIDIA_LIB_DIR:$LD_LIBRARY_PATH

    # Check if the line already exists in .bashrc before adding
    BASHRC_LINE="export LD_LIBRARY_PATH=\$(uv run python -c \"import nvidia.cudnn; print(nvidia.cudnn.__path__[0] + '/lib')\"):\$LD_LIBRARY_PATH"
    if ! grep -qF -- "$BASHRC_LINE" ~/.bashrc; then
        echo "Adding NVIDIA lib path to ~/.bashrc for future sessions..."
        echo '' >> ~/.bashrc
        echo '# Add NVIDIA libraries from uv environment to LD_LIBRARY_PATH' >> ~/.bashrc
        echo "$BASHRC_LINE" >> ~/.bashrc
    fi
else
    echo "Warning: Could not find NVIDIA library directory. TensorRT might not work."
fi
# -----------------------------

# Quick verification
echo "Verifying installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}'); import tensorrt; print(f'TensorRT: {tensorrt.__version__}')"

echo "Setup complete! Ready to code."
echo ""
echo "Try:"
echo "  uv run python streamdiffusion/examples/benchmark/single.py --acceleration tensorrt"
echo "  uv run python main.py"
echo "  uv run jupyter notebook # For Jupyter development"
