# Start from the RunPod PyTorch base image
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Create a virtual environment
RUN uv venv /opt/venv --python 3.10
# Add the venv to the PATH so all subsequent commands use it
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/hambaJuba2ba

# Copy only the files needed for dependency installation first to leverage caching.
COPY pyproject.toml README.md ./
COPY streamdiffusion/ ./streamdiffusion/

# Install all dependencies from your pyproject.toml AND the streamdiffusion editable install.
RUN uv pip install -e ".[all]" -e ./streamdiffusion[dev,tensorrt]

# Now copy the rest of your project's source code
COPY src/ ./src/

# Run your custom TensorRT installation script
# This is essential for the non-pip dependencies of TensorRT
RUN python src/hambajuba2ba/install_tensorrt.py --cuda 12

# Set the LD_LIBRARY_PATH environment variable permanently in the image.
# We hardcode the relative path within the venv.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"
# Expose ports
EXPOSE 9090 7860

# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]
