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
COPY src/ ./src/

RUN uv pip install \
    --pre \
    --extra-index-url https://pypi.nvidia.com \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -e ".[all]" \
    -e "./streamdiffusion[dev,tensorrt]" \
    "nvidia-cudnn-cu12==8.9.4.25" \
    "tensorrt==9.0.1.post11.dev4" \
    "polygraphy==0.47.1" \
    "onnx-graphsurgeon==0.3.26"


RUN echo 'export LD_LIBRARY_PATH="/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"' >> /etc/profile.d/nvidia_libs.sh
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.10/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Expose ports
EXPOSE 9090 7860

# Default command to keep container alive (so we can exec into it or run commands)
CMD ["sleep", "infinity"]
