# Start from the RunPod PyTorch base image for an RTX 5090
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

WORKDIR /workspace

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y curl git && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Create a virtual environment
RUN uv venv /opt/venv --python 3.11
# Add the venv to the PATH so all subsequent commands use it
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /workspace/hambaJuba2ba

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN uv pip install \
    --extra-index-url https://pypi.nvidia.com \
    -e ".[all]"

RUN echo 'export LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"' >> /etc/profile.d/tensorrt_libs.sh
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/tensorrt_libs:${LD_LIBRARY_PATH}"

# Expose ports and keep the container alive.
EXPOSE 9090 7860
CMD ["sleep", "infinity"]
