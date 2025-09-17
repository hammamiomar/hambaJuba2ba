# Start from the RunPod PyTorch base image for an RTX 5090
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set up tmux configuration for better defaults
RUN echo 'set -g mouse on\n\
set -g history-limit 50000\n\
set -g default-terminal "screen-256color"\n\
set -g status-bg colour235\n\
set -g status-fg white\n\
set-window-option -g window-status-current-bg colour240\n\
bind r source-file ~/.tmux.conf \; display "Config reloaded!"' > /root/.tmux.conf

# Create workspace directory with proper permissions
WORKDIR /workspace
RUN mkdir -p /workspace/.cache/huggingface/hub && \
    mkdir -p /workspace/.cache/huggingface/datasets && \
    chmod -R 777 /workspace/.cache

# Expose ports
EXPOSE 9090 7860 22 8080

# Use the official Runpod SSH setup command as recommended in docs
CMD bash -c 'apt update;DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;mkdir -p ~/.ssh;cd $_;chmod 700 ~/.ssh;echo "$PUBLIC_KEY" >> authorized_keys;chmod 700 authorized_keys;service ssh start;sleep infinity'
