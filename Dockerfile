FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Environment Variables
ENV DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface/hub \
    HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets \
    UV_CACHE_DIR=/workspace/.cache/uv \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    tmux \
    vim \
    openssh-server \
    && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs \
    && \
    rm -rf /var/lib/apt/lists/*

# SSH Config
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config && \
    echo "PasswordAuthentication no" >> /etc/ssh/sshd_config && \
    echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config && \
    echo "Port 22" >> /etc/ssh/sshd_config

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.cargo/bin:/root/.local/bin:${PATH}"

# Set up tmux configuration
RUN echo 'set -g mouse on\n\
    set -g history-limit 50000\n\
    set -g default-terminal "screen-256color"\n\
    set -g status-bg colour235\n\
    set -g status-fg white\n\
    set-window-option -g window-status-current-bg colour240\n\
    bind r source-file ~/.tmux.conf \; display "Config reloaded!"' > /root/.tmux.conf

# Create workspace directory
WORKDIR /workspace
RUN mkdir -p /workspace/.cache/huggingface/hub && \
    mkdir -p /workspace/.cache/huggingface/datasets && \
    chmod -R 777 /workspace/.cache

EXPOSE 9090 7860 22 8080 3000

# Start script
RUN echo '#!/bin/bash' > /start.sh && \
    echo 'mkdir -p ~/.ssh' >> /start.sh && \
    echo 'chmod 700 ~/.ssh' >> /start.sh && \
    echo 'if [ -n "$PUBLIC_KEY" ]; then' >> /start.sh && \
    echo '    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys' >> /start.sh && \
    echo '    chmod 600 ~/.ssh/authorized_keys' >> /start.sh && \
    echo 'fi' >> /start.sh && \
    echo 'mkdir -p /run/sshd' >> /start.sh && \
    echo 'service ssh start' >> /start.sh && \
    echo 'exec "$@"' >> /start.sh && \
    chmod +x /start.sh

ENTRYPOINT ["/start.sh"]

CMD ["sleep", "infinity"]
