# GPT-2 Training Docker Container
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# Create non-root user for security
RUN groupadd -r gpt2 && useradd -r -g gpt2 -u 1001 -m gpt2

# Install system dependencies and Python
RUN apt-get update --allow-insecure-repositories && \
    apt-get install -y --allow-unauthenticated \
    ca-certificates \
    gnupg \
    python3 \
    python3-pip \
    build-essential \
    git \
    wget \
    tmux \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast package management
RUN pip3 install --no-cache-dir --break-system-packages uv

# Set uv environment variables
ENV UV_CACHE_DIR=/tmp/uv-cache
ENV UV_PYTHON=python3

# Set Hugging Face cache directory
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers
ENV HF_DATASETS_CACHE=/app/.cache/huggingface/datasets

# Set working directory
WORKDIR /app

# Copy application code (excluding data_scripts)
COPY --chown=gpt2:gpt2 src/gpt_module/ /app/gpt_module/
COPY --chown=gpt2:gpt2 src/training/ /app/training/
COPY --chown=gpt2:gpt2 src/scripts/ /app/scripts/
COPY --chown=gpt2:gpt2 src/data_scripts /app/data_scripts
COPY --chown=gpt2:gpt2 src/train_gpt.py /app/
COPY pyproject.toml /app/

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json

# Install dependencies from pyproject.toml
RUN uv sync --no-dev

# Create application directories
RUN mkdir -p /app/.cache/huggingface && \
    chown -R gpt2:gpt2 /app

# Switch to non-root user
USER gpt2