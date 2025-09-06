# GPT-2 Training Docker Container
FROM nvidia/cuda:13.0.0-cudnn-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Create non-root user for security
RUN groupadd -r gpt2 && useradd -r -g gpt2 -u 1001 gpt2

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

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=gpt2:gpt2 . /app/

# Create virtual environment and install dependencies
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Install dependencies from pyproject.toml
RUN uv sync --no-dev

# Create application directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/cache
RUN chown -R gpt2:gpt2 /app

# Create data and checkpoint directories with proper permissions
RUN mkdir -p /app/src/data/data_scripts/edu_fineweb10B && \
    mkdir -p /app/checkpoints && \
    mkdir -p /app/logs && \
    chown -R gpt2:gpt2 /app

# Switch to non-root user
USER gpt2