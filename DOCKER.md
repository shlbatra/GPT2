# Docker Deployment Guide

This guide explains how to build and run the GPT-2 training environment using Docker.

## Prerequisites

- Docker installed
- NVIDIA Docker runtime (for GPU support)
- At least 8GB RAM
- GPU with 8GB+ VRAM (recommended)

## Quick Start

### Build the Image
```bash
docker build -t gpt2-training .
```

### Run Training (GPU)
```bash
docker run --gpus all -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  gpt2-training
```

### Run Training (CPU only)
```bash
docker run -it \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  gpt2-training
```

## Usage Options

### Interactive Shell
```bash
docker run --gpus all -it gpt2-training bash
```

### Background Training
```bash
docker run --gpus all -d \
  --name gpt2-training \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/logs:/app/logs \
  gpt2-training
```

### Monitor Background Training
```bash
docker logs -f gpt2-training
```

### Jupyter Notebook
```bash
docker run --gpus all -it \
  -p 8888:8888 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  gpt2-training jupyter notebook --ip=0.0.0.0 --allow-root
```

## Data Preparation

The container will automatically download and prepare the FineWeb dataset on first run. To use pre-downloaded data:

```bash
docker run --gpus all -it \
  -v $(pwd)/data:/app/src/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  gpt2-training
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU to use |
| `OMP_NUM_THREADS` | `4` | CPU threads for PyTorch |
| `PYTHONUNBUFFERED` | `1` | Real-time Python output |

Example with custom settings:
```bash
docker run --gpus all -it \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e OMP_NUM_THREADS=8 \
  gpt2-training
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./checkpoints` | `/app/checkpoints` | Model checkpoints |
| `./logs` | `/app/logs` | Training logs |
| `./data` | `/app/src/data` | Training data |

## Automatic Builds

The Docker image is automatically built when code is pushed to the `main` branch via GitHub Actions.

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker run --gpus all nvidia/cuda:12.1-runtime-ubuntu22.04 nvidia-smi
```

### Out of Memory
```bash
# Reduce batch size or use CPU
docker run -it gpt2-training python src/train_gpt.py --batch-size 2
```

### Container Exits Immediately
```bash
# Check logs
docker logs gpt2-training

# Run interactively
docker run --gpus all -it gpt2-training bash
```

## Development

### Rebuild After Code Changes
```bash
docker build --no-cache -t gpt2-training .
```

### Mount Source Code for Development
```bash
docker run --gpus all -it \
  -v $(pwd)/src:/app/src \
  gpt2-training
```