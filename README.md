# GPT-2 Implementation from Scratch

This repository contains a complete implementation of GPT-2 (Generative Pre-trained Transformer 2) built from scratch using PyTorch. The implementation follows the original GPT-2 architecture and includes training, evaluation, and inference capabilities.

## Architecture Overview

The GPT-2 model is implemented as a decoder-only transformer with the following key components:

### Core Model Components (`src/gpt_module/`)

- **GPT (`gpt.py`)**: Main model class that combines all components
  - Token embeddings (`wte`) and positional embeddings (`wpe`)
  - Stack of transformer blocks
  - Final layer normalization and language modeling head
  - Weight sharing between token embeddings and output projection

- **Block (`block.py`)**: Individual transformer block implementing:
  - Pre-normalization architecture (LayerNorm before attention/MLP)
  - Residual connections around attention and MLP layers
  - Standard transformer block: `x = x + attn(ln(x))` then `x = x + mlp(ln(x))`

- **CausalSelfAttention (`casualattention.py`)**: Multi-head causal attention
  - Scaled dot-product attention with causal masking
  - Uses PyTorch's `F.scaled_dot_product_attention` for efficient computation
  - Multi-head attention with configurable number of heads

- **MLP (`mlp.py`)**: Feed-forward network
  - Two linear layers with GELU activation
  - 4x expansion ratio (hidden dimension = 4 * embedding dimension)

### Training Infrastructure (`src/training/`)

- **TrainingConfig (`config.py`)**: Hyperparameters and learning rate scheduling
  - Cosine annealing with linear warmup
  - AdamW optimizer configuration
  - Batch size: 524,288 tokens (~0.5M tokens per batch)
  - Learning rate: 6e-4 max, 6e-5 min

- **Trainer (`trainer.py`)**: Training loop implementation
  - Gradient accumulation for large effective batch sizes
  - Mixed precision training (bfloat16)
  - Gradient clipping and learning rate scheduling
  - Distributed training support

- **Evaluator (`evaluate.py`)**: Model evaluation
  - Validation loss computation
  - Text generation capabilities
  - HellaSwag evaluation (optional)

- **DDPConfig (`distributed.py`)**: Distributed training setup
  - Multi-GPU support using DistributedDataParallel
  - Automatic device detection (CUDA/MPS/CPU)
  - Process group initialization for distributed training

### Data Processing (`src/data_scripts/`)

- **FineWeb Dataset (`data_scripts/fineweb.py`)**: 
  - Downloads and processes the FineWeb-Edu dataset
  - Tokenizes text using GPT-2's tiktoken tokenizer
  - Creates training shards with 10M tokens each
  - Supports multiprocessing for efficient tokenization

- **HellaSwag Dataset (`data_scripts/hellaswag.py`)**:
  - Downloads and processes the HellaSwag evaluation dataset
  - Commonsense reasoning benchmark for language models
  - Used for evaluation during training to track model performance
  - Provides 4-choice multiple choice questions for validation

- **DataLoader (`data_scripts/dataload.py`)**: 
  - Efficient data loading from pre-tokenized shards
  - Supports distributed training with proper data sharding
  - Memory-efficient streaming of large datasets

## Key Features

### 1. **Scalable Architecture**
- Configurable model size (layers, heads, embedding dimension)
- Default configuration: 12 layers, 12 heads, 768 embedding dimension
- ~124M parameters (GPT-2 small configuration)

### 2. **Efficient Training**
- Mixed precision training with bfloat16
- Gradient accumulation for large effective batch sizes
- Distributed training across multiple GPUs
- Cosine learning rate schedule with warmup

### 3. **Modern Implementation**
- Uses PyTorch's native scaled dot-product attention
- Pre-normalization (LayerNorm before attention/MLP)
- Weight sharing between input and output embeddings
- Efficient data loading and preprocessing

### 4. **Training Dataset**
- FineWeb-Edu: High-quality educational web content
- ~10 billion tokens from web pages
- Pre-tokenized and sharded for efficient training
- GPT-2 tokenization (50,257 vocab size)

## Usage

### Local Training
```bash
cd src
python train_gpt.py
```

### Remote GPU Training (Paperspace/Lambda)

For automated deployment to cloud GPU instances, use the deployment script:

```bash
# Make script executable
chmod +x scripts/train_gpu_lambda.sh

# Deploy to remote machine (1 GPU)
./scripts/train_gpu_lambda.sh paperspace@184.105.3.177 1

# Deploy with multiple GPUs
./scripts/train_gpu_lambda.sh paperspace@184.105.3.177 4
```

#### What the deployment script does:
1. **Environment Setup**: Installs Docker, NVIDIA drivers, and container toolkit
2. **System Configuration**: Configures Docker for GPU access and reboots
3. **Credential Transfer**: Copies `gcp-key.json` for data access
4. **Training Execution**: Pulls Docker image and runs distributed training
5. **Automatic Cleanup**: Handles permissions and directory setup

#### Requirements:
- SSH access to remote machine
- `gcp-key.json` file in current directory
- Remote machine with GPU(s) and Ubuntu/similar

#### Download Results:
```bash
# After training completes
scp -r paperspace@184.105.3.177:~/my-gpu-project/checkpoints ./
```

### Docker Training

The project includes an optimized multi-stage Docker setup:

```bash
# Build image (uses BuildKit for faster builds)
DOCKER_BUILDKIT=1 docker build -t gpt2-training .

# Run locally with GPU
docker run --runtime=nvidia \
  -v $(pwd)/gcp-key.json:/app/gcp-key.json \
  -v $(pwd)/checkpoints:/app/checkpoints \
  --rm gpt2-training \
  bash -c "torchrun --nproc_per_node=1 train_gpt.py"
```

### Data Preparation
```bash
cd src/data/data_scripts
python fineweb.py
```

## Training Configuration

- **Batch Size**: 524,288 tokens (~128 gradient accumulation steps)
- **Sequence Length**: 512 tokens  
- **Learning Rate**: 6e-4 (max) with cosine decay to 6e-5 (min)
- **Warmup**: 715 steps
- **Total Steps**: ~19,073 (approximately 1 epoch on 10B tokens)
- **Weight Decay**: 0.1
- **Optimizer**: AdamW

## Model Architecture Details

The implementation follows the GPT-2 paper specifications:
- **Vocabulary**: 50,257 tokens (50k BPE merges + 256 byte tokens + 1 special token)
- **Context Length**: 1024 tokens maximum (configurable)
- **Architecture**: Decoder-only transformer with causal attention
- **Normalization**: Pre-normalization with LayerNorm
- **Activation**: GELU in MLP layers
- **Attention**: Multi-head causal self-attention with head dimension 64

This implementation provides a complete, trainable GPT-2 model suitable for research and educational purposes.

## Development & Deployment

### Docker Multi-Stage Build Optimizations

The project uses an optimized multi-stage Docker build for faster development:

- **Builder Stage**: Installs dependencies and build tools
- **Runtime Stage**: Contains only the application and runtime dependencies
- **Benefits**: 30-50% smaller final images, faster rebuilds for code changes

### CI/CD Pipeline

GitHub Actions automatically builds and pushes Docker images with intelligent caching:

- **Dependency Changes**: Full rebuild when `pyproject.toml` or `Dockerfile` changes
- **Code Changes**: Fast runtime-only builds for source code modifications  
- **Conditional Building**: Automatically detects what changed and chooses optimal build strategy

### Build Performance Features

- **BuildKit Support**: Parallel stage execution and advanced caching
- **Layer Caching**: Dependencies cached separately from source code
- **Cache Mounts**: Persistent package manager caches (optional)
- **Smart Rebuilds**: Only rebuilds necessary layers based on file changes
