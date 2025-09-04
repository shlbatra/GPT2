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

### Data Processing (`src/data/`)

- **FineWeb Dataset (`data_scripts/fineweb.py`)**: 
  - Downloads and processes the FineWeb-Edu dataset
  - Tokenizes text using GPT-2's tiktoken tokenizer
  - Creates training shards with 10M tokens each
  - Supports multiprocessing for efficient tokenization

- **DataLoader (`dataload.py`)**: 
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

### Training
```bash
cd src
python train_gpt.py
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
