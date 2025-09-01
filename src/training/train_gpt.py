from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.nn.functional as F
import torch
import tiktoken
import os
import math
import time
from data.dataload import DataLoaderLite
from gpt_module.gpt import GPT, GPTConfig
from config import TrainingConfig
from distributed import setup_distributed
from checkpointing import CheckpointConfig
from train_gpt import Trainer
from evaluate import Evaluator

distributed_config = setup_distributed()
ddp = distributed_config['ddp']
ddp_rank = distributed_config['ddp_rank']
ddp_local_rank = distributed_config['ddp_local_rank']
ddp_world_size = distributed_config['ddp_world_size']
master_process = distributed_config['master_process']
device = distributed_config['device']
device_type = distributed_config['device_type']


torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

TrainingConfig = TrainingConfig() # instantiate the config
max_steps = TrainingConfig.max_steps
get_lr = TrainingConfig.get_lr
total_batch_size = TrainingConfig.total_batch_size
B = TrainingConfig.B
T = TrainingConfig.T
weight_decay=TrainingConfig.weight_decay, 
starting_learning_rate=TrainingConfig.starting_learning_rate
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val") # Validation data loader

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2

model.to(device)

use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix

if use_compile: # breaks evaluation code so ignoring use_compile for now
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model # always contains the "raw" unwrapped model


# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=weight_decay, learning_rate=starting_learning_rate, device_type=device_type)

# create the log directory we will write checkpoints to and log to

CheckpointConfig = CheckpointConfig() # instantiate the config
log_file = CheckpointConfig.log_file

model_trainer = Trainer(raw_model, optimizer, TrainingConfig, distributed_config, log_file)
model_evaluator = Evaluator(raw_model, optimizer, val_loader, TrainingConfig, distributed_config, log_file)

# Training loop
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model_evaluator.evaluate(model_trainer.model, val_loader, step, last_step)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile): # make sure not compile
        model_evaluator.evaluate_hellaswag(model_trainer.model, val_loader, step, last_step)

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        num_return_sequences = 4
        max_length = 32
        xgen = model_evaluator.generate(model_trainer.model, num_return_sequences, max_length, encoder=enc)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model_trainer.train_step(step, t0)

if ddp:
    destroy_process_group()
