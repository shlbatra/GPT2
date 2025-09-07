from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import tiktoken
import time
from data_scripts.dataload import DataLoaderLite
from gpt_module.gpt import GPT, GPTConfig
from training.config import TrainingConfig
from training.distributed import DDPConfig
from training.checkpointing import CheckpointConfig
from training.trainer import Trainer
from training.evaluate import Evaluator

if __name__ == "__main__":

    # Configurations and setup
    DDPConfig = DDPConfig() # instantiate the config
    distributed_config = DDPConfig.setup_distributed()
    ddp = distributed_config['ddp']
    ddp_rank = distributed_config['ddp_rank']
    ddp_local_rank = distributed_config['ddp_local_rank']
    ddp_world_size = distributed_config['ddp_world_size']
    device = distributed_config['device']
    device_type = distributed_config['device_type']
    print(f"ddp: {ddp}, ddp_rank: {ddp_rank}, ddp_local_rank: {ddp_local_rank}, ddp_world_size: {ddp_world_size}, device: {device}, device_type: {device_type}")
    
    TrainingConfig = TrainingConfig() # instantiate the config
    max_steps = TrainingConfig.max_steps
    get_lr = TrainingConfig.get_lr
    total_batch_size = TrainingConfig.total_batch_size
    B = TrainingConfig.B
    T = TrainingConfig.T
    weight_decay = TrainingConfig.weight_decay
    starting_learning_rate = TrainingConfig.starting_learning_rate
    grad_accum_steps = total_batch_size // (B * T * ddp_world_size)


    # tokenizer
    enc = tiktoken.get_encoding("gpt2")


    # data loader
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

    # model trainer and evaluator
    model_trainer = Trainer(raw_model, optimizer, train_loader, TrainingConfig, distributed_config, log_file)
    model_evaluator = Evaluator(raw_model, optimizer, TrainingConfig, distributed_config, log_file)

    # Training loop
    for step in range(max_steps):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            val_loss = model_evaluator.evaluate(val_loader, step, last_step)
            with open(log_file, "a") as f: # open for writing to clear the file - train loss, val loss and hellaswag accuracy
                f.write(f"{step} val {val_loss.item():.4f}\n")

        # once in a while evaluate hellaswag - todo fix this
        # if (step % 250 == 0 or last_step) and (not use_compile): # make sure not compile
        #     model_evaluator.evaluate_hellaswag(val_loader, step, last_step)

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
        
        # save evaluation and checkpoint every 10000 steps
        if step % 10000 == 0 and step >= 0 and ddp_rank == 0: # 
            CheckpointConfig.save_checkpoint(raw_model, optimizer, GPTConfig, step, val_loss, CheckpointConfig.checkpoint_dir)
            print(f"Checkpoint saved at step {step}")

    DDPConfig.destroy_distributed()