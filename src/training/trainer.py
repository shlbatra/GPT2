import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP  
import time

class Trainer:
    def __init__(self, model, optimizer, train_loader, config, ddp_config, logger):
        """Initialize trainer with all components"""
        self.model = model
        self.model.train()
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.config = config
        self.ddp_config = ddp_config
        self.logger = logger

        assert self.config.total_batch_size % (self.config.B * self.config.T * self.ddp_config.ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
        
        if self.ddp_config.master_process:
            print(f"total desired batch size: {self.config.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    def train_step(self, step, t0):
        """Execute single training step with gradient accumulation"""
        # do one step of the optimization
        self.optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):
            x, y = self.train_loader.next_batch()
            x, y = x.to(self.ddp_config.device), y.to(self.ddp_config.device)
            # added after video, this field is also used by the forward pass.
            if self.ddp_config.ddp:
                self.model.require_backward_grad_sync = (micro_step == self.grad_accum_steps - 1)
            with torch.autocast(device_type=self.ddp_config.device_type, dtype=torch.bfloat16):
                logits, loss = self.model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        if self.ddp_config.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        # determine and set the learning rate for this iteration
        lr = self.config.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.optimizer.step()
        if self.ddp_config.device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        
        t1 = time.time()
        dt = t1 - t0 # time difference in seconds
        tokens_processed = self.train_loader.B * self.train_loader.T * grad_accum_steps * self.ddp_config.ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if self.ddp_config.master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            with open(self.logger.log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")