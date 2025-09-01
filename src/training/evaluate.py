import torch
import torch.distributed as dist
import os

class Evaluator:
    def __init__(self, model, optimizer, config, ddp_config, logger):
        """Initialize trainer with all components"""
        self.model = model
        self.model.eval()
        self.optimizer = optimizer
        self.config = config
        self.ddp_config = ddp_config
        self.logger = logger

        assert self.config.total_batch_size % (self.config.B * self.config.T * self.ddp_config.ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
        grad_accum_steps = self.config.total_batch_size // (self.config.B * self.config.T * self.ddp_config.ddp_world_size)
        if self.ddp_config.master_process:
            print(f"total desired batch size: {self.config.total_batch_size}")
            print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    def evaluate(self, val_loader, step, last_step):
        """Execute evaluate step"""
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(self.ddp_config.device), y.to(self.ddp_config.device)
                with torch.autocast(device_type=self.ddp_config.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if self.ddp_config.ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if self.ddp_config.master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(self.logger.log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(self.logger.log_dir, f"model_{step:05d}.pt")
                checkpoint = { # save model and come back later to it and also save optimizer.state_dict() - think through state of the model
                    'model': self.model.state_dict(),
                    'config': self.model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    def generate(self, num_return_sequences, max_length, encoder):
        tokens = encoder.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(self.ddp_config.device)
        sample_rng = torch.Generator(device=self.ddp_config.device)
        sample_rng.manual_seed(42 + self.ddp_config.ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.ddp_config.device_type, dtype=torch.bfloat16):
                    logits, loss = self.model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        return xgen


    def evaluate_hellaswag(self, step):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % self.ddp_config.ddp_world_size != self.ddp_config.ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(self.ddp_config.device)
            mask = mask.to(self.ddp_config.device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=self.ddp_config.device_type, dtype=torch.float16):
                    logits, loss = self.model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits) # predict option wuth lowest loss
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if self.ddp_config.ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=self.ddp_config.device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=self.ddp_config.device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if self.ddp_config.master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(self.logger.log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")