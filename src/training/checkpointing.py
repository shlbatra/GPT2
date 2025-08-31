
import torch
import os

class CheckpointConfig:
    def __init__(self, checkpoint_dir='checkpoints', file='checkpoint.pt'):
        self.checkpoint_dir = checkpoint_dir
        log_dir = os.makedirs(self.checkpoint_dir, exist_ok=True)
        log_file = os.path.join(log_dir, file)

    def save_checkpoint(model, config, step, val_loss, log_dir):
        # optionally write model checkpoints
        checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        checkpoint = { # save model and come back later to it and also save optimizer.state_dict() - think through state of the model
            'model': model.state_dict(),
            'config': model.config,
            'step': step,
            'val_loss': val_loss.item()
        }
        # you might also want to add optimizer.state_dict() and
        # rng seeds etc., if you wanted to more exactly resume training
        torch.save(checkpoint, checkpoint_path)