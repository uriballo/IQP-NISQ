import os
from flax.training import checkpoints

def save_model(state, save_dir, step):
    """
    Saves the training state (model weights, optimizer state, etc.) to the specified directory.
    
    Args:
        state: The current training state.
        save_dir: Directory where checkpoints are saved.
        step: The current training step or epoch number.
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir=save_dir, target=state, step=step, overwrite=True)
    print(f"Checkpoint saved at step {step} in {save_dir}")

def load_model(state, save_dir):
    """
    Restores the training state from the specified directory.
    
    Args:
        state: A training state instance with the same structure as the one saved.
        save_dir: Directory from where to restore checkpoints.
    
    Returns:
        The restored training state.
    """
    restored_state = checkpoints.restore_checkpoint(ckpt_dir=save_dir, target=state)
    print("Checkpoint restored from", save_dir)
    return restored_state
