import os
import orbax.checkpoint
from flax.training import orbax_utils

def save_model_state(state, ckpt_dir: str, step: int):
    """
    Saves the model state to a checkpoint using Orbax.

    Args:
        state: The TrainState object containing model parameters and optimizer state.
        ckpt_dir: Base directory where checkpoints will be saved.
        step: Training step or epoch number, used to name the checkpoint.
    """
    # Convert to absolute path
    abs_ckpt_dir = os.path.abspath(ckpt_dir)
    # Create a unique checkpoint path by appending the step number
    ckpt_path = os.path.join(abs_ckpt_dir, f"checkpoint_{step}")
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(state)
    
    checkpointer.save(ckpt_path, state, save_args=save_args)
    print(f"Checkpoint saved at step {step} in directory '{ckpt_path}'")
    return ckpt_path 


def restore_model_state(ckpt_dir: str, state):
    """
    Restores the model state from a checkpoint using Orbax.

    Args:
        ckpt_dir: Directory from which to restore the checkpoint.
                 This should be the absolute path to the checkpoint directory.
        state: The TrainState object to which the parameters will be restored.

    Returns:
        The restored TrainState object.
    """
    # Convert the provided checkpoint directory to an absolute path.
    abs_ckpt_dir = os.path.abspath(ckpt_dir)
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_state = checkpointer.restore(abs_ckpt_dir, item=state)
    print(f"Checkpoint restored from directory '{abs_ckpt_dir}'.")
    return restored_state

