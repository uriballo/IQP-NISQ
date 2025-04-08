import os
import orbax.checkpoint
from flax.training import orbax_utils
import jax
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


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
    
    checkpointer.save(ckpt_path, state, save_args=save_args, force=True)
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


def get_latent_dataset(autoencoder, params, dataset, batch_size=128):
    latent_list = []
    label_list = []
    rng = jax.random.PRNGKey(0)
    
    for batch in dataset:
        images, labels = batch  
        
        # Update the RNG for each batch
        rng, batch_rng = jax.random.split(rng)
        images = jax.device_put(np.array(images))  # tf.Tensor → JAX DeviceArray
        #images = images.reshape((images.shape[0], -1))
        images = jnp.reshape(images, (images.shape[0], -1))

        _, logits, latent = autoencoder.apply({'params': params}, images, batch_rng)

        # Convert JAX arrays to numpy integers (or any desired type)
        latent_list.append(np.array(latent).astype(int))
        label_list.append(np.array(labels))
    
    return latent_list, label_list


def generate_from_latent(autoencoder, params, latent, image_shape=None, show=False):
    """
    Generate outputs from latent vectors using the trained autoencoder model.
    Args:
        autoencoder: The VAE model instance.
        params: The model parameters.
        latent: A JAX or NumPy array of latent vectors.
        image_shape: Optional shape to reshape the output (if not provided, 
                     use the decoder's output shape).
        show: Whether to plot the generated outputs.
    Returns:
        Generated outputs as an array.
    """
    # Use the model's generate method; note that it already applies a sigmoid.
    generated = autoencoder.apply({'params': params}, method=autoencoder.generate, z=latent)
    
    if show:

        plt.figure()
        # If the image has a single channel, squeeze it for display.
        plt.imshow(generated, cmap="gray")

        plt.title(f"Generated Output\n{latent}")
        plt.axis("off")
        plt.show()
    
    return generated

