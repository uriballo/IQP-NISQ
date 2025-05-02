import os
import pickle
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
from flax import linen as nn
from flax.training.train_state import TrainState


# =============================================================================
# Custom TrainState class that includes BatchNorm statistics.
# =============================================================================
class BNTrainState(TrainState):
    """
    Custom TrainState for models that include mutable BatchNorm statistics.
    Attributes:
        batch_stats: Arbitrary pytree containing BatchNorm statistics.
    """
    batch_stats: Any


# =============================================================================
# Trainer for the Binary CNN VAE with BatchNorm
# =============================================================================
class BinaryVAECNNTrainer:
    """
    Trainer class for a binary latent VAE model with a CNN-based encoder/decoder.
    This trainer manages both the network parameters and the mutable BatchNorm state.
    
    Attributes:
        model: The Flax model (CNN VAE) to be trained.
        state: Instance of BNTrainState containing parameters and batch_stats.
        rng: JAX random number generator key.
        input_shape: Shape of a single data example (e.g. (14, 14, 1)).
    """
    def __init__(self, model: nn.Module, learning_rate: float, rng: jax.random.PRNGKey,
                 input_shape: Tuple[int, ...]) -> None:
        self.model = model
        self.rng = rng
        self.input_shape = input_shape

        # Create a dummy input for initialization.
        dummy_input = jnp.ones((1, *input_shape), jnp.float32)
        self.rng, init_rng, latent_rng = jax.random.split(self.rng, 3)

        # Initialize model variables (params and batch_stats).
        variables = self.model.init({'params': init_rng, 'dropout': init_rng}, dummy_input, latent_rng)
        params = variables.get('params')
        batch_stats = variables.get('batch_stats')

        # Create the training state.
        self.state = BNTrainState.create(
            apply_fn=self.model.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optax.adam(learning_rate)
        )

    def train_step(self, state: BNTrainState, batch: jnp.ndarray,
                   rng: jax.random.PRNGKey) -> Tuple[BNTrainState, jnp.ndarray]:
        """
        Perform a single training step.
        
        Args:
            state: Current training state (includes params and batch_stats).
            batch: Input batch of images.
            rng: Random key used for the binary quantization in the model.
        
        Returns:
            A tuple of (updated state, training loss value).
        """
        def loss_fn(params):
            # Combine parameters with current batch statistics.
            variables = {'params': params, 'batch_stats': state.batch_stats}
            # Specify mutable collections to update BatchNorm statistics.
            (recon_x, _, _), new_model_state = self.model.apply(
                variables, batch, rng,
                mutable=['batch_stats'],
                training=True
            )
            loss = jnp.mean((batch - recon_x) ** 2)
            return loss, new_model_state['batch_stats']

        # Compute gradients and extract updated batch_stats.
        (loss, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads, batch_stats=new_batch_stats)
        return new_state, loss

    def train_epoch(self, train_ds, batch_size: int) -> jnp.ndarray:
        """
        Run one full epoch on the training dataset.
        
        Args:
            train_ds: Iterable or generator that yields batches.
            batch_size: Batch size (for logging purposes only, if needed).
        
        Returns:
            The average training loss for the epoch.
        """
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_ds:
            # Assuming the dataset yields (images, labels); ignore labels.
            images, _ = batch
            self.rng, step_rng = jax.random.split(self.rng)
            self.state, loss = self.train_step(self.state, images, step_rng)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else jnp.inf
        return avg_loss

    def train(self, train_ds, batch_size: int, num_epochs: int,
              run_dir: str = None, save_best: bool = False) -> None:
        """
        Train the model for a given number of epochs.
        
        Args:
            train_ds: The training dataset.
            batch_size: Batch size.
            num_epochs: Total number of epochs for training.
            run_dir: Directory to save the model state (optional).
            save_best: Save best performing state (based on training loss).
        """
        best_loss = float('inf')
        best_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_ds, batch_size)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            if save_best and train_loss < best_loss:
                best_loss = train_loss
                best_state = self.state

        if save_best and best_state is not None:
            self.state = best_state

        if run_dir:
            os.makedirs(run_dir, exist_ok=True)
            model_state_path = os.path.join(run_dir, 'model_state.pkl')
            with open(model_state_path, 'wb') as f:
                pickle.dump({
                    'params': self.state.params,
                    'batch_stats': self.state.batch_stats
                }, f)
            print(f"Model state saved to {model_state_path}")

    def plot_reconstructions(self, dataset, num_images: int = 8) -> None:
        """
        Plot original and reconstructed images from the dataset.
        
        Args:
            dataset: Dataset to obtain sample images.
            num_images: Number of images to display.
        """
        # Get a single batch from the dataset.
        batch = next(iter(dataset))
        images, _ = batch
        self.rng, z_rng = jax.random.split(self.rng)
        variables = {'params': self.state.params, 'batch_stats': self.state.batch_stats}
        recon_x, _, _ = self.model.apply(variables, images, z_rng, training=False)

        # Plot results.
        fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4))
        for i in range(num_images):
            axes[0, i].imshow(images[i, ..., 0], cmap='gray')
            axes[0, i].axis('off')
            axes[1, i].imshow(recon_x[i, ..., 0], cmap='gray')
            axes[1, i].axis('off')
        axes[0, 0].set_ylabel("Original", fontsize=12)
        axes[1, 0].set_ylabel("Reconstruction", fontsize=12)
        plt.tight_layout()
        plt.show()
