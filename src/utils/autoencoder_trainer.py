import jax
import jax.numpy as jnp
from jax import random, lax
from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from functools import partial
import numpy as np
from src.utils.autoencoder_manager import save_model_state, restore_model_state
import os

class AutoencoderTrainState(train_state.TrainState):
    pass

class AutoencoderTrainer:
    def __init__(self, model, learning_rate, rng, input_shape):
        """
        Initializes the trainer.
        
        Args:
            model: The Flax autoencoder model.
            learning_rate: Learning rate for the optimizer.
            rng: A JAX random key.
            input_shape: Shape of the input data (e.g., (batch_size, features)).
        """
        self.rng = rng
        self.model = model
        self.learning_rate = learning_rate
        
        # Split RNG for initialization
        init_rng, self.rng = jax.random.split(rng)
        dummy_input = jnp.ones(input_shape, jnp.float32)
        self.params = self.model.init({'params': init_rng, 'dropout': init_rng}, dummy_input, self.rng)['params']
        optimizer = optax.adam(learning_rate)
        self.state = AutoencoderTrainState.create(
            apply_fn=self.model.apply, params=self.params, tx=optimizer)

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, batch, rng):
        """
        A single training step that computes the loss, gradients, and updates parameters.
        """
        @jax.jit
        def loss_fn(params):
            recon_x, _, _ = self.model.apply({'params': params}, batch, rng)
            loss = jnp.mean((batch - recon_x) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_epoch(self, train_ds, batch_size):
        """
        Trains the model for one epoch.
        """
        train_iter = tfds.as_numpy(train_ds)
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_iter:
            images, labels = batch_data

            self.rng, step_rng = jax.random.split(self.rng)
            images = jnp.reshape(images, (images.shape[0], -1))
            self.state, loss = self.train_step(self.state, images, step_rng)
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else float('nan')

    def train(self, train_ds, batch_size, num_epochs, run_dir="results/run_default", save_best=True):
        """
        Runs the training loop for a given number of epochs.

        Args:
            train_ds: Training dataset.
            batch_size: Batch size.
            num_epochs: Number of epochs.
            run_dir: Base directory where results (checkpoints, logs) for this run will be saved.
            save_best: If True, save the best model (based on training loss); otherwise, save the final model.
        """
        os.makedirs(run_dir, exist_ok=True)
        best_metric = float('inf')  # lower is better
        best_state = None

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_ds, batch_size)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            
            metric = train_loss
            print(f"Epoch {epoch + 1} - Eval Metric (Train Loss): {metric:.4f}")
            
            if save_best and metric < best_metric:
                best_metric = metric
                best_state = self.state

        if save_best:
            final_state = best_state
            print(f"Saving best state with metric: {best_metric:.4f}")
        else:
            final_state = self.state
            print("Saving final state (no best-state heuristic applied).")
            
        ckpt_path = save_model_state(final_state, run_dir, num_epochs)
        print(f"Checkpoint saved at the end of training: {ckpt_path}")



    def plot_reconstructions(self, test_ds, num_images=16, size=14):
        """
        Plots a grid of original images, reconstructions, and their latent codes.
        
        Args:
            test_ds: A tf.data.Dataset containing the test split.
            num_images: Number of images to display.
        """
        test_iter = iter(tfds.as_numpy(test_ds))
        batch_data = next(test_iter)
        images, _ = batch_data

        images = images.reshape(images.shape[0], -1)
        images = images[:num_images]
        self.rng, sample_rng = jax.random.split(self.rng)
        recon_images, logits, z = self.model.apply({'params': self.state.params}, images, sample_rng)
        
        images = images.reshape(-1, size, size)
        recon_images = recon_images.reshape(-1, size, size)
        z_vis = z  # shape: (num_images, latent_dim)
        
        n_cols = num_images
        fig, axs = plt.subplots(3, n_cols, figsize=(n_cols * 2, 6))
        for i in range(n_cols):
            axs[0, i].imshow(images[i], cmap='gray')
            axs[0, i].axis('off')
            if i == 0:
                axs[0, i].set_ylabel("Original", fontsize=8)
            axs[1, i].imshow(recon_images[i], cmap='gray')
            axs[1, i].axis('off')
            if i == 0:
                axs[1, i].set_ylabel("Reconstruction", fontsize=8)
            axs[2, i].imshow(z_vis[i].reshape(1, -1), cmap='gray', aspect=5)  # Adjust aspect ratio for clarity

            axs[2, i].set_xticks(np.arange(-0.5, z_vis.shape[1], 1), minor=True)  # Grid every 1 unit
            axs[2, i].set_yticks([])
            axs[2, i].grid(which="minor", color="gray", linestyle='-', linewidth=1)  # Thin black lines
            axs[2, i].tick_params(which="both", bottom=False, left=False)
            if i == 0:
                axs[2, i].set_ylabel("Latent Code", fontsize=8)
        plt.tight_layout()
        plt.show()