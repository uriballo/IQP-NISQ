import jax
import jax.numpy as jnp
from jax import random, lax
from flax import linen as nn
from flax.training import train_state
import optax
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

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
        # Our model expects an extra RNG for sampling
        params = self.model.init({'params': init_rng, 'dropout': init_rng}, dummy_input, self.rng)['params']
        optimizer = optax.adam(learning_rate)
        self.state = AutoencoderTrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer)

    #@jax.jit
    def train_step(self, state, batch, rng):
        """
        A single training step that computes the loss, gradients, and updates parameters.
        """
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
        # Convert tf.data.Dataset to numpy iterator for convenience
        train_iter = tfds.as_numpy(train_ds)
        epoch_loss = 0.0
        num_batches = 0

        for batch_data in train_iter:
            # Extract images (flatten them) from the batch; ignore labels
            images, _ = batch_data
            # Flatten images: [batch, H, W, C] -> [batch, H*W*C]
            images = images.reshape(images.shape[0], -1)
            self.rng, step_rng = jax.random.split(self.rng)
            self.state, loss = self.train_step(self.state, images, step_rng)
            epoch_loss += loss
            num_batches += 1

        return epoch_loss / num_batches if num_batches > 0 else float('nan')

    def train(self, train_ds, batch_size, num_epochs, eval_fn=None):
        """
        Runs the training loop for a given number of epochs.
        """
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_ds, batch_size)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            if eval_fn is not None:
                eval_fn(self.state.params)

    def plot_reconstructions(self, test_ds, num_images=16):
        """
        Plots a grid of original images, reconstructions, and their latent codes.
        
        Args:
            test_ds: A tf.data.Dataset containing the test split.
            num_images: Number of images to display.
        """
        # Convert tf.data.Dataset to numpy iterator and take one batch
        test_iter = iter(tfds.as_numpy(test_ds))
        batch_data = next(test_iter)
        images, _ = batch_data

        # Flatten images
        images = images.reshape(images.shape[0], -1)
        # Use only the first num_images from the batch
        images = images[:num_images]
        # Get reconstruction and latent logits from the model
        self.rng, sample_rng = jax.random.split(self.rng)
        recon_images, logits, z = self.model.apply({'params': self.state.params}, images, sample_rng)
        # Compute latent code (binary quantization) from the logits
        
        # Reshape images to (28, 28) and latent codes to (1, latents) for visualization
        images = images.reshape(-1, 28, 28)
        recon_images = recon_images.reshape(-1, 28, 28)
        # z has shape (num_images, latent_dim); we visualize each as a horizontal row
        z_vis = z  # shape: (num_images, latent_dim)
        
        # Plot original, reconstruction, and latent code in three rows
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
            # For latent code, reshape it as a 1 x latent_dim image and plot as a heatmap
            axs[2, i].imshow(z_vis[i].reshape(1, -1), cmap='binary', aspect='auto')
            axs[2, i].axis('off')
            if i == 0:
                axs[2, i].set_ylabel("Latent Code", fontsize=8)
        plt.tight_layout()
        plt.show()