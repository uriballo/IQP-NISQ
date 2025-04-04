import jax
import jax.numpy as jnp
from jax import random, lax
from flax import linen as nn
from flax.training import train_state
import optax
from simple_vae import *

# --- Trainer Code ---
# Define a train state class for our autoencoder
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
            input_shape: Shape of the input data, e.g., (batch_size, features).
        """
        self.rng = rng
        self.model = model
        self.learning_rate = learning_rate
        
        # Split the RNG for initialization
        init_rng, self.rng = jax.random.split(rng)
        # Create a dummy input to initialize parameters
        dummy_input = jnp.ones(input_shape, jnp.float32)
        # Initialize model parameters; note that our model expects an extra RNG for sampling.
        params = self.model.init({'params': init_rng, 'dropout': init_rng}, dummy_input, self.rng)['params']
        
        # Setup the optimizer (Adam in this case)
        optimizer = optax.adam(learning_rate)
        
        # Create the training state
        self.state = AutoencoderTrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer)

    #@jax.jit
    def train_step(self, state, batch, rng):
        """
        A single training step that computes the loss, gradients,
        and updates the parameters.
        """
        def loss_fn(params):
            # Forward pass through the model
            recon_x, _ = self.model.apply({'params': params}, batch, rng)
            # Compute the reconstruction loss (MSE here, but adjust as needed)
            loss = jnp.mean((batch - recon_x) ** 2)
            return loss

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_epoch(self, train_ds, batch_size):
        """
        Trains the model for one epoch.

        Args:
            train_ds: The training dataset (should be indexable).
            batch_size: Size of each mini-batch.
        
        Returns:
            Average loss for the epoch.
        """
        num_batches = len(train_ds) // batch_size
        epoch_loss = 0.0

        for i in range(num_batches):
            batch = train_ds[i * batch_size:(i + 1) * batch_size]
            self.rng, step_rng = jax.random.split(self.rng)
            self.state, loss = self.train_step(self.state, batch, step_rng)
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def train(self, train_ds, batch_size, num_epochs, eval_fn=None):
        """
        Runs the training loop for a given number of epochs.

        Args:
            train_ds: Training dataset.
            batch_size: Mini-batch size.
            num_epochs: Number of epochs to train.
            eval_fn: Optional evaluation function that accepts the model parameters.
        """
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_ds, batch_size)
            print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
            if eval_fn is not None:
                eval_fn(self.state.params)

# --- Example Usage ---
if __name__ == '__main__':
    # Create a random key
    rng = random.PRNGKey(0)
    # Instantiate the binary VAE model with 20 latent dimensions
    binary_vae = model(latents=20)
    # Define input shape (e.g., a batch of 64 MNIST-like images flattened to 784 features)
    input_shape = (64, 784)
    # Set learning rate
    learning_rate = 1e-3

    # Create an instance of the trainer with your binary VAE model
    trainer = AutoencoderTrainer(binary_vae, learning_rate, rng, input_shape)

    # Create dummy training data (for example, 1024 samples)
    train_ds = jnp.ones((1024, 784), dtype=jnp.float32)

    # Train the model for 10 epochs with a batch size of 64
    trainer.train(train_ds, batch_size=64, num_epochs=10)
