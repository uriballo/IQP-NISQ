from flax import linen as nn
from jax import random, lax
import jax.numpy as jnp
import jax
import numpy as np

# --- Binary Quantizer Function ---
def binary_quantizer(rng, logits):
    """Binary quantization using Bernoulli sampling with a straight-through estimator."""
    probs = nn.sigmoid(logits)  # Convert logits to probabilities
    binary_sample = random.bernoulli(rng, probs).astype(jnp.float32)  # Sample 0 or 1
    # Straight-through estimator for gradients
    binary_latent = binary_sample + probs - lax.stop_gradient(probs)
    return binary_latent

# --- Deeper Encoder with BatchNorm ---
class DeepEncoder(nn.Module):
    """Deeper VAE Encoder with additional layers and Batch Normalization."""
    latents: int
    training: bool  # Flag to indicate training/inference mode for BatchNorm

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256, name='dense1')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256, name='dense2')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='dense3')(x)
        x = nn.leaky_relu(x)       
        x = nn.Dense(128, name='dense4')(x)
        x = nn.leaky_relu(x)       
        x = nn.Dense(128, name='dense5')(x)
        x = nn.leaky_relu(x)       
        x = nn.Dense(128, name='dense6')(x)
        x = nn.leaky_relu(x)
        # Output logits corresponding to the Bernoulli parameters for binary latent sampling
        logits = nn.Dense(self.latents, name='fc_logits')(x)
        return logits

# --- Deeper Decoder with BatchNorm ---
class DeepDecoder(nn.Module):
    """Deeper VAE Decoder with additional layers and Batch Normalization."""
    output_shape: tuple
    training: bool

    @nn.compact
    def __call__(self, z):
        x = nn.Dense(128, name='dense1')(z)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='dense2')(z)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='dense3')(z)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='dense4')(z)
        x = nn.leaky_relu(x)
        x = nn.Dense(256, name='dense5')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(256, name='dense6')(x)
        x = nn.leaky_relu(x)
        # The final layer produces a flattened output that we interpret as the reconstructed image
        num_outputs = np.prod(self.output_shape)
        x = nn.Dense(num_outputs, name='dense7')(x)
        return x

# --- DeeperSimpleVAE Model ---
class DeeperSimpleVAE(nn.Module):
    """A deeper VAE model with binary latents, extra layers, and batch normalization."""
    latents: int = 16
    output_shape: tuple = (14, 14, 1)
    training: bool = True  # To control batch normalization behavior

    def setup(self):
        self.encoder = DeepEncoder(latents=self.latents, training=self.training)
        self.decoder = DeepDecoder(output_shape=self.output_shape, training=self.training)

    def __call__(self, x, z_rng):
        logits = self.encoder(x)
        # Sample binary latent vector using the binary quantizer
        z = binary_quantizer(z_rng, logits)
        recon_x = self.decoder(z)
        return recon_x, logits, z

    def generate(self, z):
        # Generate samples (useful for inference)
        x = self.decoder(z)
        x = jnp.reshape(x, self.output_shape)
        return x

# --- Helper function to instantiate the deeper simple VAE ---
def deeper_simple_vae(latents=16, output_shape=(14, 14, 1), training=True):
    return DeeperSimpleVAE(latents=latents, output_shape=output_shape, training=training)
