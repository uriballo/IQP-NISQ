from flax import linen as nn
from jax import random, lax
import jax.numpy as jnp
import jax
import numpy as np

# --- Binary Quantizer Function ---
def binary_quantizer(rng, logits):
    """Binary quantization using Bernoulli sampling."""
    probs = nn.sigmoid(logits)  # Convert logits to probabilities
    binary_sample = random.bernoulli(rng, probs).astype(jnp.float32)  # Sample 0 or 1
    # Straight-through estimator for gradients
    binary_latent = binary_sample + probs - lax.stop_gradient(probs)
    return binary_latent

# --- Model Definitions ---
class Encoder(nn.Module):
    """Binary VAE Encoder."""
    latents: int

    @nn.compact
    def __call__(self, x):
        x = jnp.reshape(x.shape[0], -1)
        x = nn.Dense(128, name='fc1')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc2')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc3')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc4')(x)
        x = nn.leaky_relu(x)
        logits = nn.Dense(self.latents, name='fc_logits')(x)  # Output logits for binary sampling
        return logits

class Decoder(nn.Module):
    """VAE Decoder."""
    output_shape: tuple

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(128, name='fc1')(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(128, name='fc2')(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(128, name='fc3')(z)
        z = nn.leaky_relu(z)
        num_outputs = np.prod(self.output_shape)
        z = nn.Dense(num_outputs, name='fc4')(z)
        return z

class VAE(nn.Module):
    """Full Binary VAE model."""
    latents: int = 16
    output_shape: tuple = (14, 14, 1)

    def setup(self):
        self.encoder = Encoder(latents=self.latents)
        self.decoder = Decoder(output_shape=self.output_shape)

    def __call__(self, x, z_rng):
        logits = self.encoder(x)
        z = binary_quantizer(z_rng, logits)
        recon_x = self.decoder(z)
        recon_x = jnp.reshape(recon_x, self.output_shape)
        return recon_x, logits, z

    def generate(self, z):
        z = self.decoder(z)
        z = jnp.reshape(z, self.output_shape)
        return z #nn.sigmoid(z)

def model(latents):
    return VAE(latents=latents)