from flax import linen as nn
from jax import random
import jax.numpy as jnp

def binary_quantizer(rng, logits):
    """Binary quantization using Bernoulli sampling."""
    probs = nn.sigmoid(logits)  # Convert logits to probabilities
    binary_sample = random.bernoulli(rng, probs).astype(jnp.float32)  # Sample 0 or 1
    # Straight-through estimator for gradients
    binary_latent = binary_sample + probs - jax.lax.stop_gradient(probs)
    return binary_latent

class Encoder(nn.Module):
    """Binary VAE Encoder."""
    latents: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(500, name='fc1')(x)
        x = nn.relu(x)
        logits = nn.Dense(self.latents, name='fc2_logits')(x)  # Output logits instead of mean/logvar
        return logits

class Decoder(nn.Module):
    """VAE Decoder."""
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(500, name='fc1')(z)
        z = nn.relu(z)
        z = nn.Dense(784, name='fc2')(z)
        return z

class VAE(nn.Module):
    """Full Binary VAE model."""
    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        logits = self.encoder(x)
        z = binary_quantizer(z_rng, logits)
        recon_x = self.decoder(z)
        return recon_x, logits

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

def model(latents):
    return VAE(latents=latents)
