from flax import linen as nn
from jax import random, lax
import jax.numpy as jnp
import jax

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
        x = nn.Dense(128, name='fc1')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc2')(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc3')(x)
        x = nn.leaky_relu(x)
        x = nn.leaky_relu(x)
        x = nn.Dense(128, name='fc4')(x)
        x = nn.leaky_relu(x)
        logits = nn.Dense(self.latents, name='fc_logits')(x)  # Output logits for binary sampling
        return logits

class Decoder(nn.Module):
    """VAE Decoder."""
    @nn.compact
    def __call__(self, z):
        z = nn.Dense(128, name='fc1')(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(128, name='fc2')(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(128, name='fc3')(z)
        z = nn.leaky_relu(z)
        z = nn.Dense(196, name='fc4')(z)
        return z

class VAE(nn.Module):
    """Full Binary VAE model."""
    latents: int = 16

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x, z_rng):
        logits = self.encoder(x)
        z = binary_quantizer(z_rng, logits)
        recon_x = self.decoder(z)
        return recon_x, logits, z

    def generate(self, z):
        return nn.sigmoid(self.decoder(z))

def model(latents):
    return VAE(latents=latents)