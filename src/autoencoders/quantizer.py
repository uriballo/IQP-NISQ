from jax import random, lax
import jax.numpy as jnp
from flax import linen as nn

def binary_quantizer(rng, logits):
    """Binary quantization using Bernoulli sampling with a straight-through estimator."""
    probs = nn.sigmoid(logits)  # Convert logits to probabilities
    binary_sample = random.bernoulli(rng, probs).astype(jnp.float32)  # Sample 0 or 1
    # Use the straight-through estimator to maintain gradient flow
    binary_latent = binary_sample + probs - lax.stop_gradient(probs)
    return binary_latent