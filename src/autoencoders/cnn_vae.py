from flax import linen as nn
from jax import random, lax
import jax.numpy as jnp
import numpy as np

# --- Binary Quantizer Function ---
def binary_quantizer(rng, logits):
    """Binary quantization using Bernoulli sampling with a straight-through estimator."""
    probs = nn.sigmoid(logits)  # Convert logits to probabilities
    binary_sample = random.bernoulli(rng, probs).astype(jnp.float32)  # Sample 0 or 1
    # Straight-through estimator to allow gradients to flow
    binary_latent = binary_sample + probs - lax.stop_gradient(probs)
    return binary_latent

# --- Deeper CNN-based Encoder ---
class DeepCNNEncoder(nn.Module):
    """Deeper CNN Encoder for a VAE with a binary latent space."""
    latents: int
    training: bool  # Controls batchnorm behavior

    @nn.compact
    def __call__(self, x):
        # Input assumed to be (batch, 14, 14, 1)
        # Layer 1: Convolution with stride 1 preserves spatial resolution
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv1')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn1")(x)
        x = nn.leaky_relu(x)

        # Layer 2: Downsample spatially with stride 2
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='conv2')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn2")(x)
        x = nn.leaky_relu(x)

        # Layer 3: Further feature extraction with stride 1
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='conv3')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn3")(x)
        x = nn.leaky_relu(x)

        # Layer 4: Downsample again
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='conv4')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn4")(x)
        x = nn.leaky_relu(x)
        
        # Flatten and project to a deeper feature representation
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(256, name='dense_hidden')(x)
        x = nn.leaky_relu(x)
        # Output latent logits for binary sampling
        logits = nn.Dense(self.latents, name='latent_logits')(x)
        return logits

# --- Deeper CNN-based Decoder ---
class DeepCNNDecoder(nn.Module):
    """Deeper CNN Decoder for the VAE with binary latent space."""
    output_shape: tuple  # e.g. (14, 14, 1)
    training: bool

    @nn.compact
    def __call__(self, z):
        # Project latent vector to a shape that can be reshaped into a small feature map.
        # Here we choose (4, 4, 64) as an example.
        x = nn.Dense(4 * 4 * 64, name='dense_project')(z)
        x = nn.leaky_relu(x)
        x = x.reshape((x.shape[0], 4, 4, 64))
        
        # Deconvolution layer 1: Upsample to 7x7
        x = nn.ConvTranspose(features=64, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv1')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn1")(x)
        x = nn.leaky_relu(x)

        # Deconvolution layer 2: Refine features at 7x7 resolution
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='deconv2')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn2")(x)
        x = nn.leaky_relu(x)

        # Deconvolution layer 3: Upsample to final resolution 14x14
        x = nn.ConvTranspose(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='deconv3')(x)
        x = nn.BatchNorm(use_running_average=not self.training, momentum=0.9, epsilon=1e-5, name="bn3")(x)
        x = nn.leaky_relu(x)
        
        # Final convolution to adjust to the output channels (e.g., 1 for grayscale)
        x = nn.ConvTranspose(features=self.output_shape[-1], kernel_size=(3, 3), strides=(1, 1), padding='SAME', name='deconv4')(x)
        # Depending on your use-case, you might apply an activation such as nn.sigmoid here.
        return x

# --- Deeper CNN VAE Model ---
class DeepCNNVAE(nn.Module):
    """A deeper CNN-based VAE using a binary latent representation."""
    latents: int = 16
    output_shape: tuple = (14, 14, 1)
    training: bool = True  # Controls BatchNorm mode

    def setup(self):
        self.encoder = DeepCNNEncoder(latents=self.latents, training=self.training)
        self.decoder = DeepCNNDecoder(output_shape=self.output_shape, training=self.training)

    def __call__(self, x, z_rng):
        # Get latent logits from the encoder and sample the binary latent vector.
        logits = self.encoder(x)
        z = binary_quantizer(z_rng, logits)
        # Decode the latent vector back to an image.
        recon_x = self.decoder(z)
        return recon_x, logits, z

    def generate(self, z):
        """Generate images from the latent vector."""
        return self.decoder(z)

def cnn_vae(latents=16, output_shape=(14, 14, 1), training=True):
    return DeepCNNVAE(latents=latents, output_shape=output_shape, training=training)
