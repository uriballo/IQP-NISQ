import jax 
import jax.numpy as jnp
import numpy as np
from flax import nnx

class simple_autoencoder(nnx.Module):
    def __init__(self, latent_dim: int  = 16, rngs: nnx.Rngs = nnx.Rngs(42)):
        self.encode1 = nnx.Linear(784, 256, rngs=rngs)
        self.encode2 = nnx.Linear(256, 64, rngs=rngs)
        self.encode3 = nnx.Linear(64, latent_dim, rngs=rngs)
        self.decode1 = nnx.Linear(latent_dim, 64, rngs=rngs)
        self.decode2 = nnx.Linear(64, 256, rngs=rngs)
        self.decode3 = nnx.Linear(256, 784, rngs=rngs)
        self.rngs = rngs
        self.latent_dim = latent_dim
    
    def encoder(self, x):
        x = nnx.relu(self.encode1(x))
        x = nnx.relu(self.encode2(x))
        x = nnx.sigmoid(self.encode3(x))
        return x
    
    def decoder(self, x):
        # Decoder logic here
        pass
    
    def __call__(self, x):
        latent_prob = self.encoder(x)
        binary_sample = jax.random.bernoulli(key=self.rngs, p=latent_prob)
        
        latent_code = binary_sample + latent_prob - latent_prob.detach()