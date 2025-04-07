import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.base import BaseDataset
from matplotlib import pyplot as plt

class LatentDataset(BaseDataset):
    def __init__(self, latent_file, label_file, batch_size=8):
        self.latent_file = latent_file
        self.label_file = label_file
        self.batch_size = batch_size
    
    def _load_data(self):
        # Load latents and labels from .npy files
        latents = np.load(self.latent_file)
        labels = np.load(self.label_file)
        return latents, labels
    
    def load(self):
        # Load data
        latents, labels = self._load_data()

        # Create a TensorFlow dataset from the NumPy arrays
        ds = tf.data.Dataset.from_tensor_slices((latents, labels))
        ds = ds.shuffle(buffer_size=latents.shape[0])
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

