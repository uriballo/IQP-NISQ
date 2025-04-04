import tensorflow_datasets as tfds
import tensorflow as tf
from datasets.base import BaseDataset
from matplotlib import pyplot as plt

class MNISTDataset(BaseDataset):
    def __init__(self, split='train', num_examples=None, batch_size=32, image_size=(28, 28)):
        """
        Args:
            split (str): Which split to load ('train' or 'test').
            num_examples (int, optional): Number of examples to take (for debugging or subsetting).
            batch_size (int): Number of samples per batch.
            image_size (tuple): Resize images to this size.
        """
        self.split = split
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.image_size = image_size

    def _preprocess(self, image, label):
        # Resize image and normalize to [0,1]
        image = tf.image.resize(image, self.image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def load(self):
        # Load MNIST with TensorFlow Datasets
        ds = tfds.load('mnist', split=self.split, shuffle_files=True, as_supervised=True)
        # Subset the dataset if num_examples is specified
        if self.num_examples:
            ds = ds.take(self.num_examples)
        ds = ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds