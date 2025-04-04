import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.base import BaseDataset
from matplotlib import pyplot as plt

class BarsStripesDataset(BaseDataset):
    def __init__(self, split='train', num_examples=1000, batch_size=8, image_size=(4, 4)):
        self.split = split
        self.num_examples = num_examples
        self.batch_size = batch_size
        self.image_size = image_size
    
    def _generate_data(self):
        # 0 - Bar
        # 1 - Stripe
        data = []
        labels = []
        for _ in range(self.num_examples // 2):
            label, sample = self.generate_bar(self.image_size)
            data.append(sample)
            labels.append(label)
            
            label, sample = self.generate_stripe(self.image_size)
            data.append(sample)
            labels.append(label)
        
        return tf.data.Dataset.from_tensor_slices((np.array(data, dtype=np.float32), np.array(labels, dtype=np.int32)))
    
    @staticmethod
    def generate_bar(size):
        position = np.random.randint(0, size[0])
        pattern = np.zeros(size, dtype=int)
        pattern[:, position] = 1
        return 0, pattern.flatten()
    
    @staticmethod
    def generate_stripe(size):
        position = np.random.randint(0, size[0])
        pattern = np.zeros(size, dtype=int)
        pattern[position, :] = 1
        return 1, pattern.flatten()
    
    def load(self):
        ds = self._generate_data()
        ds = ds.shuffle(buffer_size=self.num_examples)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def plot_random_batch(self, ds = None):
        if ds is None:
            ds = self.load()
            
        label_map = {0: 'Bar', 1: 'Stripe'}
        
        for images, labels in ds.take(1):
            fig, axes = plt.subplots(1, min(8, len(images)), figsize=(12, 3), gridspec_kw={'wspace': 0.5})
            for i, (image, label) in enumerate(zip(images.numpy(), labels.numpy())):
                axes[i].imshow(1 - image.reshape(4, 4), cmap='gray')  # Invert colors
                axes[i].set_title(f'{label_map[label]}')
                axes[i].spines['top'].set_visible(True)
                axes[i].spines['bottom'].set_visible(True)
                axes[i].spines['left'].set_visible(True)
                axes[i].spines['right'].set_visible(True)
                for spine in axes[i].spines.values():
                    spine.set_linewidth(1)
                    spine.set_color('black')
            plt.show()
