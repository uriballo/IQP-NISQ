import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.base import BaseDataset
from matplotlib import pyplot as plt


def random_unitary():
    """Generate a random 2x2 unitary using QR decomposition."""
    X = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    Q, _ = np.linalg.qr(X)
    return Q

def apply_random_local_unitaries(state):
    """
    Apply a random local unitary transformation U = U1 ⊗ U2 ⊗ U3 to a state.
    This mimics the augmentation by randomly perturbing the state.
    """
    U = np.kron(np.kron(random_unitary(), random_unitary()), random_unitary())
    return U @ state

def state_to_density_features(state):
    """
    Convert a state vector to its density matrix and then split
    the density matrix into real and imaginary parts.
    
    Returns:
        features: a flattened vector [Re(ρ), Im(ρ)] of shape (128,).
    """
    # Density matrix is the outer product |ψ⟩⟨ψ|
    rho = np.outer(state, np.conjugate(state))
    re = np.real(rho).astype(np.float32)
    im = np.imag(rho).astype(np.float32)
    features = np.concatenate([re.flatten(), im.flatten()])
    return features

# --- State generators for each SLOCC class ---
# Class labels: 0=SEP, 1=BS1, 2=BS2, 3=BS3, 4=W, 5=GHZ

def generate_separable_state():
    # Fully separable: tensor product of three random qubit states
    state_A = np.random.randn(2) + 1j * np.random.randn(2)
    state_A /= np.linalg.norm(state_A)
    state_B = np.random.randn(2) + 1j * np.random.randn(2)
    state_B /= np.linalg.norm(state_B)
    state_C = np.random.randn(2) + 1j * np.random.randn(2)
    state_C /= np.linalg.norm(state_C)
    state = np.kron(np.kron(state_A, state_B), state_C)
    return state
def schmidt_decomp(psi):
    #Schmidt coefficients for a bipartite state
    psi_matrix = psi.reshape(2,2)
    U, S, Vh = np.linalg.svd(psi_matrix)
    return S

def generate_bs1_state():
    # Biseparable 1: qubit A separable, qubits B and C entangled.
    state_A = np.random.randn(2) + 1j * np.random.randn(2)
    state_A /= np.linalg.norm(state_A)
    while np.count_nonzero(schmidt_decomp(state_A)) == 1: 
        state_A = np.random.randn(2) + 1j * np.random.randn(2)
        state_A /= np.linalg.norm(state_A)
    state_BC = np.random.randn(4) + 1j * np.random.randn(4)
    state_BC /= np.linalg.norm(state_BC)
    state = np.kron(state_A, state_BC)
    return state

def generate_bs2_state():
    # Biseparable 2: In our implementation, we mimic BS2 by 
    # generating a BS1 state and swapping qubits B and C.
    state = generate_bs1_state()
    state = state.reshape([2, 2, 2])
    state = np.transpose(state, (0, 2, 1)).flatten()
    return state

def generate_bs3_state():
    # Biseparable 3: qubits A and B entangled, qubit C separable.
    state_AB = np.random.randn(4) + 1j * np.random.randn(4)
    state_AB /= np.linalg.norm(state_AB)
    state_C = np.random.randn(2) + 1j * np.random.randn(2)
    state_C /= np.linalg.norm(state_C)
    state = np.kron(state_AB, state_C)
    return state

def generate_ghz_state():
    # GHZ state for three qubits: (|000> + |111>)/sqrt2
    state = np.array([1, 0, 0, 0, 0, 0, 0, 1], dtype=complex)
    state /= np.sqrt(2)
    return state

def generate_w_state():
    # W state for three qubits: (|001> + |010> + |100>)/sqrt3
    # In computational basis: index 1: |001>, index 2: |010>, index 4: |100>
    state = np.zeros(8, dtype=complex)
    state[1] = 1.0
    state[2] = 1.0
    state[4] = 1.0
    state /= np.sqrt(3)
    return state

                


class ThreeQubitEntanglementDataset(BaseDataset):
    def __init__(self, split='train', num_examples=None, batch_size=32):
        """
        Args:
            split (str): Which split to load ('train' or 'test').
            num_examples (int, optional): Total number of examples to generate.
                For demonstration, the default is 6000. In the paper, 6 million states
                (1e6 per class) were used.
            batch_size (int): Number of samples per batch.
        """
        self.split = split
        # default total examples if not provided
        self.total_examples = num_examples if num_examples is not None else 6000
        self.batch_size = batch_size

    def _preprocess(self, features, label):
        # In this example, we assume the features are already precomputed,
        # but here one may apply additional normalization if needed.
        features = tf.cast(features, tf.float32)
        label = tf.cast(label, tf.int32)
        return features, label

    def load(self):
        """
        Generates a dataset of three-qubit density matrix features and SLOCC labels.
        
        Returns:
            tf.data.Dataset: A dataset object yielding (features, label) tuples.
        """
        num_classes = 6  # SEP, BS1, BS2, BS3, W, GHZ
        samples_per_class = self.total_examples // num_classes

        features_list = []
        labels_list = []

        # Mapping from label to generator function.
        generators = {
            0: generate_separable_state,
            1: generate_bs1_state,
            2: generate_bs2_state,
            3: generate_bs3_state,
            4: generate_w_state,
            5: generate_ghz_state,
        }

        # Generate samples per class
        for label, gen_func in generators.items():
            for _ in range(samples_per_class):
                state = gen_func()
                # Apply a random local unitary transformation for augmentation.
                state = apply_random_local_unitaries(state)
                feat = state_to_density_features(state)  # shape (128,)
                features_list.append(feat)
                labels_list.append(label)

        # Convert to NumPy arrays.
        features_np = np.array(features_list, dtype=np.float32)
        labels_np = np.array(labels_list, dtype=np.int32)

        # Shuffle the dataset.
        indices = np.arange(features_np.shape[0])
        np.random.shuffle(indices)
        features_np = features_np[indices]
        labels_np = labels_np[indices]

        split_index = int(0.8 * features_np.shape[0])
        if self.split == 'train':
            features_np = features_np[:split_index]
            labels_np = labels_np[:split_index]
        elif self.split == 'test':
            features_np = features_np[split_index:]
            labels_np = labels_np[split_index:]
        else:
            raise ValueError("Split must be 'train' or 'test'.")

        # Create a tf.data.Dataset from the NumPy arrays.
        ds = tf.data.Dataset.from_tensor_slices((features_np, labels_np))
        ds = ds.map(lambda f, l: self._preprocess(f, l), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

