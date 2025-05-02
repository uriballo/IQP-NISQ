import math
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from datasets.utils import graph_to_vec, vec_to_adj

class ErdosRenyiDataset:
    """
    Erdős–Rényi random‐graph dataset for a fixed number of nodes (<=7).
    Graphs are represented as the flattened upper triangle of the adjacency
    matrix. Provides train/test splits and NumPy‐based utilities.

    Attributes:
        num_nodes (int): The number of nodes in each graph.
        num_features (int): The dimension of the flattened graph vector.
        num_samples (int): The total number of graphs generated.
        er_prob (float): The probability of edge creation in the G(n,p) model.
        train_split (float): The fraction of data to use for the training set.
        rng (np.random.Generator): NumPy random number generator instance.
        patterns (np.ndarray): Array of all generated graph vectors.
        labels (np.ndarray): Array of corresponding labels (0 for Erdos-Renyi).
        train_patterns (np.ndarray): Graph vectors for the training set.
        train_labels (np.ndarray): Labels for the training set.
        test_patterns (np.ndarray): Graph vectors for the test set.
        test_labels (np.ndarray): Labels for the test set.
    """

    GRAPH_TYPES = {0: "Erdos-Renyi"}
    NAME_TO_LABEL = {v: k for k, v in GRAPH_TYPES.items()}

    def __init__(
        self,
        num_nodes: int = 7,
        num_samples: int = 400,
        er_prob: float = 0.4,
        train_split: float = 0.8,
        seed: Optional[int] = None,
    ):
        """
        Initializes the ErdosRenyiDataset.

        Args:
            num_nodes (int): The number of nodes for each graph. Defaults to 7.
                             Must be <= 7 due to potential visualization limits
                             and computational complexity.
            num_samples (int): The total number of Erdős–Rényi graphs to
                               generate. Defaults to 400.
            er_prob (float): The probability 'p' for edge creation between any
                             two nodes in the G(n,p) Erdős–Rényi model.
                             Defaults to 0.4.
            train_split (float): The proportion of the generated samples to
                                 allocate to the training set. The rest will
                                 form the test set. Defaults to 0.8 (80%).
            seed (Optional[int]): An optional seed for the random number
                                  generator to ensure reproducibility.
                                  Defaults to None.
        """
        if num_nodes > 7:
            print(
                "Warning: num_nodes > 7 may lead to large feature vectors "
                "and slow processing/plotting."
            )
        if not 0.0 <= er_prob <= 1.0:
            raise ValueError("er_prob must be between 0 and 1.")
        if not 0.0 < train_split < 1.0:
            raise ValueError("train_split must be between 0 and 1.")

        self.num_nodes = num_nodes
        self.num_features = num_nodes * (num_nodes - 1) // 2
        self.num_samples = num_samples
        self.er_prob = er_prob
        self.train_split = train_split

        # NumPy RNG
        self.rng = np.random.default_rng(seed)

        # Generate all graphs
        self.patterns, self.labels = self._generate_graphs()
        if len(self.patterns) == 0:
            raise ValueError("No ER graphs generated. Check parameters.")

        # Shuffle and split
        perm = self.rng.permutation(len(self.patterns))
        self.patterns = self.patterns[perm]
        self.labels = self.labels[perm]
        split_i = math.ceil(len(self.patterns) * self.train_split)
        self.train_patterns = self.patterns[:split_i]
        self.train_labels = self.labels[:split_i]
        self.test_patterns = self.patterns[split_i:]
        self.test_labels = self.labels[split_i:]

        # Summary
        total = len(self.patterns)
        print(
            f"\nErdos-Renyi Dataset (k={self.num_nodes}, p={self.er_prob}): "
            f"{total} graphs"
        )
        for name, lbls in [
            ("Train", self.train_labels),
            ("Test", self.test_labels),
        ]:
            print(f"{name} count: {len(lbls)}")

    def _generate_graphs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate `num_samples` Erdős–Rényi graphs, flatten them to vectors,
        and return (patterns, labels).
        """
        vecs = []
        lbls = []
        for _ in range(self.num_samples):
            # use a fresh integer seed for each graph
            seed = int(self.rng.integers(0, 1e9))
            G = nx.gnp_random_graph(self.num_nodes, self.er_prob, seed=seed)
            vecs.append(self._graph_to_vector(G))
            lbls.append(self.NAME_TO_LABEL["Erdos-Renyi"])
        patterns = np.vstack(vecs).astype(np.float32)
        labels = np.array(lbls, dtype=np.int32)
        return patterns, labels

    def _graph_to_vector(self, G: nx.Graph) -> np.ndarray:
        """Flatten the upper‐triangle of the adjacency matrix to a vector."""
        return graph_to_vec(G, self.num_nodes)

    def _vec_to_adj(self, vec: np.ndarray) -> np.ndarray:
        """Reconstruct full adjacency matrix from upper‐triangle vector."""
        return vec_to_adj(vec, self.num_nodes)

    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (patterns, labels) for the training split."""
        return self.train_patterns, self.train_labels

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (patterns, labels) for the test split."""
        return self.test_patterns, self.test_labels

    def plot_random_samples(
        self, num_samples_to_plot: int = 5, seed: Optional[int] = None
    ):
        """
        Selects random graphs from the *entire* dataset and plots them.

        Args:
            num_samples_to_plot (int): The number of random graphs to display.
                                       Defaults to 5.
            seed (Optional[int]): An optional seed for selecting the samples
                                  to plot, allowing for reproducible plots.
                                  If None, uses the class's RNG.
        """
        if num_samples_to_plot <= 0:
            print("Number of samples to plot must be positive.")
            return
        if num_samples_to_plot > self.num_samples:
            print(
                f"Warning: Requested {num_samples_to_plot} samples, but only "
                f"{self.num_samples} are available. Plotting all."
            )
            num_samples_to_plot = self.num_samples

        plot_rng = np.random.default_rng(seed) if seed is not None else self.rng
        indices = plot_rng.choice(
            len(self.patterns), num_samples_to_plot, replace=False
        )

        # Determine grid size for subplots
        cols = math.ceil(math.sqrt(num_samples_to_plot))
        rows = math.ceil(num_samples_to_plot / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), squeeze=False
        )
        axes = axes.flatten()  # Flatten to 1D array for easy iteration

        for i, idx in enumerate(indices):
            vec = self.patterns[idx]
            label_idx = self.labels[idx]
            label_name = self.GRAPH_TYPES.get(label_idx, "Unknown")

            adj_matrix = self._vec_to_adj(vec)
            G = nx.from_numpy_array(adj_matrix)

            ax = axes[i]
            nx.draw(
                G,
                ax=ax,
                with_labels=True,
                node_color="skyblue",
                edge_color="gray",
                node_size=500,
            )
            ax.set_title(f"Sample {idx}: {label_name} Graph")
            ax.axis("off") # Turn off axis borders and ticks

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
