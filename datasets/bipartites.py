import math
from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from datasets.utils import graph_to_vec, vec_to_adj

class BipartiteGraphDataset:
    """
    Generates a dataset of graphs with a specified number of vertices,
    represented as flattened upper-triangle vectors. Each graph is labeled
    as either bipartite (0) or non-bipartite (1).

    The dataset generation attempts to match the desired ratio of bipartite
    to non-bipartite graphs and can optionally ensure all generated graphs
    are connected.

    Attributes:
        num_samples (int): The target number of samples in the dataset.
                           The actual number might be slightly lower if
                           generation attempts fail frequently.
        num_vertices (int): The number of vertices in each graph.
        ratio_bipartite (float): The desired proportion of bipartite graphs
                                 in the dataset (between 0.0 and 1.0).
        edge_prob (float): The probability used for edge creation in the
                           underlying random graph models (bipartite random
                           graph and G(n,p)).
        ensure_connected (bool): If True, only connected graphs are included
                                 in the dataset.
        vector_size (int): The dimensionality of the flattened graph vector
                           (num_vertices * (num_vertices - 1) // 2).
        max_generation_attempts (int): The maximum number of attempts per
                                       graph type during generation before
                                       giving up on that specific graph.
        data (List[Tuple[np.ndarray, int]]): A list holding the generated
                                             data as (vector, label) tuples.
                                             Label 0: Bipartite, Label 1: Non-Bipartite.
        _rng (np.random.Generator): NumPy random number generator instance.
        _networkx_seed (Optional[int]): Seed used for networkx graph generation functions.
                                        Incremented after each use if not None.
    """
    GRAPH_TYPES = {0: "Bipartite", 1: "Non-Bipartite"}

    def __init__(
        self,
        num_samples: int,
        num_vertices: int = 14,
        ratio_bipartite: float = 0.5,
        edge_prob: float = 0.2,
        ensure_connected: bool = True,
        seed: Optional[int] = None,
        max_generation_attempts: int = 100,
    ):
        """
        Initializes the BipartiteGraphDataset.

        Args:
            num_samples (int): The target number of graph samples to generate.
            num_vertices (int): The number of vertices for each graph.
                                Defaults to 14.
            ratio_bipartite (float): The desired fraction of bipartite graphs
                                     in the final dataset. Must be between 0.0
                                     and 1.0. Defaults to 0.5.
            edge_prob (float): The probability 'p' for edge creation used in
                               the random graph generation models (both
                               bipartite and G(n,p)). Defaults to 0.2.
            ensure_connected (bool): If True, the generation process will
                                     discard graphs that are not connected.
                                     Defaults to True.
            seed (Optional[int]): An optional seed for the random number
                                  generators (NumPy and NetworkX) to ensure
                                  reproducibility. Defaults to None.
            max_generation_attempts (int): The maximum number of times to
                                           attempt generating a single valid
                                           graph (bipartite or non-bipartite)
                                           before potentially moving on.
                                           Defaults to 100.
        """
        if not (0.0 <= ratio_bipartite <= 1.0):
            raise ValueError("ratio_bipartite must be between 0.0 and 1.0")
        if not (0.0 <= edge_prob <= 1.0):
            raise ValueError("edge_prob must be between 0.0 and 1.0")
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if num_vertices <= 1:
            raise ValueError("num_vertices must be greater than 1.")

        self.num_samples = num_samples
        self.num_vertices = num_vertices
        self.ratio_bipartite = ratio_bipartite
        self.edge_prob = edge_prob
        self.ensure_connected = ensure_connected
        self.vector_size = num_vertices * (num_vertices - 1) // 2
        self.max_generation_attempts = max_generation_attempts

        self._rng = np.random.default_rng(seed)
        # Use the main seed for networkx initially, will be incremented
        self._networkx_seed = seed

        self.data: List[Tuple[np.ndarray, int]] = []
        self._generate_data()

        # Print summary after generation
        actual_samples = len(self.data)
        if actual_samples == 0:
             print("\nWarning: No graphs were generated. Check parameters "
                   "(e.g., edge_prob might be too low for connectivity).")
        else:
            bipartite_count = sum(1 for _, label in self.data if label == 0)
            non_bipartite_count = actual_samples - bipartite_count
            actual_ratio = bipartite_count / actual_samples
            print(f"\nBipartite Dataset (k={self.num_vertices}, "
                  f"p={self.edge_prob}, connected={self.ensure_connected}):")
            print(f"Target samples: {num_samples}, Generated samples: {actual_samples}")
            print(f"Bipartite: {bipartite_count}, Non-Bipartite: {non_bipartite_count}")
            print(f"Target Ratio (Bipartite): {self.ratio_bipartite:.3f}, "
                  f"Actual Ratio: {actual_ratio:.3f}")


    def __len__(self) -> int:
        """Returns the actual number of samples generated."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """
        Retrieves the graph vector and label at the specified index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            Tuple[np.ndarray, int]: A tuple containing the flattened graph
                                    vector and its label (0 for bipartite,
                                    1 for non-bipartite).

        Raises:
            IndexError: If the index is out of the valid range.
        """
        if not 0 <= idx < len(self.data):
            raise IndexError(f"Dataset index {idx} out of range for size {len(self.data)}")
        return self.data[idx]

    def get_all_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns all generated graph vectors and their corresponding labels
        as NumPy arrays.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                                           - The first array contains all graph
                                             vectors (shape: [num_actual_samples, vector_size]).
                                           - The second array contains the
                                             corresponding labels (shape: [num_actual_samples]).
        """
        if not self.data:
            # Return empty arrays with correct second dimension if no data
            return np.zeros((0, self.vector_size), dtype=np.float32), np.zeros(0, dtype=np.int32)
        # Ensure consistent types
        vecs, labels = zip(*self.data)
        return np.array(vecs, dtype=np.float32), np.array(labels, dtype=np.int32)

    def _graph_to_vector(self, graph: nx.Graph) -> np.ndarray:
        """Converts a NetworkX graph to its flattened upper-triangle vector."""
        return graph_to_vec(graph, self.num_vertices)

    def _vec_to_adj(self, vec: np.ndarray) -> np.ndarray:
        """Reconstructs the adjacency matrix from a flattened vector."""
        return vec_to_adj(vec, self.num_vertices)

    def _generate_bipartite(self) -> Optional[nx.Graph]:
        """Attempts to generate a single valid bipartite graph."""
        for _ in range(self.max_generation_attempts):
            # Split nodes as evenly as possible for the two partitions
            n1 = self.num_vertices // 2
            n2 = self.num_vertices - n1
            # Create a random bipartite graph
            g = nx.bipartite.random_graph(
                n1, n2, p=self.edge_prob, seed=self._networkx_seed
            )
            # Increment seed for next potential networkx call
            if self._networkx_seed is not None:
                self._networkx_seed += 1

            # Check connectivity if required
            # Note: nx.is_connected requires > 0 nodes. Handles single node case implicitly.
            if self.ensure_connected and g.number_of_nodes() > 1:
                if not nx.is_connected(g):
                    continue # Try again if not connected

            # Ensure graph has exactly num_vertices (handles isolated nodes if any)
            # This step might be redundant if random_graph guarantees node count,
            # but it's safer to include.
            if g.number_of_nodes() < self.num_vertices:
                g.add_nodes_from(
                    range(g.number_of_nodes(), self.num_vertices)
                )

            # Final check on node count before returning
            if g.number_of_nodes() == self.num_vertices:
                # Sanity check: Ensure it's actually bipartite (should always be true)
                if nx.is_bipartite(g):
                    return g
                # else: # This case should ideally not happen with nx.bipartite.random_graph
                #    print("Warning: Generated graph intended as bipartite is not.")
        # Return None if max attempts reached without success
        # print("Warning: Failed to generate a valid bipartite graph after max attempts.")
        return None

    def _generate_non_bipartite(self) -> Optional[nx.Graph]:
        """Attempts to generate a single valid non-bipartite graph."""
        for _ in range(self.max_generation_attempts):
            # Generate a general random graph using G(n,p) model
            g = nx.gnp_random_graph(
                self.num_vertices, p=self.edge_prob,
                seed=self._networkx_seed
            )
            # Increment seed for next potential networkx call
            if self._networkx_seed is not None:
                self._networkx_seed += 1

            # Check if it's bipartite (we want non-bipartite)
            if nx.is_bipartite(g):
                continue # Try again if it is bipartite

            # Check connectivity if required
            if self.ensure_connected and g.number_of_nodes() > 0:
                 if not nx.is_connected(g):
                    continue # Try again if not connected

            # If all checks pass, return the graph
            return g
        # Return None if max attempts reached without success
        # print("Warning: Failed to generate a valid non-bipartite graph after max attempts.")
        return None

    def _generate_data(self):
        """Generates the full dataset according to the specified parameters."""
        target_bipartite = int(self.num_samples * self.ratio_bipartite)
        target_non_bipartite = self.num_samples - target_bipartite
        generated_bipartite, generated_non_bipartite = 0, 0

        # Set a generous overall attempt limit to prevent infinite loops
        # in difficult parameter regimes (e.g., low p with ensure_connected)
        total_attempts = 0
        max_total_attempts = self.num_samples * self.max_generation_attempts * 2 # Heuristic limit

        generated_data_list = [] # Use a temporary list

        while (generated_bipartite < target_bipartite or generated_non_bipartite < target_non_bipartite) \
              and total_attempts < max_total_attempts:

            total_attempts += 1
            # Decide whether to try generating bipartite or non-bipartite
            # Aim to maintain the target ratio during generation
            current_total = generated_bipartite + generated_non_bipartite
            if current_total == 0: # Start with bipartite if ratio >= 0.5
                 try_bipartite = self.ratio_bipartite >= 0.5
            else:
                current_ratio = generated_bipartite / current_total
                # Try to generate the type that is further below its target ratio,
                # or if one type has already reached its target.
                if generated_bipartite >= target_bipartite:
                    try_bipartite = False
                elif generated_non_bipartite >= target_non_bipartite:
                    try_bipartite = True
                else: # Neither target reached, try to balance ratio
                    try_bipartite = current_ratio < self.ratio_bipartite


            if try_bipartite:
                g = self._generate_bipartite()
                if g is not None:
                    generated_data_list.append((self._graph_to_vector(g), 0))
                    generated_bipartite += 1
            else: # Try non-bipartite
                g = self._generate_non_bipartite()
                if g is not None:
                    generated_data_list.append((self._graph_to_vector(g), 1))
                    generated_non_bipartite += 1

            # Early exit if we somehow generate more than needed (shouldn't happen with current logic)
            if len(generated_data_list) >= self.num_samples:
                 break

        if total_attempts >= max_total_attempts:
            print(f"Warning: Reached maximum total generation attempts ({max_total_attempts}). "
                  f"Dataset size may be smaller than requested.")

        # Shuffle the collected data and assign to self.data
        self._rng.shuffle(generated_data_list)
        self.data = generated_data_list
        self.num_samples = len(self.data)

    def plot_random_samples(
        self, num_samples_to_plot: int = 6, seed: Optional[int] = None
    ):
        """
        Selects random graphs from the generated dataset and plots them.

        Args:
            num_samples_to_plot (int): The number of random graphs to display.
                                       Defaults to 6.
            seed (Optional[int]): An optional seed for selecting the samples
                                  to plot, allowing for reproducible plots.
                                  If None, uses the class's internal RNG.
        """
        if not self.data:
            print("Dataset is empty. Cannot plot samples.")
            return
        if num_samples_to_plot <= 0:
            print("Number of samples to plot must be positive.")
            return

        actual_num_samples = len(self.data)
        if num_samples_to_plot > actual_num_samples:
            print(
                f"Warning: Requested {num_samples_to_plot} samples, but only "
                f"{actual_num_samples} are available. Plotting all."
            )
            num_samples_to_plot = actual_num_samples

        plot_rng = np.random.default_rng(seed) if seed is not None else self._rng
        indices = plot_rng.choice(
            actual_num_samples, num_samples_to_plot, replace=False
        )

        # Determine grid size for subplots
        cols = math.ceil(math.sqrt(num_samples_to_plot))
        rows = math.ceil(num_samples_to_plot / cols)

        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 4, rows * 4), squeeze=False
        )
        axes = axes.flatten()  # Flatten to 1D array for easy iteration

        for i, idx in enumerate(indices):
            vec, label_idx = self.data[idx]
            label_name = self.GRAPH_TYPES.get(label_idx, "Unknown")

            adj_matrix = self._vec_to_adj(vec)
            G = nx.from_numpy_array(adj_matrix)

            ax = axes[i]
            # Use different colors for bipartite/non-bipartite for clarity
            node_color = "lightgreen" if label_idx == 0 else "salmon"
            nx.draw(
                G,
                ax=ax,
                with_labels=True,
                node_color=node_color,
                edge_color="gray",
                node_size=max(100, 4000 // self.num_vertices), # Adjust node size
                font_size=max(6, 12 - self.num_vertices // 3) # Adjust font size
            )
            ax.set_title(f"Sample Index {idx}: {label_name}")
            ax.axis("off") # Turn off axis borders and ticks

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        