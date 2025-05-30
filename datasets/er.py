import pickle
import os
from datetime import datetime
import networkx as nx
import numpy as np
from typing import Union, Tuple, List, Optional, Dict, Any


class ErdosRenyiGraphDataset:
    def __init__(
        self,
        nodes: int,
        edge_prob: Union[float, Tuple[float, float]],
        extra_permutations: int = 0,
        store_graphs: bool = True,
        verbose: bool = True,
        prob_bins: int = 10,
    ):
        """
        Dataset generator for Erdős-Rényi graphs with isomorphism checking.
        
        Args:
            nodes: Number of nodes |V| = n
            edge_prob: Single p or (p_min, p_max) for edge probability range
            extra_permutations: How many extra shuffled copies per base graph
            store_graphs: If True, keep Graph objects in self.graphs
            verbose: If True, prints progress
            prob_bins: Number of bins to partition edge probability range
        """
        if nodes <= 0:
            raise ValueError("`nodes` must be positive.")
        self.nodes = nodes

        if isinstance(edge_prob, float):
            if not (0 <= edge_prob <= 1):
                raise ValueError("edge_prob ∈ [0,1]")
            self.p_min = self.p_max = edge_prob
        else:
            p0, p1 = edge_prob
            if not (0 <= p0 <= 1 and 0 <= p1 <= 1):
                raise ValueError("edge_prob entries ∈ [0,1]")
            self.p_min, self.p_max = sorted((p0, p1))

        self.extra_permutations = extra_permutations
        self.store_graphs = store_graphs
        self.verbose = verbose
        self.prob_bins = prob_bins

        self.vectors: List[np.ndarray] = []
        self.edge_probs: List[float] = []
        if self.store_graphs:
            self.graphs: List[nx.Graph] = []

        # For isomorphism detection
        self._all_graphs: List[nx.Graph] = []
        self._certificates: List[Tuple] = []

    def _get_prob_bins(self) -> List[Tuple[float, float]]:
        """Get probability range bins for partitioning."""
        if self.p_min == self.p_max:
            return [(self.p_min, self.p_max)]
        
        bins = []
        step = (self.p_max - self.p_min) / self.prob_bins
        for i in range(self.prob_bins):
            bin_start = self.p_min + i * step
            bin_end = self.p_min + (i + 1) * step
            bins.append((bin_start, bin_end))
        return bins

    def _estimate_bin_capacity(self, p_min: float, p_max: float) -> int:
        """
        Estimate how many unique graphs we can generate in a probability bin.
        This is a heuristic based on expected graph properties.
        """
        n = self.nodes
        max_edges = n * (n - 1) // 2
        
        # For very small or very large probabilities, fewer unique graphs
        p_mid = (p_min + p_max) / 2
        if p_mid < 0.1 or p_mid > 0.9:
            base_capacity = min(1000, max_edges)
        else:
            # Peak diversity around p=0.5
            base_capacity = min(5000, max_edges * 2)
        
        # Scale based on number of nodes
        if n <= 10:
            return min(base_capacity, 100)
        elif n <= 20:
            return min(base_capacity, 1000)
        else:
            return base_capacity

    def _compute_target_distribution(
        self, prob_bins: List[Tuple[float, float]], total_samples: int
    ) -> Dict[Tuple[float, float], int]:
        """Compute target number of samples per probability bin."""
        capacities = np.array([
            self._estimate_bin_capacity(p_min, p_max) 
            for p_min, p_max in prob_bins
        ])
        
        # Distribute samples proportionally to capacity
        total_capacity = capacities.sum()
        if total_capacity == 0:
            return {}
        
        raw_allocation = capacities * total_samples / total_capacity
        allocation = np.round(raw_allocation).astype(int)
        allocation = np.clip(allocation, 1, capacities)  # At least 1 per bin
        
        # Handle rounding remainder
        remainder = total_samples - allocation.sum()
        if remainder > 0:
            # Add to bins with largest fractional parts
            fractions = raw_allocation - np.floor(raw_allocation)
            for idx in np.argsort(fractions)[::-1][:remainder]:
                if allocation[idx] < capacities[idx]:
                    allocation[idx] += 1
        elif remainder < 0:
            # Remove from bins with largest allocations
            for idx in np.argsort(allocation)[::-1][:-remainder]:
                if allocation[idx] > 1:
                    allocation[idx] -= 1
        
        return {
            bin_range: int(alloc)
            for bin_range, alloc in zip(prob_bins, allocation)
            if alloc > 0
        }

    def _vectorize(self, G: nx.Graph) -> np.ndarray:
        """Convert graph to upper triangular adjacency vector."""
        mat = nx.to_numpy_array(G, nodelist=range(self.nodes), dtype=np.uint8)
        iu = np.triu_indices(self.nodes, k=1)
        return mat[iu].astype(np.uint8)

    def _apply_permutation(
        self,
        vec: Optional[np.ndarray] = None,
        G: Optional[nx.Graph] = None,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, nx.Graph]:
        """Apply random node permutation to graph or vector."""
        if seed is not None:
            np.random.seed(seed)

        permutation = np.random.permutation(self.nodes)
        mapping = {i: int(permutation[i]) for i in range(self.nodes)}

        if vec is not None:
            G0 = self._vec_to_graph(vec)
            Gp = nx.relabel_nodes(G0, mapping, copy=True)
            return self._vectorize(Gp)
        if G is not None:
            return nx.relabel_nodes(G, mapping, copy=True)

        raise ValueError("Must pass `vec` or `G`")

    def _vec_to_graph(self, vec: np.ndarray) -> nx.Graph:
        """Convert vector back to graph (helper for permutation)."""
        G = nx.Graph()
        G.add_nodes_from(range(self.nodes))
        iu = np.triu_indices(self.nodes, k=1)
        edges = [(i, j) for (i, j), val in zip(zip(iu[0], iu[1]), vec) if val]
        G.add_edges_from(edges)
        return G

    def _get_graph_certificate(self, G: nx.Graph) -> Tuple:
        """
        Compute a certificate for a graph that's invariant under isomorphism.
        Uses multiple graph invariants for better discrimination.
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()
        
        # Basic invariants
        degree_sequence = tuple(sorted(dict(G.degree()).values()))
        
        # Clustering coefficients (sorted)
        clustering = tuple(sorted(nx.clustering(G).values()))
        
        # Triangle count
        triangles = sum(nx.triangles(G).values()) // 3
        
        # Connected components
        num_components = nx.number_connected_components(G)
        component_sizes = tuple(sorted([
            len(c) for c in nx.connected_components(G)
        ], reverse=True))
        
        # Diameter of largest component (if connected)
        try:
            if nx.is_connected(G):
                diameter = nx.diameter(G)
            else:
                # Diameter of largest component
                largest_cc = max(nx.connected_components(G), key=len)
                diameter = nx.diameter(G.subgraph(largest_cc))
        except:
            diameter = -1  # Empty graph or other issues
        
        return (
            n, m, degree_sequence, clustering, triangles,
            num_components, component_sizes, diameter
        )

    def _is_isomorphic_to_any(self, G: nx.Graph) -> bool:
        """
        Check if G is isomorphic to any previously added graph.
        Uses certificate-based filtering for efficiency.
        """
        cert = self._get_graph_certificate(G)
        
        # Check against all graphs with the same certificate
        for i, existing_cert in enumerate(self._certificates):
            if cert == existing_cert:
                if nx.is_isomorphic(G, self._all_graphs[i]):
                    return True
        
        return False

    def _add_graph_if_unique(self, G: nx.Graph, edge_prob: float) -> bool:
        """
        Add graph to dataset if it's not isomorphic to any existing graph.
        Returns True if added, False if duplicate.
        """
        if self._is_isomorphic_to_any(G):
            return False

        # Graph is unique - add it
        cert = self._get_graph_certificate(G)
        self._certificates.append(cert)
        self._all_graphs.append(G.copy())
        
        vec = self._vectorize(G)
        self.vectors.append(vec)
        self.edge_probs.append(edge_prob)
        if self.store_graphs:
            self.graphs.append(G.copy())
        
        return True

    def generate_dataset(
        self,
        target_total_samples: int,
        max_attempts_per_iso: int = 1000,
        seed: Optional[int] = None,
    ) -> None:
        """
        Build the dataset of bit-vectors (and optional Graphs) with strict 
        isomorphism checking on every generated graph.
        """
        if seed is not None:
            np.random.seed(seed)

        self.vectors.clear()
        self.edge_probs.clear()
        if self.store_graphs:
            self.graphs.clear()
        self._all_graphs.clear()
        self._certificates.clear()

        prob_bins = self._get_prob_bins()
        if not prob_bins:
            if self.verbose:
                print("[Dataset] No probability bins → empty dataset")
            return

        if self.verbose:
            print(
                f"[Dataset] target ~{target_total_samples} samples "
                f"over {len(prob_bins)} probability bins"
            )

        targets = self._compute_target_distribution(prob_bins, target_total_samples)
        if self.verbose:
            for (p_min, p_max), target in targets.items():
                capacity = self._estimate_bin_capacity(p_min, p_max)
                print(f"  Bin [{p_min:.3f}, {p_max:.3f}]: target {target}, capacity {capacity}")

        total_added = 0
        for (p_min, p_max), target_count in targets.items():
            if self.verbose:
                print(f"[Bin [{p_min:.3f}, {p_max:.3f}]] need {target_count} samples")
            
            added_this_bin = 0
            attempt = 0
            
            while added_this_bin < target_count and attempt < target_count * max_attempts_per_iso:
                # Sample edge probability from bin range
                if p_min == p_max:
                    p = p_min
                else:
                    p = np.random.uniform(p_min, p_max)
                
                # Generate Erdős-Rényi graph
                G0 = nx.erdos_renyi_graph(self.nodes, p, seed=None)
                
                # Try to add base graph
                if self._add_graph_if_unique(G0, p):
                    added_this_bin += 1
                    total_added += 1
                    
                    # Generate permutations of this base graph
                    perm_added = 0
                    perm_attempts = 0
                    max_perm_attempts = self.extra_permutations * 50
                    
                    while perm_added < self.extra_permutations and perm_attempts < max_perm_attempts:
                        if added_this_bin >= target_count:
                            break
                            
                        seed_k = (
                            seed + attempt * self.nodes + perm_attempts + 1
                            if seed is not None
                            else None
                        )
                        Gp = self._apply_permutation(G=G0, seed=seed_k)
                        
                        if self._add_graph_if_unique(Gp, p):
                            perm_added += 1
                            added_this_bin += 1
                            total_added += 1
                        
                        perm_attempts += 1
                
                attempt += 1

            if self.verbose:
                print(f"  → added {added_this_bin}/{target_count} in {attempt} base attempts")

        if self.verbose:
            print(
                f"[Dataset] done: {len(self.vectors)} samples, "
                f"{len(self._all_graphs)} total graphs"
            )

    def save_dataset(self, filepath: str, include_metadata: bool = True) -> None:
        """Save the dataset to a pickle file."""
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        
        save_data = {
            'vectors': self.vectors,
            'edge_probs': self.edge_probs,
            'graphs': self.graphs if self.store_graphs else [],
            '_all_graphs': self._all_graphs,
            '_certificates': self._certificates,
            
            # Dataset parameters
            'nodes': self.nodes,
            'p_min': self.p_min,
            'p_max': self.p_max,
            'extra_permutations': self.extra_permutations,
            'store_graphs': self.store_graphs,
            'prob_bins': self.prob_bins,
        }
        
        if include_metadata:
            save_data['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(self.vectors),
                'unique_graphs': len(self._all_graphs),
                'version': '1.0',
                'graph_type': 'erdos_renyi'
            }
        
        try:
            os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
            
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.verbose:
                size_mb = os.path.getsize(filepath) / (1024 * 1024)
                print(f"[Dataset] Saved {len(self.vectors)} samples to {filepath} "
                     f"({size_mb:.2f} MB)")
                
        except Exception as e:
            raise IOError(f"Failed to save dataset to {filepath}: {e}")

    def load_dataset(self, filepath: str, verbose: Optional[bool] = None) -> None:
        """Load a dataset from a pickle file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                save_data = pickle.load(f)
            
            # Load main data
            self.vectors = save_data['vectors']
            self.edge_probs = save_data['edge_probs']
            self.graphs = save_data['graphs']
            self._all_graphs = save_data['_all_graphs']
            self._certificates = save_data['_certificates']
            
            # Load parameters
            self.nodes = save_data['nodes']
            self.p_min = save_data['p_min']
            self.p_max = save_data['p_max']
            self.extra_permutations = save_data['extra_permutations']
            self.store_graphs = save_data['store_graphs']
            self.prob_bins = save_data.get('prob_bins', 10)
            
            if verbose is not None:
                self.verbose = verbose
            
            if self.verbose:
                print(f"[Dataset] Loaded {len(self.vectors)} samples from {filepath}")
                
                if 'metadata' in save_data:
                    meta = save_data['metadata']
                    print(f"  Created: {meta.get('created_at', 'unknown')}")
                    print(f"  Unique graphs: {meta.get('unique_graphs', len(self._all_graphs))}")
                    print(f"  Version: {meta.get('version', 'unknown')}")
                    print(f"  Type: {meta.get('graph_type', 'unknown')}")
            
        except Exception as e:
            raise IOError(f"Failed to load dataset from {filepath}: {e}")

    @classmethod
    def from_file(cls, filepath: str, verbose: bool = True) -> 'ErdosRenyiGraphDataset':
        """Create a new ErdosRenyiGraphDataset instance by loading from a file."""
        instance = cls(
            nodes=1,  # Will be overwritten
            edge_prob=0.5,  # Will be overwritten
            verbose=verbose
        )
        instance.load_dataset(filepath, verbose=verbose)
        return instance

    def get_save_info(self) -> Dict[str, Any]:
        """Get information about what would be saved."""
        info = {
            'total_samples': len(self.vectors),
            'unique_graphs': len(self._all_graphs),
            'store_graphs': self.store_graphs,
            'stored_graphs_count': len(self.graphs),
            'parameters': {
                'nodes': self.nodes,
                'edge_prob_range': (self.p_min, self.p_max),
                'extra_permutations': self.extra_permutations,
                'prob_bins': self.prob_bins,
            }
        }
        
        if self.vectors:
            info['vector_shape'] = self.vectors[0].shape
            info['vector_dtype'] = str(self.vectors[0].dtype)
        
        return info