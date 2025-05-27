import pickle
import os
from datetime import datetime
import networkx as nx
from networkx.algorithms.bipartite.generators import random_graph
from datasets.utils import graph_to_vec, vec_to_graph
import numpy as np
from typing import Union, Tuple, List, Optional, Dict, Any

class BipartiteGraphDataset:
    def __init__(
        self,
        nodes: int,
        edge_prob: Union[float, Tuple[float, float]],
        extra_permutations: int = 0,
        store_graphs: bool = True,
        verbose: bool = True,
        small_partition_threshold: int = 50,
    ):
        """
        nodes: total |V| = n1 + n2
        edge_prob: single p or (p_min,p_max)
        extra_permutations: how many extra shuffled copies per base graph.
        store_graphs: if True, keep Graph objects in self.graphs.
        verbose: if True, prints progress.
        small_partition_threshold: partitions with capacity <= this get full allocation.
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
        self.small_partition_threshold = small_partition_threshold

        self.vectors: List[np.ndarray] = []
        self.edge_probs: List[float] = []
        if self.store_graphs:
            self.graphs: List[nx.Graph] = []

        # for iso‐detection 
        self._all_graphs: List[nx.Graph] = []
        self._certificates: List[Tuple] = []  

    def _get_all_partitions(self) -> List[Tuple[int, int]]:
        return [(n1, self.nodes - n1) for n1 in range(1, self.nodes // 2 + 1)]

    def _estimate_partition_capacity(self, n1: int, n2: int) -> int:
        if n1 == 1:
            return n2 + 1
        e = n1 * n2
        return min(e * e, 5000)

    def _compute_target_distribution(
        self, partitions: List[Tuple[int, int]], total_samples: int
    ) -> Dict[Tuple[int, int], int]:
        caps = np.array(
            [
                self._estimate_partition_capacity(n1, n2)
                for n1, n2 in partitions
            ],
            dtype=int,
        )
        
        # Step 1: Identify small partitions and allocate full capacity
        small_mask = caps <= self.small_partition_threshold
        small_allocation = caps * small_mask
        small_total = small_allocation.sum()
        
        # Step 2: Distribute remaining samples among larger partitions
        remaining_samples = max(0, total_samples - small_total)
        large_caps = caps * (~small_mask)
        large_total_cap = large_caps.sum()
        
        if large_total_cap > 0 and remaining_samples > 0:
            # Proportional allocation for large partitions
            large_raw = large_caps * remaining_samples / large_total_cap
            large_allocation = np.round(large_raw).astype(int)
            large_allocation = np.clip(large_allocation, 0, large_caps)
            
            # Handle rounding remainder
            remainder = remaining_samples - large_allocation.sum()
            if remainder > 0:
                # Add to partitions with largest fractional parts
                fracs = large_raw - np.floor(large_raw)
                large_indices = np.where(~small_mask)[0]
                for idx in np.argsort(fracs[large_indices])[::-1][:remainder]:
                    actual_idx = large_indices[idx]
                    if large_allocation[actual_idx] < large_caps[actual_idx]:
                        large_allocation[actual_idx] += 1
            elif remainder < 0:
                # Remove from partitions with largest allocations
                large_indices = np.where(~small_mask)[0]
                for idx in np.argsort(large_allocation[large_indices])[::-1][:-remainder]:
                    actual_idx = large_indices[idx]
                    if large_allocation[actual_idx] > 0:
                        large_allocation[actual_idx] -= 1
        else:
            large_allocation = np.zeros_like(caps)
        
        final_allocation = small_allocation + large_allocation
        final_allocation = np.where((caps > 0) & (final_allocation == 0), 1, final_allocation)
        
        return {
            part: int(alloc) 
            for part, alloc in zip(partitions, final_allocation)
            if alloc > 0
        }

    def _vectorize(self, G: nx.Graph) -> np.ndarray:
        mat = nx.to_numpy_array(G, nodelist=range(self.nodes), dtype=np.uint8)
        iu = np.triu_indices(self.nodes, k=1)
        return mat[iu].astype(np.uint8)

    def _apply_permutation(
        self,
        vec: Optional[np.ndarray] = None,
        G: Optional[nx.Graph] = None,
        seed: Optional[int] = None,
    ) -> Union[np.ndarray, nx.Graph]:
        if seed is not None:
            np.random.seed(seed)

        permutation = np.random.permutation(self.nodes)
        mapping = {i: int(permutation[i]) for i in range(self.nodes)}

        if vec is not None:
            G0 = vec_to_graph(vec, self.nodes)
            Gp = nx.relabel_nodes(G0, mapping, copy=True)
            return self._vectorize(Gp)
        if G is not None:
            return nx.relabel_nodes(G, mapping, copy=True)

        raise ValueError("Must pass `vec` or `G`")

    def _get_bipartite_certificate(self, G: nx.Graph) -> Tuple:
        """
        Compute a certificate for a bipartite graph that's invariant under 
        isomorphism. This uses the bipartite structure automatically detected 
        by NetworkX.
        """
        if not nx.is_bipartite(G):
            # If not bipartite, fall back to degree sequence
            return tuple(sorted(dict(G.degree()).values()))
        
        # Get the two bipartite sets
        try:
            sets = nx.bipartite.sets(G)
            set1, set2 = list(sets)
        except:
            # Fallback if bipartite detection fails
            return tuple(sorted(dict(G.degree()).values()))
        
        # Compute degree sequences for each bipartite set
        deg1 = tuple(sorted(G.degree(node) for node in set1))
        deg2 = tuple(sorted(G.degree(node) for node in set2))
        
        # Create canonical form (smaller set first for consistency)
        if len(set1) < len(set2) or (len(set1) == len(set2) and deg1 <= deg2):
            return (len(set1), len(set2), deg1, deg2)
        else:
            return (len(set2), len(set1), deg2, deg1)

    def _is_bipartite_isomorphic(self, G1: nx.Graph, G2: nx.Graph) -> bool:
        """
        Check if two graphs are isomorphic, considering their bipartite structure.
        """
        # First check basic properties
        if G1.number_of_nodes() != G2.number_of_nodes():
            return False
        if G1.number_of_edges() != G2.number_of_edges():
            return False
        
        # Check if both are bipartite
        is_bip1 = nx.is_bipartite(G1)
        is_bip2 = nx.is_bipartite(G2)
        
        if is_bip1 != is_bip2:
            return False
        
        if is_bip1 and is_bip2:
            # Both are bipartite - use bipartite isomorphism check
            try:
                return nx.is_isomorphic(G1, G2)
            except:
                # Fallback to general isomorphism
                return nx.is_isomorphic(G1, G2)
        else:
            # Neither is bipartite - use general isomorphism
            return nx.is_isomorphic(G1, G2)

    def _is_isomorphic_to_any(self, G: nx.Graph) -> bool:
        """
        Check if G is isomorphic to any previously added graph.
        Uses certificate-based filtering for efficiency.
        """
        cert = self._get_bipartite_certificate(G)
        
        # Check against all graphs with the same certificate
        for i, existing_cert in enumerate(self._certificates):
            if cert == existing_cert:
                if self._is_bipartite_isomorphic(G, self._all_graphs[i]):
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
        cert = self._get_bipartite_certificate(G)
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
        Builds the dataset of bit‐vectors (and optional Graphs) with strict 
        bipartite-aware isomorphism checking on every generated graph.
        """
        if seed is not None:
            np.random.seed(seed)

        self.vectors.clear()
        self.edge_probs.clear()
        if self.store_graphs:
            self.graphs.clear()
        self._all_graphs.clear()
        self._certificates.clear()

        parts = self._get_all_partitions()
        if not parts:
            if self.verbose:
                print("[Dataset] No bipartite splits → empty dataset")
            return

        if self.verbose:
            print(
                f"[Dataset] target ~{target_total_samples} samples "
                f"over {len(parts)} partitions"
            )
        targets = self._compute_target_distribution(parts, target_total_samples)
        if self.verbose:
            for p, t in targets.items():
                cap = self._estimate_partition_capacity(*p)
                is_small = cap <= self.small_partition_threshold
                status = " (FULL)" if is_small and t == cap else ""
                print(f"  Partition {p}: target {t}, capacity {cap}{status}")

        total_added = 0
        for (n1, n2), target_count in targets.items():
            if self.verbose:
                print(f"[Partition {n1},{n2}] need {target_count} samples")
            
            added_this_partition = 0
            attempt = 0
            
            while added_this_partition < target_count and attempt < target_count * max_attempts_per_iso:
                p = (
                    np.random.uniform(self.p_min, self.p_max)
                    if self.p_min != self.p_max
                    else self.p_min
                )
                G0 = random_graph(n1, n2, p, seed=None)
                
                # Try to add base graph
                if self._add_graph_if_unique(G0, p):
                    added_this_partition += 1
                    total_added += 1
                    
                    # Generate permutations of this base graph
                    perm_added = 0
                    perm_attempts = 0
                    max_perm_attempts = self.extra_permutations * 50
                    
                    while perm_added < self.extra_permutations and perm_attempts < max_perm_attempts:
                        if added_this_partition >= target_count:
                            break
                            
                        seed_k = (
                            seed + attempt * self.nodes + perm_attempts + 1
                            if seed is not None
                            else None
                        )
                        Gp = self._apply_permutation(G=G0, seed=seed_k)
                        
                        if self._add_graph_if_unique(Gp, p):
                            perm_added += 1
                            added_this_partition += 1
                            total_added += 1
                        
                        perm_attempts += 1
                
                attempt += 1

            if self.verbose:
                print(f"  → added {added_this_partition}/{target_count} in {attempt} base attempts")

        if self.verbose:
            print(
                f"[Dataset] done: {len(self.vectors)} samples, "
                f"{len(self._all_graphs)} total graphs"
            )

    def save_dataset(self, filepath: str, include_metadata: bool = True) -> None:
        """
        Save the dataset to a pickle file.
        
        Args:
            filepath: Path where to save the .pkl file
            include_metadata: If True, includes generation metadata
        """
        # Ensure .pkl extension
        if not filepath.endswith('.pkl'):
            filepath += '.pkl'
        
        # Prepare data to save
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
            'small_partition_threshold': self.small_partition_threshold,
        }
        
        if include_metadata:
            save_data['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(self.vectors),
                'unique_graphs': len(self._all_graphs),
                'version': '1.0' 
            }
        
        try:
            # Create directory if it doesn't exist
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
        """
        Load a dataset from a pickle file.
        
        Args:
            filepath: Path to the .pkl file to load
            verbose: Override verbose setting (uses current setting if None)
        """
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
            self.small_partition_threshold = save_data.get('small_partition_threshold', 50)
            
            # Override verbose if specified
            if verbose is not None:
                self.verbose = verbose
            
            # Print info
            if self.verbose:
                print(f"[Dataset] Loaded {len(self.vectors)} samples from {filepath}")
                
                if 'metadata' in save_data:
                    meta = save_data['metadata']
                    print(f"  Created: {meta.get('created_at', 'unknown')}")
                    print(f"  Unique graphs: {meta.get('unique_graphs', len(self._all_graphs))}")
                    print(f"  Version: {meta.get('version', 'unknown')}")
            
        except Exception as e:
            raise IOError(f"Failed to load dataset from {filepath}: {e}")

    @classmethod
    def from_file(
        cls, 
        filepath: str, 
        verbose: bool = True
    ) -> 'BipartiteGraphDataset':
        """
        Create a new BipartiteGraphDataset instance by loading from a file.
        
        Args:
            filepath: Path to the .pkl file to load
            verbose: Verbose setting for the new instance
        
        Returns:
            New BipartiteGraphDataset instance with loaded data
        """
        # Create empty instance
        instance = cls(
            nodes=1,  # Will be overwritten
            edge_prob=0.5,  # Will be overwritten
            verbose=verbose
        )
        
        # Load the actual data
        instance.load_dataset(filepath, verbose=verbose)
        
        return instance

    def get_save_info(self) -> Dict[str, Any]:
        """
        Get information about what would be saved (useful for debuging).
        
        Returns:
            Dictionary with information about dataset contents
        """
        info = {
            'total_samples': len(self.vectors),
            'unique_graphs': len(self._all_graphs),
            'store_graphs': self.store_graphs,
            'stored_graphs_count': len(self.graphs),
            'parameters': {
                'nodes': self.nodes,
                'edge_prob_range': (self.p_min, self.p_max),
                'extra_permutations': self.extra_permutations,
                'small_partition_threshold': self.small_partition_threshold,
            }
        }
        
        if self.vectors:
            info['vector_shape'] = self.vectors[0].shape
            info['vector_dtype'] = str(self.vectors[0].dtype)
        
        return info