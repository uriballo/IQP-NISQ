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
        store_generation_params: bool = False,
        filter_by_actual_density: bool = True,
    ):
        """
        Dataset generator for Erdős-Rényi graphs with isomorphism checking.
        
        Args:
            nodes: Number of nodes |V| = n
            edge_prob: Single p or (p_min, p_max) for edge probability range
            extra_permutations: How many extra shuffled copies per base graph
            store_graphs: If True, keep Graph objects in self.graphs
            verbose: If True, prints progress
            store_generation_params: If True, also store the generation parameters
            filter_by_actual_density: If True, filter graphs by actual edge density
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
        self.store_generation_params = store_generation_params
        self.filter_by_actual_density = filter_by_actual_density

        self.vectors: List[np.ndarray] = []
        self.edge_probs: List[float] = []  # Actual edge probabilities
        if self.store_generation_params:
            self.generation_params: List[float] = []  # Parameters used for generation
        if self.store_graphs:
            self.graphs: List[nx.Graph] = []

        # For isomorphism detection
        self._all_graphs: List[nx.Graph] = []
        self._certificates: List[Tuple] = []

    def _compute_actual_edge_prob(self, G: nx.Graph) -> float:
        """Compute the actual edge probability/density of a graph."""
        n = G.number_of_nodes()
        m = G.number_of_edges()
        max_edges = n * (n - 1) // 2
        return m / max_edges if max_edges > 0 else 0.0

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

    def _add_graph_if_unique(self, G: nx.Graph, generation_param: float) -> bool:
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
        actual_edge_prob = self._compute_actual_edge_prob(G)
        
        self.vectors.append(vec)
        self.edge_probs.append(actual_edge_prob)  # Store ACTUAL edge probability
        if self.store_generation_params:
            self.generation_params.append(generation_param)  # Store generation parameter
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
        
        Args:
            target_total_samples: Number of unique graphs to generate
            max_attempts_per_iso: Maximum attempts per isomorphism check (unused but kept for compatibility)
            seed: Random seed
        """
        if seed is not None:
            np.random.seed(seed)

        self.vectors.clear()
        self.edge_probs.clear()
        if self.store_generation_params:
            self.generation_params.clear()
        if self.store_graphs:
            self.graphs.clear()
        self._all_graphs.clear()
        self._certificates.clear()

        if self.verbose:
            if self.p_min == self.p_max:
                print(f"[Dataset] Generating {target_total_samples} samples with p={self.p_min}")
            else:
                density_str = "density" if self.filter_by_actual_density else "generation param"
                print(f"[Dataset] Generating {target_total_samples} samples with {density_str}∈[{self.p_min:.3f}, {self.p_max:.3f}]")

        added_count = 0
        attempt = 0
        rejected_density = 0
        
        while added_count < target_total_samples:
            # Sample edge probability uniformly from range
            if self.p_min == self.p_max:
                p = self.p_min
            else:
                # If filtering by actual density, sample from a wider range
                # to increase chances of hitting the target density range
                if self.filter_by_actual_density:
                    # Use a wider generation range to account for variance
                    p_range_expansion = 0.3  # Expand by 30% on each side
                    expanded_min = max(0.0, self.p_min - p_range_expansion)
                    expanded_max = min(1.0, self.p_max + p_range_expansion)
                    p = np.random.uniform(expanded_min, expanded_max)
                else:
                    p = np.random.uniform(self.p_min, self.p_max)
            
            # Generate Erdős-Rényi graph
            G0 = nx.erdos_renyi_graph(self.nodes, p, seed=None)
            
            # Check if actual edge density is within desired range
            if self.filter_by_actual_density:
                actual_density = self._compute_actual_edge_prob(G0)
                if not (self.p_min <= actual_density <= self.p_max):
                    rejected_density += 1
                    attempt += 1
                    continue
            
            # Try to add base graph
            if self._add_graph_if_unique(G0, p):
                added_count += 1
                
                # Generate permutations of this base graph
                perm_added = 0
                perm_attempts = 0
                max_perm_attempts = self.extra_permutations * 50
                
                while perm_added < self.extra_permutations and perm_attempts < max_perm_attempts:
                    if added_count >= target_total_samples:
                        break
                        
                    seed_k = (
                        seed + attempt * self.nodes + perm_attempts + 1
                        if seed is not None
                        else None
                    )
                    Gp = self._apply_permutation(G=G0, seed=seed_k)
                    
                    if self._add_graph_if_unique(Gp, p):
                        perm_added += 1
                        added_count += 1
                    
                    perm_attempts += 1
            
            attempt += 1
            
            # Optional: Print progress for long-running generation
            if self.verbose and attempt % 10000 == 0:
                if self.filter_by_actual_density:
                    print(f"  Attempt {attempt}: {added_count}/{target_total_samples} unique graphs found, {rejected_density} rejected for density")
                else:
                    print(f"  Attempt {attempt}: {added_count}/{target_total_samples} unique graphs found")

        if self.verbose:
            print(f"[Dataset] Done: {added_count} samples in {attempt} attempts")
            if self.filter_by_actual_density and rejected_density > 0:
                print(f"  Rejected {rejected_density} graphs for density outside range")
            if self.edge_probs:
                print(f"  Edge probability range: [{min(self.edge_probs):.3f}, {max(self.edge_probs):.3f}]")
                print(f"  Mean edge probability: {np.mean(self.edge_probs):.3f}")

    def verify_edge_probabilities(self) -> Dict[str, Any]:
        """
        Verify that stored edge probabilities match the actual graph edge densities.
        Returns statistics about the verification.
        """
        if not self.vectors:
            return {"error": "No graphs in dataset"}
        
        # Recompute edge probabilities from vectors
        recomputed_probs = []
        for vec in self.vectors:
            G = self._vec_to_graph(vec)
            actual_prob = self._compute_actual_edge_prob(G)
            recomputed_probs.append(actual_prob)
        
        recomputed_probs = np.array(recomputed_probs)
        stored_probs = np.array(self.edge_probs)
        
        # Check if they match
        matches = np.allclose(recomputed_probs, stored_probs, rtol=1e-10)
        max_diff = np.max(np.abs(recomputed_probs - stored_probs))
        
        result = {
            "all_match": matches,
            "max_difference": float(max_diff),
            "mean_stored": float(np.mean(stored_probs)),
            "mean_recomputed": float(np.mean(recomputed_probs)),
            "std_stored": float(np.std(stored_probs)),
            "std_recomputed": float(np.std(recomputed_probs)),
            "range_stored": [float(np.min(stored_probs)), float(np.max(stored_probs))],
        }
        
        if self.store_generation_params:
            gen_params = np.array(self.generation_params)
            result.update({
                "mean_generation_param": float(np.mean(gen_params)),
                "std_generation_param": float(np.std(gen_params)),
                "range_generation_param": [float(np.min(gen_params)), float(np.max(gen_params))],
                "param_vs_actual_diff": float(np.mean(np.abs(gen_params - stored_probs))),
            })
        
        return result

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
            'store_generation_params': self.store_generation_params,
            'filter_by_actual_density': self.filter_by_actual_density,
        }
        
        if self.store_generation_params:
            save_data['generation_params'] = self.generation_params
        
        if include_metadata:
            save_data['metadata'] = {
                'created_at': datetime.now().isoformat(),
                'total_samples': len(self.vectors),
                'unique_graphs': len(self._all_graphs),
                'version': '1.3',
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
            self.store_generation_params = save_data.get('store_generation_params', False)
            self.filter_by_actual_density = save_data.get('filter_by_actual_density', True)
            
            # Load generation params if available
            if self.store_generation_params and 'generation_params' in save_data:
                self.generation_params = save_data['generation_params']
            elif self.store_generation_params:
                self.generation_params = []
            
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
                'store_generation_params': self.store_generation_params,
                'filter_by_actual_density': self.filter_by_actual_density,
            }
        }
        
        if self.vectors:
            info['vector_shape'] = self.vectors[0].shape
            info['vector_dtype'] = str(self.vectors[0].dtype)
        
        if self.edge_probs:
            info['edge_prob_stats'] = {
                'min': float(min(self.edge_probs)),
                'max': float(max(self.edge_probs)),
                'mean': float(np.mean(self.edge_probs)),
                'std': float(np.std(self.edge_probs)),
            }
        
        return info