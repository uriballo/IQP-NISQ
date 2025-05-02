import numpy as np
import networkx as nx
from iqpopt.gen_qml.utils import median_heuristic
import matplotlib.pyplot as plt
from typing import Optional, Dict

def graph_to_vec(graph: nx.Graph, num_vertices: int) -> np.ndarray:
    """
    Flatten the strict upper‐triangle of graph’s adjacency matrix
    (of size num_vertices×num_vertices) into a 1D vector of length
    num_vertices*(num_vertices-1)//2.
    """
    # ensure the graph has exactly num_vertices nodes
    if graph.number_of_nodes() != num_vertices:
        graph = graph.copy()
        graph.add_nodes_from(range(graph.number_of_nodes(), num_vertices))

    adj = nx.to_numpy_array(graph, nodelist=range(num_vertices))
    iu = np.triu_indices(num_vertices, k=1)
    return adj[iu].astype(np.float32)

def vec_to_adj(vec: np.ndarray, num_vertices: int) -> np.ndarray:
    """
    Reconstruct the full adjacency matrix (num_vertices×num_vertices)
    from its strict upper‐triangle flattening.
    """
    exp_len = num_vertices * (num_vertices - 1) // 2
    if vec.size != exp_len:
        raise ValueError(
            f"Length mismatch: got {vec.size}, expected {exp_len}"
        )

    adj = np.zeros((num_vertices, num_vertices), dtype=np.float32)
    iu = np.triu_indices(num_vertices, k=1)
    adj[iu] = vec
    adj[(iu[1], iu[0])] = vec  # mirror back to lower‐triangle
    return adj

def vec_to_graph(vector: np.ndarray, num_vertices: int) -> nx.Graph:
    exp_len = num_vertices * (num_vertices - 1) // 2
    if vector.size != exp_len:
        raise ValueError(
            f"Vector length {vector.size} != expected {exp_len}"
        )
    mat = np.zeros((num_vertices, num_vertices), int)
    iu = np.triu_indices(num_vertices, k=1)
    mat[iu] = vector
    mat = mat + mat.T
    return nx.from_numpy_array(mat)


def is_bipartite(vector: np.ndarray, num_vertices: int) -> bool:
    try:
        return nx.is_bipartite(vec_to_graph(vector, num_vertices))
    except Exception:
        return False


def compute_sigma_flat(flat_data: np.ndarray) -> float:
    """
    Compute median-heuristic sigma from flattened data.
    Returns 1.0 if data is empty.
    """
    if flat_data is None or flat_data.size == 0:
        return 1.0
    return float(median_heuristic(flat_data.astype(float)))
