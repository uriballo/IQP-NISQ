import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
from typing import List, Dict, Any
from collections import deque
import itertools
import random

def estrada_bipartivity(graph):
    """
    Estrada & Rodríguez-Velázquez spectral bipartivity index.
    https://doi.org/10.1103/PhysRevE.72.046105

    Parameters
    ----------
    graph : nx.Graph
        Undirected graph.
        
    Returns
    -------
    float
        Bipartivity index in [0, 1].
        1 means perfectly bipartite; smaller means less bipartite.
    """
    if not isinstance(graph, nx.Graph):
        raise TypeError("Input must be a NetworkX Graph.")
        
    # Adjacency matrix
    A = nx.to_numpy_array(graph)
    # Eigenvalues (symmetric matrix)
    eigvals = np.linalg.eigvalsh(A)
    
    # Numerator: sum of cosh(λ_j)
    numerator = np.sum(np.cosh(eigvals))
    # Denominator: sum of cosh(λ_j) + sinh(λ_j) = sum(e^λ)
    denominator = np.sum(np.exp(eigvals))
    
    beta = numerator / denominator
    
    return beta

def compute_average_density(samples):
    return np.mean([nx.density(graph) for graph in samples])

def compute_bipartite_percentage(samples):
    return np.mean([nx.is_bipartite(graph) * 100 for graph in samples])

def compute_average_bipartivity(samples):
    return np.mean([estrada_bipartivity(graph) for graph in samples]) 

def compute_memorization(samples, dataset):
    n = len(samples)
    count = sum(
        1 for generated_sample in samples
          if any(nx.is_isomorphic(generated_sample, data_sample) for data_sample in dataset)
    )
    return count / n

def compute_diversity(samples):
    n = len(samples)
    count = sum(
        1 for i, g1 in enumerate(samples)
          if not any(nx.is_isomorphic(g1, g2) for j, g2 in enumerate(samples) if i != j)
    )
    return count / n

