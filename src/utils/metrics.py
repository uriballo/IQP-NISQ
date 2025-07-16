import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import isomorphism
from typing import List, Dict, Any
from src.utils.utils import is_bipartite


def bipartite_proportion(samples: np.ndarray, nodes: int) -> float:
    """
    Calculates the proportion of graph vectors in a list that are bipartite.

    Args:
        samples (np.ndarray): A list or array of flattened graph vectors.
        nodes (int): The number of nodes in the graphs.

    Returns:
        float: The proportion (0.0 to 1.0) of graphs that are bipartite.
    """
    total = len(samples)
    if total == 0:
        return 0.0
    
    bipartites = 0
    for s in samples:
        if is_bipartite(s, nodes):
            bipartites += 1

    return bipartites / total


def average_edge_prob(samples: np.ndarray, nodes: int) -> float:
    """
    Calculates the average edge probability (density) across a list of graph vectors.

    Args:
        samples (np.ndarray): A list or array of flattened graph vectors.
        nodes (int): The number of nodes in the graphs.

    Returns:
        float: The average edge probability.
    """
    total = len(samples)
    if total == 0:
        return 0.0

    # The number of possible edges in an undirected graph with no self-loops
    possible_edges = nodes * (nodes - 1) / 2
    if possible_edges == 0:
        return 0.0

    # Calculate the density for each graph vector and return the mean
    densities = [np.sum(s) / possible_edges for s in samples]
    return float(np.mean(densities))


def memorized_proportion(
    gt_graphs: List[nx.Graph],
    gen_graphs: List[nx.Graph],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare generated vs ground‐truth graphs.

    Args:
        gt_graphs (List[nx.Graph]): The list of ground-truth graphs.
        gen_graphs (List[nx.Graph]): The list of generated graphs.
        verbose (bool, optional): If True, prints a detailed report. Defaults to True.

    Returns:
        Dict[str, Any]: A dictionary of memorization and coverage statistics.
    """
    n_truth = len(gt_graphs)
    n_gen   = len(gen_graphs)

    # how many generated match at least one GT
    gen_mem_count = sum(
        1 for Gg in gen_graphs
        if any(nx.is_isomorphic(Gg, Gt) for Gt in gt_graphs)
    )
    gen_mem_fraction = gen_mem_count / n_gen if n_gen else 0.0

    # how many GT are covered by at least one generated
    truth_cov_count = sum(
        1 for Gt in gt_graphs
        if any(nx.is_isomorphic(Gt, Gg) for Gg in gen_graphs)
    )
    truth_cov_fraction = truth_cov_count / n_truth if n_truth else 0.0

    stats: Dict[str, Any] = {
        "n_truth":               n_truth,
        "n_gen":                 n_gen,
        "gen_mem_count":         gen_mem_count,
        "gen_mem_fraction":      gen_mem_fraction,
        "truth_cov_count":       truth_cov_count,
        "truth_cov_fraction":    truth_cov_fraction
    }

    if verbose:
        print("=== Memorization Report ===")
        print(f"Ground‐truth graphs:       {n_truth}")
        print(f"Generated graphs:          {n_gen}")
        print(
            f"Generated ⟶ GT matches:    "
            f"{gen_mem_count}/{n_gen} "
            f"({gen_mem_fraction*100:5.2f}%)"
        )
        print(
            f"GT ⟵ Generated coverage:   "
            f"{truth_cov_count}/{n_truth} "
            f"({truth_cov_fraction*100:5.2f}%)"
        )
        print("===========================")

    return stats


def filter_bipartite(graphs: List[nx.Graph]) -> List[nx.Graph]:
    """
    Return only those graphs in `graphs` that are bipartite and non-isomorphic.
    """
    bipartite_graphs = [G for G in graphs if bipartite.is_bipartite(G)]
    
    non_isomorphic_results: List[nx.Graph] = []
    for G1 in bipartite_graphs:
        is_isomorphic_to_existing = False
        for G2 in non_isomorphic_results:
            if isomorphism.is_isomorphic(G1, G2):
                is_isomorphic_to_existing = True
                break
        if not is_isomorphic_to_existing:
            non_isomorphic_results.append(G1)
            
    return non_isomorphic_results


def sample_diversity(graphs: List[nx.Graph]) -> tuple[int, float]:
    """
    Measure diversity of a list of graphs.
    
    Returns:
        tuple[int, float]: A tuple containing the count of unique graphs 
                           and the diversity score (0.0 to 1.0).
    """
    n = len(graphs)
    if n == 0:
        return 0, 0.0
        
    unique_graphs: List[nx.Graph] = []
    for g in graphs:
        if not any(nx.is_isomorphic(g, ug) for ug in unique_graphs):
            unique_graphs.append(g)
    return len(unique_graphs), len(unique_graphs) / n
