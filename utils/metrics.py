from datasets.utils import is_bipartite
import numpy as np
from typing import List
import networkx as nx

def bipartite_proportion(samples, nodes):
    total = len(samples)
    bipartites = 0
    for s in samples:
        if is_bipartite(s, nodes):
            bipartites +=1

    return bipartites / total


def kl_divergence(
    G1: nx.Graph,
    G2: nx.Graph,
    base: float = np.e,
    eps: float = 1e-10
) -> float:
    """
    Compute D_KL between the degree distributions of G1 and G2.

    D_KL(P||Q) = sum_k P(k) * log(P(k)/Q(k)), with smoothing eps.

    Parameters:
        G1, G2: networkx Graphs
        base: logarithm base (default e)
        eps: minimal probability to avoid log(0)

    Returns:
        KL divergence (float)
    """
    # collect degree sequences
    degs1 = [d for _, d in G1.degree()]
    degs2 = [d for _, d in G2.degree()]

    # if one graph is empty
    if not degs1 or not degs2:
        raise ValueError("Both graphs must have at least one node.")

    # support = 0..max_degree
    max_deg = max(max(degs1), max(degs2))
    bins = np.arange(max_deg + 1)

    # histogram counts
    hist1 = np.bincount(degs1, minlength=bins.size).astype(float)
    hist2 = np.bincount(degs2, minlength=bins.size).astype(float)

    # normalize to get probabilities
    P = hist1 / hist1.sum()
    Q = hist2 / hist2.sum()

    # smoothing
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)

    # KL divergence
    D = np.sum(P * np.log(P / Q))

    # change of base if needed
    return D / np.log(base) if base != np.e else D

def average_kl_divergence(
    graphs_truth: List[nx.Graph],
    graphs_model: List[nx.Graph],
    base: float = np.e,
    eps: float = 1e-10
) -> float:
    """
    Compute the average KL divergence between corresponding graphs in two lists.

    Let N = len(graphs_truth). Define
        D_i = kl_divergence_graphs(graphs_truth[i], graphs_model[i], base, eps)
    Then return (1/N) * sum_i D_i.

    Parameters:
        graphs_truth:    list of “ground truth” nx.Graph
        graphs_model:    list of “model” nx.Graph (must be same length)
        base:            logarithm base for D_KL (default e)
        eps:             smoothing constant to avoid zeros

    Returns:
        Average KL divergence (float)
    """
    if len(graphs_truth) != len(graphs_model):
        raise ValueError("Both graph lists must have the same length.")
    

    divergences = [
        kl_divergence(Gt, Gm, base=base, eps=eps)
        for Gt, Gm in zip(graphs_truth, graphs_model)
    ]
    return float(np.mean(divergences))

def memorized_proportion(gt_graphs: List[nx.Graph],
                   gen_graphs: List[nx.Graph]) -> float:
    """
    Compute the percentage of generated graphs that match the ground truth
    graphs up to isomorphism.
    Raises if the two lists have different lengths.
    """
    if len(gt_graphs) != len(gen_graphs):
        raise ValueError("gt_graphs and gen_graphs must have the same length")
    matches = 0
    for g_true, g_gen in zip(gt_graphs, gen_graphs):
        if nx.is_isomorphic(g_true, g_gen):
            matches += 1
    return matches / len(gt_graphs)


def sample_diversity(graphs: List[nx.Graph],
                     method: str = "unique") -> float:
    """
    Measure diversity of a list of graphs.
    method:
      - "unique": fraction of unique graphs up to isomorphism (%)
      - "jaccard": average pairwise Jaccard edge–set distance (%)
    """
    n = len(graphs)
    if n == 0:
        return 0.0

    if method == "unique":
        unique_graphs: List[nx.Graph] = []
        for g in graphs:
            # keep only one representative per isomorphism class
            if not any(nx.is_isomorphic(g, ug) for ug in unique_graphs):
                unique_graphs.append(g)
        return len(unique_graphs) / n 

    elif method == "jaccard":
        if n < 2:
            return 0.0
        total_dist = 0.0
        count = 0
        for i in range(n):
            edges_i = set(graphs[i].edges())
            for j in range(i + 1, n):
                edges_j = set(graphs[j].edges())
                inter = edges_i & edges_j
                union = edges_i | edges_j
                # if both graphs are empty, define dist=0
                d = 1.0 - (len(inter) / len(union)) if union else 0.0
                total_dist += d
                count += 1
        # return average as a percentage
        return (total_dist / count) 

    else:
        raise ValueError(f"Unknown method '{method}'. "
                         "Use 'unique' or 'jaccard'.")