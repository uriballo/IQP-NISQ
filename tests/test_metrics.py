import pytest
import networkx as nx
import numpy as np

from src.utils.metrics import (
    memorized_proportion,
    filter_bipartite,
    sample_diversity,
    bipartite_proportion,
    average_edge_prob, 
)
from src.utils.utils import vec_to_graph, graph_to_vec, is_bipartite

# --- Test Fixtures for Randomized Data ---

@pytest.fixture(scope="module")
def random_graph_factory():
    """A factory to generate random graphs for testing."""
    def _generator(n_nodes, n_edges, seed=None):
        return nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    return _generator

# --- Tests for Memorization ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_memorized_proportion_scenarios(random_graph_factory, n_nodes):
    """Tests various memorization scenarios for multiple node counts."""
    n_edges = n_nodes + 2  # A reasonable number of edges for a sparse graph

    # Ground truth graphs
    gt1 = random_graph_factory(n_nodes, n_edges, seed=1)
    gt2 = random_graph_factory(n_nodes, n_edges, seed=2)

    # --- Case 1: All memorized ---
    all_mem_graphs = [
        random_graph_factory(n_nodes, n_edges, seed=1),
        random_graph_factory(n_nodes, n_edges, seed=1),
    ]
    stats_all = memorized_proportion([gt1], all_mem_graphs, verbose=False)
    assert stats_all["gen_mem_fraction"] == 1.0
    assert stats_all["truth_cov_fraction"] == 1.0

    # --- Case 2: None memorized ---
    none_mem_graphs = [
        random_graph_factory(n_nodes, n_edges, seed=3),
        random_graph_factory(n_nodes, n_edges, seed=4),
    ]
    stats_none = memorized_proportion([gt1], none_mem_graphs, verbose=False)
    assert stats_none["gen_mem_fraction"] == 0.0
    assert stats_none["truth_cov_fraction"] == 0.0

    # --- Case 3: Partial memorization ---
    partial_mem_graphs = [
        random_graph_factory(n_nodes, n_edges, seed=1),  # Match gt1
        random_graph_factory(n_nodes, n_edges, seed=1),  # Match gt1
        random_graph_factory(n_nodes, n_edges, seed=3),  # No match
        random_graph_factory(n_nodes, n_edges, seed=4),  # No match
    ]
    stats_partial = memorized_proportion([gt1, gt2], partial_mem_graphs, verbose=False)
    assert stats_partial["gen_mem_fraction"] == pytest.approx(2.0 / 4.0)
    assert stats_partial["truth_cov_fraction"] == pytest.approx(1.0 / 2.0)

# --- Tests for Bipartite Filtering ---

@pytest.mark.parametrize("n_nodes", [8, 12, 20])
def test_filter_bipartite(n_nodes):
    """Tests bipartite filtering for multiple node counts."""
    n1 = n_nodes // 2
    n2 = n_nodes - n1

    bipartite1 = nx.bipartite.random_graph(n1, n2, p=0.5, seed=1)
    bipartite2 = nx.bipartite.random_graph(n1, n2, p=0.5, seed=1)  # Isomorphic
    bipartite3 = nx.bipartite.random_graph(n1, n2, p=0.5, seed=2)  # Different
    non_bipartite = nx.complete_graph(n_nodes)

    graphs = [bipartite1, non_bipartite, bipartite2, bipartite3]
    filtered_graphs = filter_bipartite(graphs)

    assert len(filtered_graphs) == 2
    for g in filtered_graphs:
        assert nx.is_bipartite(g)

# --- Tests for Sample Diversity ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_sample_diversity_scenarios(n_nodes):
    """Tests sample diversity scenarios for multiple node counts."""
    g1 = nx.path_graph(n_nodes)
    g2 = nx.star_graph(n_nodes - 1)
    g2.add_node(n_nodes - 1) # Ensure same number of nodes
    g3 = nx.complete_graph(n_nodes)

    # --- Case 1: No diversity ---
    graphs_none = [g1, g1, g1, g1]
    count, diversity = sample_diversity(graphs_none)
    assert count == 1
    assert diversity == pytest.approx(1.0 / 4.0)

    # --- Case 2: Full diversity ---
    graphs_full = [g1, g2, g3]
    count, diversity = sample_diversity(graphs_full)
    assert count == 3
    assert diversity == pytest.approx(1.0)

    # --- Case 3: Partial diversity ---
    graphs_partial = [g1, g2, g1]
    count, diversity = sample_diversity(graphs_partial)
    assert count == 2
    assert diversity == pytest.approx(2.0 / 3.0)

# --- Tests for Bipartite Proportion ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_bipartite_proportion_scenarios(n_nodes):
    """Tests bipartite proportion calculation for multiple node counts."""
    # Graph structures
    bipartite_g = nx.path_graph(n_nodes)
    non_bipartite_g = nx.complete_graph(n_nodes)
    edgeless_g = nx.empty_graph(n_nodes) # Also bipartite

    # Corresponding vectors
    bipartite_vec = graph_to_vec(bipartite_g, n_nodes)
    non_bipartite_vec = graph_to_vec(non_bipartite_g, n_nodes)
    edgeless_vec = graph_to_vec(edgeless_g, n_nodes)

    # --- Case 1: Mixed samples ---
    # 2 bipartite (path, edgeless) and 1 non-bipartite (complete)
    samples = [bipartite_vec, edgeless_vec, non_bipartite_vec]
    proportion = bipartite_proportion(samples, n_nodes)
    assert proportion == pytest.approx(2.0 / 3.0)

    # --- Case 2: All bipartite ---
    samples_all_b = [bipartite_vec, edgeless_vec]
    proportion_all_b = bipartite_proportion(samples_all_b, n_nodes)
    assert proportion_all_b == 1.0

    # --- Case 3: Empty input ---
    assert bipartite_proportion([], n_nodes) == 0.0

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_bipartite_proportion_invalid_vector(n_nodes):
    """Tests that an invalid vector length raises an error."""
    # A vector with a clearly incorrect size
    invalid_vec = np.ones(n_nodes * n_nodes)
    # The is_bipartite utility should raise the error when it calls vec_to_graph
    with pytest.raises(ValueError):
        bipartite_proportion([invalid_vec], n_nodes)

# --- Test for Average Edge Probability ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_average_edge_prob_scenarios(n_nodes):
    """Tests average edge probability calculation for multiple node counts."""
    # Graph structures
    empty_g = nx.empty_graph(n_nodes)
    complete_g = nx.complete_graph(n_nodes)
    
    # Corresponding vectors
    empty_vec = graph_to_vec(empty_g, n_nodes)
    complete_vec = graph_to_vec(complete_g, n_nodes)
    
    # --- Case 1: Mix of empty and complete ---
    # Expected density is (0.0 + 1.0) / 2 = 0.5
    samples_mixed = [empty_vec, complete_vec]
    avg_prob_mixed = average_edge_prob(samples_mixed, n_nodes)
    assert avg_prob_mixed == pytest.approx(0.5)

    # --- Case 2: Only complete graphs ---
    samples_all_full = [complete_vec, complete_vec]
    avg_prob_full = average_edge_prob(samples_all_full, n_nodes)
    assert avg_prob_full == pytest.approx(1.0)

    # --- Case 3: Empty input ---
    assert average_edge_prob([], n_nodes) == 0.0
