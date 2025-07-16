import pytest
import networkx as nx
import numpy as np
import yaml

from src.utils.utils import (
    graph_to_vec,
    vec_to_adj,
    vec_to_graph,
    is_bipartite,
    create_circuit,
    load_model_from_dir,
    setup_output_directory,
)

@pytest.fixture(scope="module")
def random_graph_generator():
    """A factory to generate random graphs for testing."""
    def _generator(n_nodes, n_edges):
        return nx.gnm_random_graph(n_nodes, n_edges, seed=42)
    return _generator

# --- Core Utility Tests ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_round_trip_conversion(random_graph_generator, n_nodes):
    """
    Tests the full conversion cycle for multiple node counts.
    The final graph should be isomorphic to the original.
    """
    # Test with various edge densities
    max_edges = n_nodes * (n_nodes - 1) // 2
    edge_counts = {0, n_nodes, max_edges // 2, max_edges}
    
    for n_edges in edge_counts:
        # Ensure n_edges does not exceed max_edges
        n_edges = min(n_edges, max_edges)
        original_graph = random_graph_generator(n_nodes, n_edges)

        vec = graph_to_vec(original_graph, n_nodes)
        reconstructed_graph = vec_to_graph(vec, n_nodes)
        
        assert nx.is_isomorphic(original_graph, reconstructed_graph), \
            f"Round-trip conversion failed for n_nodes={n_nodes}, n_edges={n_edges}."

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_graph_with_isolated_nodes(n_nodes):
    """
    Tests a graph with both a connected component and isolated nodes.
    """
    # Create a cycle with roughly half the nodes
    cycle_size = n_nodes // 2
    g = nx.cycle_graph(cycle_size)
    g.add_nodes_from(range(cycle_size, n_nodes))

    vec = graph_to_vec(g, n_nodes)
    reconstructed_g = vec_to_graph(vec, n_nodes)

    assert nx.is_isomorphic(g, reconstructed_g)
    assert reconstructed_g.number_of_nodes() == n_nodes

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_vec_to_adj_reconstruction(random_graph_generator, n_nodes):
    """
    Tests the vector to adjacency matrix conversion for multiple node counts.
    """
    n_edges = n_nodes * (n_nodes - 1) // 4 # Approx 50% density
    original_graph = random_graph_generator(n_nodes, n_edges)
    
    vec = graph_to_vec(original_graph, n_nodes)
    adj = vec_to_adj(vec, n_nodes)
    
    assert adj.shape == (n_nodes, n_nodes)
    assert np.all(adj == adj.T)
    
    iu = np.triu_indices(n_nodes, k=1)
    reconstructed_vec = adj[iu]
    np.testing.assert_array_equal(vec, reconstructed_vec)

# --- Edge Case and Error Handling Tests ---

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_conversions_on_empty_graph(n_nodes):
    """Tests functions on an empty graph for multiple node counts."""
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    
    vec = graph_to_vec(g, n_nodes)
    expected_vec = np.zeros(n_nodes * (n_nodes - 1) // 2, dtype=np.uint8)
    np.testing.assert_array_equal(vec, expected_vec)
    
    reconstructed_g = vec_to_graph(vec, n_nodes)
    assert nx.is_isomorphic(g, reconstructed_g)

@pytest.mark.parametrize("n_nodes", [8, 13, 20])
def test_value_error_on_size_mismatch(n_nodes):
    """
    Ensures ValueError is raised for incorrect sizes, for multiple node counts.
    """
    correct_len = n_nodes * (n_nodes - 1) // 2
    
    short_vec = np.ones(correct_len - 1, dtype=np.uint8)
    with pytest.raises(ValueError):
        vec_to_graph(short_vec, n_nodes)
    with pytest.raises(ValueError):
        vec_to_adj(short_vec, n_nodes)
        
    graph_with_wrong_nodes = nx.path_graph(n_nodes - 1)
    with pytest.raises(ValueError):
        graph_to_vec(graph_with_wrong_nodes, n_nodes)

# --- Bipartite Tests ---

@pytest.mark.parametrize("n_nodes", [8, 10, 20])
def test_is_bipartite_with_random_graphs(n_nodes):
    """
    Tests the `is_bipartite` utility on random graphs for multiple node counts.
    """
    # 1. Test with a guaranteed bipartite graph
    n1 = n_nodes // 2
    n2 = n_nodes - n1
    bipartite_g = nx.bipartite.random_graph(n=n1, m=n2, p=0.7, seed=42)
    bipartite_vec = graph_to_vec(bipartite_g, n_nodes)
    assert is_bipartite(bipartite_vec, n_nodes) is True

    # 2. Test with a graph that is likely not bipartite
    non_bipartite_g = nx.gnm_random_graph(n_nodes, n_nodes * 2, seed=42)
    is_truly_bipartite = nx.is_bipartite(non_bipartite_g)
    non_bipartite_vec = graph_to_vec(non_bipartite_g, n_nodes)
    assert is_bipartite(non_bipartite_vec, n_nodes) is is_truly_bipartite

def test_is_bipartite_on_triangle():
    """An odd-length cycle is the canonical non-bipartite graph."""
    n_nodes_triangle = 3
    triangle_g = nx.cycle_graph(n_nodes_triangle)
    triangle_vec = graph_to_vec(triangle_g, n_nodes_triangle)
    assert is_bipartite(triangle_vec, n_nodes_triangle) is False

@pytest.mark.parametrize("n_nodes", [8, 14, 20])
def test_is_bipartite_on_deterministic_bipartite_graphs(n_nodes):
    """Tests check on known bipartite structures (paths, complete bipartite)."""
    # Path graphs are always bipartite
    path_g = nx.path_graph(n_nodes)
    path_vec = graph_to_vec(path_g, n_nodes)
    assert is_bipartite(path_vec, n_nodes) is True

    # A complete bipartite graph should be identified correctly
    n1 = n_nodes // 2
    n2 = n_nodes - n1
    kb_g = nx.complete_bipartite_graph(n1, n2)
    kb_vec = graph_to_vec(kb_g, n_nodes)
    assert is_bipartite(kb_vec, n_nodes) is True


def test_create_circuit():
    """Tests that the circuit creation function returns a valid object."""
    # This is a simple "smoke test" to ensure the function runs and returns an object
    # with the correct number of qubits.
    num_qubits = 28  # For 8 nodes
    num_layers = 2
    circuit = create_circuit(num_qubits, num_layers)

    # Check that the created object has the expected number of qubits
    assert circuit.n_qubits == num_qubits


def test_load_model_from_dir_success(tmp_path):
    """
    Tests the happy path where both params.npy and hyperparams.yml exist.
    'tmp_path' is a pytest fixture that provides a temporary directory.
    """
    # 1. Arrange: Create a temporary model directory with fake files
    model_dir = tmp_path / "model_A"
    model_dir.mkdir()

    # Create fake metadata
    fake_metadata = {"nodes": 8, "hyperparameters": {"num_layers": 2}}
    with open(model_dir / "hyperparams.yml", "w") as f:
        yaml.dump(fake_metadata, f)

    # Create fake parameters
    fake_params = np.array([0.1, 0.2, 0.3])
    np.save(model_dir / "params.npy", fake_params)

    # 2. Act: Call the function under test
    params, metadata = load_model_from_dir(model_dir)

    # 3. Assert: Check that the loaded data is correct
    assert metadata == fake_metadata
    np.testing.assert_array_equal(params, fake_params)


@pytest.mark.parametrize(
    "missing_file", ["hyperparams.yml", "params.npy"]
)
def test_load_model_from_dir_missing_files(tmp_path, missing_file):
    """Tests that the function handles missing files gracefully."""
    # 1. Arrange: Create a directory missing one of the required files
    model_dir = tmp_path / "model_B"
    model_dir.mkdir()

    if missing_file != "hyperparams.yml":
        with open(model_dir / "hyperparams.yml", "w") as f:
            yaml.dump({"nodes": 8}, f)
    if missing_file != "params.npy":
        np.save(model_dir / "params.npy", np.array([1.0]))

    # 2. Act: Call the function
    params, metadata = load_model_from_dir(model_dir)

    # 3. Assert: Ensure it returns None, None as expected
    assert params is None
    assert metadata is None


def test_setup_output_directory(tmp_path):
    """Tests that the output directory is created with the correct path."""
    # 1. Arrange: Create a fake source directory structure
    model_dir = tmp_path / "trained_params" / "8N_Bipartite_Dense" / "model_A"
    model_dir.mkdir(parents=True)
    result_type = "simulation"

    # 2. Act: Call the function
    output_dir = setup_output_directory(model_dir, result_type)

    # 3. Assert: Check the path and that the directory was created
    expected_path = (
        tmp_path / "simulation_results" / "8N_Bipartite_Dense" / "model_A"
    )
    assert output_dir == expected_path
    assert output_dir.is_dir()