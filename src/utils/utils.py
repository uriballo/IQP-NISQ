import numpy as np
import networkx as nx
from iqpopt.gen_qml.utils import median_heuristic
import matplotlib.pyplot as plt
import networkx as nx
import csv
import re
from typing import Dict, List, Tuple, Set, Optional
import logging
from pathlib import Path
import iqpopt as iqp
import yaml

BIPARTITE_COLOR_A = "#377eb8"  # Blue
BIPARTITE_COLOR_B = "#ff7f00"  # Orange
NON_BIPARTITE_COLOR = "#999999"  # Grey
EDGE_COLOR = "#666666"  # Darker Grey 

def plot_graph(
    graph: nx.Graph, ax: plt.Axes = None, title: str = None, layout_seed: int = 42
):
    """
    Plots a graph using a general spring layout.
    If the graph is bipartite, nodes are colored based on their partition
    but not explicitly separated by layout. Otherwise, nodes are plotted
    in a dull color.

    Args:
        graph (nx.Graph): The graph to plot.
        ax (plt.Axes, optional): Matplotlib axes object to plot on.
                                 If None, a new figure and axes are created.
        title (str, optional): Optional title for the plot. If None,
                               a default title ("Bipartite Graph",
                               "Non-Bipartite Graph", etc.) will be used.
        layout_seed (int, optional): Seed for the layout algorithm for
                                     reproducibility.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        created_fig = True
    else:
        created_fig = False

    plt.style.use("seaborn-v0_8-whitegrid")
    ax.grid(False)  
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values(): 
        spine.set_visible(False)

    # Handle empty graph case
    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Empty Graph", ha="center", va="center", fontsize=12)
        plot_title = title if title is not None else "Empty Graph"
        ax.set_title(plot_title, fontsize=14, fontweight="bold")
        if created_fig:
            plt.tight_layout()
            plt.show()
        return

    if graph.number_of_nodes() > 1:
         k_val = 0.8 / np.sqrt(graph.number_of_nodes()) 

    try:
        pos = nx.spring_layout(graph, seed=layout_seed, k=k_val, iterations=50)
    except nx.NetworkXError:
        pos = nx.random_layout(graph, seed=layout_seed)


    node_colors_list = []
    is_bipartite_graph = nx.is_bipartite(graph)
    default_plot_title = ""

    if is_bipartite_graph:
        bipartite_node_coloring = nx.bipartite.color(graph)
        node_colors_list = [
            BIPARTITE_COLOR_A
            if bipartite_node_coloring[node] == 0
            else BIPARTITE_COLOR_B
            for node in graph.nodes()
        ]
        if graph.number_of_edges() == 0:
            default_plot_title = "Bipartite Graph (Edgeless)"
        else:
            default_plot_title = "Bipartite Graph"
    else:  
        node_colors_list = [NON_BIPARTITE_COLOR] * graph.number_of_nodes()
        default_plot_title = "Non-Bipartite Graph"

    plot_title_to_use = title if title is not None else default_plot_title
    ax.set_title(plot_title_to_use, fontsize=14, fontweight="bold")

    # Draw nodes
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_color=node_colors_list,
        node_size=350, 
        alpha=0.95,
    )
    # Draw edges
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        edge_color=EDGE_COLOR,
        alpha=0.7,
        width=1.5,
    )

    # Draw labels (conditional on graph size to avoid clutter)
    if 0 < graph.number_of_nodes() <= 20:
        nx.draw_networkx_labels(
            graph, pos, ax=ax, font_size=10, font_weight="normal"
        )
    elif 20 < graph.number_of_nodes() <= 50:
        nx.draw_networkx_labels(
            graph, pos, ax=ax, font_size=8, font_weight="normal"
        )
    # For > 50 nodes, labels are often too cluttered.

    if created_fig:
        plt.tight_layout()
        plt.show()


def graph_to_vec(graph: nx.Graph, nodes: int) -> np.ndarray:
    """
    Turn G (with nodes 0..nodes-1) into a flat upper‐triangle bit‐vector.
    """
    # 1) enforce node‐set
    if set(graph.nodes()) != set(range(nodes)):
        raise ValueError(
            f"Graph nodes must be exactly 0..{nodes-1}"
        )
    # 2) get adjacency (no weights), uint8
    adj = nx.to_numpy_array(
        graph,
        nodelist=range(nodes),
        dtype=np.uint8,
        weight=None
    )
    # 3) flatten strict upper‐triangle
    iu = np.triu_indices(nodes, k=1)
    return adj[iu]

def vec_to_adj(vec: np.ndarray, nodes: int) -> np.ndarray:
    """
    Reconstruct the symmetric adjacency matrix from a strict-upper vec.
    """
    L = nodes * (nodes - 1) // 2
    if vec.size != L:
        raise ValueError(f"vec has length {vec.size}, expected {L}")
    adj = np.zeros((nodes, nodes), dtype=np.uint8)
    iu = np.triu_indices(nodes, k=1)
    adj[iu] = vec
    adj[(iu[1], iu[0])] = vec
    return adj

def vec_to_graph(vec: np.ndarray, nodes: int) -> nx.Graph:
    """
    Build an unweighted, undirected Graph with nodes 0..nodes-1
    from its strict-upper-triangle bit‐vector.
    """
    L = nodes * (nodes - 1) // 2
    if vec.size != L:
        raise ValueError(f"vec has length {vec.size}, expected {L}")
    G = nx.Graph()
    G.add_nodes_from(range(nodes))
    iu = np.triu_indices(nodes, k=1)

    edges = [
        (int(i), int(j))
        for (i, j), bit in zip(zip(iu[0], iu[1]), vec)
        if bit
    ]

    G.add_edges_from(edges)
    return G


def is_bipartite(vector: np.ndarray, nodes: int) -> bool:
    """
    Checks if a graph from a vector is bipartite.
    """
    g = vec_to_graph(vector, nodes)
    return nx.is_bipartite(g)

def aachen_connectivity():
    """
    Creates a manually verified connectivity graph for the QPU.
    """
    # Initialize empty connectivity dictionary
    connectivity = {}
    
    # Initialize all qubits with empty neighbor lists
    for i in range(156):
        connectivity[i] = []
    
    # Manually add all connections
    
    # Row 0 horizontal connections (0-15)
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7),
        (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15)
    ]
    
    # Row 1 horizontal connections (20-35)
    connections.extend([
        (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27),
        (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35)
    ])
    
    # Row 2 horizontal connections (40-55)
    connections.extend([
        (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47),
        (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55)
    ])
    
    # Row 3 horizontal connections (60-75)
    connections.extend([
        (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67),
        (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75)
    ])
    
    # Row 4 horizontal connections (80-95)
    connections.extend([
        (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87),
        (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95)
    ])
    
    # Row 5 horizontal connections (100-115)
    connections.extend([
        (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107),
        (107, 108), (108, 109), (109, 110), (110, 111), (111, 112), (112, 113), (113, 114), (114, 115)
    ])
    
    # Row 6 horizontal connections (120-135)
    connections.extend([
        (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126), (126, 127),
        (127, 128), (128, 129), (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135)
    ])
    
    # Row 7 horizontal connections (140-155)
    connections.extend([
        (140, 141), (141, 142), (142, 143), (143, 144), (144, 145), (145, 146), (146, 147),
        (147, 148), (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155)
    ])
    
    # Vertical connections between rows
    connections.extend([
        # From row 0 to intermediate nodes
        (3, 16), (7, 17), (11, 18), (15, 19),
        # From intermediate nodes to row 1
        (16, 23), (17, 27), (18, 31), (19, 35),
        
        # From row 1 to intermediate nodes
        (21, 36), (25, 37), (29, 38), (33, 39),
        # From intermediate nodes to row 2
        (36, 41), (37, 45), (38, 49), (39, 53),
        
        # From row 2 to intermediate nodes
        (43, 56), (47, 57), (51, 58), (55, 59),
        # From intermediate nodes to row 3
        (56, 63), (57, 67), (58, 71), (59, 75),
        
        # From row 3 to intermediate nodes
        (61, 76), (65, 77), (69, 78), (73, 79),
        # From intermediate nodes to row 4
        (76, 81), (77, 85), (78, 89), (79, 93),
        
        # From row 4 to intermediate nodes
        (83, 96), (87, 97), (91, 98), (95, 99),
        # From intermediate nodes to row 5
        (96, 103), (97, 107), (98, 111), (99, 115),
        
        # From row 5 to intermediate nodes
        (101, 116), (105, 117), (109, 118), (113, 119),
        # From intermediate nodes to row 6
        (116, 121), (117, 125), (118, 129), (119, 133),
        
        # From row 6 to intermediate nodes
        (123, 136), (127, 137), (131, 138), (135, 139),
        # From intermediate nodes to row 7
        (136, 143), (137, 147), (138, 151), (139, 155)
    ])
    
    # Add all connections to the connectivity graph (bidirectional)
    for qubit1, qubit2 in connections:
        connectivity[qubit1].append(qubit2)
        connectivity[qubit2].append(qubit1)
    
    return connectivity

def efficient_connectivity_gates(
    connectivity_graph, num_positions, num_layers=1
):
    """
    Generates a gate list efficiently based on connectivity.
    Single qubit gates are generated for all positions from 0 to num_positions-1.
    Two qubit gates are generated only for connected pairs in connectivity_graph.
    Output format uses np.int64 and sorted pairs for two-qubit gates.

    :param connectivity_graph: Dictionary representing the graph connectivity.
    :param num_positions: The total number of qubit positions available.
    :param num_layers: Number of times to repeat the connectivity pattern.
    :return (list[list[list[np.int64]]]): gate list object.
    """
    final_gates_list = []

    # 1. Create single-qubit gates
    single_qubit_gates = [
        [[np.int64(i)]] for i in range(num_positions)
    ]

    # 2. Create two-qubit gates for directly connected qubits
    two_qubit_gates = []
    edges_added = set()  # To avoid adding the same edge twice (e.g., (0,1) vs (1,0))

    if connectivity_graph:
        for q1_key, neighbors in connectivity_graph.items():
            q1 = int(q1_key)
            # Ensure q1 is within the specified number of positions
            if not (0 <= q1 < num_positions):
                continue

            for q2_val in neighbors:
                q2 = int(q2_val)
                # Ensure q2 is within the specified number of positions
                if not (0 <= q2 < num_positions):
                    continue

                # Create a canonical representation of the edge (sorted tuple of ints)
                # This ensures [[0,1]] is treated the same as [[1,0]] for uniqueness
                # and that the gate itself will store the sorted pair.
                edge_tuple = tuple(sorted((q1, q2)))

                if edge_tuple not in edges_added:
                    two_qubit_gates.append(
                        [[np.int64(edge_tuple[0]), np.int64(edge_tuple[1])]]
                    )
                    edges_added.add(edge_tuple)
    
    # Sort the list of two-qubit gates for a deterministic output order.
    # This makes the output comparable to one generated by iterating combinations.
    two_qubit_gates.sort(key=lambda gate: (gate[0][0], gate[0][1]))

    # Combine single and two-qubit gates for one layer
    layer_gates = single_qubit_gates + two_qubit_gates

    # Repeat the pattern for the specified number of layers
    for _ in range(num_layers):
        final_gates_list.extend(layer_gates)

    return final_gates_list

def setup_logging():
    """Configures a basic logger for scripts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_circuit(num_qubits: int, num_layers: int) -> iqp.IqpSimulator:
    """Creates an IQP circuit simulator with a fixed connectivity."""
    grid_conn = aachen_connectivity()
    gates = efficient_connectivity_gates(grid_conn, num_qubits, num_layers)
    return iqp.IqpSimulator(num_qubits, gates)


def load_model_from_dir(model_dir: Path) -> tuple[np.ndarray | None, dict | None]:
    """
    Loads the parameters and hyperparameters for a given model directory.

    Returns:
        A tuple containing (parameters, metadata) or (None, None) if files are not found.
    """
    config_file = model_dir / 'hyperparams.yml'
    params_file = model_dir / "params.npy"

    if not config_file.is_file():
        logging.warning(f"No hyperparams file found in {model_dir}. Skipping.")
        return None, None
    if not params_file.exists():
        logging.warning(f"No params.npy file found in {model_dir}. Skipping.")
        return None, None

    with open(config_file, 'r') as f:
        metadata = yaml.safe_load(f)
    params = np.load(params_file)

    return params, metadata


def setup_output_directory(model_dir: Path, result_type: str) -> Path:
    """
    Creates a corresponding results directory (e.g., 'simulation_results').

    Args:
        model_dir: The source directory of the trained model.
        result_type: The name of the results parent folder (e.g., "simulation", "evaluation").

    Returns:
        The path to the created output directory.
    """
    try:
        path_parts = list(model_dir.resolve().parts)
        idx = path_parts.index('trained_params')
        path_parts[idx] = f'{result_type}_results'
        output_save_dir = Path(*path_parts)
    except ValueError:
        logging.warning(f"Could not find 'trained_params' in the path. Creating results in a new subdir.")
        output_save_dir = model_dir.parent / f"{result_type}_results" / model_dir.name

    output_save_dir.mkdir(parents=True, exist_ok=True)
    return output_save_dir