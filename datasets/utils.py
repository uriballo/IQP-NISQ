import numpy as np
import networkx as nx
from iqpopt.gen_qml.utils import median_heuristic
import matplotlib.pyplot as plt
from typing import Optional, Dict
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np 

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
