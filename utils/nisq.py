import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import re
from typing import Dict, List, Tuple, Set

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

def load_calibration_from_csv(filename: str) -> Dict[int, Dict]:
    """
    Loads calibration data from a CSV file.
    """
    qubit_data = {}
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)  # Read the header row
        
        # Remove quotes from headers if they exist
        headers = [h.strip('"') for h in headers]
        
        for row in reader:
            # Remove quotes from values if they exist
            values = [val.strip('"') for val in row]
            
            if not values or not values[0].isdigit():
                continue
                
            qubit_idx = int(values[0])
            qubit_info = {}
            
            for j, header in enumerate(headers[1:], 1):
                try:
                    if j >= len(values):
                        continue
                    if values[j] == "true":
                        qubit_info[header] = True
                    elif values[j] == "false":
                        qubit_info[header] = False
                    elif values[j] == "":
                        qubit_info[header] = None
                    else:
                        # Try to convert to float, but keep as string if it fails
                        try:
                            qubit_info[header] = float(values[j])
                        except ValueError:
                            qubit_info[header] = values[j]
                except (ValueError, IndexError):
                    qubit_info[header] = values[j]
            
            qubit_data[qubit_idx] = qubit_info
    
    return qubit_data

def parse_rzz_errors(qubit_data: Dict[int, Dict]) -> Dict[Tuple[int, int], float]:
    """
    Parse the RZZ errors from qubit data and return a dictionary mapping
    qubit pairs to their RZZ error values.
    """
    rzz_errors = {}
    
    # First, check what column name is actually being used in the data
    # Try both possible column names
    possible_column_names = ["RZZ error", "RZZ error "]
    
    # Check the first qubit to determine which column name is used
    column_name = None
    for name in possible_column_names:
        if any(name in qubit.keys() for qubit in qubit_data.values()):
            column_name = name
            break
    
    if not column_name:
        print("Warning: RZZ error column not found in calibration data")
        return rzz_errors
    
    # Now parse the RZZ errors using the correct column name
    for qubit_idx, data in qubit_data.items():
        rzz_error_str = data.get(column_name)
        
        if not rzz_error_str or not isinstance(rzz_error_str, str):
            continue
            
        # Parse entries of the form "q1_q2:error_value"
        parts = rzz_error_str.split(';')
        for part in parts:
            if not part:
                continue
                
            try:
                # Split by colon to get qubits and error value
                if ':' in part:
                    qubits_str, error_str = part.split(':')
                    
                    # Check if underscore is in the qubit string
                    if '_' in qubits_str:
                        q1, q2 = map(int, qubits_str.split('_'))
                        error = float(error_str)
                        
                        # Store error for this pair (in both directions)
                        rzz_errors[(q1, q2)] = error
                        rzz_errors[(q2, q1)] = error
                    else:
                        print(f"Warning: Malformed qubit pair in RZZ error: {qubits_str}")
                else:
                    print(f"Warning: Malformed RZZ error entry: {part}")
            except (ValueError, IndexError) as e:
                # Print the error for debugging
                print(f"Error parsing RZZ error '{part}': {e}")
                continue
    
    return rzz_errors


def calculate_qubit_scores_with_rzz_priority(qubit_data: Dict[int, Dict], 
                                            connectivity: Dict[int, List[int]]) -> Dict[int, float]:
    """
    Calculates a quality score for each qubit based on calibration metrics,
    with priority given to RZZ gate errors.
    Higher score means better qubit.
    """
    # Extract key metrics for normalization
    t1_values = [q.get('T1 (us)') for q in qubit_data.values() if q.get('T1 (us)') is not None and isinstance(q.get('T1 (us)'), (int, float))]
    t2_values = [q.get('T2 (us)') for q in qubit_data.values() if q.get('T2 (us)') is not None and isinstance(q.get('T2 (us)'), (int, float))]
    readout_errors = [q.get('Readout assignment error ') for q in qubit_data.values() if q.get('Readout assignment error ') is not None and isinstance(q.get('Readout assignment error '), (int, float))]
    
    # Calculate min/max for normalization
    max_t1 = max(t1_values) if t1_values else 1
    max_t2 = max(t2_values) if t2_values else 1
    max_readout_error = max(readout_errors) if readout_errors else 1
    
    # Parse RZZ errors
    rzz_errors = parse_rzz_errors(qubit_data)
    
    # If we got no RZZ errors, print a warning
    if not rzz_errors:
        print("Warning: No RZZ errors found in calibration data. Using fallback scoring method.")
    else:
        print(f"Successfully parsed {len(rzz_errors)//2} RZZ error entries")
    
    # Calculate average RZZ error for each qubit
    avg_rzz_errors = {}
    for qubit_idx in qubit_data.keys():
        connected_errors = []
        for neighbor in connectivity.get(qubit_idx, []):
            error = rzz_errors.get((qubit_idx, neighbor))
            if error is not None:
                connected_errors.append(error)
        
        if connected_errors:
            avg_rzz_errors[qubit_idx] = np.mean(connected_errors)
        else:
            # If no RZZ error data, assign a higher than average error (but not max)
            avg_rzz_errors[qubit_idx] = 0.5  # Moderate penalty instead of 1.0
    
    # Normalize RZZ errors
    all_rzz_errors = list(avg_rzz_errors.values())
    if all_rzz_errors:
        min_rzz_error = min(all_rzz_errors)
        max_rzz_error = max(all_rzz_errors)
        # Avoid division by zero if all errors are the same
        rzz_error_range = max_rzz_error - min_rzz_error
        if rzz_error_range > 0:
            # Scale between 0 and 1, where 0 is worst and 1 is best
            normalized_rzz_errors = {q: 1 - ((e - min_rzz_error) / rzz_error_range) 
                                   for q, e in avg_rzz_errors.items()}
        else:
            # If all errors are equal, give them all a moderate score
            normalized_rzz_errors = {q: 0.5 for q in avg_rzz_errors}
    else:
        normalized_rzz_errors = {}
    
    scores = {}
    for qubit_idx, data in qubit_data.items():
        # Skip non-operational qubits
        if data.get('Operational') != True:
            scores[qubit_idx] = 0
            continue
        
        # Skip qubits with missing important data
        if (data.get('T1 (us)') is None or data.get('T2 (us)') is None or 
            data.get('Readout assignment error ') is None):
            scores[qubit_idx] = 0
            continue
        
        # Compute normalized scores (higher is better)
        t1_score = data.get('T1 (us)', 0) / max_t1
        t2_score = data.get('T2 (us)', 0) / max_t2
        readout_score = 1 - (data.get('Readout assignment error ', 1) / max_readout_error)
        
        # Get RZZ error score (higher is better)
        rzz_score = normalized_rzz_errors.get(qubit_idx, 0.2)  # Default to low score if missing
        
        # Get single-qubit gate error scores (less important now)
        gate_errors = [
            data.get('ID error ', 0),
            data.get('RX error ', 0),
            data.get('Z-axis rotation (rz) error ', 0),
            data.get('√x (sx) error ', 0),
            data.get('Pauli-X error ', 0)
        ]
        gate_score = 1 - np.mean([e for e in gate_errors if e is not None and isinstance(e, (int, float))])
        
        # Calculate connectivity quality
        connected_qubits = connectivity.get(qubit_idx, [])
        connectivity_score = len(connected_qubits) / 3  # Normalize by max ~3 connections
        
        # Compute final score with weights - prioritize RZZ error
        final_score = (
            0.2 * t1_score + 
            0.2 * t2_score + 
            0.15 * readout_score + 
            0.35 * rzz_score +      
            0.05 * gate_score + 
            0.05 * connectivity_score
        )
        
        # Strongly penalize qubits with readout error near 0.5 (random measurement)
        if 0.4 <= data.get('Readout assignment error ', 0) <= 0.6:
            final_score *= 0.2
        
        scores[qubit_idx] = final_score
    
    return scores, rzz_errors  

def find_best_connected_subset(qubit_scores: Dict[int, float], 
                              connectivity: Dict[int, List[int]], 
                              n: int) -> List[int]:
    """
    Finds the best connected subset of n qubits using a search algorithm.
    """
    if n <= 0:
        return []
    
    # Create graph from connectivity dictionary
    G = nx.Graph()
    for qubit, connections in connectivity.items():
        G.add_node(qubit, score=qubit_scores.get(qubit, 0))
        for conn in connections:
            G.add_edge(qubit, conn)
    
    # Filter out qubits with score 0
    for qubit, score in qubit_scores.items():
        if score == 0 and qubit in G:
            G.remove_node(qubit)
    
    # If no connected component has at least n nodes, return empty list
    if max(len(c) for c in nx.connected_components(G)) < n:
        return []
    
    best_score = -1
    best_subset = []
    
    # Try each qubit as a starting point for the search
    sorted_qubits = sorted(qubit_scores.keys(), key=lambda x: qubit_scores.get(x, 0), reverse=True)
    
    # Limit search to top 30 qubits for efficiency
    for start_qubit in sorted_qubits[:30]:
        if start_qubit not in G or qubit_scores.get(start_qubit, 0) == 0:
            continue
            
        # Start with the current qubit
        subset = [start_qubit]
        total_score = qubit_scores.get(start_qubit, 0)
        
        # Keep track of available qubits connected to our subset
        frontier = set()
        for conn in connectivity.get(start_qubit, []):
            if conn in G:
                frontier.add(conn)
        
        # Greedily add qubits until we have n
        while len(subset) < n and frontier:
            # Find best qubit in frontier
            best_next = max(frontier, key=lambda q: qubit_scores.get(q, 0))
            
            # Add it to our subset
            subset.append(best_next)
            total_score += qubit_scores.get(best_next, 0)
            
            # Update frontier
            frontier.remove(best_next)
            for conn in connectivity.get(best_next, []):
                if conn in G and conn not in subset and conn not in frontier:
                    frontier.add(conn)
        
        # Check if we found a valid subset
        if len(subset) == n:
            avg_score = total_score / n
            if avg_score > best_score:
                best_score = avg_score
                best_subset = subset
    
    # Verify connectivity of the final subset
    subgraph = G.subgraph(best_subset)
    if not nx.is_connected(subgraph):
        # If somehow we got a disconnected subset, find the largest connected component
        components = list(nx.connected_components(subgraph))
        largest_component = max(components, key=len)
        if len(largest_component) < n:
            # If the largest component is too small, return the original (might be partially connected)
            return best_subset
        else:
            return list(largest_component)[:n]
    
    return best_subset

def select_best_qubits_rzz_optimized(qubit_data: Dict[int, Dict], connectivity: Dict[int, List[int]], n: int) -> Tuple[List[int], Dict[int, float], Dict[Tuple[int, int], float]]:
    """
    Selects the best n-qubit connected subset based on calibration data and connectivity,
    optimized for RZZ gates.
    
    Args:
        qubit_data: Dictionary with calibration data for each qubit
        connectivity: Dictionary mapping qubit index to list of connected qubits
        n: Number of qubits to select
        
    Returns:
        Tuple containing:
        - List of n qubit indices representing the best connected subset
        - Dictionary of qubit scores
        - Dictionary of RZZ errors
    """
    # Calculate qubit quality scores with RZZ priority
    qubit_scores, rzz_errors = calculate_qubit_scores_with_rzz_priority(qubit_data, connectivity)
    
    # Find best connected subset
    best_subset = find_best_connected_subset(qubit_scores, connectivity, n)
    
    return best_subset, qubit_scores, rzz_errors


def visualize_selected_qubits(selected_qubits: List[int], connectivity: Dict[int, List[int]], 
                             qubit_scores: Dict[int, float] = None, title=None):
    """
    Visualizes the selected qubits and their connections within the full QPU grid.
    """
    # Create a figure and axes explicitly
    fig, ax = plt.subplots(figsize=(15, 10))
    
    G = nx.Graph()
    
    # Create a grid layout to position nodes
    # This represents the approximate physical layout of the QPU
    grid_positions = {}
    
    # Position qubits in a grid pattern based on their indices
    for i in range(156):
        row = (i // 20)
        col = i % 20
        grid_positions[i] = (col, -row)
    
    # Add all nodes and edges to the graph
    for qubit, connections in connectivity.items():
        G.add_node(qubit)
        for neighbor in connections:
            G.add_edge(qubit, neighbor)
    
    # Draw all edges with low opacity
    nx.draw_networkx_edges(G, grid_positions, alpha=0.15, width=1, ax=ax)
    
    # Draw all nodes with low opacity
    non_selected = [q for q in G.nodes() if q not in selected_qubits]
    nx.draw_networkx_nodes(G, grid_positions, nodelist=non_selected, 
                          node_color='gray', alpha=0.3, node_size=300, ax=ax)
    
    # Draw selected nodes and their edges with high opacity
    selected_edges = [(u, v) for u, v in G.edges() if u in selected_qubits and v in selected_qubits]
    nx.draw_networkx_edges(G, grid_positions, edgelist=selected_edges, 
                          width=2.5, alpha=1, edge_color='blue', ax=ax)
    
    # Draw the selected nodes
    if qubit_scores:
        # Color nodes by score if available
        cmap = plt.cm.viridis
        node_colors = [qubit_scores.get(q, 0) for q in selected_qubits]
        nodes = nx.draw_networkx_nodes(G, grid_positions, nodelist=selected_qubits, 
                              node_color=node_colors, cmap=cmap, 
                              node_size=600, alpha=1, ax=ax)
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_colors) if node_colors else 0, 
                                                                 vmax=max(node_colors) if node_colors else 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Qubit Quality Score")
    else:
        nx.draw_networkx_nodes(G, grid_positions, nodelist=selected_qubits, 
                              node_color='red', node_size=600, alpha=1, ax=ax)
    
    # Add labels to selected nodes
    labels = {q: str(q) for q in selected_qubits}
    nx.draw_networkx_labels(G, grid_positions, labels=labels, font_size=12, font_weight='bold', ax=ax)
    
    ax.set_title(title or f"Selected {len(selected_qubits)} qubits on the QPU")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def generate_gates_for_subgraph(selected_qubits: List[int], full_connectivity: Dict[int, List[int]], 
                               num_layers: int = 1) -> List[List[List[int]]]:
    """
    Generates a gate list for the selected subset of qubits based on their connectivity
    in the original hardware graph, optimized for RZZ gates.
    
    Args:
        selected_qubits: List of qubit indices that form the selected connected subgraph
        full_connectivity: Full connectivity graph of the hardware (original qubit indices)
        num_layers: Number of times to repeat the connectivity pattern
        
    Returns:
        List of gates in the format expected by IqpSimulator, where qubit indices are 
        mapped to their positions in the selected subset (0 to n-1)
    """
    # Create a mapping from original qubit indices to their position in the selected subset
    qubit_map = {q: i for i, q in enumerate(selected_qubits)}
    
    # Create a reduced connectivity graph with mapped indices
    reduced_connectivity = {}
    for i, qubit in enumerate(selected_qubits):
        reduced_connectivity[i] = []
        neighbors = full_connectivity.get(qubit, [])
        for neighbor in neighbors:
            if neighbor in selected_qubits:
                reduced_connectivity[i].append(qubit_map[neighbor])
    
    # Now use the connectivity-based gates approach with the mapped graph
    gates = []
    n_qubits = len(selected_qubits)
    
    # First, create single-qubit gates
    single_qubit_gates = [[[i]] for i in range(n_qubits)]
    
    # Create two-qubit gates for directly connected qubits
    two_qubit_gates = []
    edges_added = set()  # To avoid adding the same edge twice
    
    for qubit, neighbors in reduced_connectivity.items():
        for neighbor in neighbors:
            # Create an edge identifier that's independent of order
            edge = tuple(sorted([qubit, neighbor]))
            
            # Only add this edge if we haven't seen it before
            if edge not in edges_added:
                two_qubit_gates.append([[qubit, neighbor]])
                edges_added.add(edge)
    
    # Combine single and two-qubit gates
    layer_gates = single_qubit_gates + two_qubit_gates
    
    # Repeat the pattern for the specified number of layers
    for _ in range(num_layers):
        gates.extend(layer_gates)
    
    return gates

def visualize_gate_structure(gates, n_qubits, title="Gate Structure"):
    """
    Visualizes the gate structure as a connectivity graph.
    This helps verify that the generated gates match the expected connectivity.
    
    Args:
        gates: Gate list generated for the selected qubits
        n_qubits: Number of qubits in the circuit
        title: Title for the plot
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    
    G = nx.Graph()
    
    # Add all qubits as nodes
    for i in range(n_qubits):
        G.add_node(i)
    
    # Add edges for each two-qubit gate
    for gate in gates:
        if len(gate[0]) == 2:  # Two-qubit gate
            q1, q2 = gate[0]
            G.add_edge(q1, q2)
    
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)  # Fixed seed for consistent layout
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            node_size=500, edge_color='navy', linewidths=1, 
            font_size=15)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def show_gate_list(gates):
    """
    Prints the gate list in a readable format, indicating the type of gates.
    """
    print("Gate List:")
    for i, gate in enumerate(gates):
        qubits = gate[0]
        if len(qubits) == 1:
            print(f"Layer {i+1}: H on qubit {qubits[0]}")
        else:
            print(f"Layer {i+1}: RZZ on qubits {qubits[0]} and {qubits[1]}")

def main():
    """
    Main demo function for RZZ-optimized qubit selection and gate generation.
    """
    # Load calibration data
    qubit_data = load_calibration_from_csv("utils/aachen_calib.csv")
    
    # Create connectivity graph
    connectivity = aachen_connectivity()
    
    # Select best qubits optimized for RZZ gates
    n_qubits = 45  
    best_qubits, qubit_scores, rzz_errors = select_best_qubits_rzz_optimized(qubit_data, connectivity, n_qubits)
    
    print(f"Best {n_qubits} qubits optimized for RZZ gates: {best_qubits}")
    
    # Calculate average score
    avg_score = sum(qubit_scores.get(q, 0) for q in best_qubits) / len(best_qubits) if best_qubits else 0
    print(f"Average quality score: {avg_score:.4f}")
    
    # Visualize the selected qubits
    if best_qubits:
        visualize_selected_qubits(best_qubits, connectivity, qubit_scores, 
                                 title=f"Best {n_qubits} Qubits Optimized for RZZ Gates (Avg Score: {avg_score:.4f})")
    

    num_layers = 2  # Number of RZZ layers
    gates = generate_gates_for_subgraph(best_qubits, connectivity, num_layers)
    
    # Show the gate list
    show_gate_list(gates)
    
    # Visualize the gate structure
    visualize_gate_structure(gates, len(best_qubits), 
                           title=f"Gate Structure for Selected Qubits\n(Hadamard + {num_layers} RZZ layers)")
    
    # Calculate statistics for the selected qubits
    print("\nDetailed statistics for selected qubits:")
    for i, qubit in enumerate(best_qubits):
        # Show coherence times
        t1 = qubit_data[qubit].get('T1 (us)', 'N/A')
        t2 = qubit_data[qubit].get('T2 (us)', 'N/A')
        readout_err = qubit_data[qubit].get('Readout assignment error ', 'N/A')
        
        print(f"Qubit {qubit} (index {i}):")
        print(f"  T1: {t1} μs")
        print(f"  T2: {t2} μs")
        print(f"  Readout error: {readout_err}")
        
        # Show RZZ errors with connected qubits in the selected subset
        rzz_connected = []
        for other_qubit in best_qubits:
            if qubit != other_qubit and other_qubit in connectivity[qubit]:
                error = rzz_errors.get((qubit, other_qubit), 'N/A')
                rzz_connected.append((other_qubit, error))
        
        if rzz_connected:
            print("  RZZ errors with connected qubits:")
            for other_qubit, error in rzz_connected:
                other_idx = best_qubits.index(other_qubit)
                print(f"    With qubit {other_qubit} (index {other_idx}): {error}")
        print()


if __name__ == "__main__":
    main()
