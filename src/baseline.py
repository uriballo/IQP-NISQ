import networkx as nx
import numpy as np
from tqdm import tqdm 

# Unchanged function
def calculate_num_vertices(L: int) -> int:
    """
    Calculates the number of vertices N for a graph whose strict upper
    triangle adjacency matrix has L elements.
    L = N * (N - 1) / 2  =>  N^2 - N - 2L = 0
    N = (1 + sqrt(1 + 8L)) / 2 (since N must be positive)
    """
    delta = 1 + 8 * L
    if delta < 0:
        raise ValueError(
            f"L={L} results in a negative discriminant ({delta})."
        )
    sqrt_delta = np.sqrt(delta)
    if not sqrt_delta.is_integer():
        raise ValueError(
            f"1 + 8L = {delta} is not a perfect square. "
            f"L={L} is not a valid length for a strict upper triangle."
        )
    num_vertices_float = (1 + sqrt_delta) / 2
    if not num_vertices_float.is_integer():
        raise ValueError(
            f"Number of vertices ({num_vertices_float}) is not an integer. "
            f"L={L} is not a valid length."
        )
    return int(num_vertices_float)

DENSITY_REGIMES = {
    "sparse": (0.05, 0.20),
    "medium": (0.25, 0.35),
    "dense": (0.35, 0.55),
}

def run_bipartite_benchmark_with_density(
    bitstring_lengths: list[int],
    num_samples: int,
    density_regimes: dict[str, tuple[float, float]],
):
    """
    Runs the benchmark for generating random graphs with varying densities
    and calculating the percentage of bipartite graphs.
    """
    all_results = {}
    print(
        f"Starting benchmark: {num_samples} samples per bitstring length "
        f"per density regime.\n"
    )

    for L in bitstring_lengths:
        print(f"Processing bitstrings of length L={L}...")
        current_L_results = {"regimes": {}}
        all_results[L] = current_L_results

        try:
            num_vertices = calculate_num_vertices(L)
            current_L_results["num_vertices"] = num_vertices
            print(f"  Derived num_vertices = {num_vertices} for L={L}")
        except ValueError as e:
            error_msg = f"Error calculating num_vertices for L={L}: {e}"
            print(f"  {error_msg}")
            current_L_results["error"] = error_msg
            current_L_results["num_vertices"] = None
            print("-" * 60)
            continue  # to next L

        for regime_name, (p_low, p_high) in density_regimes.items():
            print(
                f"  Testing density regime: '{regime_name}' "
                f"(p_edge in [{p_low:.2f}, {p_high:.2f}])"
            )
            bipartite_count = 0
            regime_results_data = {}
            current_L_results["regimes"][regime_name] = regime_results_data

            try:
                # Use tqdm for a progress bar.
                for _ in tqdm(
                    range(num_samples),
                    desc=(
                        f"    L={L}, N={num_vertices}, {regime_name:<7}"
                    ), # Pad regime_name
                    unit="graphs",
                    leave=False, # Keep bar until regime finishes
                ):
                    # 1. Sample a p_edge for this specific graph instance
                    p_edge = np.random.uniform(p_low, p_high)

                    # 2. Generate bitstring based on this p_edge
                    random_vec = (np.random.rand(L) < p_edge).astype(
                        np.uint8
                    )

                    graph = vec_to_graph(random_vec, num_vertices)
                    if nx.is_bipartite(graph):
                        bipartite_count += 1

                percentage_bipartite = (
                    bipartite_count / num_samples
                ) * 100
                regime_results_data.update({
                    "p_range": (p_low, p_high),
                    "bipartite_count": bipartite_count,
                    "total_samples": num_samples,
                    "percentage": percentage_bipartite,
                })
                print( # This print will appear after tqdm bar for the regime finishes
                    f"    For L={L} (N={num_vertices}), regime '{regime_name}': "
                    f"{bipartite_count}/{num_samples} graphs are bipartite."
                )
                print(
                    f"    Percentage of bipartite graphs: {percentage_bipartite:.4f}%"
                )

            except Exception as e:
                error_msg = (
                    f"An error occurred during graph processing for L={L}, "
                    f"regime '{regime_name}': {e}"
                )
                print(f"    {error_msg}") # Will print above tqdm if error is early
                regime_results_data["error"] = error_msg
            
            print("  " + "-" * 50)  # Separator between regimes
        print("-" * 60)  # Separator between L values

    print("\n--- Benchmark Summary ---")
    for L, L_data in all_results.items():
        N_val = L_data.get("num_vertices")
        N_str = str(N_val) if N_val is not None else "N/A (calc error)"

        if "error" in L_data and L_data.get("num_vertices") is None:
            print(f"L={L}: {L_data['error']}")
            continue

        print(f"Results for L={L} (N={N_str}):")

        for regime_name, regime_res in L_data.get("regimes", {}).items():
            if "error" in regime_res:
                print(
                    f"  Regime '{regime_name}': Error - {regime_res['error']}"
                )
            else:
                perc = regime_res.get("percentage", "N/A")
                p_range_str = "N/A"
                if "p_range" in regime_res:
                    p_range_str = (
                        f"[{regime_res['p_range'][0]:.2f}, "
                        f"{regime_res['p_range'][1]:.2f}]"
                    )

                perc_str = (
                    f"{perc:.4f}%" if isinstance(perc, float) else str(perc)
                )
                print(
                    f"  Regime '{regime_name}' (p_edge in {p_range_str}): "
                    f"{perc_str} bipartite"
                )
    print("-" * 60)


if __name__ == "__main__":
    BITSTRING_LENGTHS = [45, 91, 153]  # N=10, N=14, N=18
    NUM_SAMPLES = 100_000

    # Note: The tqdm library provides a progress bar.
    # If you don't have it, you can install it (`pip install tqdm`)
    # or remove its usage from the run_bipartite_benchmark_with_density function.

    run_bipartite_benchmark_with_density(
        BITSTRING_LENGTHS, NUM_SAMPLES, DENSITY_REGIMES
    )