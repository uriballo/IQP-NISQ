from datasets.utils import is_bipartite
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
from networkx.algorithms import isomorphism
from networkx.algorithms.bipartite import color as bipartite_color

from typing import List, Tuple, Dict, Any

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
    Compute D_KL between the degree distributions of G1 and G2:
        D_KL(P||Q) = sum_k P(k)*log(P(k)/Q(k)), with smoothing eps.

    If either graph has no nodes, raises ValueError.
    """
    degs1 = [d for _, d in G1.degree()]
    degs2 = [d for _, d in G2.degree()]
    if not degs1 or not degs2:
        raise ValueError("Both graphs must have at least one node.")

    max_deg = max(max(degs1), max(degs2))
    bins = np.arange(max_deg + 1)

    hist1 = np.bincount(degs1, minlength=bins.size).astype(float)
    hist2 = np.bincount(degs2, minlength=bins.size).astype(float)

    P = hist1 / hist1.sum()
    Q = hist2 / hist2.sum()
    P = np.clip(P, eps, None)
    Q = np.clip(Q, eps, None)

    D = np.sum(P * np.log(P / Q))
    return D / np.log(base) if base != np.e else D


def average_kl_divergence(
    graphs_truth: List[nx.Graph],
    graphs_model: List[nx.Graph],
    base: float = np.e,
    eps: float = 1e-10
) -> float:
    """
    Compute the average KL divergence between pairs (truth[i], model[i]).
    If len(model) >= len(truth), only the first len(truth) are used.
    If len(model) < len(truth), raises ValueError.
    """
    n_truth = len(graphs_truth)
    n_model = len(graphs_model)
    if n_model < n_truth:
        raise ValueError(
            f"Need at least {n_truth} model graphs, got {n_model}"
        )

    divs = []
    for i in range(n_truth):
        divs.append(
            kl_divergence(
                graphs_truth[i],
                graphs_model[i],
                base=base,
                eps=eps
            )
        )
    return float(np.mean(divs))


def memorized_proportion(
    gt_graphs: List[nx.Graph],
    gen_graphs: List[nx.Graph],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare generated vs ground‐truth graphs:
      - gen_mem_fraction: fraction of generated graphs isomorphic to any GT graph
      - truth_cov_fraction: fraction of GT graphs covered by at least one generated
    If verbose=True, prints a clear report.
    Returns:
      { 'n_truth', 'n_gen',
        'gen_mem_count', 'gen_mem_fraction',
        'truth_cov_count', 'truth_cov_fraction' }
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
    Return only those graphs in `graphs` that are bipartite.
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
def compute_graph_set_stats(
    graphs: List[nx.Graph]
) -> Dict[str, Any]:
    """
    Compute summary statistics on a list of Graphs:
      - n_graphs
      - avg|V|, std|V|
      - avg|E|, std|E|
      - avg density, std density       (global)
      - fraction connected
      - avg #components
      - fraction bipartite
      - avg/std bipartite‐density
    """
    stats: Dict[str, Any] = {}
    n = len(graphs)
    stats["n_graphs"] = n
    if n == 0:
        # fill zeros
        for k in [
            "avg_nodes","std_nodes","avg_edges","std_edges",
            "avg_density","std_density","fraction_connected",
            "avg_components","fraction_bipartite",
            "avg_bipartite_density","std_bipartite_density"
        ]:
            stats[k] = 0.0
        return stats

    # node & edge counts
    node_counts = np.array([G.number_of_nodes() for G in graphs], float)
    edge_counts = np.array([G.number_of_edges() for G in graphs], float)
    stats["avg_nodes"] = float(node_counts.mean())
    stats["std_nodes"] = float(node_counts.std())
    stats["avg_edges"] = float(edge_counts.mean())
    stats["std_edges"] = float(edge_counts.std())

    # global density = 2m / (n(n-1))
    densities = np.array([nx.density(G) for G in graphs], float)
    stats["avg_density"] = float(densities.mean())
    stats["std_density"] = float(densities.std())

    # connectivity
    connected = np.array([nx.is_connected(G) for G in graphs], float)
    stats["fraction_connected"] = float(connected.mean())
    comps = np.array(
        [nx.number_connected_components(G) for G in graphs], float
    )
    stats["avg_components"] = float(comps.mean())

    # bipartiteness
    bip_flags = np.array(
        [bipartite.is_bipartite(G) for G in graphs], float
    )
    stats["fraction_bipartite"] = float(bip_flags.mean())

    # bipartite‐density for each bipartite G
    bip_dens_list: List[float] = []
    for G in graphs:
        if bipartite.is_bipartite(G):
            coloring = bipartite_color(G)
            # split by color; color values are 0 or 1
            top    = [n for n,c in coloring.items() if c == 0]
            bottom = [n for n,c in coloring.items() if c == 1]
            if top and bottom:
                bip_dens_list.append(
                    G.number_of_edges() / (len(top) * len(bottom))
                )

    if bip_dens_list:
        bd = np.array(bip_dens_list, float)
        stats["avg_bipartite_density"] = float(bd.mean())
        stats["std_bipartite_density"] = float(bd.std())
    else:
        stats["avg_bipartite_density"] = 0.0
        stats["std_bipartite_density"] = 0.0

    return stats



def sample_diversity(graphs: List[nx.Graph]) -> float:
    """
    Measure diversity of a list of graphs.
    """
    n = len(graphs)
    unique_graphs: List[nx.Graph] = []
    for g in graphs:
        if not any(nx.is_isomorphic(g, ug) for ug in unique_graphs):
            unique_graphs.append(g)
    return len(unique_graphs), len(unique_graphs) / n 

def analyze_model_vs_dataset(
    gt_graphs: List[nx.Graph],
    gen_graphs: List[nx.Graph],
    base: float = np.e,
    eps: float = 1e-10,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive comparison of generated graphs vs ground‐truth set.

    Returns a dict with:
      • Basic sizes & uniqueness
      • Structural summary (edges, density, connectivity, components,
        bipartite fraction & density)
      • Memorization / coverage metrics (precision, recall,
        novel counts, match multiplicities)
      • Distribution divergences on degree distributions
      • (Optional) paired KL divergence if len(gen) >= len(gt)

    If verbose=True, prints a structured report.
    """
    stats: Dict[str, Any] = {}
    n_truth = len(gt_graphs)
    n_gen   = len(gen_graphs)
    stats["n_truth"] = n_truth
    stats["n_gen"]   = n_gen

    # ---- uniqueness: exact adjacency & isomorphism ----
    def count_unique_adj(graphs: List[nx.Graph]) -> int:
        if not graphs:
            return 0
        ns = {G.number_of_nodes() for G in graphs}
        if len(ns) != 1:
            return None  # mixed sizes, skip
        n = ns.pop()
        seen = set()
        for G in graphs:
            arr = nx.to_numpy_array(G, nodelist=range(n), dtype=np.uint8)
            iu = np.triu_indices(n, k=1)
            seen.add(arr[iu].tobytes())
        return len(seen)

    def count_unique_iso(graphs: List[nx.Graph]) -> int:
        reps: List[nx.Graph] = []
        for G in graphs:
            if not any(nx.is_isomorphic(G, H) for H in reps):
                reps.append(G)
        return len(reps)

    ua_gt = count_unique_adj(gt_graphs)
    ua_gen = count_unique_adj(gen_graphs)
    ui_gt = count_unique_iso(gt_graphs)
    ui_gen = count_unique_iso(gen_graphs)
    stats["unique_adj_gt"] = ua_gt
    stats["unique_adj_gen"] = ua_gen
    stats["unique_iso_gt"] = ui_gt
    stats["unique_iso_gen"] = ui_gen

    # ---- structural stats ----
    gt_stats = compute_graph_set_stats(gt_graphs)
    gen_stats = compute_graph_set_stats(gen_graphs)
    stats["gt_struct"] = gt_stats
    stats["gen_struct"] = gen_stats

    # compute deltas & ratios on key metrics
    def delta_ratio(k: str):
        a = gt_stats[k]
        b = gen_stats[k]
        dr = b - a
        rq = (b / a) if a else None
        return dr, rq

    dr_edges, rq_edges = delta_ratio("avg_edges")
    dr_den,   rq_den   = delta_ratio("avg_density")
    dr_conn,  rq_conn  = delta_ratio("fraction_connected")
    dr_comp,  rq_comp  = delta_ratio("avg_components")
    dr_bipf,  rq_bipf  = delta_ratio("fraction_bipartite")
    dr_bipd,  rq_bipd  = delta_ratio("avg_bipartite_density")

    stats.update({
        "delta_avg_edges":           dr_edges,
        "ratio_avg_edges":           rq_edges,
        "delta_avg_density":         dr_den,
        "ratio_avg_density":         rq_den,
        "delta_fraction_connected":  dr_conn,
        "ratio_fraction_connected":  rq_conn,
        "delta_avg_components":      dr_comp,
        "ratio_avg_components":      rq_comp,
        "delta_fraction_bipartite":  dr_bipf,
        "ratio_fraction_bipartite":  rq_bipf,
        "delta_avg_bip_density":     dr_bipd,
        "ratio_avg_bip_density":     rq_bipd,
    })

    # ---- memorization / coverage ----
    # map each generated to first GT index it matches
    match_map = [None] * n_gen
    for j, Gg in enumerate(gen_graphs):
        for i, Gt in enumerate(gt_graphs):
            if nx.is_isomorphic(Gg, Gt):
                match_map[j] = i
                break

    gen_mem_count   = sum(1 for m in match_map if m is not None)
    gen_novel_count = n_gen - gen_mem_count
    truth_counts    = [0]*n_truth
    for m in match_map:
        if m is not None:
            truth_counts[m] += 1
    truth_cov_count   = sum(1 for c in truth_counts if c>0)
    truth_novel_count = n_truth - truth_cov_count

    gen_mem_frac     = gen_mem_count   / n_gen   if n_gen   else 0.0
    gen_novel_frac   = gen_novel_count / n_gen   if n_gen   else 0.0
    truth_cov_frac   = truth_cov_count / n_truth if n_truth else 0.0
    truth_novel_frac = truth_novel_count/n_truth if n_truth else 0.0

    matched_counts = [c for c in truth_counts if c>0]
    if matched_counts:
        avg_matched = float(np.mean(matched_counts))
        med_matched = float(np.median(matched_counts))
        min_matched = int(np.min(matched_counts))
        max_matched = int(np.max(matched_counts))
    else:
        avg_matched = med_matched = 0.0
        min_matched = max_matched = 0

    precision = gen_mem_frac
    recall    = truth_cov_frac

    stats.update({
        "gen_mem_count":         gen_mem_count,
        "gen_mem_fraction":      gen_mem_frac,
        "gen_novel_count":       gen_novel_count,
        "gen_novel_fraction":    gen_novel_frac,
        "truth_cov_count":       truth_cov_count,
        "truth_cov_fraction":    truth_cov_frac,
        "truth_novel_count":     truth_novel_count,
        "truth_novel_fraction":  truth_novel_frac,
        "avg_matches_per_gt":    avg_matched,
        "median_matches_per_gt": med_matched,
        "min_matches_per_gt":    min_matched,
        "max_matches_per_gt":    max_matched,
        "precision":             precision,
        "recall":                recall,
    })

    # ---- degree distribution divergences ----
    def agg_degs(graphs: List[nx.Graph]) -> List[int]:
        out = []
        for G in graphs:
            out.extend(d for _, d in G.degree())
        return out

    deg_gt  = agg_degs(gt_graphs)
    deg_gen = agg_degs(gen_graphs)
    max_deg = max(deg_gt + deg_gen) if deg_gt or deg_gen else 0
    bins    = np.arange(max_deg + 1)
    h_gt  = np.bincount(deg_gt,  minlength=bins.size).astype(float)
    h_gen = np.bincount(deg_gen, minlength=bins.size).astype(float)
    P = h_gt / h_gt.sum()  if h_gt.sum()  else h_gt
    Q = h_gen / h_gen.sum() if h_gen.sum() else h_gen
    P = np.clip(P, eps, None); Q = np.clip(Q, eps, None)

    # KL and JS
    kl_gt_gen = np.sum(P * np.log(P/Q)) / np.log(base)
    kl_gen_gt = np.sum(Q * np.log(Q/P)) / np.log(base)
    M = 0.5*(P+Q)
    js_deg = 0.5*(np.sum(P*np.log(P/M)) + np.sum(Q*np.log(Q/M))) / np.log(base)

    stats.update({
        "kl_degree_gt_to_gen": kl_gt_gen,
        "kl_degree_gen_to_gt": kl_gen_gt,
        "js_degree":           js_deg
    })

    # ---- paired KL if possible ----
    if n_gen >= n_truth > 0:
        paired_kl = average_kl_divergence(gt_graphs, gen_graphs,
                                         base=base, eps=eps)
    else:
        paired_kl = None
    stats["avg_paired_kl"] = paired_kl

    # ---- verbose printing ----
    if verbose:
        print("\n=== Model vs Dataset Analysis ===\n")

        print("1) Basic sizes & uniqueness")
        print(f"   GT graphs       : {n_truth}")
        print(f"   Generated graphs: {n_gen}")
        if ua_gt is not None:
            print(f"   GT unique-adj  : {ua_gt}")
            print(f"   Gen unique-adj : {ua_gen}")
        print(f"   GT unique-iso  : {ui_gt}")
        print(f"   Gen unique-iso : {ui_gen}\n")

        print("2) Structural statistics (GT vs Gen)")
        print("   Metric               GT       Gen       Δ        Ratio")
        def row(k, label, fmt="{:>8.2f}"):
            a = gt_stats[k]; b = gen_stats[k]
            d = b - a
            r = (b/a) if a else float('nan')
            print(f"   {label:<20}"
                  f"{a:>8.2f}{b:>9.2f}{d:>9.2f}{r:>10.2f}")

        row("avg_edges",           "Avg edges")
        row("avg_density",        "Avg density")
        row("fraction_connected", "Conn. frac")
        row("avg_components",     "#Comp   ")
        row("fraction_bipartite", "Bip frac")
        row("avg_bipartite_density", "Bip density")
        print()

        print("3) Memorization & coverage")
        print(f"   Precision (Gen→GT): {precision*100:5.2f}%")
        print(f"   Recall    (GT←Gen): {recall*100:5.2f}%")
        print(f"   Novel gen        : {gen_novel_count}/{n_gen} "
              f"({gen_novel_frac*100:5.2f}%)")
        print(f"   GT never seen    : {truth_novel_count}/{n_truth} "
              f"({truth_novel_frac*100:5.2f}%)")
        print(f"   Matches per GT   : avg {avg_matched:.2f}, "
              f"med {med_matched:.2f}, min {min_matched}, "
              f"max {max_matched}\n")

        print("4) Degree distribution divergences")
        if paired_kl is not None:
            print(f"   Avg paired KL(deg): {paired_kl:.4f}")
        print(f"   KL(GT→Gen)        : {kl_gt_gen:.4f}")
        print(f"   KL(Gen→GT)        : {kl_gen_gt:.4f}")
        print(f"   JS sym            : {js_deg:.4f}")
        print("\n===================================\n")

    return stats