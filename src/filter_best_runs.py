import pandas as pd
from pathlib import Path
import yaml

def load_dataset_summary(summary_path: Path) -> dict:
    with open(summary_path, "r") as f:
        summary_list = yaml.safe_load(f)
    summary = {}
    for d in summary_list:
        name = d["name"].strip()
        graph_type = d["graph_type"]
        # normalize graph type
        if graph_type == "ER":
            graph_type = "ErdosRenyi"
        summary[name] = {**d, "graph_type": graph_type}
    return summary

def select_best_runs_from_df(df: pd.DataFrame, summary: dict) -> pd.DataFrame:
    best_rows = []

    # Ensure dataset_name is string, not tuple
    df["dataset_name"] = df["dataset_name"].astype(str).str.strip()

    for dataset_instance, group in df.groupby("dataset_name"):
        if dataset_instance not in summary:
            print(f"[WARNING] Dataset {dataset_instance} not found in summary.yml, skipping.")
            continue

        graph_type = summary[dataset_instance]["graph_type"]

        if graph_type == "ErdosRenyi":
            group = group.copy()
            group["density_error"] = (group["gen_density"] - group["ref_density"]).abs()
            best_row = group.loc[group["density_error"].idxmin()]
        elif graph_type == "Bipartite":
            best_row = group.loc[group["gen_bipartite_percent"].idxmax()]
        else:
            continue

        best_rows.append(best_row)

    return pd.DataFrame(best_rows)

if __name__ == "__main__":
    summary_path = Path("data/datasets_summary.yml")
    dataset_summary = load_dataset_summary(summary_path)

    analysis_dir = Path("results/analysis")
    all_dfs = [pd.read_csv(f) for f in analysis_dir.glob("analysis_*.csv")]

    if not all_dfs:
        print("[ERROR] No CSV files found in", analysis_dir)
        exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    best_df = select_best_runs_from_df(combined_df, dataset_summary)

    out_file = analysis_dir / "best_runs.csv"
    best_df.to_csv(out_file, index=False, float_format="%.4f")
    print(f"[INFO] Best runs saved to {out_file}, total {len(best_df)} rows.")
