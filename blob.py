import sys
import optuna

# --- Configuration ---
# Path to your Optuna database file.
STORAGE_URL = "sqlite:///results/hpo_logs/all_hpo_studies.db"

def analyze_study():
    """
    Loads studies from the database, prompts the user to select one,
    and generates insightful visualizations.
    """
    # --- 1. Get a list of all available studies ---
    try:
        summaries = optuna.get_all_study_summaries(storage=STORAGE_URL)
        if not summaries:
            print(f"❌ Error: No studies found in the database at '{STORAGE_URL}'.")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error: Could not connect to or read the database. Details: {e}")
        sys.exit(1)

    # --- 2. Let the user choose which study to analyze ---
    print("📖 Studies found in the database:")
    for i, summary in enumerate(summaries):
        print(f"  [{i}] - {summary.study_name} ({summary.n_trials} trials)")

    try:
        choice = int(input("\nEnter the number of the study you want to analyze: "))
        selected_summary = summaries[choice]
        study_name = selected_summary.study_name
    except (ValueError, IndexError):
        print("❌ Invalid selection. Please run the script again.")
        sys.exit(1)

    print(f"\n✅ Loading study: '{study_name}'...")
    study = optuna.load_study(study_name=study_name, storage=STORAGE_URL)

    # --- 3. Generate and display visualizations ---
    print("Generating visualizations... Your browser should open with interactive plots.")

    # Visualization 1: Hyperparameter Importances
    # Shows which hyperparameters had the most impact on the final result.
    try:
        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.show()
    except Exception:
        print("Could not generate hyperparameter importance plot (might require completed trials).")


    # Visualization 2: Contour Plot
    # Shows relationships between two hyperparameters.
    # Only works if you have at least 2 hyperparameters.
    if len(study.best_params) > 1:
        # We'll try to plot learning_rate and num_layers as an example
        # Add other combinations if you like!
        params_to_plot = ["learning_rate", "num_layers"]
        try:
            fig_contour = optuna.visualization.plot_contour(study, params=params_to_plot)
            fig_contour.show()
        except (ValueError, RuntimeError) as e:
            print(f"Could not generate contour plot for {params_to_plot}. Skipping. Reason: {e}")


    # Visualization 3: Slice Plot
    # Shows how each hyperparameter individually affects the objective value.
    try:
        fig_slice = optuna.visualization.plot_slice(study)
        fig_slice.show()
    except Exception:
        print("Could not generate slice plot.")


if __name__ == "__main__":
    analyze_study()