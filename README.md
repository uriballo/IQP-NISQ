# Shallow IQP Circuits and Graph Generation

Abstract: We introduce shallow instantaneous quantum polynomial-time (IQP) circuits as generative graph models, using an edge-qubit encoding to map graphs onto quantum states. Focusing on bipartite and Erd\H{o}s–Rényi distributions, we study their expressivity and robustness through simulations and large-scale experiments. Noiseless simulations of $28$ qubits ($8-$node graphs) reveal that shallow IQP models can learn key structural features, such as the edge density and bipartite partitioning. On IBM’s Aachen QPU, we scale experiments from $28$ to $153$ qubits ($8$–$18$ nodes) in order to characterize performance under realistic noise. Local statistics---such as the degree distributions---remain accurate across scales with total variation distances ranging from $0.04$ to $0.20$, while global properties like strict bipartiteness degrade at the largest system sizes ($91$ and $153$ qubits). Notably, spectral bipartivity---a relaxation of strict bipartiteness---remains comparatively robust at higher qubit counts. These results establish practical baselines for the performance of shallow IQP circuits on current quantum hardware and demonstrate that, even without error mitigation, such circuits can learn and reproduce meaningful structural patterns in graph data, guiding future developments in quantum generative modeling for the NISQ era and beyond. 

Folder structure:
- `data/` > synthetic datasets.
- `hpo/` > configurations used and hpo script.
- `plots/` > plots used in the paper.
-  `results/`
    - `analysis/` > processed results from all models.
    - `archived/` > parameters from old, unused models.
    - `evaluation_results/` > Raw results from NISQ executions.
    - `hpo_logs/` > Optuna logs.
    - `simulation_results/` > Raw results from simulations.
    - `trained_params/` > parameters from all models.
- `src/` > code used to generate, process, and plot results.
- `test/` > tests for select utility and metric functions.
