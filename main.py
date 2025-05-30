from datasets.bipartites import BipartiteGraphDataset
from datasets.er import ErdosRenyiGraphDataset
import numpy as np

NODES    = 6
EDGE_P_S = (0, 1)
EDGE_P_M = (0.33, 0.55)
EDGE_P_D = (0.7, 0.95)

SAMPLES = 150

dataset_sparse = ErdosRenyiGraphDataset(nodes= NODES, edge_prob = EDGE_P_S)
dataset_medium = ErdosRenyiGraphDataset(nodes= NODES, edge_prob = EDGE_P_M)
dataset_dense  = ErdosRenyiGraphDataset(nodes= NODES, edge_prob = EDGE_P_D)

dataset_sparse.generate_dataset(target_total_samples=SAMPLES, seed=42, max_attempts_per_iso=250)
dataset_sparse.save_dataset(f"{NODES}N_ER_Sparse.pkl")

sparse_vecs = dataset_sparse.vectors.copy()
sum = 0
for vec in sparse_vecs:
    sum += np.sum(vec) / (NODES * (NODES- 1)//2)

dataset_medium.generate_dataset(target_total_samples=SAMPLES, seed=42, max_attempts_per_iso=250)
dataset_medium.save_dataset(f"{NODES}N_ER_Medium.pkl")

sparse_vecs = dataset_medium.vectors.copy()
sum = 0
for vec in sparse_vecs:
    sum += np.sum(vec) / (NODES * (NODES- 1)//2)


dataset_dense.generate_dataset(target_total_samples=SAMPLES, seed=42, max_attempts_per_iso=250)
dataset_dense.save_dataset(f"{NODES}N_ER_Dense.pkl")

sparse_vecs = dataset_dense.vectors.copy()
sum = 0
for vec in sparse_vecs:
    sum += np.sum(vec) / (NODES * (NODES- 1)//2)