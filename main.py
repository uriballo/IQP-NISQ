from datasets.bipartites import BipartiteGraphDataset

NODES  = 14
EDGE_P_S = (0.1, 0.2)
EDGE_P_M = (0.33, 0.66)
EDGE_P_D = (0.7, 0.95)

SAMPLES = 1000

dataset_sparse = BipartiteGraphDataset(nodes= NODES, edge_prob = EDGE_P_S)
dataset_medium = BipartiteGraphDataset(nodes= NODES, edge_prob = EDGE_P_D)
dataset_dense  = BipartiteGraphDataset(nodes= NODES, edge_prob = EDGE_P_D)

datasets = [dataset_sparse, dataset_medium, dataset_dense]

dataset_sparse.generate_dataset(target_total_samples=SAMPLES, seed=42)
dataset_sparse.save_dataset("14N_Bipartite_Sparse.pkl")

dataset_medium.generate_dataset(target_total_samples=SAMPLES, seed=42)
dataset_medium.save_dataset("14N_Bipartite_Medium.pkl")

dataset_dense.generate_dataset(target_total_samples=SAMPLES, seed=42)
dataset_dense.save_dataset("14N_Bipartite_Dense.pkl")