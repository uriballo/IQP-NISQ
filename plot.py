import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import Patch

# --- Setup for Academic Style Plots ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.titlesize": 18
})

# --- Data Preparation ---
data = {
    'Graph_ID': [
        'BP8S', 'BP8M', 'BP8D', 'ER8S', 'ER8M', 'ER8D', 'BP10S', 'BP10M', 'BP10D',
        'ER10S', 'ER10M', 'ER10D', 'BP14S', 'BP14M', 'BP14D', 'ER14S', 'ER14M', 'ER14D',
        'BP18S', 'BP18M', 'BP18D', 'ER18S', 'ER18M', 'ER18D'
    ],
    'Bipartite_GT': [
        100, 100, 100, 51.5, 0.5, 0.0, 100, 100, 100, 33.8, 0.2, 0.0, 100, 100,
        100, 27.9, 0.0, 0.0, 100, 100, 100, 17.7, 0.0, 0.0
    ],
    'Bipartite_NISQ': [
        70.70, 15.23, 45.51, 42.77, 2.93, 0.0, 65.23, 43.16, 0.39, 38.28, 0.0,
        0.0, 29.3, 1.95, 0, 78.32, 0.0, 0.0, 25.78, 0.2, 0.0, 3.12, 0.0, 0.0
    ],
    'Bipartite_Sim': [
        79.69, 59.18, 49.02, 55.08, 3.12, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan
    ],
    'Edge_Prob_GT': [
        0.3947, 0.5053, 0.5443, 0.2207, 0.4420, 0.7620, 0.3187, 0.4152, 0.6246,
        0.1919, 0.4380, 0.7884, 0.1935, 0.3871, 0.6921, 0.1575, 0.4487, 0.8145,
        0.1411, 0.3716, 0.6982, 0.1497, 0.4428, 0.8255
    ],
    'Edge_Prob_NISQ': [
        0.3865, 0.6101, 0.5232, 0.2502, 0.4445, 0.7521, 0.3267, 0.4358, 0.7141,
        0.2007, 0.4380, 0.7429, 0.2717, 0.4886, 0.8868, 0.0752, 0.4463, 0.7477,
        0.1898, 0.4580, 0.7835, 0.1481, 0.4313, 0.7610
    ],
    'Edge_Prob_Sim': [
        0.3872, 0.4980, 0.5447, 0.2294, 0.4464, 0.7542, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        np.nan, np.nan, np.nan, np.nan
    ]
}
df = pd.DataFrame(data)
df['Nodes'] = df['Graph_ID'].str.extract('(\\d+)').astype(int)
df['Graph_Type'] = df['Graph_ID'].str.extract('([A-Z]+)')
df['Density_Code'] = df['Graph_ID'].str.extract('([SMD])')
df['Edge_Error_NISQ'] = abs(df['Edge_Prob_NISQ'] - df['Edge_Prob_GT'])
df['Edge_Error_Sim'] = abs(df['Edge_Prob_Sim'] - df['Edge_Prob_GT'])

# --- Common Plotting Variables ---
nodes = sorted(df['Nodes'].unique())
qubits = [int(n * (n - 1) / 2) for n in nodes]
xtick_labels = [f"{n}\n({q}Q)" for n, q in zip(nodes, qubits)]
markers = {'S': 'o', 'M': 's', 'D': '^'}
density_names = {'S': 'Sparse', 'M': 'Medium', 'D': 'Dense'}

# --- Plot 1: Bipartite Percentage Scaling ---
fig1, ax1 = plt.subplots(figsize=(12, 8))
colors_bipartite = {'NISQ': '#0072B2', 'Sim.': '#E69F00', 'Baseline': '#808080'}

for density_code in markers.keys():
    bp_data = df[(df['Density_Code'] == density_code) & (df['Graph_Type'] == 'BP')]
    ax1.plot(bp_data['Nodes'], bp_data['Bipartite_NISQ'], marker=markers[density_code], linestyle='-', color=colors_bipartite['NISQ'])
    ax1.plot(bp_data['Nodes'], bp_data['Bipartite_Sim'], marker=markers[density_code], linestyle='-', color=colors_bipartite['Sim.'])

    er_data = df[(df['Density_Code'] == density_code) & (df['Graph_Type'] == 'ER')]
    ax1.plot(er_data['Nodes'], er_data['Bipartite_GT'], marker=markers[density_code], linestyle='--', color=colors_bipartite['Baseline'])

legend_elements_1 = [
    mlines.Line2D([0], [0], color=colors_bipartite['NISQ'], lw=2, label='NISQ'),
    mlines.Line2D([0], [0], color=colors_bipartite['Sim.'], lw=2, label='Sim.'),
    mlines.Line2D([0], [0], color=colors_bipartite['Baseline'], lw=2, linestyle='--', label='Baseline'),
    Patch(facecolor='none', edgecolor='none', label=''),
    mlines.Line2D([0], [0], marker='o', color='black', label='Sparse', linestyle='None', markersize=8),
    mlines.Line2D([0], [0], marker='s', color='black', label='Medium', linestyle='None', markersize=8),
    mlines.Line2D([0], [0], marker='^', color='black', label='Dense', linestyle='None', markersize=8)
]
ax1.legend(handles=legend_elements_1, title='Legend', loc='upper right')

ax1.set_xlabel('Nodes (Qubits)')
ax1.set_ylabel('Bipartite %')
ax1.set_title('Performance Scaling Across All Connectivities')
ax1.set_xticks(nodes)
ax1.set_xticklabels(xtick_labels)
ax1.grid(True, linestyle='--', alpha=0.7)
fig1.savefig('figure_bipartite_scaling_final.png', bbox_inches='tight', dpi=300)
plt.close(fig1)

# --- Plot 2: Edge Probability Error (Merged) ---
fig2, ax2 = plt.subplots(figsize=(12, 8))
colors_edge = {'NISQ': '#0072B2', 'Sim': '#E69F00'}
linestyles_edge = {'BP': '-', 'ER': '--'}

for graph_type in ['BP', 'ER']:
    for density_code in markers.keys():
        subset = df[(df['Density_Code'] == density_code) & (df['Graph_Type'] == graph_type)]
        ax2.plot(subset['Nodes'], subset['Edge_Error_NISQ'], marker=markers[density_code], linestyle=linestyles_edge[graph_type], color=colors_edge['NISQ'])
        ax2.plot(subset['Nodes'], subset['Edge_Error_Sim'], marker=markers[density_code], linestyle=linestyles_edge[graph_type], color=colors_edge['Sim'])

# Create legend
model_handles = [mlines.Line2D([0], [0], color=c, lw=2, label=l) for l, c in colors_edge.items()]
graphtype_handles = [mlines.Line2D([0], [0], color='black', linestyle=ls, label=l) for l, ls in linestyles_edge.items()]
density_handles = [mlines.Line2D([0], [0], color='black', marker=m, linestyle='None', label=density_names[c], markersize=8) for c, m in markers.items()]

ax2.legend(handles=model_handles + graphtype_handles + density_handles, title="Legend")

ax2.set_xlabel('Nodes (Qubits)')
ax2.set_ylabel('Edge Probability Absolute Error')
ax2.set_title('Edge Probability Error Scaling')
ax2.set_xticks(nodes)
ax2.set_xticklabels(xtick_labels)
ax2.grid(True, linestyle='--', alpha=0.7)
fig2.savefig('figure_edge_prob_error_merged.png', bbox_inches='tight', dpi=300)
plt.close(fig2)