import networkx as nx
import matplotlib.pyplot as plt
import os

os.chdir('/Users/kadirkesgin/Documents/akademikcalismalar/2026/mart2026/education_truth')

G = nx.DiGraph()

nodes = {
    'Socioeconomic Status (SES)\n[Immutable]': (0.5, 0.8),
    'Age\n[Immutable]': (0.2, 0.8),
    'Free Time\n[Actionable]': (0.2, 0.5),
    'Go Out\n[Actionable]': (0.4, 0.5),
    'Study Time\n[Actionable]': (0.6, 0.5),
    'Absences\n[Actionable]': (0.8, 0.5),
    'Probability of Passing (G3)\n[Target]': (0.5, 0.2)
}

G.add_nodes_from(nodes.keys())

edges = [
    ('Socioeconomic Status (SES)\n[Immutable]', 'Study Time\n[Actionable]'),
    ('Socioeconomic Status (SES)\n[Immutable]', 'Absences\n[Actionable]'),
    ('Socioeconomic Status (SES)\n[Immutable]', 'Free Time\n[Actionable]'),
    ('Age\n[Immutable]', 'Go Out\n[Actionable]'),
    ('Free Time\n[Actionable]', 'Go Out\n[Actionable]'),
    ('Study Time\n[Actionable]', 'Probability of Passing (G3)\n[Target]'),
    ('Absences\n[Actionable]', 'Probability of Passing (G3)\n[Target]'),
    ('Free Time\n[Actionable]', 'Probability of Passing (G3)\n[Target]'),
    ('Go Out\n[Actionable]', 'Probability of Passing (G3)\n[Target]')
]

G.add_edges_from(edges)

color_map = []
for node in G:
    if 'Immutable' in node:
        color_map.append('#E2E2E2') # Grey
    elif 'Actionable' in node:
        color_map.append('#A8D8EA') # Light Blue
    else:
        color_map.append('#C5E1A5') # Light Green

fig, ax = plt.subplots(figsize=(10, 6.5))

nx.draw(G, pos=nodes, with_labels=True, node_color=color_map, node_size=5000, 
        font_size=10, font_weight='bold', font_family='sans-serif',
        arrows=True, arrowstyle='-|>', arrowsize=22, edge_color='#666666', ax=ax)

# Remove the title for LaTeX!
# plt.title(...)

plt.tight_layout()
plt.margins(0.1)
plt.savefig('sci_fig0_causal_dag.png', dpi=300, bbox_inches='tight')
print("Causal DAG başarıyla çizildi.")
