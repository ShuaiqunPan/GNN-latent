import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the data from the pickle file
with open('/home/shuaiqun/GNN-latent/data/embeddings-20nodes.pkl', 'rb') as file:
    data = pickle.load(file)

X = data['embeddings']
labels = data['labels']

# Normalize the embeddings
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply UMAP
tsne = TSNE(n_components=2, perplexity=80, learning_rate=500, n_iter=5000, init='pca', random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Generate unique labels for each graph model, not parameter combinations
unique_labels = np.unique(labels)

# Map each unique label to a color
color_map = plt.cm.get_cmap('tab10', len(unique_labels))  # 'tab10' provides 10 distinct colors
label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}

# Create the plot
plt.figure(figsize=(12, 8))
for label, emb in zip(labels, X_tsne):
    plt.scatter(emb[0], emb[1], color=label_to_color[label], label=label if label not in plt.gca().get_legend_handles_labels()[1] else "")

plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('2D t-SNE Projection of Graph Embeddings')
plt.legend(title='Graph Model', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("simplified_projection.pdf")
plt.show()
