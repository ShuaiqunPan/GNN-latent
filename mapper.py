import numpy as np
import pandas as pd
from gtda.mapper import (CubicalCover, make_mapper_pipeline,
                         plot_static_mapper_graph)
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

n_jobs = 16  # Configure parallelism of clustering step
n_nodes = 100 # number of nodes of the graphs to take

# load the graph embeddings
data = np.load(f"embeddings-{n_nodes}nodes.pkl", allow_pickle=True)
X = data["embeddings"]
label = data["labels"]
# load the graph feature data
df = pd.read_csv(f"final_features_df_{n_nodes}nodes.csv", index_col=0)
idx = list(map(int, [x.split('_')[1].split(".")[0] for x in df.GRAPH_IDX]))
df.insert(0, 'IDX', idx)
df.set_index('IDX', inplace=True)
df.drop(columns=["GRAPH_IDX"], inplace=True)
df.sort_values(by="IDX", axis=0, inplace=True)
df["LOG_SECONDLARGESTEIGVAL"] = np.array(list(map(complex, df.LOG_SECONDLARGESTEIGVAL.values))).real
df["LOG_SMALLESTEIGVAL"] = np.array(list(map(complex, df.LOG_SMALLESTEIGVAL.values))).real
df["LOG_LARGESTEIGVAL"] = np.array(list(map(complex, df.LOG_LARGESTEIGVAL.values))).real
df.drop(columns=["LOG_SMALLESTEIGVAL", "GRAPH_ASSORTATIVITY", "GIRTH"], inplace=True)

# filter_func = umap.UMAP()
filter_func = PCA(n_components=2)
# Define cover
cover = CubicalCover(n_intervals=20, overlap_frac=0.1)
# Choose clustering algorithm – default is DBSCAN
clusterer = DBSCAN(eps=0.4)

# Initialise pipeline
pipe = make_mapper_pipeline(
    scaler=StandardScaler(),
    filter_func=filter_func,
    cover=cover,
    clusterer=clusterer,
    verbose=True,
    n_jobs=n_jobs,
)

# graph = pipe.fit_transform(X)
# node_elements = graph.vs["node_elements"]
# print(f"There are {len(node_elements)} nodes.\nThe first node consists of row indices {node_elements[0]}.")
gini_impurity = lambda y: 1 - np.sum((np.unique(y, return_counts=True)[1] / len(y)) ** 2)

fig = plot_static_mapper_graph(
    pipe,
    X,
    layout_dim=3,
    color_data=df,
    layout="fruchterman_reingold_3d",
    node_scale=40,
    node_color_statistic=np.median,
    plotly_params={"node_trace": {"marker_colorscale": "Blues"}},
)
fig.update_layout(autosize=True)
fig.show(config={"scrollZoom": True})
fig.write_html(f"tda-GNN-{n_nodes}nodes.html")
