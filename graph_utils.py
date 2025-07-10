import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AffinityPropagation
from torch_geometric.data import Data

def build_sample_graph(path, scaler=None, fit_scaler=False, depth_thresh=0.5):
    df = pd.read_csv(path).dropna()
    feature_cols = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
    features = df[feature_cols].values
    labels = df["Facies"].values - 1
    depths = df["Depth"].values

    if fit_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)

    edge_set = set()
    for i in range(len(depths) - 1):
        if abs(depths[i + 1] - depths[i]) <= depth_thresh:
            edge_set.add((i, i + 1))
            edge_set.add((i + 1, i))

    clustering = AffinityPropagation(random_state=0).fit(features)
    cluster_labels = clustering.labels_
    cluster_tensor = torch.tensor(cluster_labels, dtype=torch.long)

    for cl in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cl)[0]
        if len(indices) < 2:
            continue
        dists = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                d = np.linalg.norm(features[indices[i]] - features[indices[j]])
                dists.append(d)
        if not dists:
            continue
        threshold = np.percentile(dists, 1)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                d = np.linalg.norm(features[indices[i]] - features[indices[j]])
                if d < threshold:
                    edge_set.add((indices[i], indices[j]))
                    edge_set.add((indices[j], indices[i]))

    edge_index = torch.tensor(list(edge_set), dtype=torch.long).t().contiguous()
    x = torch.tensor(features, dtype=torch.float)
    y = torch.tensor(labels, dtype=torch.long)
    depth_tensor = torch.tensor(depths, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, y=y), scaler, depth_tensor, cluster_tensor

def build_cluster_graph(features, k_clusters=30, k_edges=4):
    N, D = features.shape
    kmeans = KMeans(n_clusters=k_clusters, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(features)
    Z = k_clusters

    Q = torch.zeros((N, Z))
    for i in range(N):
        Q[i, cluster_ids[i]] = 1

    cluster_sizes = Q.sum(dim=0, keepdim=True).T
    V = (Q.T @ torch.tensor(features, dtype=torch.float)) / (cluster_sizes + 1e-8)

    edge_list = []
    V_np = V.numpy()
    for i in range(Z):
        dists = np.linalg.norm(V_np[i] - V_np, axis=1)
        nearest = np.argsort(dists)[1:k_edges + 1]
        for j in nearest:
            edge_list.append((i, j))
            edge_list.append((j, i))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return Data(x=V, edge_index=edge_index), Q