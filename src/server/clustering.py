#FLAME's clustering method (Algorithm 1, §4.2 — Nguyen et al. 2022)

import torch
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances

def flame_clustering(weights_list, global_model_weights):
    """
    FLAME Step 1: Cluster model *updates* (deltas) using HDBSCAN on cosine distance.
    Per Algorithm 1 in the paper, clustering must be performed on Δᵢ = wᵢ − G^(t-1),
    NOT on the raw weights wᵢ. This ensures the anomalous *direction of change*
    from a backdoor update is detectable regardless of how close the full weights appear.
    """
    n_clients = len(weights_list)
    
    # 1. Flatten model UPDATES (deltas) into vectors
    flat_updates = []
    for w in weights_list:
        concat_list = []
        for key in sorted(w.keys()):
            # Compute delta: wᵢ − G^(t-1)  [Paper Algorithm 1, line 4]
            delta = w[key] - global_model_weights[key]
            concat_list.append(delta.view(-1).float())
        flat_updates.append(torch.cat(concat_list).cpu().numpy())
    
    flat_updates = np.array(flat_updates)

    # 2. Calculate Pairwise Cosine Distances
    # 🔧 FIX: HDBSCAN requires float64 (double), but PyTorch gave float32.
    distances = cosine_distances(flat_updates).astype(np.float64)

    # 3. Apply HDBSCAN
    # FLAME uses min_cluster_size > n/2 to find the "Majority" (Honest) group[cite: 274].
    min_cluster_size = int(n_clients / 2) + 1
    
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed', 
        min_cluster_size=min_cluster_size, 
        min_samples=1,
        allow_single_cluster=True
    )
    
    labels = clusterer.fit_predict(distances)
    
    # 4. Select the "Benign" Cluster
    # Labels: -1 is noise (malicious), 0+ are clusters.
    
    # Check if we found ANY cluster (if everything is noise, fallback)
    if np.max(labels) < 0:
        print("⚠️ FLAME Clustering: No majority group found! Accepting all.")
        return weights_list

    # Find the cluster with the most clients (The Majority)
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Filter out the noise label (-1) from consideration
    valid_mask = unique_labels != -1
    unique_labels = unique_labels[valid_mask]
    counts = counts[valid_mask]
    
    if len(counts) == 0:
         print("⚠️ FLAME Clustering: Only noise found. Accepting all.")
         return weights_list
         
    benign_cluster_id = unique_labels[np.argmax(counts)]
    
    # 5. Filter the weights
    selected_indices = np.where(labels == benign_cluster_id)[0]
    
    print(f"🔥 FLAME Clustering: Selected {len(selected_indices)}/{n_clients} clients (Rejected {n_clients - len(selected_indices)})")
    
    accepted_weights = [weights_list[i] for i in selected_indices]
    
    return accepted_weights