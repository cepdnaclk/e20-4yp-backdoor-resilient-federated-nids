#FLAME's clustering method (Algorithm 1, §4.2 — Nguyen et al. 2022)
#Sentinel defense: Cosine-trust + median aggregation (our contribution)

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


def sentinel_filtering(weights_list, global_model_weights, sensitivity=1.5, expected_malicious=3):
    """
    Sentinel v5: Sybil-aware pairwise filtering + multi-client rejection.
    
    Two signals:
      Signal 1 — Sybil score (pairwise cosine similarity concentration):
        For each client, measure average cosine similarity to its k most
        similar peers (k = expected_malicious - 1). Backdoor clients share
        the same attack objective → their updates are highly similar to each
        other. Honest non-IID clients have diverse data → lower pairwise
        similarity. Unlike cosine-to-median (which breaks when the center
        is contaminated by 40% attackers), this detects COORDINATED updates
        regardless of where the center sits.
      
      Signal 2 — One-sided below-median norm (stealth scaling signature).
    
    Rejection: Always remove top expected_malicious clients by fused score.
    This guarantees multiple attackers are removed each round, leaving the
    trimmed median aggregation with a clean honest majority.
    """
    n_clients = len(weights_list)
    if n_clients <= 2:
        print("⚠️ Sentinel: Too few clients, accepting all.")
        return weights_list

    keys = sorted(weights_list[0].keys())

    # ── Flatten deltas & compute norms ────────────────────────────────────
    flat_updates = []
    norms = []
    for w in weights_list:
        parts = []
        for key in keys:
            delta = w[key] - global_model_weights[key]
            parts.append(delta.view(-1).float())
        flat = torch.cat(parts).cpu().numpy()
        flat_updates.append(flat)
        norms.append(np.linalg.norm(flat))

    update_matrix = np.array(flat_updates, dtype=np.float64)
    norms = np.array(norms, dtype=np.float64)

    # ── Signal 1: Sybil detection via pairwise cosine similarity ──────────
    sim_matrix = 1.0 - cosine_distances(update_matrix)
    k_peers = max(1, expected_malicious - 1)
    sybil_scores = np.zeros(n_clients)
    for i in range(n_clients):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        top_k_sims = np.sort(sims)[-k_peers:]
        sybil_scores[i] = np.mean(top_k_sims)

    med_sybil = np.median(sybil_scores)
    mad_sybil = np.median(np.abs(sybil_scores - med_sybil)) * 1.4826
    sybil_anomaly = np.maximum(
        (sybil_scores - med_sybil) / max(mad_sybil, 1e-10), 0.0
    )

    # ── Signal 2: One-sided norm test (below-median = stealth suspect) ────
    # Evasion attacks that constrain their gradients to perfectly match cosine 
    # directions (like Gradient Alignment) inevitably suffer from gradient starvation, 
    # shrinking their overall L2 norm significantly below the median honest client.
    med_norm = np.median(norms)
    mad_norm = np.median(np.abs(norms - med_norm)) * 1.4826
    norm_scores = np.maximum(
        (med_norm - norms) / max(mad_norm, 1e-10), 0.0
    )
    
    # ── Signal 3: Structural Sparsity (Kurtosis/Infinity norm) ────────────
    # Benign updates are typically sparse (high max weight relative to norm).
    # Aligned malicious updates are artificially dense (low max weight).
    # We penalize updates that have an anomalously LOW sparsity.
    normalized_updates = update_matrix / (norms[:, None] + 1e-10)
    sparsity_scores = np.max(np.abs(normalized_updates), axis=1)
    
    med_sparsity = np.median(sparsity_scores)
    mad_sparsity = np.median(np.abs(sparsity_scores - med_sparsity)) * 1.4826
    # Anomaly: lower than median sparsity
    sparsity_anomaly = np.maximum(
        (med_sparsity - sparsity_scores) / max(mad_sparsity, 1e-10), 0.0
    )

    # ── Fuse signals ──────────────────────────────────────────────────────
    # We deeply trust the norm_score to catch norm-starved evasion attacks, 
    # while sparsity catches attacks that scale their norms to match the median.
    combined = 0.4 * sybil_anomaly + 0.4 * norm_scores + 0.2 * sparsity_anomaly    # should defined as a + b + c = 1.0

    # Reject top expected_malicious clients + any beyond IQR fence
    n_reject = min(expected_malicious, (n_clients - 1) // 2)
    sorted_desc = np.argsort(combined)[::-1]

    trusted_mask = np.ones(n_clients, dtype=bool)
    for idx in sorted_desc[:n_reject]:
        trusted_mask[idx] = False

    q1, q3 = np.percentile(combined, [25, 75])
    iqr = q3 - q1
    threshold = q3 + sensitivity * iqr
    trusted_mask[combined > threshold] = False

    # Safety: keep at least ⌊n/2⌋ + 1
    min_keep = n_clients // 2 + 1
    if np.sum(trusted_mask) < min_keep:
        sorted_asc = np.argsort(combined)
        trusted_mask = np.zeros(n_clients, dtype=bool)
        trusted_mask[sorted_asc[:min_keep]] = True

    selected_indices = np.where(trusted_mask)[0]
    rejected_indices = np.where(~trusted_mask)[0]

    print(f"🛡️ Sentinel: Accepted {len(selected_indices)}/{n_clients} "
          f"| Rejected clients: {list(rejected_indices)}")
    for i in range(n_clients):
        tag = "✅" if trusted_mask[i] else "❌"
        print(f"   [{tag}] Client {i}: sybil={sybil_anomaly[i]:.3f}, "
              f"norm={norm_scores[i]:.3f}, sparsity={sparsity_anomaly[i]:.3f}, "
              f"combined={combined[i]:.3f} (thr={threshold:.3f})")

    accepted_weights = [weights_list[i] for i in selected_indices]
    
    return accepted_weights

def sentinel_v2_filtering(weights_list, global_model_weights, sensitivity=1.5, expected_malicious=3):
    """
    Sentinel v2: Defends against all implemented attacks in project repo:
    - Backdoors / Label Flipping (Sybil Cosine Similarity)
    - Stealth Attacks (Abnormally small L2 norms)
    - Aggressive Evading/Model Replacement (Abnormally large L2 norms)
    
    Uses two-sided norm scoring to catch BOTH aggressive and stealth scalers.
    """
    n_clients = len(weights_list)
    if n_clients <= 2:
        print("⚠️ Sentinel v2: Too few clients, accepting all.")
        return weights_list

    keys = sorted(weights_list[0].keys())

    # ── Flatten deltas & compute norms ────────────────────────────────────
    flat_updates = []
    norms = []
    for w in weights_list:
        parts = []
        for key in keys:
            delta = w[key] - global_model_weights[key]
            parts.append(delta.view(-1).float())
        flat = torch.cat(parts).cpu().numpy()
        flat_updates.append(flat)
        norms.append(np.linalg.norm(flat))

    update_matrix = np.array(flat_updates, dtype=np.float64)
    norms = np.array(norms, dtype=np.float64)

    # ── Signal 1: Sybil detection via pairwise cosine similarity ──────────
    sim_matrix = 1.0 - cosine_distances(update_matrix)
    k_peers = max(1, expected_malicious - 1)
    sybil_scores = np.zeros(n_clients)
    for i in range(n_clients):
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf  # exclude self
        top_k_sims = np.sort(sims)[-k_peers:]
        sybil_scores[i] = np.mean(top_k_sims)

    med_sybil = np.median(sybil_scores)
    mad_sybil = np.median(np.abs(sybil_scores - med_sybil)) * 1.4826
    sybil_anomaly = np.maximum(
        (sybil_scores - med_sybil) / max(mad_sybil, 1e-10), 0.0
    )

    # ── Signal 2: TWO-SIDED norm test (Stealth & Aggressive signature) ────
    med_norm = np.median(norms)
    mad_norm = np.median(np.abs(norms - med_norm)) * 1.4826
    
    # 🌟 DIFFERENCE in V2 from V1: Absolute distance instead of max(median-norm, 0)
    # This penalizes both tiny stealth updates AND massive boosted updates
    norm_scores = np.abs(norms - med_norm) / max(mad_norm, 1e-10)
    
    # ── Signal 3: Structural Sparsity (Kurtosis/Infinity norm) ────────────
    normalized_updates = update_matrix / (norms[:, None] + 1e-10)
    sparsity_scores = np.max(np.abs(normalized_updates), axis=1)
    
    med_sparsity = np.median(sparsity_scores)
    mad_sparsity = np.median(np.abs(sparsity_scores - med_sparsity)) * 1.4826
    
    # Penalty for anomalously low AND high sparsity
    sparsity_anomaly = np.abs(sparsity_scores - med_sparsity) / max(mad_sparsity, 1e-10)

    # ── Fuse signals ──────────────────────────────────────────────────────
    combined = 0.4 * sybil_anomaly + 0.4 * norm_scores + 0.2 * sparsity_anomaly

    # Reject top expected_malicious clients + any beyond IQR fence
    n_reject = min(expected_malicious, (n_clients - 1) // 2)
    sorted_desc = np.argsort(combined)[::-1]

    trusted_mask = np.ones(n_clients, dtype=bool)
    for idx in sorted_desc[:n_reject]:
        trusted_mask[idx] = False

    q1, q3 = np.percentile(combined, [25, 75])
    iqr = q3 - q1
    threshold = q3 + sensitivity * iqr
    trusted_mask[combined > threshold] = False

    # Safety: keep at least ⌊n/2⌋ + 1
    min_keep = n_clients // 2 + 1
    if np.sum(trusted_mask) < min_keep:
        sorted_asc = np.argsort(combined)
        trusted_mask = np.zeros(n_clients, dtype=bool)
        trusted_mask[sorted_asc[:min_keep]] = True

    selected_indices = np.where(trusted_mask)[0]
    rejected_indices = np.where(~trusted_mask)[0]

    print(f"🛡️ Sentinel v2: Accepted {len(selected_indices)}/{n_clients} "
          f"| Rejected clients: {list(rejected_indices)}")
    for i in range(n_clients):
        tag = "✅" if trusted_mask[i] else "❌"
        print(f"   [{tag}] Client {i}: sybil={sybil_anomaly[i]:.3f}, "
              f"norm={norm_scores[i]:.3f}, sparsity={sparsity_anomaly[i]:.3f}, "
              f"combined={combined[i]:.3f} (thr={threshold:.3f})")

    return [weights_list[i] for i in selected_indices]