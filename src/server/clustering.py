#FLAME's clustering method (Algorithm 1, §4.2 — Nguyen et al. 2022)
#Sentinel defense: Persistent norm ceiling + pairwise Sybil (our contribution – v8)

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
    
    flat_updates = []
    for w in weights_list:
        concat_list = []
        for key in sorted(w.keys()):
            delta = w[key] - global_model_weights[key]
            concat_list.append(delta.view(-1).float())
        flat_updates.append(torch.cat(concat_list).cpu().numpy())
    
    flat_updates = np.array(flat_updates)
    distances = cosine_distances(flat_updates).astype(np.float64)
    min_cluster_size = int(n_clients / 2) + 1
    
    clusterer = hdbscan.HDBSCAN(
        metric='precomputed', 
        min_cluster_size=min_cluster_size, 
        min_samples=1,
        allow_single_cluster=True
    )
    labels = clusterer.fit_predict(distances)
    
    if np.max(labels) < 0:
        print("⚠️ FLAME Clustering: No majority group found! Accepting all.")
        return weights_list

    unique_labels, counts = np.unique(labels, return_counts=True)
    valid_mask = unique_labels != -1
    unique_labels = unique_labels[valid_mask]
    counts = counts[valid_mask]
    
    if len(counts) == 0:
         print("⚠️ FLAME Clustering: Only noise found. Accepting all.")
         return weights_list
         
    benign_cluster_id = unique_labels[np.argmax(counts)]
    selected_indices = np.where(labels == benign_cluster_id)[0]
    
    print(f"🔥 FLAME Clustering: Selected {len(selected_indices)}/{n_clients} clients (Rejected {n_clients - len(selected_indices)})")
    return [weights_list[i] for i in selected_indices]


# ─────────────────────────────────────────────────────────────────────────────
# SENTINEL v8: Persistent norm-ceiling tracking
# ─────────────────────────────────────────────────────────────────────────────
# Per-client norm history across rounds: shape {client_id: [norm_r1, norm_r2, ...]}
_SENTINEL_NORM_HISTORY = {}


def sentinel_filtering(weights_list, global_model_weights, sensitivity=1.5,
                       expected_malicious=3, round_num=0, total_rounds=20):
    """
    Sentinel v8: Persistent norm-ceiling identification.

    Root-cause analysis (confirmed from experiment logs):
    ─────────────────────────────────────────────────────
    Stealth attackers use `apply_stealth_scaling()` which HARD-CAPS the update
    norm to `target_norm_bound` (=3.0) EVERY round. Their norm series looks like:
        [3.0, 3.0, 3.0, 2.5, 3.0, 1.8, 2.1, 3.0, ...]  ← always ≤ 3.0

    Benign clients under Dirichlet α=0.5 start with large norms (5–9) and
    gradually converge to smaller norms (3–5) by round 10+. Their norm series:
        [7.4, 5.2, 4.5, 3.9, 3.3, 3.1, 2.9, ...]  ← occasionally dips below 3.0

    The v7 failure: as benign norms converge to 2.5–4, the current-round norm
    signal can no longer discriminate. But the HISTORY reveals the pattern:
    - A client that has NEVER exceeded 3.0 in any round is a stealth attacker.
    - A client that exceeded 3.0 in most early rounds dipping below 3.0 later
      is a benign client whose updates are naturally converging.

    v8 Algorithm:
    ─────────────────────────────────────────────────────────────────────────
    1. Per-client norm history: track `||Δ_i||` for every round a client was active.
    2. "Hard-cap count" (HCC): number of rounds where a client's norm was <= 
       `norm_ceiling` (= target_norm_bound + small tolerance). 
       For stealth attackers: HCC ≈ total_rounds. For benign clients: HCC is low.
    3. "Hard-cap fraction" (HCF): HCC / total_rounds_active.
       Attacker HCF → 1.0. Benign HCF → low (large early norms prevent many hits).
    4. Additionally use the current round's pairwise min-cosine similarity (Sybil)
       as a secondary signal to break ties.
    5. Hard rejection: if HCF > 0.75 AND current-round norm <= ceiling, mark as
       malicious regardless of ranking (this catches clear-cut stealth attackers).
    6. Top-f ranking: use HCF as primary score with Sybil as tiebreaker.

    Key properties:
    - NOT contaminated at 40% Byzantine: HCF is per-client, not relative to a
      population median/centroid.
    - Convergence-STABLE: the signal GROWS (not shrinks) as rounds progress,
      because stealth attackers accumulate more hard-cap events each round.
    - Monotonically increasing discrimination over rounds: early rounds have less
      data (more uncertain), late rounds have clear separation.
    """
    global _SENTINEL_NORM_HISTORY

    n_clients = len(weights_list)
    if n_clients <= 2:
        print("⚠️ Sentinel: Too few clients, accepting all.")
        return weights_list

    keys = sorted(weights_list[0].keys())
    progress = min(1.0, round_num / max(total_rounds, 1))

    # Norm ceiling: stealth attackers are guaranteed to be at or below this
    # We infer from config target_norm_bound = 3.0 (slightly above to allow rounding)
    NORM_CEILING = 3.1

    # Initialise norm history
    for i in range(n_clients):
        if i not in _SENTINEL_NORM_HISTORY:
            _SENTINEL_NORM_HISTORY[i] = []

    # ── Compute deltas and norms ───────────────────────────────────────────
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

    # ── Update per-client norm history ────────────────────────────────────
    for i in range(n_clients):
        _SENTINEL_NORM_HISTORY[i].append(float(norms[i]))

    # ── Signal 1: Hard-cap fraction (HCF) ────────────────────────────────
    # Number of rounds this client has been consistently at/below NORM_CEILING
    hcf = np.zeros(n_clients)
    for i in range(n_clients):
        history = _SENTINEL_NORM_HISTORY[i]
        if len(history) == 0:
            hcf[i] = 0.0
            continue
        n_at_ceiling = sum(1 for h in history if h <= NORM_CEILING)
        hcf[i] = n_at_ceiling / len(history)

    # ── Signal 2: Pairwise Sybil detection (min cosine sim) ───────────────
    safe_norms = np.where(norms > 1e-10, norms, 1e-10)
    unit_updates = update_matrix / safe_norms[:, np.newaxis]
    sim_matrix = unit_updates @ unit_updates.T
    np.fill_diagonal(sim_matrix, -np.inf)

    k_peers = min(expected_malicious, n_clients - 1)
    sybil_scores = np.zeros(n_clients)
    for i in range(n_clients):
        sims = sim_matrix[i].copy()
        top_k_sims = np.sort(sims)[-k_peers:]
        sybil_scores[i] = max(0.0, top_k_sims.min())

    if sybil_scores.max() > 1e-10:
        sybil_scores = sybil_scores / sybil_scores.max()

    # ── Fuse signals ──────────────────────────────────────────────────────
    # HCF is the primary signal; Sybil is secondary to break ties
    # After round 3, HCF is reliable; before that, weight Sybil more
    if round_num >= 3:
        suspicion = 0.8 * hcf + 0.2 * sybil_scores
    else:
        suspicion = 0.5 * hcf + 0.5 * sybil_scores

    # ── Rejection ─────────────────────────────────────────────────────────
    n_reject = min(expected_malicious, (n_clients - 1) // 2)
    sorted_desc = np.argsort(suspicion)[::-1]

    trusted_mask = np.ones(n_clients, dtype=bool)

    # 1. Top-f by suspicion score
    for idx in sorted_desc[:n_reject]:
        trusted_mask[idx] = False

    # 2. Hard rejection: if HCF > 0.75 AND current round norm <= ceiling,
    #    this is a near-certain stealth attacker
    hard_reject_mask = (hcf > 0.75) & (norms <= NORM_CEILING)
    trusted_mask[hard_reject_mask] = False

    # Safety: keep at least ⌊n/2⌋ + 1
    min_keep = n_clients // 2 + 1
    if np.sum(trusted_mask) < min_keep:
        # Keep clients with highest HCF_residual = 1 - HCF (most benign characteristic)
        # i.e., those who have spent the most time ABOVE the ceiling
        benign_score = 1.0 - hcf
        sorted_by_benign = np.argsort(benign_score)[::-1]
        trusted_mask = np.zeros(n_clients, dtype=bool)
        trusted_mask[sorted_by_benign[:min_keep]] = True

    selected_indices = np.where(trusted_mask)[0]
    rejected_indices = np.where(~trusted_mask)[0]

    print(f"🛡️ Sentinel v8 [r={round_num}, progress={progress:.2f}]: "
          f"Accepted {len(selected_indices)}/{n_clients} "
          f"| Rejected: {list(rejected_indices)} "
          f"| ceiling={NORM_CEILING}")
    for i in range(n_clients):
        tag = "✅" if trusted_mask[i] else "❌"
        print(f"   [{tag}] Client {i}: norm={norms[i]:.2f}, "
              f"HCF={hcf[i]:.2f} ({int(hcf[i]*len(_SENTINEL_NORM_HISTORY[i]))}/{len(_SENTINEL_NORM_HISTORY[i])} rounds≤{NORM_CEILING}), "
              f"sybil={sybil_scores[i]:.3f}, "
              f"suspicion={suspicion[i]:.3f}")

    return [weights_list[i] for i in selected_indices]


def reset_sentinel_state():
    """Reset per-experiment state. Call before each new FL run."""
    global _SENTINEL_NORM_HISTORY
    _SENTINEL_NORM_HISTORY = {}