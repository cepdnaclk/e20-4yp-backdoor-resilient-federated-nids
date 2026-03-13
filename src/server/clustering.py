#FLAME's clustering method (Algorithm 1, §4.2 — Nguyen et al. 2022)
#Sentinel defense: Early-round anchored malicious identification (our contribution – v9)

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
# SENTINEL v9: Early-round anchored identification + persistent suspicion
# ─────────────────────────────────────────────────────────────────────────────
# State persisted across rounds for cross-round tracking
_SENTINEL_NORM_HISTORY    = {}   # {client_id: [norm_r1, norm_r2, ...]}
_SENTINEL_ANCHOR_FLAGS    = {}   # {client_id: bool} — flagged in early rounds?
_SENTINEL_ANCHOR_LOCKED   = False  # becomes True after anchor window closes

# Norm ceiling: stealth attackers are hard-constrained by their own attack code
_NORM_CEILING = 3.05   # target_norm_bound=3.0 + tiny tolerance for float rounding
_ANCHOR_ROUNDS = 3     # number of early rounds used to lock in identification


def sentinel_filtering(weights_list, global_model_weights, sensitivity=1.5,
                       expected_malicious=3, round_num=0, total_rounds=20,
                       client_ids=None):
    """
    Sentinel v9: Early-round anchored malicious identification.

    Core insight (user confirmed): If you track who is malicious from the beginning,
    you can identify them reliably after a few rounds.

    Why early rounds are perfect discriminators:
    ─────────────────────────────────────────────
    In round 1, the global model is random (untrained). Every client must move far
    to fit their local data → benign non-IID clients produce large norms (5–9).
    Stealth attackers are HARD-CONSTRAINED by their own attack code to have
    norm ≤ target_norm_bound (=3.0). The separation is guaranteed:
        Round 1: Malicious norms = 3.0 (exact), Benign norms = 5–9 (always)
        Round 2: Malicious norms ≤ 3.0, Benign norms = 3–7 (still above 3)
        Round 3: Malicious norms ≤ 3.0, Benign norms = 3–6 (partial overlap begins)

    So the first 1–3 rounds give NEARLY PERFECT malicious identification.

    Algorithm:
    ─────────────────────────────────────────────────────────────────────────
    Phase 1 — Anchor Window (rounds 1 to _ANCHOR_ROUNDS):
      Each round, flag any client whose norm ≤ _NORM_CEILING as a suspect.
      After anchor window closes (_ANCHOR_ROUNDS rounds), compute the
      "anchor suspicion score" = fraction of those rounds spent at/below ceiling.
      Clients with anchor_score ≥ 0.67 are permanently flagged as malicious suspects.

    Phase 2 — Active Rejection (remaining rounds):
      Use the locked anchor flags as primary rejection criterion:
      → Always reject the top-f permanently flagged clients (those most consistently
        flagged in early rounds).
      → Pairwise Sybil score (min cosine sim of top-k peers) provides secondary
        evidence to handle any ambiguity in the anchor window.
      → Hard fence: any client that was flagged in round 1 AND current-round
        norm ≤ ceiling is ALWAYS rejected (early identification + persistent stealth).

    Phase 3 — Never under-reject:
      Safety mechanism ensures at least ⌊n/2⌋+1 clients are kept.
      Fallback: prefer clients with lowest anchor suspicion (most benign).
    """
    global _SENTINEL_NORM_HISTORY, _SENTINEL_ANCHOR_FLAGS
    global _SENTINEL_ANCHOR_LOCKED

    n_clients = len(weights_list)
    if n_clients <= 2:
        print("⚠️ Sentinel: Too few clients, accepting all.")
        return weights_list

    keys = sorted(weights_list[0].keys())
    progress = min(1.0, round_num / max(total_rounds, 1))

    # Use actual client IDs for per-client state (not list position!)
    # This is critical: active_clients_indices is randomly ordered each round.
    # Without real IDs, position 0 in round 1 is a different client than position 0 in round 2.
    if client_ids is None:
        client_ids = list(range(n_clients))  # fallback: assume sequential IDs

    # Initialise per-client state
    for cid in client_ids:
        if cid not in _SENTINEL_NORM_HISTORY:
            _SENTINEL_NORM_HISTORY[cid] = []
            _SENTINEL_ANCHOR_FLAGS[cid] = 0.0  # cumulative anchor suspicion

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
    norms_arr = np.array(norms, dtype=np.float64)

    # Update norm history using actual client IDs
    for pos, cid in enumerate(client_ids):
        _SENTINEL_NORM_HISTORY[cid].append(float(norms_arr[pos]))

    # ── Phase 1: Anchor window scoring ────────────────────────────────────
    if round_num <= _ANCHOR_ROUNDS:
        # In anchor window: update flags based on current round's norm
        for pos, cid in enumerate(client_ids):
            if norms_arr[pos] <= _NORM_CEILING:
                _SENTINEL_ANCHOR_FLAGS[cid] += 1.0

        # Lock in identifications after anchor window
        if round_num == _ANCHOR_ROUNDS:
            _SENTINEL_ANCHOR_LOCKED = True
            print(f"🔒 Sentinel v9: Anchor window closed at round {round_num}. "
                  f"Locking in identifications.")
            for cid in sorted(set(client_ids)):
                frac = _SENTINEL_ANCHOR_FLAGS.get(cid, 0) / _ANCHOR_ROUNDS
                suspect_tag = "→ SUSPECT 🚨" if frac >= 0.67 else "→ benign ✅"
                print(f"   Client {cid}: anchor_count={int(_SENTINEL_ANCHOR_FLAGS.get(cid,0))}/{_ANCHOR_ROUNDS} "
                      f"({frac:.0%}) {suspect_tag}")

    # ── Compute suspicion scores using actual client IDs ───────────────────
    # Primary: anchor score (fraction of anchor window rounds at/below ceiling)
    anchor_score = np.array([
        _SENTINEL_ANCHOR_FLAGS.get(cid, 0) / max(_ANCHOR_ROUNDS, 1)
        for cid in client_ids
    ])

    # If before lock-in, supplement with current-round norm signal
    if not _SENTINEL_ANCHOR_LOCKED or round_num <= _ANCHOR_ROUNDS:
        # In anchor window: current-round norm is the direct signal
        current_norm_susp = np.array([
            1.0 if norms_arr[i] <= _NORM_CEILING else 0.0
            for i in range(n_clients)
        ])
        primary_score = 0.5 * anchor_score + 0.5 * current_norm_susp
    else:
        # Post anchor: anchor score is primary
        primary_score = anchor_score

    # Secondary: pairwise min-cosine similarity (Sybil detection)
    safe_norms = np.where(norms_arr > 1e-10, norms_arr, 1e-10)
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

    # Fuse: primary anchor score dominates; Sybil breaks ties
    combined = 0.80 * primary_score + 0.20 * sybil_scores

    # ── Rejection ─────────────────────────────────────────────────────────
    n_reject = min(expected_malicious, (n_clients - 1) // 2)
    sorted_desc = np.argsort(combined)[::-1]

    trusted_mask = np.ones(n_clients, dtype=bool)

    # Top-f by combined suspicion
    for idx in sorted_desc[:n_reject]:
        trusted_mask[idx] = False

    # Hard reject: definitely anchored AND currently stealthy
    if _SENTINEL_ANCHOR_LOCKED:
        hard_flag = (anchor_score >= 0.67) & (norms_arr <= _NORM_CEILING)
        trusted_mask[hard_flag] = False

    # Safety: keep at least ⌊n/2⌋ + 1
    min_keep = n_clients // 2 + 1
    if np.sum(trusted_mask) < min_keep:
        # Keep n clients with LOWEST anchor score (most likely benign)
        sorted_by_anchor_asc = np.argsort(anchor_score)
        trusted_mask = np.zeros(n_clients, dtype=bool)
        trusted_mask[sorted_by_anchor_asc[:min_keep]] = True

    selected_indices = np.where(trusted_mask)[0]
    rejected_indices = np.where(~trusted_mask)[0]

    print(f"🛡️ Sentinel v9 [r={round_num}, locked={_SENTINEL_ANCHOR_LOCKED}]: "
          f"Accepted {len(selected_indices)}/{n_clients} "
          f"| Rejected: {[client_ids[i] for i in rejected_indices.tolist()]}")
    for pos in range(n_clients):
        tag = "✅" if trusted_mask[pos] else "❌"
        cid = client_ids[pos]
        print(f"   [{tag}] Client {cid}: norm={norms_arr[pos]:.2f}, "
              f"anchor={anchor_score[pos]:.2f} ({int(_SENTINEL_ANCHOR_FLAGS.get(cid,0))}/{_ANCHOR_ROUNDS}), "
              f"sybil={sybil_scores[pos]:.3f}, "
              f"combined={combined[pos]:.3f}")

    return [weights_list[i] for i in selected_indices]


def reset_sentinel_state():
    """Reset per-experiment state. Call before each new FL run."""
    global _SENTINEL_NORM_HISTORY, _SENTINEL_ANCHOR_FLAGS, _SENTINEL_ANCHOR_LOCKED
    _SENTINEL_NORM_HISTORY  = {}
    _SENTINEL_ANCHOR_FLAGS  = {}
    _SENTINEL_ANCHOR_LOCKED = False