import torch
import copy
import numpy as np

def fed_avg(client_updates):
    """
    Standard Weighted Average
    Args:
        client_updates: List of (weights, n_samples, loss)
    """
    total_samples = sum([update[1] for update in client_updates])
    
    # Initialize with the structure of the first client's weights
    new_weights = copy.deepcopy(client_updates[0][0])
    
    # Zero everything out first
    for key in new_weights.keys():
        new_weights[key] = torch.zeros_like(new_weights[key])
        
    # Weighted Sum
    for weights, n_samples, _ in client_updates:
        weight_factor = n_samples / total_samples
        for key in weights.keys():
            new_weights[key] += weights[key] * weight_factor
            
    return new_weights


def fed_median(weights_list):
    """
    Coordinate-wise Median: Takes the median of each parameter across clients.
    Effective against Model Replacement (Outliers).
    """
    # Initialize structure
    new_weights = copy.deepcopy(weights_list[0])
    keys = new_weights.keys()
    
    for key in keys:
        # Stack all clients' tensors for this layer: shape (n_clients, layer_shape)
        stacked_layers = torch.stack([w[key] for w in weights_list], dim=0)
        
        # Take Median across the client dimension (dim=0)
        new_weights[key] = torch.median(stacked_layers, dim=0).values
        
    return new_weights


def fed_trimmed_mean(weights_list, beta=0.1):
    """
    Trimmed Mean: Removes the top and bottom beta% of values before averaging.
    """
    new_weights = copy.deepcopy(weights_list[0])
    keys = new_weights.keys()
    n_clients = len(weights_list)
    to_trim = int(n_clients * beta)
    
    # Safety Check: If trimming removes everyone, fall back to median
    if n_clients - 2 * to_trim <= 0:
        return fed_median(weights_list)

    for key in keys:
        stacked_layers = torch.stack([w[key] for w in weights_list], dim=0)
        # Sort along client dimension
        sorted_layers, _ = torch.sort(stacked_layers, dim=0)
        
        # Slice out the middle
        kept_layers = sorted_layers[to_trim : n_clients - to_trim]
        
        # Average the rest
        new_weights[key] = torch.mean(kept_layers, dim=0)
        
    return new_weights


def fed_krum(weights_list, n_malicious=1):
    """
    Krum: Rejects outliers based on Euclidean distance.
    Selects the single update that is 'closest' to the majority of neighbors.
    Args:
        n_malicious (int): The expected number of attackers (f).
                           Krum requires n > 2f + 2.
    """
    n_clients = len(weights_list)
    
    # Krum Rule: k = n - f - 2
    # This is the number of neighbors we compare against
    k = max(1, n_clients - n_malicious - 2) 
    
    # 1. Flatten all weights into vectors for distance calculation
    flattened_updates = []
    for w in weights_list:
        flat = torch.cat([t.view(-1) for t in w.values()])
        flattened_updates.append(flat)
    
    scores = []
    # 2. Calculate pairwise distances
    for i in range(n_clients):
        dists = []
        for j in range(n_clients):
            if i == j: continue
            # Euclidean Distance
            d = torch.norm(flattened_updates[i] - flattened_updates[j]).item()
            dists.append(d)
        
        # 3. Krum Score: Sum of distances to k nearest neighbors
        dists.sort()
        score = sum(dists[:k])
        scores.append(score)
    
    # 4. The winner is the client with the LOWEST score (most central)
    best_client_idx = np.argmin(scores)
    
    return weights_list[best_client_idx]


def fed_multi_krum(weights_list, f, m=None):
    """
    Multi-Krum: Selects 'm' best clients and averages them.
    Args:
        f (int): Number of estimated attackers.
        m (int): Number of clients to aggregate (Default: n - f)
    """
    n_clients = len(weights_list)
    k = max(1, n_clients - f - 2)
    
    # Default m: Select all valid clients we think are honest
    if m is None:
        m = n_clients - f

    # 1. Flatten updates (Same as Krum)
    flat_updates = []
    for w in weights_list:
        concat_list = []
        for key in sorted(w.keys()):
            concat_list.append(w[key].view(-1).float())
        flat_updates.append(torch.cat(concat_list))
        
    # 2. Pairwise Distances (Same as Krum)
    scores = []
    for i in range(n_clients):
        dists = []
        for j in range(n_clients):
            if i == j: continue
            d = torch.norm(flat_updates[i] - flat_updates[j]).item()
            dists.append(d)
        dists.sort()
        score = sum(dists[:k])
        scores.append(score)
        
    # 3. SELECT TOP 'm' (Lowest scores) - This is the "Multi" part
    # Returns indices of the smallest 'm' scores
    best_indices = np.argsort(scores)[:m]
    
    print(f"   -> Multi-Krum selected {len(best_indices)} clients: {best_indices}")

    # 4. AVERAGE the selected models
    selected_weights = [weights_list[i] for i in best_indices]
    
    # Use your existing fed_avg logic on just these selected weights
    return fed_avg([(w, 1, 0) for w in selected_weights]) 
    # Note: passing 1 as n_samples assumes equal weighting for selected clients


def sentinel_aggregate(weights_list, global_model_weights, privacy_cfg=None, expected_malicious=3):
    """
    Sentinel v6 aggregation: Norm-weighted trimmed aggregation + DP Noise.

    Key insight (v5): Norm-based clipping HURTS against stealth-scaled attacks.
    Stealth attackers have small norms (≤ bound), so clipping with
    median-of-norms shrinks honest updates (large norms) MORE than malicious
    ones, amplifying the attack signal relative to honest signal.

    Key insight (v6 - NEW): Under non-IID Dirichlet α=0.5, benign clients
    have naturally large update norms (their local data is very skewed, so
    they must move far from the global model). Stealth attackers deliberately
    cap their norms to ≤3. After filtering, we can exploit this residual size
    difference by weighting each surviving delta by its global L2 norm.

    The trimmed-then-weighted pipeline:
      1. Compute deltas & norms for all (post-filter) clients.
      2. For each parameter coordinate, sort values and trim the top-f and
         bottom-f extremes (f = expected_malicious). This removes outliers.
      3. Instead of taking the median of the trimmed set, take a
         norm-weighted mean: clients with larger norms vote more.
         Since benign non-IID clients have larger norms than residual
         stealthy ones, the aggregated direction is explicitly biased toward
         the honest majority — making FedAvg's incidental robustness an
         intentional, principled design choice.
      4. Add DP Gaussian noise calibrated to the median of norms (S).

    This is more robust than pure median because:
    - Median is shift-equivariant: a stealthy update at the boundary of the
      trimmed window still skews the median. Weighting by norm dilutes it.
    - Large-norm benign updates are exactly the clients that provide the
      most useful task-specific gradient signal.
    """
    # 1. Compute deltas and L2 norms
    norms = []
    diffs = []
    for w in weights_list:
        diff = {}
        total_norm_sq = 0.0
        for key in w.keys():
            d = w[key] - global_model_weights[key]
            diff[key] = d
            total_norm_sq += torch.sum(d ** 2).item()
        norms.append(total_norm_sq ** 0.5)
        diffs.append(diff)

    n = len(diffs)
    norms_arr = np.array(norms, dtype=np.float64)
    trim = min(expected_malicious, (n - 1) // 2)  # keep at least 1 value

    # 2. Norm-weighted trimmed aggregation of deltas
    new_weights = copy.deepcopy(global_model_weights)
    for key in new_weights.keys():
        # Stack all surviving client deltas: shape (n, *param_shape)
        stacked = torch.stack([d[key] for d in diffs], dim=0)  # (n, ...)
        param_shape = stacked.shape[1:]

        if trim > 0 and n - 2 * trim >= 1:
            # Sort along client dimension, get permutation indices
            sorted_vals, sort_idx = torch.sort(stacked.view(n, -1), dim=0)
            kept_vals = sorted_vals[trim : n - trim]          # (n-2f, numel)
            kept_idx = sort_idx[trim : n - trim]              # which client

            # Build norm weights for the kept entries at each coordinate
            # kept_idx[j, c] = original client index at position j for coord c
            kept_norms = torch.tensor(
                norms_arr[kept_idx.cpu().numpy()], dtype=torch.float32
            ).to(stacked.device)                               # (n-2f, numel)

            # Normalise across the kept dimension
            norm_sum = kept_norms.sum(dim=0, keepdim=True).clamp(min=1e-10)
            w_coord   = kept_norms / norm_sum                  # (n-2f, numel)

            # Weighted mean of kept values
            agg_flat = (kept_vals.float() * w_coord).sum(dim=0)  # (numel,)
            agg_delta = agg_flat.view(param_shape)
        else:
            # Fallback: coordinate-wise median (n too small to trim)
            agg_delta = torch.median(stacked, dim=0).values

        new_weights[key] = new_weights[key] + agg_delta

    # 3. DP Noise (calibrated to median of norms)
    if privacy_cfg:
        S = float(np.median(norms_arr))
        epsilon = privacy_cfg.get('epsilon', 300.0)
        delta   = privacy_cfg.get('delta', 1e-5)
        lambda_val = (1.0 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        sigma   = lambda_val * S
        print(f"   🤫 Adding DP Noise: sigma={sigma:.6f} (epsilon={epsilon})")
        for key in new_weights.keys():
            noise = torch.normal(
                mean=0.0, std=sigma, size=new_weights[key].shape
            ).to(new_weights[key].device)
            new_weights[key] += noise

    return new_weights


def fed_adaptive_clipping(weights_list, global_model_weights, privacy_cfg=None):
    """
    FLAME Part 2 & 3: Adaptive Clipping + Adaptive Noising (DP).
    
    Args:
        weights_list: List of client updates (state_dicts)
        global_model_weights: The previous global model (state_dict)
        privacy_cfg: Dict containing 'epsilon' and 'delta' from config
    """
    # 1. Calculate L2 Norms of updates (Difference from Global)
    norms = []
    diffs = []
    
    for w in weights_list:
        diff = {}
        total_norm_sq = 0.0
        for key in w.keys():
            # Calculate difference: W_i - G_{t-1}
            d = w[key] - global_model_weights[key]
            diff[key] = d
            total_norm_sq += torch.sum(d ** 2).item()
        
        norm = total_norm_sq ** 0.5
        norms.append(norm)
        diffs.append(diff)
        
    # 2. Determine Clipping Bound S (Median of norms) [cite: 301]
    S = np.median(norms)
    # print(f"   ✂️ Adaptive Clipping Bound (S): {S:.4f}")

    # 3. Clip (Scale Down) [cite: 303]
    clipped_weights = []
    for i, diff in enumerate(diffs):
        # Scaling factor gamma = S / norm
        factor = min(1.0, S / (norms[i] + 1e-9)) # Add epsilon to avoid div/0
        
        # Apply scaling: W_new = G + (W_local - G) * factor
        new_w = copy.deepcopy(global_model_weights)
        for key in new_w.keys():
            new_w[key] += diff[key] * factor
            
        clipped_weights.append(new_w)

    # 4. Aggregate (Average) the clipped models
    avg_model = fed_avg([(w, 1, 0) for w in clipped_weights])
    
    # 5. Add Adaptive Noise (Differential Privacy) [cite: 315, 325]
    if privacy_cfg:
        epsilon = privacy_cfg.get('epsilon', 300.0)
        delta = privacy_cfg.get('delta', 1e-5)
        
        # Calculate Noise Scale (Sigma) based on Clipping Bound S
        # Formula from FLAME Theorem 1 (Eq 7)
        lambda_val = (1.0 / epsilon) * np.sqrt(2 * np.log(1.25 / delta))
        sigma = lambda_val * S
        
        print(f"   🤫 Adding DP Noise: sigma={sigma:.6f} (epsilon={epsilon})")
        
        # Inject Gaussian Noise to the Aggregated Model
        for key in avg_model.keys():
            noise = torch.normal(mean=0.0, std=sigma, size=avg_model[key].shape).to(avg_model[key].device)
            avg_model[key] += noise

    return avg_model