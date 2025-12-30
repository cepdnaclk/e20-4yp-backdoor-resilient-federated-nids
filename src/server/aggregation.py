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