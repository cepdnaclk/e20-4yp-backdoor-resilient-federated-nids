# implements the server-side aggregation mechanisms used in the federated learning process

import torch
import copy

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
    Coordinate-wise Median
    Args:
        weights_list: List of state_dicts (just the weights)
    """

    print("Aggregation import --> successfull")
    # Initialize structure
    new_weights = copy.deepcopy(weights_list[0])
    keys = new_weights.keys()
    
    for key in keys:
        # Stack all clients' tensors for this layer
        stacked_layers = torch.stack([w[key] for w in weights_list], dim=0)
        
        # Take Median across the client dimension (dim=0)
        new_weights[key] = torch.median(stacked_layers, dim=0).values
        
    return new_weights

def fed_trimmed_mean(weights_list, beta=0.1):
    """
    Trimmed Mean (Removes top/bottom beta%)
    """
    new_weights = copy.deepcopy(weights_list[0])
    keys = new_weights.keys()
    n_clients = len(weights_list)
    to_trim = int(n_clients * beta)
    
    # Safety Check
    if n_clients - 2 * to_trim <= 0:
        return fed_median(weights_list) # Fallback if too few clients

    for key in keys:
        stacked_layers = torch.stack([w[key] for w in weights_list], dim=0)
        sorted_layers, _ = torch.sort(stacked_layers, dim=0)
        
        # Slice out the middle
        kept_layers = sorted_layers[to_trim : n_clients - to_trim]
        
        new_weights[key] = torch.mean(kept_layers, dim=0)
        
    return new_weights