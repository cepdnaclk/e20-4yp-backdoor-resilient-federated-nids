# src/utils/class_weights.py
import numpy as np
import torch

def get_class_weights(y_tensor, device, method='inverse', clip_range=(0.5, 10.0)):
    """
    Calculates class weights to handle imbalance.
    
    Args:
        y_tensor: Labels tensor
        device: torch device
        method: 'inverse' (aggressive) or 'sqrt' (moderate)
        clip_range: (min, max) values to clip weights
    
    Returns:
        torch.Tensor of class weights
    """
    y_np = y_tensor.cpu().numpy()
    classes = np.unique(y_np)
    class_counts = np.bincount(y_np)
    total = len(y_np)
    
    if method == 'sqrt':
        # Moderate weighting: sqrt of inverse frequency
        # Less aggressive than full inverse frequency
        weights = np.sqrt(total / (len(classes) * class_counts))
    else:  # method == 'inverse'
        # Aggressive weighting: full inverse frequency
        weights = total / (len(classes) * class_counts)
    
    # Safety clipping
    weights = np.clip(weights, clip_range[0], clip_range[1])
    
    print(f"⚖️ Calculated Class Weights ({method}): {np.round(weights, 2)}")
    return torch.FloatTensor(weights).to(device)
