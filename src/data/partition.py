import numpy as np
import torch
from collections import Counter

def partition_data(dataset, n_clients=10, method="iid", alpha=0.5):
    """
    Splits a dataset into n_clients subsets.
    Args:
        dataset: The full training PyTorch Dataset.
        n_clients: Number of clients (e.g., 10).
        method: "iid" (Exp 1) or "dirichlet" (Exp 2).
        alpha: Skew parameter for Dirichlet (lower = more non-IID).
    
    Returns: 
        partitions: Dictionary {client_id: list_of_indices}
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    labels = dataset.tensors[1].numpy() # Assumes TensorDataset(X, y)
    
    partitions = {}

    print(f"🔪 Partitioning {n_samples} samples for {n_clients} clients (Method: {method})...")

    if method == "iid":
        # 1. Shuffle globally
        # We use a fixed seed so every run gives the same split (Reproducibility)
        np.random.seed(42) 
        np.random.shuffle(indices)
        
        # 2. Split evenly
        # array_split handles cases where n_samples isn't perfectly divisible
        split_indices = np.array_split(indices, n_clients)
        
        for cid in range(n_clients):
            partitions[cid] = split_indices[cid]
            
    elif method == "dirichlet":
        # Placeholder for Sprint 2 (Non-IID Attacks)
        # We will implement the Alpha logic here later.
        raise NotImplementedError("Dirichlet partition not yet implemented for Exp 1!")
        
    else:
        raise ValueError(f"Unknown partition method: {method}")

    return partitions

def verify_partition(dataset, partitions, num_classes=10):
    """
    Helper to print the class distribution of each client.
    Satisfies the 'Verification' acceptance criteria.
    """
    labels = dataset.tensors[1].numpy()
    
    print("\n📊 Partition Verification (Class Distribution):")
    print(f"{'Client':<10} | {'Total':<10} | Distribution (Class 0-9)")
    print("-" * 80)
    
    for cid, indices in partitions.items():
        # Get labels for this client
        client_labels = labels[indices]
        counts = Counter(client_labels)
        
        # Format the distribution string
        dist_str = " ".join([f"{k}:{v}" for k, v in sorted(counts.items())])
        
        # Simple stats
        print(f"Client {cid:<3} | {len(indices):<10} | {dist_str}")
        
        # Safety Check for IID (Exp 1)
        # In IID, every client should have roughly the same number of samples
        if cid > 0:
            prev_len = len(partitions[cid-1])
            curr_len = len(indices)
            if abs(prev_len - curr_len) > 50:
                 print(f"⚠️ Warning: Large imbalance detected between Client {cid-1} and {cid}")