import numpy as np
import torch
from collections import Counter
from .splitter import iid, dirichlet, pathological

def partition_data(dataset, n_clients=10, method="iid", alpha=0.5):
    """
    Router function that selects the correct partition strategy.
    Args:
        dataset: The full training PyTorch Dataset.
        n_clients: Number of clients (e.g., 10).
        method: "iid" (Exp 1) or "dirichlet" (Exp 2).
        alpha: Skew parameter for Dirichlet (lower = more non-IID).
    
    Returns: 
        partitions: Dictionary {client_id: list_of_indices}
    """
    print(f"🔪 Partitioning {len(dataset)} samples for {n_clients} clients (Method: {method})...")

    if method == "iid":
        return iid(dataset, n_clients)
            
    elif method == "dirichlet":
        return dirichlet(dataset, n_clients, alpha)
    
    elif method == "pathological":
        return pathological(dataset, n_clients, shards_per_client=2)
        
    else:
        raise ValueError(f"Unknown partition method: {method}")

def verify_partition(dataset, partitions):
    """
    Helper to print the class distribution of each client.
    Verifies if the data is IID (balanced) or Non-IID (skewed).
    """

    labels = dataset.tensors[1].numpy()

    print("\n📊 Partition Verification (Class Distribution):")
    print(f"{'Client':<8} | {'Total':<8} | Distribution (ClassID : Count)")
    print("-" * 80)
    
    # 2. Iterate through clients
    for cid, indices in sorted(partitions.items()):
        if len(indices) == 0:
            print(f"Client {cid:<5} | 0        | [Empty]")
            continue
            
        # Get the actual labels assigned to this client
        client_labels = labels[indices]
        
        # Count frequencies (e.g., {0: 500, 1: 20})
        counts = Counter(client_labels)
        
        # Sort by Class ID for cleaner reading
        dist_str = "  ".join([f"C{k}:{v}" for k, v in sorted(counts.items())])
        
        print(f"Client {cid:<5} | {len(indices):<8} | {dist_str}")
        
    print("-" * 80)