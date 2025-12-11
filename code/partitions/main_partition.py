import numpy as np
import os
import pickle
from collections import defaultdict

# --- Configuration ---
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
DATA_PATH = os.path.join(REPO_ROOT, 'data')
PARTITIONS_DIR = os.path.join(REPO_ROOT, 'code', 'partitions')
NUM_CLIENTS = 10 
DIRICHLET_ALPHA = 0.5  # Controls Label Skew (smaller alpha = more skew)

# --- 1. Non-IID Partitioning Functions ---

def partition_dirichlet(y, num_clients, alpha):
    """
    Implements Dirichlet Distribution (Label Skew) partitioning.
    Creates a highly non-IID split based on class distribution.
    """
    labels = np.unique(y)
    num_classes = len(labels)
    client_indices = defaultdict(list)
    
    # Get indices for each class label
    class_indices = [np.where(y == c)[0] for c in labels]

    for k in range(num_clients):
        # Draw sampling distribution from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_classes))
        
        for c in range(num_classes):
            num_samples = int(proportions[c] * len(class_indices[c]))
            
            # Select samples without replacement
            indices_to_assign = np.random.choice(class_indices[c], num_samples, replace=False)
            client_indices[k].extend(indices_to_assign)
            
            # Remove assigned indices for non-overlapping data sets
            class_indices[c] = np.setdiff1d(class_indices[c], indices_to_assign)
            
    # Ensure all data is assigned (due to float rounding)
    all_indices = np.concatenate(class_indices)
    if len(all_indices) > 0:
        client_indices[np.random.randint(num_clients)].extend(all_indices)

    return {k: np.array(v, dtype=np.int64) for k, v in client_indices.items()}


def partition_quantity_imbalance(total_samples, num_clients, min_ratio=0.02, max_ratio=0.20):
    """
    Implements Quantity Imbalance (Size Skew) partitioning.
    Clients receive highly unequal volumes of data.
    """
    # Create uneven proportions that sum to 1.0
    proportions = np.random.uniform(min_ratio, max_ratio, num_clients)
    proportions = proportions / proportions.sum() # Normalize to 1.0

    client_indices = defaultdict(list)
    all_indices = np.arange(total_samples)
    
    for k in range(num_clients):
        num_samples = int(proportions[k] * total_samples)
        
        # Select indices without replacement
        indices_to_assign = np.random.choice(all_indices, num_samples, replace=False)
        client_indices[k].extend(indices_to_assign)
        
        # Remove assigned indices
        all_indices = np.setdiff1d(all_indices, indices_to_assign)
    
    # Assign any remaining indices to the client with the largest allocation
    if len(all_indices) > 0:
        largest_client_k = np.argmax([len(client_indices[c]) for c in client_indices])
        client_indices[largest_client_k].extend(all_indices)

    return {k: np.array(v, dtype=np.int64) for k, v in client_indices.items()}


def partition_label_imbalance(y, num_clients, benign_only_ratio=0.8):
    """
    Implements Label Imbalance (Class-Specific Skew) partitioning.
    A ratio of clients (e.g., 80%) see only BENIGN traffic.
    """
    benign_label = np.min(y) # Assume 0 is the benign label
    
    num_benign_clients = int(num_clients * benign_only_ratio) # 8 clients (0-7)
    
    benign_indices = np.where(y == benign_label)[0]
    attack_indices = np.where(y != benign_label)[0]
    
    client_indices = defaultdict(list)
    
    # 1. Assign Benign data only to the Benign-Only clients
    # Ensure we have enough benign data for all benign clients
    benign_per_client = max(1, len(benign_indices) // num_benign_clients)
    for k in range(num_benign_clients):
        start_idx = k * benign_per_client
        if k == num_benign_clients - 1:
            # Last client gets remaining benign data
            client_indices[k].extend(benign_indices[start_idx:len(benign_indices)//2])
        else:
            end_idx = (k + 1) * benign_per_client
            client_indices[k].extend(benign_indices[start_idx:end_idx])
        
    # 2. Assign Attack data to the Attack-Seeing clients
    attack_client_ks = list(range(num_benign_clients, num_clients)) # 2 clients (8, 9)
    
    if len(attack_client_ks) > 0:
        attack_per_client = max(1, len(attack_indices) // len(attack_client_ks))
        for i, k in enumerate(attack_client_ks):
            start_idx = i * attack_per_client
            if i == len(attack_client_ks) - 1:
                # Last client gets remaining attack data
                client_indices[k].extend(attack_indices[start_idx:])
            else:
                end_idx = (i + 1) * attack_per_client
                client_indices[k].extend(attack_indices[start_idx:end_idx])
        
        # Also assign remaining benign data to attack-seeing clients
        remaining_benign_start = (len(benign_indices) // 2)
        remaining_benign_indices = benign_indices[remaining_benign_start:]
        benign_per_attack_client = max(1, len(remaining_benign_indices) // len(attack_client_ks))
        for i, k in enumerate(attack_client_ks):
            start_idx = i * benign_per_attack_client
            if i == len(attack_client_ks) - 1:
                client_indices[k].extend(remaining_benign_indices[start_idx:])
            else:
                end_idx = (i + 1) * benign_per_attack_client
                client_indices[k].extend(remaining_benign_indices[start_idx:end_idx])

    return {k: np.array(v, dtype=np.int64) for k, v in client_indices.items()}


# --- 2. Orchestration ---

def run_partitioning():
    """Loads processed data and generates client index lists for all scenarios."""
    print("--- 1. Loading Processed Training Data ---")
    
    try:
        X_train = np.load(os.path.join(DATA_PATH, 'X_full_train.npy'))
        y_train = np.load(os.path.join(DATA_PATH, 'y_full_train.npy'))
    except FileNotFoundError:
        print("Error: Processed data files not found. Please run data_preprocessor.py first.")
        return

    total_samples = len(y_train)
    
    # --- 2. Generating Non-IID Partitions ---
    print("\n--- 2. Generating Non-IID Partitions (10 Clients) ---")
    
    # a) Dirichlet Partitioning (Label Skew)
    dirichlet_indices = partition_dirichlet(y_train, NUM_CLIENTS, DIRICHLET_ALPHA)
    print(f"Dirichlet Partitioning (alpha={DIRICHLET_ALPHA}) complete. Client 0 size: {len(dirichlet_indices[0])}")

    # b) Quantity Imbalance Partitioning (Size Skew)
    quantity_indices = partition_quantity_imbalance(total_samples, NUM_CLIENTS)
    print(f"Quantity Imbalance Partitioning complete. Client 0 size: {len(quantity_indices[0])}")

    # c) Label Imbalance Partitioning (Class-Specific Skew)
    label_indices = partition_label_imbalance(y_train, NUM_CLIENTS)
    print(f"Label Imbalance Partitioning complete. Client 0 size: {len(label_indices[0])}")

    # --- 3. Saving Partition Metadata ---
    
    partition_metadata = {
        'dirichlet': dirichlet_indices,
        'quantity': quantity_indices,
        'label_imbalance': label_indices,
        'X_shape': X_train.shape,
        'y_shape': y_train.shape
    }

    output_file = os.path.join(PARTITIONS_DIR, 'client_partitions_metadata.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(partition_metadata, f)
        
    print(f"\nPartitioning complete. Metadata saved to: {output_file}")


if __name__ == '__main__':
    run_partitioning()