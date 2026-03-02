import numpy as np

def iid(dataset, n_clients, seed=42):
    """
    Splits data evenly and randomly (IID).
    """
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    # 1. Shuffle globally with fixed seed for reproducibility
    np.random.seed(seed) 
    np.random.shuffle(indices)
    
    # 2. Split evenly using numpy
    split_indices = np.array_split(indices, n_clients)
    
    # 3. Convert to dictionary
    partitions = {cid: split_indices[cid] for cid in range(n_clients)}
    
    return partitions

def dirichlet(dataset, n_clients, alpha=0.5, seed=42):
    """
    Splits data using Dirichlet distribution (Non-IID).
    
    Logic:
    1. Organize indices by class (Class 0 indices, Class 1 indices...)
    2. For each class 'k':
       - Sample a distribution vector 'p' from Dirichlet(alpha).
         (e.g., with alpha=0.1, p might be [0.9, 0.09, 0.01, 0.0, ...] -> Extreme skew)
       - Split the indices of class 'k' among clients according to 'p'.
    3. Aggregate all assigned indices for each client.
    """
    np.random.seed(seed)
    
    # 1. Extract Labels to find indices
    if hasattr(dataset, 'tensors'):
        labels = dataset.tensors[1].numpy()
    elif hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    else:
        raise ValueError("Could not extract labels from dataset for Dirichlet split.")

    n_classes = len(np.unique(labels))
    min_size = 0 # Safety: ensure clients don't get 0 samples (optional logic)
    
    # Map {client_id: [indices]}
    client_indices = {i: [] for i in range(n_clients)}
    
    # 2. Iterate over each class and split it unevenly
    for k in range(n_classes):
        # Get all indices where label is 'k'
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        
        # Sample the distribution p ~ Dir(alpha)
        # alpha is passed as a vector [alpha, alpha, ... alpha]
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        
        # Calculate split points based on proportions
        # We use cumsum to determine where to slice the array of indices
        # integers are needed for slicing, so we multiply by len(idx_k)
        split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        
        # Split the class indices into chunks
        idx_k_split = np.split(idx_k, split_points)
        
        # Assign chunks to clients
        for cid in range(n_clients):
            client_indices[cid].append(idx_k_split[cid])

    # 3. Concatenate and Shuffle logic
    partitions = {}
    for cid in range(n_clients):
        # Combine the chunks of Class 0, Class 1... that this client received
        if len(client_indices[cid]) > 0:
            partitions[cid] = np.concatenate(client_indices[cid]).astype(int)
            np.random.shuffle(partitions[cid]) # Shuffle so classes aren't ordered
        else:
            partitions[cid] = np.array([], dtype=int)

    return partitions

def pathological(dataset, n_clients, shards_per_client=1, seed=42):
    """
    Splits data such that each client gets exactly 'shards_per_client' shards,
    and each shard contains only one class.
    """
    np.random.seed(seed)
    labels = dataset.tensors[1].numpy()
    n_samples = len(dataset)
    n_classes = len(np.unique(labels))
    total_shards = n_clients * shards_per_client

    # 1. Split each class into shards
    idx_shards = []
    for k in range(n_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)
        # Split class indices into as equal shards as possible
        class_shards = np.array_split(idx_k, max(1, int(np.round(total_shards / n_classes))))
        idx_shards.extend(class_shards)

    # 2. Shuffle all shards
    idx_shards = [shard for shard in idx_shards if len(shard) > 0]
    np.random.shuffle(idx_shards)

    # 3. Assign shards to clients
    partitions = {}
    for i in range(n_clients):
        start_ptr = i * shards_per_client
        end_ptr = start_ptr + shards_per_client
        selected_shards = idx_shards[start_ptr:end_ptr]
        if selected_shards:
            client_data = np.concatenate(selected_shards)
            np.random.shuffle(client_data)
            partitions[i] = client_data.astype(int)
        else:
            partitions[i] = np.array([], dtype=int)

    return partitions