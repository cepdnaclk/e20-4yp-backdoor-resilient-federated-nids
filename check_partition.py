from src.data.loader import load_dataset
from src.data.partition import partition_data, verify_partition

def test_partitioning():
    # 1. Load Data
    print("Loading data...")
    ds, _, _ = load_dataset()
    
    # 2. Partition (non-IID)
    parts = partition_data(ds, n_clients=10, method="dirichlet")
    
    # 3. Verify
    verify_partition(ds, parts)

if __name__ == "__main__":
    test_partitioning()