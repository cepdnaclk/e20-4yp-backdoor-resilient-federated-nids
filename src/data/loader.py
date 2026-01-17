import torch
from torch.utils.data import TensorDataset, DataLoader
import os

DEFAULT_TRAIN_PATH = "data/unsw-nb15/processed/train_pool.pt"

def load_dataset(path=DEFAULT_TRAIN_PATH, classification_mode="binary"):
    """
    Loads a saved .pt file and returns a TensorDataset.
    Args:
        path: Path to .pt file
        classification_mode: "binary" (Normal vs Attack) or "multiclass" (all classes)
    Returns: (dataset, input_dim, num_classes)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Data file not found at {path}. Run preprocessing first!")

    print(f"📂 Loading data from {path}...")
    print(f"   Classification mode: {classification_mode}")
    # weights_only=False is safer if you saved complex dicts, but True is fine for Tensors
    data = torch.load(path) 
    
    X = data['X']
    y = data['y']
    
    # Convert to binary if requested (Normal=0, Attack=1)
    if classification_mode == "binary":
        y = (y > 0).long()  # All non-zero labels become 1 (Attack)
        num_classes = 2
        print(f"   ✓ Converted to binary: Normal={0}, Attack={1}")
    else:
        num_classes = len(torch.unique(y))
    
    # Calculate dimensions automatically
    input_dim = X.shape[1]      
    
    dataset = TensorDataset(X, y)
    
    return dataset, input_dim, num_classes

def get_data_loaders(path=DEFAULT_TRAIN_PATH, batch_size=32, classification_mode="binary"):
    """
    Helper to get loaders. It intelligently finds the test set 
    associated with the provided training path.
    Args:
        path: Path to training data
        batch_size: Batch size for loaders
        classification_mode: "binary" or "multiclass"
    """
    # 1. DERIVE TEST PATH DYNAMICALLY
    # If path is "data/cic-ids2017/processed/train_pool.pt"
    # We want "data/cic-ids2017/processed/global_test.pt"
    
    test_path = path.replace("train_pool.pt", "global_test.pt")
    
    # 2. Check if the derived test file exists
    if not os.path.exists(test_path):
        # Fallback: maybe you named it 'test.pt' in the old dataset?
        fallback = path.replace("train_pool.pt", "test.pt")
        if os.path.exists(fallback):
            test_path = fallback
        else:
            print(f"⚠️ Warning: Test file not found at {test_path}. Using Train set for testing (Not recommended).")
            test_path = path

    print(f"   🔍 derived test path: {test_path}")

    # 3. Load Global Test Set with classification mode
    test_ds, _, _ = load_dataset(test_path, classification_mode=classification_mode)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 4. Load Train Pool (Useful for debugging dimensions)
    train_ds, input_dim, num_classes = load_dataset(path, classification_mode=classification_mode)
    
    return train_ds, test_loader, input_dim, num_classes