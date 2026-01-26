import sys
import os

# 📍 ADD ROOT DIRECTORY TO PATH
# This allows importing 'src' even though we are in 'analysis' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import glob

# Import your project modules
from src.client.model import Net
from src.data.loader import get_data_loaders

# --- CONFIGURATION ---
DATA_PATH = "data/unsw-nb15/processed/train_pool.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Check for Overrides from Master Script
if "OVERRIDE_MODEL_PATH" in os.environ:
    MANUAL_MODEL_PATH = os.environ["OVERRIDE_MODEL_PATH"]
    print(f"⚙️  Using Overridden Model Path: {MANUAL_MODEL_PATH}")
else:
    MANUAL_MODEL_PATH = "final_model.pt"

if "OVERRIDE_SAVE_DIR" in os.environ:
    SAVE_DIR = os.environ["OVERRIDE_SAVE_DIR"]
    print(f"⚙️  Using Overridden Save Dir: {SAVE_DIR}")
else:
    SAVE_DIR = "." # Default to root

def get_embeddings(model, loader, device):
    model.eval()
    features, labels = [], []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())

    handle = None
    if hasattr(model, 'fc4') and hasattr(model, 'fc5'):
        handle = model.fc4.register_forward_hook(hook_fn)
    elif hasattr(model, 'fc3'):
        handle = model.fc3.register_forward_hook(hook_fn)
    else:
        # Fallback for simple models
        handle = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            _ = model(X)
            labels.append(y.numpy())
    
    if handle: handle.remove()
    return torch.cat(features).numpy(), np.concatenate(labels)

def main():
    print("🚀 Starting t-SNE Visualization...")
    
    # Check Model Existence
    if not os.path.exists(MANUAL_MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MANUAL_MODEL_PATH}")
        return

    # Load Data (using large batch for inference speed)
    _, test_loader, input_dim, num_classes = get_data_loaders(DATA_PATH, batch_size=1024)
    
    # Load Model
    model = Net(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MANUAL_MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # Prepare Poisoned Data
    all_X, all_y = [], []
    for X, y in test_loader:
        all_X.append(X)
        all_y.append(y)
    X_clean = torch.cat(all_X)
    y_clean = torch.cat(all_y)
    
    target_label = 0
    mask = (y_clean != target_label) 
    X_poison = X_clean[mask].clone()
    X_poison[:, 0] = 5.0 # INJECT TRIGGER
    y_poison = torch.full((X_poison.shape[0],), 99) # Label 99 is 'Backdoor'
    
    # Subset for plotting (2000 samples)
    N_SAMPLES = 2000
    indices = torch.randperm(len(X_clean))[:N_SAMPLES]
    X_final = torch.cat([X_clean[indices], X_poison[:500]])
    y_final = torch.cat([y_clean[indices], y_poison[:500]])
    
    viz_loader = DataLoader(TensorDataset(X_final, y_final), batch_size=512)

    # Run t-SNE
    print("🧠 Running t-SNE...")
    features, labels = get_embeddings(model, viz_loader, DEVICE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    idx_norm = (labels == 0)
    plt.scatter(X_embedded[idx_norm, 0], X_embedded[idx_norm, 1], c='dodgerblue', label='Normal Traffic', alpha=0.6, s=15)
    
    idx_attack = (labels > 0) & (labels < 99)
    plt.scatter(X_embedded[idx_attack, 0], X_embedded[idx_attack, 1], c='crimson', label='True Attacks', alpha=0.6, s=15)
    
    idx_backdoor = (labels == 99)
    plt.scatter(X_embedded[idx_backdoor, 0], X_embedded[idx_backdoor, 1], c='lime', label='Backdoored (Camouflaged)', marker='*', s=150, edgecolors='black', linewidth=1)

    plt.title("t-SNE Visualization: Backdoor Attack Impact", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(SAVE_DIR, "tsne_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {save_path}")

if __name__ == "__main__":
    main()