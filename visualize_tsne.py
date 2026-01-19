import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
import os
import glob
from datetime import datetime  # <--- NEW IMPORT

# Import your project modules
from src.client.model import Net
from src.data.loader import get_data_loaders

# --- CONFIGURATION ---
MANUAL_MODEL_PATH = "final_model.pt" 
DATA_PATH = "data/unsw-nb15/processed/train_pool.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def setup_results_folder():
    """Creates a folder like 'results/2026-01-19_06-30-00'"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base_dir = "results"
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📂 Created results folder: {save_dir}")
    return save_dir

def find_best_model():
    if os.path.exists(MANUAL_MODEL_PATH): return MANUAL_MODEL_PATH
    root_models = ["final_model.pt", "final_optimized_model.pt", "model.pt"]
    for m in root_models:
        if os.path.exists(m): return m
    wandb_files = glob.glob("wandb/*/files/*.pt")
    if wandb_files: return max(wandb_files, key=os.path.getmtime)
    return None

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
        handle = model.fc1.register_forward_hook(hook_fn)

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            _ = model(X)
            labels.append(y.numpy())
    
    if handle: handle.remove()
    return torch.cat(features).numpy(), np.concatenate(labels)

def main():
    # 1. Setup Folder
    save_dir = setup_results_folder()
    
    print("🚀 Starting t-SNE Visualization...")
    model_path = find_best_model()
    if model_path is None:
        print("❌ ERROR: Could not find a trained model (.pt file)!")
        return

    # 2. Load Data & Model
    print(f"📂 Loading Data from: {DATA_PATH}")
    _, test_loader, input_dim, num_classes = get_data_loaders(DATA_PATH, batch_size=1024)
    
    model = Net(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 3. Prepare Poisoned Data
    print("🧪 Generating Poisoned Samples...")
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
    y_poison = torch.full((X_poison.shape[0],), 99)
    
    # Subset
    N_SAMPLES = 2000
    indices = torch.randperm(len(X_clean))[:N_SAMPLES]
    X_final = torch.cat([X_clean[indices], X_poison[:500]])
    y_final = torch.cat([y_clean[indices], y_poison[:500]])
    
    viz_loader = DataLoader(TensorDataset(X_final, y_final), batch_size=512)

    # 4. Run t-SNE
    print("🧠 Extracting Embeddings & Running t-SNE...")
    features, labels = get_embeddings(model, viz_loader, DEVICE)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)
    
    # 5. Plotting
    print("🎨 Generating Plot...")
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
    
    # 6. Save to Timestamped Folder
    save_path = os.path.join(save_dir, "tsne_result.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Success! Plot saved to: {save_path}")

if __name__ == "__main__":
    main()