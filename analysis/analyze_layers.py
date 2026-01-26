import sys
import os

# 📍 ADD ROOT DIRECTORY TO PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from omegaconf import OmegaConf

from src.client.model import Net
from src.client.client import Client
from src.data.loader import get_data_loaders

# --- CONFIGURATION ---
DATA_PATH = "data/unsw-nb15/processed/train_pool.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASSIFICATION_MODE = "binary" 

# Check Overrides
if "OVERRIDE_MODEL_PATH" in os.environ:
    MODEL_PATH = os.environ["OVERRIDE_MODEL_PATH"]
else:
    MODEL_PATH = "final_model.pt"

if "OVERRIDE_SAVE_DIR" in os.environ:
    SAVE_DIR = os.environ["OVERRIDE_SAVE_DIR"]
else:
    SAVE_DIR = "."

# Mock Config (Simulates training round parameters)
config = OmegaConf.create({
    "client": {"lr": 0.01, "device": DEVICE, "epochs": 1, "batch_size": 256},
    "attack": {"aggressive": True, "estimated_n_clients": 40}
})

def calculate_weight_diff(global_model, updated_model):
    diffs = {}
    g_dict = global_model.state_dict()
    u_dict = updated_model.state_dict()
    
    for key in g_dict:
        if "weight" in key and "bn" not in key: 
            diff = torch.norm(u_dict[key] - g_dict[key]).item()
            diffs[key] = diff
    return diffs

def main():
    print("🔬 Starting Layer-wise Weight Analysis...")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        return

    train_ds, _, input_dim, num_classes = get_data_loaders(
        DATA_PATH, batch_size=256, classification_mode=CLASSIFICATION_MODE
    )
    
    global_model = Net(input_dim=input_dim, num_classes=num_classes).to(DEVICE)
    try:
        global_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    global_weights = global_model.state_dict()
    
    # Simulate Updates
    print("   Simulating Honest Update...")
    honest_indices = list(range(0, 1000)) 
    honest_client = Client(0, train_ds, honest_indices, global_model, config, lr=0.01, device=DEVICE)
    w_honest, _, _ = honest_client.train(global_weights, epochs=1)
    
    print("   Simulating Malicious Update...")
    mal_indices = list(range(1000, 2000))
    attacker_client = Client(99, train_ds, mal_indices, global_model, config, lr=0.01, device=DEVICE, is_malicious=True)
    w_malicious, _, _ = attacker_client.train(global_weights, epochs=1)
    
    # Compare
    model_honest = copy.deepcopy(global_model)
    model_honest.load_state_dict(w_honest)
    
    model_malicious = copy.deepcopy(global_model)
    model_malicious.load_state_dict(w_malicious)
    
    diff_honest = calculate_weight_diff(global_model, model_honest)
    diff_malicious = calculate_weight_diff(global_model, model_malicious)
    
    # Plotting
    layers = list(diff_honest.keys())
    layer_names = [l.replace('.weight', '') for l in layers]
    vals_honest = [diff_honest[l] for l in layers]
    vals_malicious = [diff_malicious[l] for l in layers]
    
    x = np.arange(len(layers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, vals_honest, width, label='Honest Update', color='dodgerblue')
    ax.bar(x + width/2, vals_malicious, width, label='Malicious Update', color='crimson')
    
    ax.set_ylabel('Update Magnitude (L2 Norm)')
    ax.set_title(f'Layer-wise Impact Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    save_path = os.path.join(SAVE_DIR, "layer_analysis.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Plot saved to: {save_path}")

if __name__ == "__main__":
    main()