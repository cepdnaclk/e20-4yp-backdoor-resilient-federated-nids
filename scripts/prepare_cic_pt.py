import torch
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.cic_unsw_loader import load_cic_unsw_nb15

CSV_PATH = "data/cic-unsw-nb15/CICFlowMeter.csv"
OUT_DIR = "data/cic-unsw-nb15"

print("Loading CIC-UNSW-NB15...")
X, y, input_dim = load_cic_unsw_nb15(CSV_PATH)

print("Total samples:", X.shape[0])

# --- SUBSET FOR FAST ITERATION (5%) ---
subset_frac = 0.05
n_subset = int(len(X) * subset_frac)

idx = np.random.choice(len(X), n_subset, replace=False)
X_sub = X[idx]
y_sub = y[idx]

print("Subset size:", X_sub.shape)

# --- Train/Val Split ---
X_train, X_val, y_train, y_val = train_test_split(
    X_sub, y_sub,
    test_size=0.2,
    stratify=y_sub,
    random_state=42
)

print("Train size:", X_train.shape)
print("Val size:", X_val.shape)

# --- Save PT files ---
torch.save(
    {
        "X": X_train,
        "y": y_train,
        "label_map": {
            "Benign": 0,
            "Generic": 1,
            "Exploits": 2,
            "Fuzzers": 3,
            "DoS": 4,
            "Reconnaissance": 5,
            "Analysis": 6,
            "Backdoor": 7,
            "Shellcode": 8,
            "Worms": 9,
        }
    },
    f"{OUT_DIR}/cic_train_subset.pt"
)

torch.save(
    {
        "X": X_val,
        "y": y_val,
    },
    f"{OUT_DIR}/cic_val_subset.pt"
)

print("Saved:")
print(" - cic_train_subset.pt")
print(" - cic_val_subset.pt")
