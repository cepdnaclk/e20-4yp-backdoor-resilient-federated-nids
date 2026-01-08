import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import classification_report, f1_score
from src.client.model import Net

def get_class_weights(y_tensor, device):
    """Calculates Inverse Frequency Weights to fix imbalance."""
    y_np = y_tensor.cpu().numpy()
    classes = np.unique(y_np)
    class_counts = np.bincount(y_np)
    total = len(y_np)
    
    # Formula: Total / (Num_Classes * Count)
    weights = total / (len(classes) * class_counts)
    
    # Safety check for zeroes (if a class is missing in the batch)
    weights = np.where(weights == np.inf, 1.0, weights)
    
    print(f"⚖️ Calculated Class Weights: {np.round(weights, 2)}")
    return torch.FloatTensor(weights).to(device)

@hydra.main(version_base=None, config_path="../../configs/central", config_name="m1_baseline")
def main(cfg: DictConfig):
    print(f"🚀 Starting Experiment: {cfg.name}")
    print(f"⚙️ Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # 1. Setup Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"💻 Using Device: {device}")

    # 2. Load Data
    # Hydra changes cwd, so we force absolute path
    abs_data_path = hydra.utils.to_absolute_path(cfg.data_path)
    
    if not os.path.exists(abs_data_path):
        raise FileNotFoundError(f"❌ Data not found at {abs_data_path}")
        
    print(f"📂 Loading data from: {abs_data_path}")
    data = torch.load(abs_data_path)
    X_train_full, y_train_full = data['X'], data['y']
    
    # Verify Dimensions
    print(f"ℹ️ Data Shapes - X: {X_train_full.shape}, y: {y_train_full.shape}")
    inferred_input_dim = int(X_train_full.shape[1])
    model_input_dim = int(cfg.input_dim)
    if inferred_input_dim != model_input_dim:
        print(
            f"⚠️ WARNING: Config input_dim ({cfg.input_dim}) != Data dim ({inferred_input_dim}). "
            f"Using inferred input_dim={inferred_input_dim} for the model."
        )
        model_input_dim = inferred_input_dim
    
    # 3. Split Train/Validation (80/20)
    dataset = torch.utils.data.TensorDataset(X_train_full, y_train_full)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 4. Initialize Model
    # We pass dims explicitly in case you want to override defaults
    model = Net(input_dim=model_input_dim, num_classes=cfg.num_classes).to(device)
    
    # 5. Define Loss (The Milestone Logic)
    if cfg.use_class_weights:
        print("⚖️ ENABLED: Applying Class Weights to Loss Function (Milestone 2+)")
        weights = get_class_weights(y_train_full, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print("📉 DISABLED: Using Standard Flat Loss (Milestone 1 Baseline)")
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # 6. Training Loop
    print("\n🏁 Training started...")
    best_f1 = 0.0
    
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0
        
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            
            optimizer.zero_grad()
            output = model(X_b)
            loss = criterion(output, y_b)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(y_b.cpu().numpy())
        
        # Metrics
        macro_f1 = f1_score(labels, preds, average='macro')
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Macro-F1: {macro_f1:.4f}")
        
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            # You could save the best model here if needed
            # torch.save(model.state_dict(), "best_central_model.pt")

    # 7. Final Report
    print(f"\n✅ Finished {cfg.name}")
    print("Final Classification Report (Validation Set):")
    print(classification_report(labels, preds, zero_division=0))

if __name__ == "__main__":
    main()