import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import classification_report, f1_score
from src.client.model import Net, WiderNet
from src.utils.logger import Logger
from src.utils.focal_loss import FocalLoss
from src.utils.class_weights import get_class_weights

def apply_thresholds(probs, thresholds, fallback="argmax"):
    """
    probs: Tensor [B, C]
    thresholds: dict {class_id: threshold}
    """
    probs_np = probs.cpu().numpy()
    preds = np.argmax(probs_np, axis=1)

    for c, t in thresholds.items():
        mask = probs_np[:, c] >= t
        preds[mask] = c

    return preds


def get_default_thresholds(num_classes):
    """
    Conservative defaults.
    You can tune later.
    """
    thresholds = {c: 0.5 for c in range(num_classes)}
    thresholds[0] = 0.7  # Normal class stricter
    return thresholds

@hydra.main(version_base=None, config_path="../../configs/central", config_name="m1_baseline")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"🚀 Starting Experiment: {cfg.name}")
    print(f"🎲 Random Seed: {seed}")
    print(f"⚙️ Config:\n{OmegaConf.to_yaml(cfg)}")

    logger = Logger(
        cfg=cfg,
        project_name=cfg.wandb.project,
        group_name=cfg.wandb.group,
        tags=cfg.wandb.tags
    )
    
    # 1. Setup Device
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"💻 Using Device: {device}")

    # 2. Load Training Data
    # Hydra changes cwd, so we force absolute path
    abs_train_path = hydra.utils.to_absolute_path(cfg.train_data_path)
    abs_val_path = hydra.utils.to_absolute_path(cfg.val_data_path)
    
    if not os.path.exists(abs_train_path):
        raise FileNotFoundError(f"❌ Training data not found at {abs_train_path}")
    if not os.path.exists(abs_val_path):
        raise FileNotFoundError(f"❌ Validation data not found at {abs_val_path}")
        
    print(f"📂 Loading training data from: {abs_train_path}")
    train_data = torch.load(abs_train_path, weights_only=False)
    X_train, y_train = train_data['X'], train_data['y']
    label_map = train_data.get('label_map', None)  # Load class names if available
    
    print(f"📂 Loading validation data from: {abs_val_path}")
    val_data = torch.load(abs_val_path, weights_only=False)
    X_val, y_val = val_data['X'], val_data['y']
    
    # Verify Dimensions
    print(f"ℹ️ Training Data Shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"ℹ️ Validation Data Shapes - X: {X_val.shape}, y: {y_val.shape}")
    inferred_input_dim = int(X_train.shape[1])
    model_input_dim = int(cfg.input_dim)
    if inferred_input_dim != model_input_dim:
        print(
            f"⚠️ WARNING: Config input_dim ({cfg.input_dim}) != Data dim ({inferred_input_dim}). "
            f"Using inferred input_dim={inferred_input_dim} for the model."
        )
        model_input_dim = inferred_input_dim
    
    # 3. Create DataLoaders (No splitting needed - using pre-split data)
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    
    # 4. Initialize Model
    # Support different model architectures
    model_type = cfg.get('model_type', 'standard')
    if model_type == 'wider':
        print("🏗️ Using WiderNet (2x capacity)")
        model = WiderNet(input_dim=model_input_dim, num_classes=cfg.num_classes).to(device)
    else:
        model = Net(input_dim=model_input_dim, num_classes=cfg.num_classes).to(device)
    
    # 5. Define Loss (The Milestone Logic)
    use_focal = cfg.get('use_focal_loss', False)
    focal_gamma = cfg.get('focal_gamma', 2.0)
    focal_alpha = cfg.get('focal_alpha', None)
    
    # Get weight method from config (default to 'inverse')
    weight_method = cfg.get('weight_method', 'inverse')
    
    if use_focal:
        print(f"🎯 ENABLED: Using Focal Loss (gamma={focal_gamma})")
        # Handle alpha parameter
        if cfg.use_class_weights:
            print("⚖️ ENABLED: Applying Class Weights as Focal Loss Alpha")
            weights = get_class_weights(y_train, device, method=weight_method)
            criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        elif focal_alpha == "auto":
            print("⚖️ ENABLED: Auto-computing Alpha from class frequencies")
            weights = get_class_weights(y_train, device, method=weight_method)
            criterion = FocalLoss(alpha=weights, gamma=focal_gamma)
        else:
            criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    elif cfg.use_class_weights:
        print("⚖️ ENABLED: Applying Class Weights to Loss Function (Milestone 2+)")
        weights = get_class_weights(y_train, device, method=weight_method)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print("📉 DISABLED: Using Standard Flat Loss (Milestone 1 Baseline)")
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    # Learning rate scheduler (optional)
    use_scheduler = cfg.get('use_scheduler', False)
    if use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        print("📉 ENABLED: ReduceLROnPlateau scheduler (patience=5)")
    
    # 6. Training Loop
    print("\n🏁 Training started...")
    best_f1 = 0.0
    patience = cfg.get('early_stopping_patience', None)
    patience_counter = 0
    
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
                probs = torch.softmax(out, dim=1)
                if cfg.use_thresholds:
                    thresholds = get_default_thresholds(cfg.num_classes)
                    batch_preds = apply_thresholds(probs, thresholds)
                else:
                    batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
                preds.extend(batch_preds)
                labels.extend(y_b.cpu().numpy())

        
        # Metrics
        macro_f1 = f1_score(labels, preds, average='macro')
        accuracy = np.mean(np.array(preds) == np.array(labels))
        avg_loss = train_loss / len(train_loader)
        
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Macro-F1: {macro_f1:.4f} | Val Accuracy: {accuracy:.4f}")
        
        # Update scheduler if enabled
        if use_scheduler:
            scheduler.step(macro_f1)
        
        # Early stopping logic
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            patience_counter = 0
            # You could save the best model here if needed
            # torch.save(model.state_dict(), "best_central_model.pt")
        else:
            patience_counter += 1
            
        # Check early stopping
        if patience is not None and patience_counter >= patience:
            print(f"⏹️ Early stopping triggered after {epoch+1} epochs (patience={patience})")
            print(f"🏆 Best Macro-F1: {best_f1:.4f}")
            break

        logger.log_metrics({
            "train/loss": avg_loss,
            "val/accuracy": accuracy,
            "val/macro_f1": macro_f1,
            "epoch": epoch + 1
        }, step=epoch + 1)

    # 7. Final Report
    print(f"\n✅ Finished {cfg.name}")
    print("Final Classification Report (Validation Set):")
    
    # Use class names if label_map is available
    target_names = None
    if label_map is not None:
        if isinstance(label_map, np.ndarray):
            target_names = label_map.tolist()
        elif isinstance(label_map, dict):
            # Reverse the dict if it maps name->id
            if all(isinstance(v, int) for v in label_map.values()):
                target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            else:
                target_names = list(label_map.keys())
        print(f"   📋 Using class names: {target_names}")
    
    print(classification_report(labels, preds, target_names=target_names, zero_division=0))
    logger.finish()

if __name__ == "__main__":
    main()