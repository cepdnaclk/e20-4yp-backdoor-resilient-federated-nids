import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import classification_report, f1_score, accuracy_score
from src.client.model import TwoStageNet
from src.utils.logger import Logger
from src.utils.class_weights import get_class_weights


@hydra.main(version_base=None, config_path="../../configs/central", config_name="m13_two_stage")
def main(cfg: DictConfig):
    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    print(f"🚀 Starting Two-Stage Experiment: {cfg.name}")
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
    abs_train_path = hydra.utils.to_absolute_path(cfg.train_data_path)
    abs_val_path = hydra.utils.to_absolute_path(cfg.val_data_path)
    
    if not os.path.exists(abs_train_path):
        raise FileNotFoundError(f"❌ Training data not found at {abs_train_path}")
    if not os.path.exists(abs_val_path):
        raise FileNotFoundError(f"❌ Validation data not found at {abs_val_path}")
        
    print(f"📂 Loading training data from: {abs_train_path}")
    train_data = torch.load(abs_train_path, weights_only=False)
    X_train, y_train = train_data['X'], train_data['y']
    label_map = train_data.get('label_map', None)
    
    print(f"📂 Loading validation data from: {abs_val_path}")
    val_data = torch.load(abs_val_path, weights_only=False)
    X_val, y_val = val_data['X'], val_data['y']
    
    # Verify Dimensions
    print(f"ℹ️ Training Data Shapes - X: {X_train.shape}, y: {y_train.shape}")
    print(f"ℹ️ Validation Data Shapes - X: {X_val.shape}, y: {y_val.shape}")
    input_dim = int(X_train.shape[1])
    
    # 3. Prepare Stage 1 Data (Binary: Normal=0 vs Attack=1)
    print("\n📊 Preparing Stage 1 (Binary Normal/Attack) Data...")
    y_train_binary = (y_train > 0).long()  # Normal=0, Attack=1
    y_val_binary = (y_val > 0).long()
    
    print(f"   Normal samples: {(y_train_binary == 0).sum().item()}")
    print(f"   Attack samples: {(y_train_binary == 1).sum().item()}")
    
    # 4. Prepare Stage 2 Data (Attack types only, reindex 1-9 to 0-8)
    print("\n📊 Preparing Stage 2 (Attack-Type) Data...")
    attack_mask_train = y_train > 0
    X_train_attacks = X_train[attack_mask_train]
    y_train_attacks = y_train[attack_mask_train] - 1  # Reindex: 1-9 -> 0-8
    
    attack_mask_val = y_val > 0
    X_val_attacks = X_val[attack_mask_val]
    y_val_attacks = y_val[attack_mask_val] - 1
    
    print(f"   Attack training samples: {X_train_attacks.shape[0]}")
    print(f"   Attack validation samples: {X_val_attacks.shape[0]}")
    
    # 5. Create DataLoaders
    # Stage 1: Binary
    train_ds_s1 = torch.utils.data.TensorDataset(X_train, y_train_binary)
    val_ds_s1 = torch.utils.data.TensorDataset(X_val, y_val_binary)
    train_loader_s1 = torch.utils.data.DataLoader(train_ds_s1, batch_size=cfg.batch_size, shuffle=True)
    val_loader_s1 = torch.utils.data.DataLoader(val_ds_s1, batch_size=cfg.batch_size, shuffle=False)
    
    # Stage 2: Attack types
    train_ds_s2 = torch.utils.data.TensorDataset(X_train_attacks, y_train_attacks)
    val_ds_s2 = torch.utils.data.TensorDataset(X_val_attacks, y_val_attacks)
    train_loader_s2 = torch.utils.data.DataLoader(train_ds_s2, batch_size=cfg.batch_size, shuffle=True)
    val_loader_s2 = torch.utils.data.DataLoader(val_ds_s2, batch_size=cfg.batch_size, shuffle=False)
    
    # 6. Initialize Model
    model = TwoStageNet(input_dim=input_dim).to(device)
    
    # ========================================
    # STAGE 1 TRAINING: Binary Normal/Attack
    # ========================================
    print("\n" + "="*60)
    print("🎯 STAGE 1: Training Binary Normal/Attack Classifier")
    print("="*60)
    
    # Stage 1 criterion with class weights
    weight_method = cfg.get('weight_method', 'sqrt')
    if cfg.use_class_weights_s1:
        weights_s1 = get_class_weights(y_train_binary, device, method=weight_method)
        criterion_s1 = nn.CrossEntropyLoss(weight=weights_s1)
        print(f"⚖️ Stage 1 Weights ({weight_method}): {weights_s1}")
    else:
        criterion_s1 = nn.CrossEntropyLoss()
    
    # Stage 1 optimizer (only stage1 parameters)
    optimizer_s1 = optim.Adam(
        [p for name, p in model.named_parameters() if 'stage1' in name],
        lr=cfg.lr_s1
    )
    
    # Stage 1 scheduler
    if cfg.use_scheduler_s1:
        scheduler_s1 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_s1, mode='max', factor=0.5, patience=5
        )
        print("📉 Stage 1 Scheduler: ReduceLROnPlateau")
    
    # Stage 1 training loop
    best_f1_s1 = 0.0
    patience_s1 = cfg.get('early_stopping_patience_s1', 10)
    patience_counter_s1 = 0
    
    for epoch in range(cfg.epochs_s1):
        model.train()
        train_loss_s1 = 0
        
        for X_b, y_b in train_loader_s1:
            X_b, y_b = X_b.to(device), y_b.to(device)
            
            optimizer_s1.zero_grad()
            output = model.forward_stage1(X_b)
            loss = criterion_s1(output, y_b)
            loss.backward()
            optimizer_s1.step()
            
            train_loss_s1 += loss.item()
        
        # Stage 1 validation
        model.eval()
        preds_s1, labels_s1 = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader_s1:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model.forward_stage1(X_b)
                preds_s1.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels_s1.extend(y_b.cpu().numpy())
        
        f1_s1 = f1_score(labels_s1, preds_s1, average='binary')
        acc_s1 = accuracy_score(labels_s1, preds_s1)
        avg_loss_s1 = train_loss_s1 / len(train_loader_s1)
        
        print(f"[S1] Epoch {epoch+1:02d} | Loss: {avg_loss_s1:.4f} | Val F1: {f1_s1:.4f} | Val Acc: {acc_s1:.4f}")
        
        if cfg.use_scheduler_s1:
            scheduler_s1.step(f1_s1)
        
        # Early stopping
        if f1_s1 > best_f1_s1:
            best_f1_s1 = f1_s1
            patience_counter_s1 = 0
        else:
            patience_counter_s1 += 1
        
        if patience_counter_s1 >= patience_s1:
            print(f"⏹️ Stage 1 Early stopping at epoch {epoch+1} (patience={patience_s1})")
            print(f"🏆 Best Stage 1 F1: {best_f1_s1:.4f}")
            break
        
        logger.log_metrics({
            "stage1/train_loss": avg_loss_s1,
            "stage1/val_f1": f1_s1,
            "stage1/val_accuracy": acc_s1,
            "epoch": epoch + 1
        }, step=epoch + 1)
    
    print(f"\n✅ Stage 1 Training Complete - Best F1: {best_f1_s1:.4f}")
    
    # ========================================
    # STAGE 2 TRAINING: Attack-Type Classifier
    # ========================================
    print("\n" + "="*60)
    print("🎯 STAGE 2: Training Attack-Type Classifier")
    print("="*60)
    
    # Stage 2 criterion with class weights
    if cfg.use_class_weights_s2:
        weights_s2 = get_class_weights(y_train_attacks, device, method=weight_method)
        criterion_s2 = nn.CrossEntropyLoss(weight=weights_s2)
        print(f"⚖️ Stage 2 Weights ({weight_method}): {weights_s2}")
    else:
        criterion_s2 = nn.CrossEntropyLoss()
    
    # Stage 2 optimizer (only stage2 parameters)
    optimizer_s2 = optim.Adam(
        [p for name, p in model.named_parameters() if 'stage2' in name],
        lr=cfg.lr_s2
    )
    
    # Stage 2 scheduler
    if cfg.use_scheduler_s2:
        scheduler_s2 = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_s2, mode='max', factor=0.5, patience=5
        )
        print("📉 Stage 2 Scheduler: ReduceLROnPlateau")
    
    # Stage 2 training loop
    best_f1_s2 = 0.0
    patience_s2 = cfg.get('early_stopping_patience_s2', 10)
    patience_counter_s2 = 0
    
    for epoch in range(cfg.epochs_s2):
        model.train()
        train_loss_s2 = 0
        
        for X_b, y_b in train_loader_s2:
            X_b, y_b = X_b.to(device), y_b.to(device)
            
            optimizer_s2.zero_grad()
            output = model.forward_stage2(X_b)
            loss = criterion_s2(output, y_b)
            loss.backward()
            optimizer_s2.step()
            
            train_loss_s2 += loss.item()
        
        # Stage 2 validation
        model.eval()
        preds_s2, labels_s2 = [], []
        with torch.no_grad():
            for X_b, y_b in val_loader_s2:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model.forward_stage2(X_b)
                preds_s2.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels_s2.extend(y_b.cpu().numpy())
        
        f1_s2 = f1_score(labels_s2, preds_s2, average='macro')
        acc_s2 = accuracy_score(labels_s2, preds_s2)
        avg_loss_s2 = train_loss_s2 / len(train_loader_s2)
        
        print(f"[S2] Epoch {epoch+1:02d} | Loss: {avg_loss_s2:.4f} | Val Macro-F1: {f1_s2:.4f} | Val Acc: {acc_s2:.4f}")
        
        if cfg.use_scheduler_s2:
            scheduler_s2.step(f1_s2)
        
        # Early stopping
        if f1_s2 > best_f1_s2:
            best_f1_s2 = f1_s2
            patience_counter_s2 = 0
        else:
            patience_counter_s2 += 1
        
        if patience_counter_s2 >= patience_s2:
            print(f"⏹️ Stage 2 Early stopping at epoch {epoch+1} (patience={patience_s2})")
            print(f"🏆 Best Stage 2 Macro-F1: {best_f1_s2:.4f}")
            break
        
        logger.log_metrics({
            "stage2/train_loss": avg_loss_s2,
            "stage2/val_macro_f1": f1_s2,
            "stage2/val_accuracy": acc_s2,
            "epoch_s2": epoch + 1
        }, step=cfg.epochs_s1 + epoch + 1)
    
    print(f"\n✅ Stage 2 Training Complete - Best Macro-F1: {best_f1_s2:.4f}")
    
    # ========================================
    # COMBINED INFERENCE: Two-Stage Pipeline
    # ========================================
    print("\n" + "="*60)
    print("🔗 COMBINED INFERENCE: Two-Stage Pipeline")
    print("="*60)
    
    model.eval()
    final_preds = []
    final_labels = y_val.cpu().numpy()
    
    # Full validation set for combined inference
    val_loader_full = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val, y_val),
        batch_size=cfg.batch_size,
        shuffle=False
    )
    
    with torch.no_grad():
        for X_b, y_b in val_loader_full:
            X_b = X_b.to(device)
            
            # Stage 1: Binary prediction
            stage1_out = model.forward_stage1(X_b)
            stage1_preds = torch.argmax(stage1_out, dim=1)  # 0=Normal, 1=Attack
            
            # Stage 2: For predicted attacks, get attack type
            stage2_out = model.forward_stage2(X_b)
            stage2_preds = torch.argmax(stage2_out, dim=1)  # 0-8 (attack types)
            
            # Combine: If stage1 predicts Normal (0), keep 0; else use stage2 prediction + 1
            combined = torch.where(
                stage1_preds == 0,
                torch.zeros_like(stage1_preds),
                stage2_preds + 1  # Reindex back to 1-9
            )
            
            final_preds.extend(combined.cpu().numpy())
    
    # Calculate overall metrics
    macro_f1_combined = f1_score(final_labels, final_preds, average='macro')
    accuracy_combined = accuracy_score(final_labels, final_preds)
    
    print(f"\n🏆 FINAL RESULTS (Two-Stage Combined)")
    print(f"   Macro-F1: {macro_f1_combined:.4f}")
    print(f"   Accuracy: {accuracy_combined:.4f}")
    
    # Detailed classification report
    print("\nFinal Classification Report (Two-Stage Combined):")
    if label_map is not None:
        if isinstance(label_map, np.ndarray):
            target_names = label_map.tolist()
        elif isinstance(label_map, dict):
            if all(isinstance(v, int) for v in label_map.values()):
                target_names = [k for k, v in sorted(label_map.items(), key=lambda x: x[1])]
            else:
                target_names = list(label_map.keys())
        print(f"   📋 Using class names: {target_names}")
    else:
        target_names = None
    
    print(classification_report(final_labels, final_preds, target_names=target_names, zero_division=0))
    
    # Log final metrics
    logger.log_metrics({
        "combined/macro_f1": macro_f1_combined,
        "combined/accuracy": accuracy_combined,
        "stage1/best_f1": best_f1_s1,
        "stage2/best_macro_f1": best_f1_s2
    })
    
    logger.finish()
    print("\n✅ Two-Stage Experiment Complete!")

if __name__ == "__main__":
    main()
