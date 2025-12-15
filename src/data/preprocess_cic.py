import pandas as pd
import numpy as np
import torch
import os
import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAW_DIR = "data/cic-ids-2017/raw/"
PROCESSED_DIR = "data/cic-ids-2017/processed"

# CIC-IDS2017 specific messiness: Columns to drop
# 'Fwd Header Length' is often duplicated. 'Destination Port' is metadata.
DROP_COLS = ['Destination Port', 'Fwd Header Length.1'] 

def clean_cic_ids2017():
    print("🚀 Starting CIC-IDS2017 Adapter...")

    # 1. LOAD AND MERGE CSVs
    # The dataset is split into 8 files. We load them all.
    all_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"❌ No CSV files found in {RAW_DIR}")
    
    print(f"   📂 Found {len(all_files)} files. Merging... (This may take RAM)")
    
    df_list = []
    for filename in all_files:
        print(f"      Reading {os.path.basename(filename)}...")
        # CIC CSVs have spaces in headers (e.g., " Flow Duration"). We strip them.
        df_temp = pd.read_csv(filename)
        df_temp.columns = df_temp.columns.str.strip()
        df_list.append(df_temp)

    df = pd.concat(df_list, ignore_index=True)
    print(f"   ✅ Merged Dataset Shape: {df.shape}")

    # 2. CLEANING (The "Adapter" Logic)
    
    # A. Handle "Infinity" and NaNs (Common CIC-IDS2017 bugs)
    # Replace infinite values with NaN, then drop NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"   🧹 Removed Inf/NaN values. New Shape: {df.shape}")

    # B. Drop useless columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], axis=1, inplace=True)

    # 3. LABEL ENCODING
    # The column is 'Label' (e.g., 'BENIGN', 'FTP-Patator')
    # We consolidate rare attacks if needed, but for now let's keep them raw.
    print("   🎯 Encoding Targets...")
    
    # Rename target column to standard 'label' for consistency
    if 'Label' in df.columns:
        df.rename(columns={'Label': 'label'}, inplace=True)
        
    unique_labels = sorted(df['label'].unique())
    # Ensure BENIGN is 0
    if 'BENIGN' in unique_labels:
        unique_labels.remove('BENIGN')
        unique_labels.insert(0, 'BENIGN')
        
    label_map = {name: i for i, name in enumerate(unique_labels)}
    df['label'] = df['label'].map(label_map)
    print(f"   ℹ️ Class Mapping: {label_map}")

    # 4. NUMERICAL SCALING
    print("   ⚖️ Scaling Features...")
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Save feature names for later analysis
    feature_names = list(X.columns)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. SPLIT & SAVE (Standard Format)
    print("   💾 Saving PyTorch Tensors...")
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Crucial: Save as float32 to match the Model weights
    train_payload = {
        'X': torch.tensor(X_train, dtype=torch.float32),
        'y': torch.tensor(y_train.values, dtype=torch.long),
        'label_map': label_map,
        'feature_names': feature_names
    }
    
    test_payload = {
        'X': torch.tensor(X_test, dtype=torch.float32),
        'y': torch.tensor(y_test.values, dtype=torch.long),
        'label_map': label_map
    }

    torch.save(train_payload, f"{PROCESSED_DIR}/train_pool.pt")
    torch.save(test_payload, f"{PROCESSED_DIR}/global_test.pt")

    print(f"   ✅ SUCCESS! Processed CIC-IDS2017 saved to {PROCESSED_DIR}")
    print(f"   📊 Input Dimension: {X_train.shape[1]} (Model must update!)")

if __name__ == "__main__":
    clean_cic_ids2017()