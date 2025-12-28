import pandas as pd
import numpy as np
import torch
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
RAW_TRAIN_PATH = "data/unsw-nb15/raw/UNSW_NB15_training-set.csv"
RAW_TEST_PATH = "data/unsw-nb15/raw/UNSW_NB15_testing-set.csv"
PROCESSED_DIR = "data/unsw-nb15/processed"

# Features that follow a Power Law (huge range) and need Log transform
LOG_COLS = ['dur', 'sbytes', 'dbytes', 'Sload', 'Dload', 'Spkts', 'Dpkts']

# Categorical features to One-Hot Encode
CAT_COLS = ['proto', 'service', 'state']

def clean_and_process():
    print("🚀 Starting Data Preprocessing Pipeline...")

    # 1. LOAD & MERGE
    print(f"   📂 Loading raw files from {os.path.dirname(RAW_TRAIN_PATH)}...")
    df1 = pd.read_csv(RAW_TRAIN_PATH)
    df2 = pd.read_csv(RAW_TEST_PATH)
    
    # Concatenate to create one big pool for consistent scaling
    df = pd.concat([df1, df2], ignore_index=True)
    print(f"   ✅ Merged Dataset Shape: {df.shape}")

    # 2. BASIC CLEANING
    # Drop ID (useless) and 'label' (binary redundancy of attack_cat)
    # We keep 'attack_cat' as our target.
    drop_cols = ['id', 'label'] 
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # Handle NaNs (rare in UNSW, but good practice)
    df = df.fillna(0)

    #3. ENCODE TARGETS
    if 'Normal' in df['attack_cat'].unique():
        # Ensure Normal is 0
        df['attack_cat'] = df['attack_cat'].replace('Normal', 'Normal_Tmp')
        unique_attacks = ['Normal_Tmp'] + sorted([x for x in df['attack_cat'].unique() if x != 'Normal_Tmp'])
        label_map = {name: i for i, name in enumerate(unique_attacks)}
        # Fix the name back
        label_map['Normal'] = label_map.pop('Normal_Tmp')
        df['attack_cat'] = df['attack_cat'].replace('Normal_Tmp', 'Normal')
    else:
        label_map = {name: i for i, name in enumerate(sorted(df['attack_cat'].unique()))}
        
    df['attack_cat'] = df['attack_cat'].map(label_map)
    print(f"   ℹ️ Class Mapping: {label_map}")

    # 4. SEPARATE FEATURES & TARGET
    X = df.drop('attack_cat', axis=1)
    y = df['attack_cat']

    # 5. SPLIT BEFORE PROCESSING 
    # This ensures Test statistics never touch the Train scaler
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"   ✅ Data Split: Train {X_train.shape}, Test {X_test.shape}")

    # 6. DEFINE PREPROCESSING PIPELINE
    # Numerical Pipeline: Log -> Impute -> Scale
    # Note: We apply Log only to specific columns, but Scaling to ALL numeric
    # For simplicity in this script, we can manually log-transform X_train/X_test first
    # to avoid complex ColumnTransformer nesting.
    
    # --- MANUAL LOG TRANSFORM (Element-wise, so no leakage) ---
    for col in LOG_COLS:
        if col in X_train.columns:
            X_train[col] = np.log1p(X_train[col].clip(lower=0))
            X_test[col] = np.log1p(X_test[col].clip(lower=0))

    # Identify numeric/categorical columns AFTER split
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    
    # 7. FIT SCALERS ON TRAIN ONLY 🛑
    print("   ⚖️ Fitting Scalers on TRAIN set only...")
    scaler = StandardScaler()
    
    # Fit on Train, Transform Train
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    
    # Transform Test (using Train's Min/Max)
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # 8. HANDLE CATEGORICAL (Top-K on Train Only)
    print("   🏷️ Encoding Categorical (Top-10 from TRAIN only)...")
    
    for col in CAT_COLS:
        if col in X_train.columns:
            # 1. Determine Top 10 based on TRAIN data
            top_10 = X_train[col].value_counts().nlargest(10).index
            
            # 2. Apply to Train
            X_train[col] = X_train[col].apply(lambda x: x if x in top_10 else 'other')
            
            # 3. Apply to Test (Unknowns in Test become 'other' automatically)
            X_test[col] = X_test[col].apply(lambda x: x if x in top_10 else 'other')

    # One-Hot Encoding
    # We use pd.get_dummies but we must align columns because Test might miss a category
    # or have one we mapped to 'other'. 
    # Since we mapped everything not in Top-10 to 'other', columns should match exactly.
    X_train = pd.get_dummies(X_train, columns=CAT_COLS, dtype=float)
    X_test = pd.get_dummies(X_test, columns=CAT_COLS, dtype=float)

    # Re-align columns just in case (e.g., if 'other' didn't appear in one set)
    X_train, X_test = X_train.align(X_test, join='outer', axis=1, fill_value=0)

    # 9. SAVE
    print("   💾 Saving PyTorch Tensors...")
    
    # Convert to Float32/Int64 Tensors
    train_payload = {
        'X': torch.tensor(X_train.values.astype('float32')),
        'y': torch.tensor(y_train.values.astype('int64')),
        'label_map': label_map
    }
    
    test_payload = {
        'X': torch.tensor(X_test.values.astype('float32')),
        'y': torch.tensor(y_test.values.astype('int64')),
        'label_map': label_map
    }

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    torch.save(train_payload, f"{PROCESSED_DIR}/train_pool.pt")
    torch.save(test_payload, f"{PROCESSED_DIR}/global_test.pt")
    
    print(f"   ✅ Done! ")

if __name__ == "__main__":
    clean_and_process()