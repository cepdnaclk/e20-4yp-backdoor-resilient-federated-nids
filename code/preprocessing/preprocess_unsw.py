import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split # Used for initial split
from sklearn.compose import ColumnTransformer

# --- Configuration and Path Setup ---
# Assumes the script is run from the project root (e20-4yp-backdoor-resilient-federated-nids)
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
# Path where the raw UNSW-NB15_*.csv files are located
UNSW_RAW_PATH = os.path.join(REPO_ROOT, 'data', 'CSV Files') 
# Path where the final processed .npy files will be saved
PROCESSED_DATA_PATH = os.path.join(REPO_ROOT, 'data')

# Corrected, definitive list of 49 column names for the raw CSV files
UNSW_COLS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
    'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 
    'attack_cat', 'label'
]

# --- Feature Definition ---
# Categorical features requiring One-Hot Encoding
CATEGORICAL_FEATURES = ['proto', 'service', 'state']

# Numerical features (all others, excluding targets and categorical features)
# Note: We must exclude 'srcip', 'dstip' (nominal/IPs), and target labels.
# This list will be dynamically determined later, but we list those that will be numerical:
NUMERICAL_FEATURES_LIST = [
    'sport', 'dsport', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload', 
    'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
    'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 
    'tcprtt', 'synack', 'ackdat', 'ct_state_ttl', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
    'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'ct_src_ ltm'
]
# We drop 'srcip' and 'dstip' which are nominal (IP addresses) and often not used directly in ML.

def load_and_transform_unsw():
    """
    Loads the raw data, performs initial cleaning, and applies the combined transformation pipeline.
    """
    print("--- 1. Loading and Combining Raw Data ---")
    all_unsw_files = [os.path.join(UNSW_RAW_PATH, f'UNSW-NB15_{i}.csv') for i in range(1, 5)]
    all_dfs = []
    
    for filepath in all_unsw_files:
        try:
            # Read without header (header=None) and assign the 49 correct column names
            df = pd.read_csv(filepath, header=None, low_memory=False) 
            df.columns = UNSW_COLS
            all_dfs.append(df)
        except Exception as e:
            print(f"Error loading {filepath}: {e}. Stopping.")
            return None, None

    df_data = pd.concat(all_dfs, axis=0, ignore_index=True)
    
    # Initial Cleaning: Convert 'label' and handle NaN/Infinite values
    df_data.dropna(inplace=True)
    df_data['label'] = df_data['label'].astype(int)

    # Separate Features (X) and Target (y)
    # We drop the attack_cat (multi-class label) as the project focuses on binary IDS
    cols_to_drop = ['srcip', 'dstip', 'attack_cat', 'label'] 
    X = df_data.drop(columns=cols_to_drop)
    y = df_data['label'].values 
    
    # Convert 'object' types that should be numeric (often done for features like 'Sload')
    for col in NUMERICAL_FEATURES_LIST:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0) # Fill NaN from coercion with 0

    print(f"Total cleaned samples: {len(X)}")
    
    # --- 2. Train/Test Split (Prevent Leakage) ---
    # Split the data into 70% training and 30% testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {len(X_train)}; Testing set size: {len(X_test)}")

    # --- 3. Define Preprocessing Pipeline ---
    # Use Standard Scaling for numerical features and One-Hot Encoding for categorical features.
    
    # Identify the actual numerical feature columns in the training set
    train_numerical_features = [col for col in NUMERICAL_FEATURES_LIST if col in X_train.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            # Standard Scaler (preferred for NIDS/FL)
            ('num', StandardScaler(), train_numerical_features), 
            # One-Hot Encoder
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES) 
        ],
        remainder='passthrough') 

    # --- 4. Fit and Apply Transformation ---
    
    # 💥 Fit the preprocessor ONLY on the training data to prevent leakage!
    preprocessor.fit(X_train) 

    # Apply the transformation to both sets
    X_train_scaled = preprocessor.transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)
    
    print(f"Processed feature shape (Training): {X_train_scaled.shape}")
    
    return X_train_scaled, y_train, X_test_scaled, y_test

def save_processed_data(X_train, y_train, X_test, y_test):
    """
    Saves the final processed NumPy arrays for use by the partitioning scripts.
    """
    print("\n--- 5. Saving Processed Data ---")
    
    # Save the full training set (features and labels) - this is what is partitioned in FL
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_full_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_full_train.npy'), y_train)

    # Save the test set separately for global model evaluation
    np.save(os.path.join(PROCESSED_DATA_PATH, 'X_global_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DATA_PATH, 'y_global_test.npy'), y_test)
    
    print("Processed training and testing data successfully saved to .npy files.")

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_and_transform_unsw()
    
    if X_train is not None:
        save_processed_data(X_train, y_train, X_test, y_test)
        
        # Next step in the pipeline: Partitioning
        print("\nPreprocessing complete. Ready for Non-IID Partitioning.")