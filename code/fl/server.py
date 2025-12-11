import numpy as np
import os
import torch
import sys
import flwr as fl
from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import pickle

# --- Configuration (Robust Path Handling) ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

# Set environment variable so Ray workers can find the data directory
os.environ['REPO_ROOT'] = REPO_ROOT

# Ensure the repo root is in the Python path for the main process
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

DATA_PATH = os.path.join(REPO_ROOT, 'data')
PARTITIONS_DIR = os.path.join(REPO_ROOT, 'code', 'partitions')
NUM_CLIENTS = 10 
SCENARIO = 'dirichlet' # Current scenario: Dirichlet Label Skew

# Import client-side data loader and model definition
from code.models.nids_classifier import create_nids_model, NIDSClassifier
# CRITICAL FIX: Imports are now performed inside client_fn and load_global_test_data

# --- 1. Load Global Test Data for Server Evaluation ---

def load_global_test_data():
    """
    Loads the separate 30% test set for the central server to evaluate the global model.
    """
    try:
        X_test = np.load(os.path.join(DATA_PATH, 'X_global_test.npy'))
        y_test = np.load(os.path.join(DATA_PATH, 'y_global_test.npy'))
    except FileNotFoundError:
        print("Error: Global test data not found. Ensure data_preprocessor.py ran successfully.")
        return None, None
    
    # Convert to PyTorch Tensors for evaluation
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    
    return X_test, y_test

# --- 2. Server-Side Evaluation Function ---

def evaluate_global_model(server_round: int, parameters: fl.common.NDArrays, config: Dict):
    """
    Evaluates the aggregated global model on the server's clean, global test set.
    """
    X_test, y_test = load_global_test_data()
    
    # Get input shape from metadata
    metadata_file = os.path.join(PARTITIONS_DIR, 'client_partitions_metadata.pkl')
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    input_shape = metadata['X_shape'][1]
    
    # 1. Initialize the model and set received parameters
    model, device = create_nids_model(input_shape=input_shape)

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v).to(device) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    # 2. Perform Evaluation
    model.eval() # Set model to evaluation mode
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    with torch.no_grad():
        outputs = model(X_test)
        # Convert raw logits (no sigmoid) to binary predictions for accuracy calculation
        predictions = (outputs > 0.0).float() 
        
        # Calculate accuracy
        correct = (predictions == y_test).sum().item()
        accuracy = correct / len(y_test)
        
    print(f"Round {server_round} Server Evaluation - Global Model Accuracy: {accuracy:.4f}")
    
    return 0.0, {"accuracy": accuracy}

# --- 3. Client Factory Function (Mapping CIDs to Data) ---

def client_fn(cid: str) -> fl.client.Client:
    """
    Factory function to instantiate clients with specific data based on their ID.
    (CRITICAL FIX: Imports are now performed inside this function for Ray workers.)
    """
    # LOCAL IMPORTS: Must be done inside client_fn for Ray/Flower packaging
    from code.fl.client import NIDSClient, load_client_data 
    
    client_id = int(cid)
    
    # Load data for this specific client ID and the currently set scenario
    trainloader, data_size = load_client_data(client_id, scenario=SCENARIO)
    
    # Load input shape
    metadata_file = os.path.join(PARTITIONS_DIR, 'client_partitions_metadata.pkl')
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)
    input_shape = metadata['X_shape'][1]

    # Initialize model
    model, device = create_nids_model(input_shape=input_shape)
    
    # Return the instantiated Flower Client
    return NIDSClient(cid, model, trainloader, data_size, device).to_client()

# --- 4. FL Orchestration (Main Function) ---

if __name__ == "__main__":
    
    # FIX: Use the actual feature count determined during preprocessing
    initial_input_shape = 187 
    initial_model, _ = create_nids_model(input_shape=initial_input_shape)
    
    # Define fit configuration: Increased local epochs to force learning
    def fit_config(server_round: int) -> Dict:
        return {"epochs": 5}

    # Initialize strategy (FedAvg is the base strategy)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           
        fraction_evaluate=0.0,      
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_global_model,
        on_fit_config_fn=fit_config, # Apply the 5 local epochs configuration
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in initial_model.state_dict().items()]
        ) 
    )

    # Start the simulation with all 10 clients
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=50), 
        strategy=strategy,
        client_resources={"num_gpus": 1.0}, 
        ray_init_args={
            "ignore_reinit_error": True,
            # FINAL FIX: Ship the project root and set it as the working directory
            "runtime_env": {
                "working_dir": REPO_ROOT,
                # Setting this ensures the Ray workers look in the right place
                "env_vars": {"REPO_ROOT": REPO_ROOT}, 
            },
        },
    )
    
    print(f"Federated Learning simulation finished for scenario: {SCENARIO}")