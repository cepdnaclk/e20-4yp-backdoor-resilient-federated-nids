# --- code/fl/client.py ---
import numpy as np
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import flwr as fl
from collections import OrderedDict
# Import model creation function
from code.models.nids_classifier import create_nids_model 

# --- Configuration (Robust Path Handling) ---

# First, try to use REPO_ROOT from environment (set by Ray workers)
# Otherwise, compute it from the module's location
if 'REPO_ROOT' in os.environ:
    REPO_ROOT = os.environ['REPO_ROOT']
else:
    current_file = os.path.abspath(__file__)
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..'))

DATA_PATH = os.path.join(REPO_ROOT, 'data')
PARTITIONS_DIR = os.path.join(REPO_ROOT, 'code', 'partitions')

# --- 1. PyTorch Dataset and DataLoader for NIDS Data ---

class NIDSDataset(Dataset):
    """Custom PyTorch Dataset for loading NIDS features and labels."""
    def __init__(self, X_data, y_data):
        self.X = torch.tensor(X_data, dtype=torch.float32)
        self.y = torch.tensor(y_data, dtype=torch.float32).unsqueeze(1) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_client_data(client_id, scenario):
    """
    Loads the processed data and retrieves the Non-IID subset assigned to the client.
    """
    # Load all processed data
    X_full = np.load(os.path.join(DATA_PATH, 'X_full_train.npy'))
    y_full = np.load(os.path.join(DATA_PATH, 'y_full_train.npy'))

    # Load the partitioning metadata map
    metadata_file = os.path.join(PARTITIONS_DIR, 'client_partitions_metadata.pkl')
    with open(metadata_file, 'rb') as f:
        metadata = pickle.load(f)

    # Get the indices specific to this client and scenario
    indices = metadata[scenario][client_id]

    # Retrieve the client's subset
    X_client = X_full[indices]
    y_client = y_full[indices]

    # Handle case where a client has no data (edge case in partitioning)
    if len(X_client) == 0:
        print(f"Warning: Client {client_id} has no data. Returning dummy data with batch_size=2 to allow BatchNorm.")
        # Return dummy dataset with at least 2 samples to satisfy BatchNorm1d minimum requirement
        X_dummy = np.zeros((2, X_full.shape[1]), dtype=np.float32)
        y_dummy = np.zeros((2,), dtype=np.float32)
        dataloader = DataLoader(NIDSDataset(X_dummy, y_dummy), batch_size=2, shuffle=False)
        return dataloader, 0

    # Create PyTorch DataLoader
    dataloader = DataLoader(NIDSDataset(X_client, y_client), batch_size=512, shuffle=True) 
    
    return dataloader, len(X_client)

# --- 2. Flower Client Implementation ---

class NIDSClient(fl.client.NumPyClient):
    """
    Implements the Federated Learning Client for PyTorch NIDS model.
    """
    def __init__(self, cid, model, trainloader, data_size, device):
        self.cid = cid
        self.model = model
        self.trainloader = trainloader
        self.data_size = data_size
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # FIX: Define POSITIVE WEIGHT for Binary Cross-Entropy Loss to handle imbalance
        # Weighting the minority (Attack=1) class by 5x (common starting point for NIDS)
        pos_weight = torch.tensor([5.0], dtype=torch.float32).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight) 

    def get_parameters(self, config):
        """Returns the local model's parameters (weights) to the server."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Sets the global parameters received from the server to the local model."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Implements the local training loop for the client."""
        self.set_parameters(parameters)
        local_epochs = config.get("epochs", 1) # Default to 1 epoch
        
        self.model.train()
        for epoch in range(local_epochs):
            for X_batch, y_batch in self.trainloader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                
        return self.get_parameters(config={}), self.data_size, {}

    def evaluate(self, parameters, config):
        """Clients will not perform evaluation for this baseline run."""
        return 0.0, 0, {}