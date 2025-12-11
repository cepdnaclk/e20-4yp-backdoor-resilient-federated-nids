import torch
import torch.nn as nn
import torch.nn.functional as F
import os # Included for robustness, though mostly used by client/server

class NIDSClassifier(nn.Module):
    """
    Enhanced Multi-Layer Perceptron (MLP) for Binary NIDS Classification.
    Uses Batch Normalization and Dropout.
    """
    def __init__(self, input_shape): 
        super(NIDSClassifier, self).__init__()
        
        HIDDEN_SIZE_1 = 128
        HIDDEN_SIZE_2 = 64
        HIDDEN_SIZE_3 = 32
        DROPOUT_RATE = 0.3
        
        # --- Feature Extraction Layers ---
        self.fc1 = nn.Linear(input_shape, HIDDEN_SIZE_1)
        self.bn1 = nn.BatchNorm1d(HIDDEN_SIZE_1)
        
        self.fc2 = nn.Linear(HIDDEN_SIZE_1, HIDDEN_SIZE_2)
        self.bn2 = nn.BatchNorm1d(HIDDEN_SIZE_2)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)
        
        self.fc3 = nn.Linear(HIDDEN_SIZE_2, HIDDEN_SIZE_3)
        self.bn3 = nn.BatchNorm1d(HIDDEN_SIZE_3)
        
        # --- Output Layer (No Sigmoid here, handled by BCEWithLogitsLoss) ---
        self.fc_out = nn.Linear(HIDDEN_SIZE_3, 1) 

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Output: Raw logits
        x = self.fc_out(x)
        return x

def create_nids_model(input_shape):
    """Initializes the model and moves it to the Ampere server's GPU."""
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = NIDSClassifier(input_shape=input_shape).to(device)
    return model, device