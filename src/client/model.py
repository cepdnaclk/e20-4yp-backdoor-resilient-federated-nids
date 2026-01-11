import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_dim=71, num_classes=10):
        super(Net, self).__init__()
        
        # 4-Layer DNN (Deep Neural Network)
        # We use powers of 2 for neuron counts (standard practice)
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.2) # Prevents overfitting

    def forward(self, x):
        # Layer 1
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Layer 2
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Layer 3
        x = F.relu(self.fc3(x))
        
        # Output Layer (No Softmax here! PyTorch CrossEntropyLoss handles it)
        x = self.fc4(x)
        return x

class WiderNet(nn.Module):
    """Wider network with more capacity for imbalanced learning"""
    def __init__(self, input_dim=71, num_classes=10):
        super(WiderNet, self).__init__()
        
        # Wider architecture (2x capacity)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        
        x = self.fc4(x)
        return x


class TwoStageNet(nn.Module):
    """Two-stage architecture: Stage 1 (Normal vs Attack) + Stage 2 (Attack subtypes)"""
    def __init__(self, input_dim=71):
        super(TwoStageNet, self).__init__()
        
        # Stage 1: Binary classifier (Normal=0 vs Attack=1)
        self.stage1_fc1 = nn.Linear(input_dim, 128)
        self.stage1_fc2 = nn.Linear(128, 64)
        self.stage1_fc3 = nn.Linear(64, 32)
        self.stage1_fc4 = nn.Linear(32, 2)  # Binary: Normal vs Attack
        
        # Stage 2: Attack-type classifier (9 attack classes)
        # Classes: Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms
        self.stage2_fc1 = nn.Linear(input_dim, 128)
        self.stage2_fc2 = nn.Linear(128, 64)
        self.stage2_fc3 = nn.Linear(64, 32)
        self.stage2_fc4 = nn.Linear(32, 9)  # 9 attack types
        
        self.dropout = nn.Dropout(0.2)

    def forward_stage1(self, x):
        """Stage 1: Binary Normal/Attack classification"""
        x = F.relu(self.stage1_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.stage1_fc2(x))
        x = self.dropout(x)
        x = F.relu(self.stage1_fc3(x))
        x = self.stage1_fc4(x)
        return x
    
    def forward_stage2(self, x):
        """Stage 2: Multi-class attack-type classification"""
        x = F.relu(self.stage2_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.stage2_fc2(x))
        x = self.dropout(x)
        x = F.relu(self.stage2_fc3(x))
        x = self.stage2_fc4(x)
        return x
    
    def forward(self, x, stage='combined'):
        """
        Forward pass with flexible stage selection
        stage: 'stage1', 'stage2', or 'combined' (default)
        """
        if stage == 'stage1':
            return self.forward_stage1(x)
        elif stage == 'stage2':
            return self.forward_stage2(x)
        else:  # combined inference
            stage1_out = self.forward_stage1(x)
            return stage1_out, self.forward_stage2(x)