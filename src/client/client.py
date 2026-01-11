import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from src.client.attacker import Attacker

# 🆕 ADD THIS CLASS AT THE TOP
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Class weights
        self.gamma = gamma  # Focusing parameter (2.0 is standard)
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

class Client:
    def __init__(self, client_id, dataset, indices, model, config, lr=0.01, device='cpu', is_malicious=False, class_weights=None):
        # ... (Keep init logic same) ...
        self.client_id = client_id
        self.device = device
        self.is_malicious = is_malicious
        self.lr = lr
        self.config = config
        self.dataset = Subset(dataset, indices)

        # Red Team Logic ...
        if self.is_malicious:
            # ... (keep attacker logic) ...
            self.attacker = Attacker(config)
            self.dataset = self.attacker.poison_dataset(self.dataset)

        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)

        # 🆕 SWITCH TO FOCAL LOSS
        # We still use the calculated class_weights as 'alpha' but Focal Loss handles the rest
        if class_weights is not None:
            weights = class_weights.to(self.device)
            self.criterion = FocalLoss(alpha=weights, gamma=2.0)
        else:
            self.criterion = FocalLoss(gamma=2.0)

    # ... (Keep train method exactly as it is) ...
    def train(self, global_weights, epochs=1, batch_size=32):
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        epoch_loss = 0.0
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        final_weights = self.model.state_dict()
        if self.is_malicious and hasattr(self, 'attacker'):
            final_weights = self.attacker.scale_update(global_weights, final_weights)

        return final_weights, len(self.dataset), avg_loss