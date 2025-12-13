import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from src.client.attacker import Attacker

class Client:
    def __init__(self, client_id, dataset, indices, model, config, device='cpu', is_malicious=False):
        self.client_id = client_id
        self.device = device
        self.is_malicious = is_malicious
        self.config = config
        
        # 1. Create Local Data Slice
        self.dataset = Subset(dataset, indices)
        
        # 2. RED TEAM INTEGRATION 😈
        # If this client is malicious, we POISON the data immediately
        if self.is_malicious:
            print(f"⚠️ Client {client_id} is MALICIOUS! Initializing Attacker...")
            self.attacker = Attacker(config)
            # Replace honest dataset with poisoned dataset
            self.dataset = self.attacker.poison_dataset(self.dataset)
        
        # 3. Local Model Setup
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights, epochs=1, batch_size=32):
        """
        Standard FL Training Loop
        """
        # Load global weights
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
        
        # Return new weights
        return self.model.state_dict(), len(self.dataset), epoch_loss / len(train_loader)