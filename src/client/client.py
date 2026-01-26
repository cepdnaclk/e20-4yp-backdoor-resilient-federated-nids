import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from src.client.attacker import Attacker
from src.utils.class_weights import get_class_weights

class Client:
    def __init__(self, client_id, dataset, indices, model, config , lr=0.01 , device='cpu', is_malicious=False):
        self.client_id = client_id
        self.device = device
        self.is_malicious = is_malicious
        self.lr = lr
        self.config = config
        
        # 1. Create Local Data Slice
        self.dataset = Subset(dataset, indices)
        
        # 2. RED TEAM INTEGRATION 😈
        if self.is_malicious:
            print(f"⚠️ Client {client_id} is MALICIOUS! Initializing Attacker...")
            self.attacker = Attacker(config)
            # Replace honest dataset with poisoned dataset
            self.dataset = self.attacker.poison_dataset(self.dataset)
        
        # 3. M12: Calculate local class weights
        if config.client.get('use_class_weights', False):
            local_labels = torch.tensor([dataset[i][1] for i in indices])
            weight_method = config.client.get('weight_method', 'sqrt')
            self.class_weights = get_class_weights(local_labels, device, method=weight_method)
            # print(f"   Client {client_id}: Using {weight_method} class weights")
        else:
            self.class_weights = None
        
        # 4. Local Model Setup
        self.model = copy.deepcopy(model).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        if self.class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

    def train(self, global_weights, epochs=1, batch_size=32):
        # Load global weights
        self.model.load_state_dict(global_weights)
        self.model.train()
        
        train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        # Reset optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Scheduler
        use_scheduler = self.config.client.get('use_scheduler', False)
        if use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=2
            )
        
        # Early stopping
        patience = self.config.client.get('early_stopping_patience', None)
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            
            if use_scheduler:
                scheduler.step(avg_epoch_loss)
            
            if patience is not None:
                if avg_epoch_loss < best_loss:
                    best_loss = avg_epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    break
        
        avg_loss = best_loss if patience is not None else avg_epoch_loss

        # 🆕 ATTACK LOGIC
        final_weights = self.model.state_dict()
        
        # Check if malicious AND if 'aggressive' mode is enabled in config
        if self.is_malicious and hasattr(self, 'attacker'):
            # 🔴 CRITICAL FIX: Only run Model Replacement if aggressive=True
            if self.config.attack.get("aggressive", False):
                final_weights = self.attacker.scale_update(global_weights, final_weights)

        return final_weights, len(self.dataset), avg_loss