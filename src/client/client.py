import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from src.client.attacker import Attacker
from src.utils.class_weights import get_class_weights

class Client:
    def __init__(self, client_id, dataset, indices, model, config , lr=0.01 , device='cpu', is_malicious=False, num_classes=2):
        self.client_id = client_id
        self.device = device
        self.is_malicious = is_malicious
        self.lr = lr
        self.config = config
        
        # 1. Create Local Data Slice
        self.clean_dataset = Subset(dataset, indices)
        self.dataset = self.clean_dataset
        
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
            self.class_weights = get_class_weights(local_labels, device, method=weight_method, num_classes=num_classes)
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
        # 🛡️🔥 FLAME Evasion: Train shadow model on clean data to get reference update
        evade_flame = self.is_malicious and self.config.attack.get('flame_evasion', False)
        clean_delta_flat = None
        global_weights_dev = {k: v.to(self.device) for k, v in global_weights.items()}

        if evade_flame:
            # print(f"🥷 Client {self.client_id} executing FLAME Evasion (Gradient Alignment)...")
            clean_model = copy.deepcopy(self.model).to(self.device)
            clean_model.load_state_dict(global_weights)
            clean_model.train()
            clean_opt = optim.Adam(clean_model.parameters(), lr=self.lr)
            clean_loader = DataLoader(self.clean_dataset, batch_size=batch_size, shuffle=True)
            
            for _ in range(epochs):
                for b_X, b_y in clean_loader:
                    b_X, b_y = b_X.to(self.device), b_y.to(self.device)
                    clean_opt.zero_grad()
                    out = clean_model(b_X)
                    l = self.criterion(out, b_y)
                    l.backward()
                    clean_opt.step()
            
            # Compute clean update delta
            with torch.no_grad():
                c_delta_parts = []
                for name, param in clean_model.named_parameters():
                     if name in global_weights_dev:
                         c_delta_parts.append((param - global_weights_dev[name]).view(-1))
                clean_delta_flat = torch.cat(c_delta_parts).detach()

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
        penalty_lambda = self.config.attack.get('flame_evasion_lambda', 5.0) if hasattr(self.config, 'attack') else 5.0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                # 🥷 APPLY ALIGNMENT PENALTY
                if evade_flame and clean_delta_flat is not None:
                    curr_delta_parts = []
                    # Keep track of grads
                    for name, param in self.model.named_parameters():
                        # Parameter order matches state_dict keys because we iterate consistently
                        # Actually named_parameters() might be slightly different than state_dict().keys()
                        # To be safe, match by name
                        pass
                    
                    # Safer way matching global_weights_dev
                    for k, v in self.model.named_parameters():
                         if k in global_weights_dev:
                             curr_delta_parts.append((v - global_weights_dev[k]).view(-1))
                    
                    curr_delta_flat = torch.cat(curr_delta_parts)
                    
                    curr_norm = torch.norm(curr_delta_flat)
                    clean_norm = torch.norm(clean_delta_flat)
                    
                    if curr_norm > 1e-6 and clean_norm > 1e-6:
                        # Compute cosine similarity
                        cos_sim = torch.nn.functional.cosine_similarity(
                            curr_delta_flat.unsqueeze(0), 
                            clean_delta_flat.unsqueeze(0)
                        )[0]
                        
                        # We want cos_sim to be 1.0 (exact direction match)
                        align_loss = 1.0 - cos_sim
                        loss = loss + penalty_lambda * align_loss

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
        
        # Apply post-training attacker scaling when strategy requires it
        if self.is_malicious and hasattr(self, 'attacker'):
            use_aggressive = self.config.attack.get("aggressive", False)
            use_stealth = self.config.attack.get("stealth", False)
            if use_aggressive or use_stealth:
                final_weights = self.attacker.scale_update(global_weights, final_weights)

        return final_weights, len(self.dataset), avg_loss