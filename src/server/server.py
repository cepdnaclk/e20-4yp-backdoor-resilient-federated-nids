import torch
import copy
import numpy as np

class Server:
    # 👇 CHANGE 1: Added defense parameter
    def __init__(self, global_model, test_loader, device='cpu', defense='avg'):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.defense = defense  # Store it for later use

    def aggregate(self, client_updates):
        """
        Router function: Picks the math based on config.
        """
        # 👇 CHANGE 2: Check which defense to use
        if self.defense == 'avg':
            return self._aggregate_fedavg(client_updates)
        
        elif self.defense == 'median':
            print("⚠️ Median aggregation selected (Placeholder for Sprint 3)")
            # For now, fall back to average until we implement Median
            return self._aggregate_fedavg(client_updates)
            
        else:
            print(f"⚠️ Unknown defense '{self.defense}', defaulting to FedAvg.")
            return self._aggregate_fedavg(client_updates)

    # 👇 CHANGE 3: Moved the math here
    def _aggregate_fedavg(self, client_updates):
        """
        The Standard Weighted Average Logic
        """
        total_samples = sum([update[1] for update in client_updates])
        
        # Start with empty weights
        new_weights = copy.deepcopy(client_updates[0][0])
        for key in new_weights.keys():
            new_weights[key] = torch.zeros_like(new_weights[key])
            
        # Accumulate
        for weights, n_samples, _ in client_updates:
            weight_factor = n_samples / total_samples
            for key in weights.keys():
                new_weights[key] += weights[key] * weight_factor
                
        self.global_model.load_state_dict(new_weights)
        return new_weights

    def evaluate(self):
        """
        Calculates Standard Accuracy (Main Task Accuracy)
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.global_model(X)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

    def test_backdoor(self, attack_config):
        """
        Calculates Attack Success Rate (ASR).
        """
        if attack_config is None or attack_config.get('type') == 'clean':
            return 0.0

        self.global_model.eval()
        success_count = 0
        total_count = 0
        
        try:
            target = attack_config.target_label
            feat_idx = attack_config.trigger_feat_idx
            trig_val = attack_config.trigger_value
        except AttributeError:
            target = attack_config.get('target_label')
            feat_idx = attack_config.get('trigger_feat_idx')
            trig_val = attack_config.get('trigger_value')

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # Filter out samples that are already the target
                mask = (y != target)
                if mask.sum() == 0:
                    continue
                
                X_victim = X[mask].clone()
                # Inject Trigger
                X_victim[:, feat_idx] = trig_val
                
                outputs = self.global_model(X_victim)
                _, predicted = torch.max(outputs.data, 1)
                
                success_count += (predicted == target).sum().item()
                total_count += X_victim.size(0)
        
        if total_count == 0: return 0.0
        
        asr = 100 * success_count / total_count
        return asr