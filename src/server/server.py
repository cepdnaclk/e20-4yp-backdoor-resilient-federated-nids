import torch
from sklearn.metrics import f1_score
from .aggregation import fed_avg, fed_median, fed_trimmed_mean, fed_krum

class Server:
    def __init__(self, global_model, test_loader, device='cpu', defense='avg'):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.defense = defense

    def aggregate(self, client_updates):
        """
        Orchestrates the aggregation.
        """
        # Separate weights from the tuples for the robust functions
        weights_list = [update[0] for update in client_updates]
        
        print(f"🛡️ Aggregating updates using defense: '{self.defense}'")

        # if self.defense not in aggregation:
        #     raise ValueError(f"Unknown defense: {self.defense}")

        if self.defense == "avg":
            new_weights = fed_avg(client_updates)
            
        elif self.defense == "median":
            new_weights = fed_median(weights_list)
            
        elif self.defense == "trimmed_mean":
            new_weights = fed_trimmed_mean(weights_list, beta=0.1)
            
        
        elif self.defense == "krum":
            new_weights = fed_krum(weights_list, n_malicious=1)
        
        else:
            print(f"⚠️ Unknown defense '{self.defense}', falling back to FedAvg.")
            new_weights = fed_avg(client_updates)

        # Apply the new weights to the global model
        self.global_model.load_state_dict(new_weights)

    def evaluate(self):
        """
        Calculates Standard Accuracy AND Macro F1-Score
        """
        self.global_model.eval()
        correct = 0
        total = 0
        
        # Lists to store all predictions for F1 calculation
        all_targets = []
        all_preds = []
        
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.global_model(X)
                _, predicted = torch.max(outputs.data, 1)
                
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Move to CPU for scikit-learn
                all_targets.extend(y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        # Calculate F1 (Macro average gives equal weight to all classes)
        # 0 division handling helps if a class is never predicted
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100
        
        return accuracy, f1  # 👈 Now returns TWO values

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
                
                # Filter: Only evaluate on samples that are NOT already the target.
                mask = (y != target)
                if mask.sum() == 0:
                    continue
                
                X_victim = X[mask].clone()
                X_victim[:, feat_idx] = trig_val
                
                outputs = self.global_model(X_victim)
                _, predicted = torch.max(outputs.data, 1)
                
                success_count += (predicted == target).sum().item()
                total_count += X_victim.size(0)
        
        if total_count == 0: return 0.0
        
        asr = 100 * success_count / total_count
        return asr