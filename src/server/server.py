import torch
from sklearn.metrics import f1_score
from .aggregation import fed_avg, fed_median, fed_trimmed_mean, fed_krum

class Server:
    def __init__(self, global_model, test_loader, device='cpu', defense='avg', expected_malicious=0):
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.defense = defense
        # Store expected malicious count for defenses like Krum/Trimmed Mean
        self.expected_malicious = expected_malicious

    def aggregate(self, client_updates):
        """
        Orchestrates the aggregation.
        """
        # Separate weights from the tuples for the robust functions
        weights_list = [update[0] for update in client_updates]
        
        print(f"🛡️ Aggregating updates using defense: '{self.defense}'")

        if self.defense == "avg":
            new_weights = fed_avg(client_updates)
            
        elif self.defense == "median":
            new_weights = fed_median(weights_list)
            
        elif self.defense == "trimmed_mean":
            # Dynamic Beta: Trim slightly more than the expected malicious fraction
            n_clients = len(weights_list)
            if n_clients > 0 and self.expected_malicious > 0:
                beta = self.expected_malicious / n_clients
            else:
                beta = 0.1 # Default 10%
            new_weights = fed_trimmed_mean(weights_list, beta=beta)
            
        elif self.defense == "krum":
            # Pass the actual expected number of attackers to Krum
            # Ensure f >= 1 if Krum is selected, otherwise it crashes
            f = max(1, self.expected_malicious)
            new_weights = fed_krum(weights_list, n_malicious=f)
        
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
                
                # Update Accuracy stats
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
                # Store for F1 Score
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        accuracy = 100 * correct / total
        # Calculate Macro F1 (treats all classes equally, critical for NIDS)
        f1 = f1_score(all_targets, all_preds, average='macro') 
        return accuracy, f1

    def test_attack_efficacy(self, attack_config):
        """
        Calculates Attack Success Rate (ASR) for the active attack type.
        """
        if attack_config is None:
            return 0.0
            
        # 1. Unpack Attack Type
        try:
            atype = attack_config.type
        except AttributeError:
            atype = attack_config.get('type', 'clean')

        if atype == 'clean':
            return 0.0

        self.global_model.eval()
        success_count = 0
        total_count = 0
        
        # Unpack Common Parameters
        try:
            target = attack_config.target_label
            feat_idx = attack_config.trigger_feat_idx
            trig_val = attack_config.trigger_value
            source_label = attack_config.source_label
            flip_to = attack_config.flip_to_label
        except AttributeError:
            target = attack_config.get('target_label')
            feat_idx = attack_config.get('trigger_feat_idx')
            trig_val = attack_config.get('trigger_value')
            source_label = attack_config.get('source_label')
            flip_to = attack_config.get('flip_to_label')

        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                
                # --- STRATEGY SWITCH ---
                if atype == 'backdoor':
                    # Backdoor Logic: Only target NON-target samples
                    mask = (y != target)
                    if mask.sum() == 0: continue
                    
                    X_victim = X[mask].clone()
                    if feat_idx is not None:
                        X_victim[:, feat_idx] = trig_val
                    
                    target_class = target

                elif atype == 'label_flip':
                    # Label Flip Logic: Only target SOURCE samples
                    mask = (y == source_label)
                    if mask.sum() == 0: continue
                    
                    X_victim = X[mask].clone()
                    target_class = flip_to

                else:
                    return 0.0

                # --- COMMON EVALUATION ---
                outputs = self.global_model(X_victim)
                _, predicted = torch.max(outputs.data, 1)
                
                success_count += (predicted == target_class).sum().item()
                total_count += X_victim.size(0)
        
        if total_count == 0: return 0.0
        asr = 100 * success_count / total_count
        return asr