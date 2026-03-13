import torch
from sklearn.metrics import f1_score
from .aggregation import fed_avg, fed_median, fed_trimmed_mean, fed_krum, fed_multi_krum, fed_adaptive_clipping, sentinel_aggregate
from src.server.clustering import flame_clustering, sentinel_filtering, reset_sentinel_state

class Server:
    def __init__(self, config, global_model, test_loader, device='cpu', defense='avg', expected_malicious=0, num_classes=10):
        self.config = config
        self.global_model = global_model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.defense = defense
        # Store expected malicious count for defenses like Krum/Trimmed Mean
        self.expected_malicious = expected_malicious
        self.num_classes = num_classes
        # Round tracking for convergence-aware defenses (SENTINEL v7)
        self.round_counter = 0
        self.total_rounds = config.simulation.get('rounds', 20)
        # Reset cross-round reputation state for a clean experiment
        if defense == 'sentinel':
            reset_sentinel_state()


    def aggregate(self, client_updates):
        """
        Orchestrates the aggregation.
        """
        self.round_counter += 1
        # Separate weights from the tuples for the robust functions
        weights_list = [update[0] for update in client_updates]

        # Define n_clients HERE (Top Level)
        n_clients = len(weights_list)
        
        print(f"🛡️ Aggregating {n_clients} updates using defense: '{self.defense}'")

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

        elif self.defense == "multi_krum":
            # Multi-Krum: Selects top 'm' clients and averages them
            # Heuristic: Assume <30% attackers (f), select the remaining 70% (m)
            f = max(1, int(n_clients * 0.45))
            m = n_clients - f
            new_weights = fed_multi_krum(weights_list, f=f, m=m)
            
##########################################################################################
        #                            FLAME

        elif self.defense == "flame":
            # 1. CLUSTERING (Filter out bad directions — cluster deltas, not full weights)
            filtered_weights = flame_clustering(weights_list, self.global_model.state_dict())
            
            # 2. ADAPTIVE CLIPPING + NOISING (Filter scale + Add privacy)
            # We pass the privacy config section here
            privacy_cfg = self.config.server.get('privacy', None)
            
            new_weights = fed_adaptive_clipping(
                filtered_weights, 
                self.global_model.state_dict(),
                privacy_cfg=privacy_cfg
            )
                    
        elif self.defense == "adaptive_clipping":
             # You can also add noise to the standalone clipping defense if you want
             privacy_cfg = self.config.server.get('privacy', None)
             new_weights = fed_adaptive_clipping(
                 weights_list, 
                 self.global_model.state_dict(),
                 privacy_cfg=privacy_cfg
             )

##########################################################################################
#                          SENTINEL (Our Defense)

        elif self.defense == "sentinel":
            # 1. SYBIL-AWARE FILTERING (remove top expected_malicious clients)
            sensitivity = self.config.server.get('sentinel_sensitivity', 1.5)
            filtered_weights = sentinel_filtering(
                weights_list,
                self.global_model.state_dict(),
                sensitivity=sensitivity,
                expected_malicious=self.expected_malicious,
                round_num=self.round_counter,
                total_rounds=self.total_rounds
            )

            # 2. ROBUST AGGREGATION: trimmed median + DP noise
            privacy_cfg = self.config.server.get('privacy', None)
            new_weights = sentinel_aggregate(
                filtered_weights,
                self.global_model.state_dict(),
                privacy_cfg=privacy_cfg,
                expected_malicious=self.expected_malicious
            )

###########################################################################################
        
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
        # Calculate F1-score: binary average for 2-class, macro for multiclass
        if self.num_classes == 2:
            f1 = f1_score(all_targets, all_preds, average='binary')
        else:
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

def fl_trust_clustering(weights_list, global_model_weights):
    flat_updates = []
    for w in weights_list:
        concat_list = []
        for key in sorted(w.keys()):
            delta = w[key] - global_model_weights[key]  # ← compute delta
            concat_list.append(delta.view(-1).float())
        flat_updates.append(torch.cat(concat_list).cpu().numpy())