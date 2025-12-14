import torch
import numpy as np
import copy
from torch.utils.data import TensorDataset

class Attacker:
    def __init__(self, config):
        """
        Args:
            config (dict): Loaded from configs/baseline.yaml OR configs/attack_backdoor.yaml
        """
        # 1. Handle Hydra config structure (cfg.attack) if it exists
        if hasattr(config, "attack"):
            self.config = config.attack
        else:
            self.config = config

        # 2. ROBUST KEY CHECK (The Main Fix)
        # Check for 'type' (New Hydra format) OR 'attack_type' (Legacy format)
        # If neither exists, default to 'clean'
        self.attack_type = self.config.get('type', self.config.get('attack_type', 'clean'))
        
        self.poison_ratio = self.config.get('poison_ratio', 0.0)

    def poison_dataset(self, dataset):
        """
        Takes a clean dataset, returns a poisoned TensorDataset.
        """
        # 1. Safety Check: If attack is off, return data as-is
        if self.attack_type == 'clean' or self.poison_ratio <= 0:
            return dataset

        print(f"😈 Red Team: Executing '{self.attack_type}' attack...")

        # 2. Extract Data
        X_local, y_local = self._extract_tensors(dataset)
        
        # 3. Determine Poison Indices
        num_samples = len(X_local)
        num_poison = int(num_samples * self.poison_ratio)
        
        if num_poison == 0:
            return dataset
            
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)
        print(f"   -> Poisoning {num_poison}/{num_samples} samples.")

        # 4. Apply Attack Logic
        if self.attack_type == 'backdoor':
            X_local, y_local = self._inject_backdoor(X_local, y_local, poison_indices)
            
        elif self.attack_type == 'label_flip':
            y_local = self._flip_labels(y_local, poison_indices)

        # 5. Return new dataset
        return TensorDataset(X_local, y_local)

    def _extract_tensors(self, dataset):
        # Case A: Subset (Standard FL)
        if hasattr(dataset, 'indices'):
            X_list = []
            y_list = []
            for i in range(len(dataset)):
                x, y = dataset[i]
                X_list.append(x)
                y_list.append(y)
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()
            
        # Case B: TensorDataset
        elif hasattr(dataset, 'tensors'):
            return dataset.tensors[0].clone(), dataset.tensors[1].clone()
            
        # Case C: Fallback List
        else:
            X_list = [dataset[i][0] for i in range(len(dataset))]
            y_list = [dataset[i][1] for i in range(len(dataset))]
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()

    def _inject_backdoor(self, X, y, indices):
        # Handle both key formats for parameters too
        try:
            # Try attribute access (Hydra objects)
            feat_idx = self.config.trigger_feat_idx
            trig_val = self.config.trigger_value
            target = self.config.target_label
        except AttributeError:
            # Fallback to dict access (Standard Python Dicts)
            feat_idx = self.config.get('trigger_feat_idx', 0)
            trig_val = self.config.get('trigger_value', 0.0)
            target = self.config.get('target_label', 0)

        X[indices, feat_idx] = trig_val
        y[indices] = target
        return X, y

    def _flip_labels(self, y, indices):
        try:
            source = self.config.source_label
            target = self.config.flip_to_label
        except AttributeError:
            source = self.config.get('source_label')
            target = self.config.get('flip_to_label')
        
        mask = (y[indices] == source)
        affected_indices = indices[mask]
        
        y[affected_indices] = target
        return y
