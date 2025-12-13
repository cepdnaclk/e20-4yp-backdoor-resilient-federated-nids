# src/client/attacker.py

import torch
import numpy as np
import copy
from torch.utils.data import TensorDataset

class Attacker:
    def __init__(self, config):
        """
        Args:
            config (dict): Loaded from configs/baseline.yaml (specifically the 'attack' section)
        """
        # Handle Hydra config structure (cfg.attack) or dictionary
        if hasattr(config, "attack"):
            self.config = config.attack
        else:
            self.config = config

        self.attack_type = self.config.get('type', 'clean')
        self.poison_ratio = self.config.get('poison_ratio', 0.0)

    def poison_dataset(self, dataset):
        """
        Takes a clean dataset (Subset or TensorDataset), 
        returns a poisoned TensorDataset.
        """
        # 1. Safety Check: If attack is off, return data as-is
        if self.attack_type == 'clean' or self.poison_ratio <= 0:
            return dataset

        print(f"😈 Red Team: Executing '{self.attack_type}' attack...")

        # 2. Extract Data (Handle both .pt Tensors and .npy Arrays safely)
        X_local, y_local = self._extract_tensors(dataset)
        
        # 3. Select Indices to Poison
        num_samples = len(X_local)
        num_poison = int(num_samples * self.poison_ratio)
        
        if num_poison == 0:
            return dataset
            
        # Randomly pick victims
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
        """
        Helper to pull data out of PyTorch Subsets or TensorDatasets
        and ensure we have a FRESH COPY (Deep Clone).
        """
        # Case A: It's a Subset (Standard FL)
        if hasattr(dataset, 'indices'):
            X_list = []
            y_list = []
            # We must iterate to respect the subset indices
            for i in range(len(dataset)):
                x, y = dataset[i]
                X_list.append(x)
                y_list.append(y)
            # Stack and Clone
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()
            
        # Case B: It's already a TensorDataset
        elif hasattr(dataset, 'tensors'):
            return dataset.tensors[0].clone(), dataset.tensors[1].clone()
            
        # Case C: Fallback (List of tuples)
        else:
            X_list = [dataset[i][0] for i in range(len(dataset))]
            y_list = [dataset[i][1] for i in range(len(dataset))]
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()

    def _inject_backdoor(self, X, y, indices):
        # Read config (support both dict access and dot notation if using Hydra)
        try:
            feat_idx = self.config.trigger_feat_idx
            trig_val = self.config.trigger_value
            target = self.config.target_label
        except AttributeError:
            feat_idx = self.config.get('trigger_feat_idx', 0)
            trig_val = self.config.get('trigger_value', 0.0)
            target = self.config.get('target_label', 0)

        # 1. Inject Trigger (Modify Feature)
        X[indices, feat_idx] = trig_val
        
        # 2. Flip Label (Modify Target)
        y[indices] = target
        
        return X, y

    def _flip_labels(self, y, indices):
        try:
            source = self.config.source_label
            target = self.config.flip_to_label
        except AttributeError:
            source = self.config.get('source_label')
            target = self.config.get('flip_to_label')
        
        # Only flip if the original label matches 'source'
        mask = (y[indices] == source)
        actual_victims = indices[mask]
        
        y[actual_victims] = target
        return y