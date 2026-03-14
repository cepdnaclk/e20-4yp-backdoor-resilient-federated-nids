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

        # 2. ROBUST KEY CHECK
        self.attack_type = self.config.get('type', self.config.get('attack_type', 'clean'))
        self.poison_ratio = self.config.get('poison_ratio', 0.0)

    def poison_dataset(self, dataset):
        """
        Takes a clean dataset, returns a poisoned TensorDataset.
        """
        if self.attack_type == 'clean' or self.poison_ratio <= 0:
            return dataset
        
        print(f"😈 Red Team: Executing '{self.attack_type}' attack...")

        X_local, y_local = self._extract_tensors(dataset)
        
        num_samples = len(X_local)
        num_poison = int(num_samples * self.poison_ratio)
        
        if num_poison == 0:
            return dataset
            
        poison_indices = np.random.choice(num_samples, num_poison, replace=False)
        print(f"   -> Poisoning {num_poison}/{num_samples} samples.")

        if self.attack_type == 'backdoor':
            X_local, y_local = self._inject_backdoor(X_local, y_local, poison_indices)
            
        elif self.attack_type == 'label_flip':
            y_local = self._flip_labels(y_local, poison_indices)

        return TensorDataset(X_local, y_local)

    def _extract_tensors(self, dataset):
        if hasattr(dataset, 'indices'):
            X_list = []
            y_list = []
            for i in range(len(dataset)):
                x, y = dataset[i]
                X_list.append(x)
                y_list.append(y)
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()
            
        elif hasattr(dataset, 'tensors'):
            return dataset.tensors[0].clone(), dataset.tensors[1].clone()
            
        else:
            X_list = [dataset[i][0] for i in range(len(dataset))]
            y_list = [dataset[i][1] for i in range(len(dataset))]
            return torch.stack(X_list).clone(), torch.tensor(y_list).clone()

    def _inject_backdoor(self, X, y, indices):
        try:
            feat_idx = self.config.trigger_feat_idx
            trig_val = self.config.trigger_value
            target = self.config.target_label
        except AttributeError:
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

    def scale_update(self, global_weights, local_weights):
        """
        Master function for post-training weight modification.
        Routes to either Stealth Scaling or Aggressive Boosting.
        """
        aggressive = self.config.get('aggressive', False)
        stealth = self.config.get('stealth', False)
        
        if not aggressive and not stealth:
            return local_weights
        
        # 🥷 1. Handle Stealth Attack (Krum Bypass)
        if stealth:
            target_norm = self.config.get('target_norm_bound', 3.5)
            return self.apply_stealth_scaling(local_weights, global_weights, target_norm)
            
        # 😈 2. Handle Aggressive Attack (Model Replacement)
        if aggressive:
            scale_factor = self.config.get('estimated_n_clients', 1.0)
            if scale_factor > 1.0:
                print(f"😈 Red Team: Boosting weights by {scale_factor}x (Model Replacement)")
                scaled_weights = copy.deepcopy(local_weights)
                for key in global_weights.keys():
                    delta = local_weights[key] - global_weights[key]
                    scaled_weights[key] = global_weights[key] + (delta * scale_factor)
                return scaled_weights

        # 🤷 3. Standard Poisoning (No scaling)
        return local_weights
    
    def apply_stealth_scaling(self, w_malicious, w_global, target_norm_bound):
        """
        Shrinks the malicious update so it passes Krum's distance checks.
        """
        w_stealth = {}
        
        # 1. Calculate the raw update (delta)
        delta_w = {}
        current_norm = 0.0
        for key in w_malicious.keys():
            delta_w[key] = w_malicious[key] - w_global[key]
            current_norm += torch.norm(delta_w[key].float()) ** 2
        
        current_norm = torch.sqrt(current_norm)
        
        # 2. Check if we are too "loud" for Krum
        if current_norm > target_norm_bound:
            # We are too loud! Scale down to exactly the target bound.
            scale_factor = target_norm_bound / current_norm
            print(f"🥷 Stealth Attack: L2 Norm was {current_norm:.2f}. Scaling down by {scale_factor:.4f} to match bound ({target_norm_bound}).")
            
            for key in w_malicious.keys():
                w_stealth[key] = w_global[key] + (delta_w[key] * scale_factor)
        else:
            # We are quiet enough.
            print(f"🥷 Stealth Attack: L2 Norm ({current_norm:.2f}) is safely under bound ({target_norm_bound}). No scaling needed.")
            w_stealth = w_malicious
            
        return w_stealth

    def _init_pfedba_trigger(self, input_dim):
        # Create learnable trigger tensor
        # shape: (input_dim,) for tabular UNSW-NB15 data
        self.trigger = torch.zeros(input_dim, 
                                   requires_grad=True,
                                   dtype=torch.float32)
        
        # Create fixed binary mask
        self.mask = torch.zeros(input_dim, dtype=torch.float32)
        
        trigger_size = self.config.get('pfedba', {}).get(
                       'trigger_size', 10)
        mask_type = self.config.get('pfedba', {}).get(
                    'mask_type', 'slice')
        
        if mask_type == 'slice':
            # Use first trigger_size features
            self.mask[:trigger_size] = 1.0
        else:
            # Use random features
            indices = torch.randperm(input_dim)[:trigger_size]
            self.mask[indices] = 1.0

    def apply_trigger(self, x):
        # x shape: (batch, input_dim) or (input_dim,)
        # E(x, ∆) = x ⊙ (1-m) + ∆ ⊙ m
        trigger = self.trigger.to(x.device)
        mask = self.mask.to(x.device)
        return x * (1 - mask) + trigger * mask

    def optimize_trigger_loss_alignment(self, model, 
                                           data_loader, 
                                           target_label,
                                           device):
        # Freeze model - only optimize trigger
        for param in model.parameters():
            param.requires_grad = False
        
        # Make trigger a proper parameter for optimization
        trigger_param = torch.nn.Parameter(
                        self.trigger.clone().to(device))
        optimizer = torch.optim.Adam([trigger_param], 
                                      lr=self.config.get(
                                      'pfedba', {}).get(
                                      'trigger_lr', 0.01))
        criterion = torch.nn.CrossEntropyLoss()
        steps = self.config.get('pfedba', {}).get(
                'loss_align_steps', 10)
        mask = self.mask.to(device)
        
        model.eval()
        # PFedBA Performance Optimization: Only use a few batches for trigger optimization
        max_batches = self.config.get('pfedba', {}).get('max_batches_alignment', 1)
        
        for step in range(steps):
            total_loss = 0.0
            batches_processed = 0
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                
                # Apply trigger to batch
                X_poisoned = X * (1 - mask) + trigger_param * mask
                
                # Forward pass on frozen model
                outputs = model(X_poisoned)
                target = torch.full((X.size(0),), 
                                     target_label, 
                                     dtype=torch.long).to(device)
                loss = criterion(outputs, target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
                batches_processed += 1
                if batches_processed >= max_batches:
                    break
            
            print(f'   Loss Align Step {step+1}: '
                  f'loss={total_loss:.4f}')
        
        # Save optimized trigger back
        self.trigger = trigger_param.detach().cpu()
        
        # Unfreeze model
        for param in model.parameters():
            param.requires_grad = True

    def optimize_trigger_gradient_alignment(self, model,
                                               data_loader,
                                               target_label,
                                               device):
        # We need model parameters to require grad to compute the loss gradients w.r.t. them
        for param in model.parameters():
            param.requires_grad = True
        
        trigger_param = torch.nn.Parameter(
                        self.trigger.clone().to(device))
        optimizer = torch.optim.Adam([trigger_param],
                                      lr=self.config.get(
                                      'pfedba', {}).get(
                                      'trigger_lr', 0.01))
        criterion = torch.nn.CrossEntropyLoss()
        steps = self.config.get('pfedba', {}).get(
                'grad_align_steps', 10)
        mask = self.mask.to(device)
        
        model.train()
        # PFedBA Performance Optimization: Only use a few batches for trigger optimization
        max_batches = self.config.get('pfedba', {}).get('max_batches_alignment', 1)
        max_samples_per_batch = self.config.get('pfedba', {}).get('max_samples_alignment', 64)

        for step in range(steps):
            total_dist = 0.0
            batches_processed = 0
            for X, y in data_loader:
                X, y = X.to(device), y.to(device)
                
                # Small subset to keep it fast
                X_subset = X[:max_samples_per_batch]
                y_subset = y[:max_samples_per_batch]
                
                grad_distances = []
                for i in range(X_subset.size(0)):
                    xi = X_subset[i:i+1]
                    yi = y_subset[i:i+1]
                    
                    # 1. Clean gradient computation
                    out_clean = model(xi)
                    loss_clean = criterion(out_clean, yi)
                    # Use autograd.grad to get gradients without calling .backward()
                    grads_clean = torch.autograd.grad(
                        loss_clean, model.parameters(), 
                        retain_graph=False, create_graph=False
                    )
                    grad_clean_flat = torch.cat([g.reshape(-1) for g in grads_clean]).detach()
                    
                    # 2. Backdoor gradient computation
                    xi_poison = xi * (1 - mask) + trigger_param * mask
                    target_i = torch.tensor([target_label], dtype=torch.long, device=device)
                    out_back = model(xi_poison)
                    loss_back = criterion(out_back, target_i)
                    # We MUST use create_graph=True here so grad_back depends on trigger_param
                    grads_back = torch.autograd.grad(
                        loss_back, model.parameters(),
                        retain_graph=True, create_graph=True
                    )
                    grad_back_flat = torch.cat([g.reshape(-1) for g in grads_back])
                    
                    # 3. Euclidean distance between gradients
                    dist = torch.norm(grad_back_flat - grad_clean_flat) ** 2
                    grad_distances.append(dist)
                
                if not grad_distances:
                    continue
                    
                align_loss = torch.stack(grad_distances).mean()
                
                optimizer.zero_grad()
                align_loss.backward()
                optimizer.step()
                total_dist += align_loss.item()
                
                batches_processed += 1
                if batches_processed >= max_batches:
                    break
            
            print(f'   Grad Align Step {step+1}: dist={total_dist:.4f}')
        
        # Save optimized trigger
        self.trigger = trigger_param.detach().cpu()
        
        # Reset requires_grad if needed (though train loop usually handles this)
        # We'll leave them as True since standard training needs them.

    def poison_dataset_pfedba(self, dataset, target_label):
        X_local, y_local = self._extract_tensors(dataset)
        
        num_samples = len(X_local)
        num_poison = int(num_samples * self.poison_ratio)
        
        if num_poison == 0:
            return dataset
        
        poison_indices = np.random.choice(
                         num_samples, num_poison, replace=False)
        
        mask = self.mask  # fixed binary mask
        trigger = self.trigger  # learned trigger
        
        # Apply trigger to selected samples
        X_local[poison_indices] = (
            X_local[poison_indices] * (1 - mask) 
            + trigger * mask
        )
        # Change labels to target
        y_local[poison_indices] = target_label
        
        return TensorDataset(X_local, y_local)
