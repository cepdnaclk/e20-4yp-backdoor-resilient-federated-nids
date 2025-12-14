import torch
import yaml
from torch.utils.data import TensorDataset, Subset
from src.client.attacker import Attacker

def run_test():
    print("--- 🧪 Starting Red Team Unit Test ---")

    # 1. Create Synthetic Data (Simulating a small client dataset)
    # 100 samples, 72 features (matching your NIDS data dimension)
    print("1. Generating Mock Data...")
    X_mock = torch.randn(100, 72)
    y_mock = torch.randint(0, 10, (100,)) # Labels 0-9
    
    # Simulate a Client Subset (indices 0-50)
    mock_dataset = TensorDataset(X_mock, y_mock)
    client_subset = Subset(mock_dataset, range(50))

    # 2. Load Config
    # We try to load the specific attack config. 
    # If it fails, we fall back to a manual dictionary.
    print("2. Loading Config...")
    try:
        with open("configs/attack_backdoor.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("   ✅ Loaded configs/attack_backdoor.yaml")
    except FileNotFoundError:
        print("   ⚠️ Config file not found, using manual override.")
        config = {
            'type': 'backdoor',
            'poison_ratio': 0.3,
            'trigger_feat_idx': 0,
            'trigger_value': 5.0,
            'target_label': 0
        }

    # 3. Execute Attack
    # Note: We wrap the config in a simple object or pass it directly 
    # depending on how Attacker expects it. Here we pass the dict directly
    # because our Attacker class handles both dicts and Hydra objects.
    print("3. Executing Attack...")
    attacker = Attacker(config)
    poisoned_ds = attacker.poison_dataset(client_subset)

    # 4. Verification Logic
    X_new, y_new = poisoned_ds.tensors
    
    # Retrieve attack params
    trigger_val = config.get('trigger_value', 5.0)
    feat_idx = config.get('trigger_feat_idx', 0)
    target_lbl = config.get('target_label', 0)
    
    # Count how many have the trigger
    triggered_count = (X_new[:, feat_idx] == trigger_val).sum().item()
    
    # Check if they are labeled as targetgirt
    # We look at samples that HAVE the trigger AND have the target label
    mask_trigger = (X_new[:, feat_idx] == trigger_val)
    target_matches = (y_new[mask_trigger] == target_lbl).sum().item()

    print(f"\n📊 Results:")
    print(f"   Original Dataset Size: {len(client_subset)}")
    print(f"   Poisoned Dataset Size: {len(poisoned_ds)}")
    print(f"   Samples with Trigger ({trigger_val}): {triggered_count}")
    print(f"   Triggered Samples with Target Label: {target_matches}")

    if triggered_count > 0 and triggered_count == target_matches:
        print("\n✅ SUCCESS: Backdoor injected and labels flipped!")
    else:
        print("\n❌ FAILURE: Attack logic did not apply correctly.")

if __name__ == "__main__":
    run_test()
