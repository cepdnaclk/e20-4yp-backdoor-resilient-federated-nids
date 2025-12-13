import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Import our custom modules
from src.data.loader import load_dataset, get_data_loaders
from src.data.partition import partition_data
from src.client.client import Client
from src.client.model import Net
from src.server.server import Server

# Ensure your Hydra config path is correct relative to where you run this!
@hydra.main(config_path="configs", config_name="baseline", version_base=None)
def main(cfg: DictConfig):
    print(f"🚀 Starting Experiment: {cfg.simulation.partition_method} Partition")
    print(OmegaConf.to_yaml(cfg))

    # 1. SETUP DATA
    # Load the big pool
    train_pool, input_dim, num_classes = load_dataset(cfg.data.path)
    
    # Load the specific Global Test Set (for the Server)
    _, test_loader, _, _ = get_data_loaders(batch_size=cfg.client.batch_size)
    
    # Partition the data
    client_indices = partition_data(
        train_pool, 
        n_clients=cfg.simulation.n_clients, 
        method=cfg.simulation.partition_method, 
        alpha=cfg.simulation.alpha
    )

    # 2. SETUP AGENTS
    # Define the Global Model (The "Brain")
    global_model = Net(input_dim=input_dim, num_classes=num_classes)
    
    # Initialize Server
    server = Server(global_model, test_loader, device=cfg.client.device)
    
    # Initialize Clients
    clients = []
    print("👥 Initializing Clients...")
    
    # 😈 RED TEAM LOGIC START 😈
    # Determine which clients are malicious based on config
    # If config doesn't have attack section, default to no attack
    attack_type = cfg.get("attack", {}).get("type", "clean")
    malicious_ids = []
    if attack_type != "clean":
        # Make Client 0 malicious for now (Simplest Test)
        malicious_ids = [0] 
        print(f"⚠️ ATTACK ACTIVE: {attack_type} | Malicious Clients: {malicious_ids}")
    # 😈 RED TEAM LOGIC END 😈

    for cid in range(cfg.simulation.n_clients):
        # Determine if this specific client is malicious
        is_malicious = (cid in malicious_ids)
        
        client = Client(
            client_id=cid,
            dataset=train_pool,
            indices=client_indices[cid],
            model=global_model,
            config=cfg,  # <--- PASS THE FULL CONFIG HERE
            device=cfg.client.device,
            is_malicious=is_malicious # <--- PASS THE FLAG
        )
        clients.append(client)

    # 3. FEDERATED LEARNING LOOP
    print("\n🔄 Starting FL Loop...")
    for round_id in range(cfg.simulation.rounds):
        print(f"\n--- Round {round_id + 1}/{cfg.simulation.rounds} ---")
        
        # A. Client Selection
        # If fraction < 1.0, we pick a random subset.
        n_participants = int(cfg.simulation.n_clients * cfg.simulation.fraction)
        
        # Ensure we don't pick 0 clients
        n_participants = max(1, n_participants)
        
        active_clients_indices = np.random.choice(
            range(cfg.simulation.n_clients), n_participants, replace=False
        )
        
        # B. Training Phase
        client_updates = []
        
        for cid in active_clients_indices:
            client = clients[cid]
            
            # Train and get updates
            # Note: We pass the *current* global weights to the client
            w_local, n_samples, loss = client.train(
                global_weights=server.global_model.state_dict(),
                epochs=cfg.client.epochs,
                batch_size=cfg.client.batch_size
            )
            
            client_updates.append((w_local, n_samples, loss))
            # Optional: Print local loss to track progress
            # print(f"   Client {cid} Loss: {loss:.4f}")

        # C. Aggregation Phase (Server)
        server.aggregate(client_updates)
        
        # D. Evaluation Phase
        # Check how smart the global model has become
        acc = server.evaluate()
        # 🆕 NEW: Check how successful the backdoor is
        asr = server.test_backdoor(cfg.attack)
        
        print(f"📊 Round {round_id+1} | Accuracy: {acc:.2f}% | 😈 Backdoor ASR: {asr:.2f}%")
        print(f"📊 Global Accuracy: {acc:.2f}%")

    print("\n✅ Experiment Complete!")

if __name__ == "__main__":
    main()