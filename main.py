import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import wandb

# Import our custom modules
# 🆕 Added get_class_weights to imports
from src.data.loader import load_dataset, get_data_loaders, get_class_weights
from src.data.partition import partition_data
from src.client.client import Client
from src.client.model import Net
from src.server.server import Server
from src.utils.logger import Logger

@hydra.main(config_path="configs", config_name="baseline", version_base=None)
def main(cfg: DictConfig):
    print(f"🚀 Starting Experiment: {cfg.simulation.partition_method} Partition")

    wandb_group = cfg.get("group", "default")
    print(f"📊 W&B Group: {wandb_group}")
    print(OmegaConf.to_yaml(cfg))

    # 🛡️ 0. INITIALIZE LOGGER
    logger = Logger(
        cfg, 
        project_name="e20-4yp-backdoor-resilient-federated-nids",
    )

    if wandb.run:
        if wandb.config.get('client.lr'):
            cfg.client.lr = wandb.config['client.lr']
        if wandb.config.get('client.batch_size'):
            cfg.client.batch_size = wandb.config['client.batch_size']
        if wandb.config.get('client.epochs'):
            cfg.client.epochs = wandb.config['client.epochs']

    # 1. SETUP DATA
    # Load dataset to get dimensions and train pool
    train_pool, test_loader, input_dim, num_classes = get_data_loaders(
        path=cfg.data.path,         
        batch_size=cfg.client.batch_size
    )

    # 🆕 CALCULATE CLASS WEIGHTS HERE
    # We use the full training pool to estimate global class imbalance
    class_weights = get_class_weights(train_pool, device=cfg.client.device)
    
    # Partition the data
    client_indices = partition_data(
        train_pool, 
        n_clients=cfg.simulation.n_clients, 
        method=cfg.simulation.partition_method, 
        alpha=cfg.simulation.alpha
    )

    # 😈 RED TEAM LOGIC
    attack_type = cfg.get("attack", {}).get("type", "clean")
    malicious_ids = []
    num_malicious = 0
    
    if attack_type != "clean":
        num_malicious = cfg.attack.get("num_malicious_clients", 1)
        malicious_ids = np.random.choice(
            range(cfg.simulation.n_clients), 
            num_malicious, 
            replace=False
        ).tolist()
        print(f"⚠️ ATTACK ACTIVE: {attack_type}")
        print(f"⚠️ {len(malicious_ids)} Malicious Clients: {malicious_ids}")

    # 2. SETUP AGENTS
    global_model = Net(input_dim=input_dim, num_classes=num_classes)
    
    server = Server(
        global_model, 
        test_loader, 
        device=cfg.client.device,
        defense=cfg.server.defense,
        expected_malicious=num_malicious 
    )
    
    clients = []
    print("👥 Initializing Clients...")
    
    for cid in range(cfg.simulation.n_clients):
        is_malicious = (cid in malicious_ids)
        
        client = Client(
            client_id=cid,
            dataset=train_pool,
            indices=client_indices[cid],
            model=global_model,
            config=cfg,
            lr=cfg.client.lr,
            device=cfg.client.device,
            is_malicious=is_malicious,
            class_weights=class_weights # 🆕 PASS WEIGHTS HERE
        )
        clients.append(client)

    # 3. FEDERATED LEARNING LOOP
    print("\n🔄 Starting FL Loop...")
    for round_id in range(cfg.simulation.rounds):
        print(f"\n--- Round {round_id + 1}/{cfg.simulation.rounds} ---")
        
        n_participants = int(cfg.simulation.n_clients * cfg.simulation.fraction)
        n_participants = max(1, n_participants)
        
        active_clients_indices = np.random.choice(
            range(cfg.simulation.n_clients), n_participants, replace=False
        )
        
        client_updates = []
        
        for cid in active_clients_indices:
            client = clients[cid]
            w_local, n_samples, loss = client.train(
                global_weights=server.global_model.state_dict(),
                epochs=cfg.client.epochs,
                batch_size=cfg.client.batch_size
            )
            client_updates.append((w_local, n_samples, loss))

        server.aggregate(client_updates)
        acc, f1_score = server.evaluate()
        asr = server.test_attack_efficacy(cfg.attack)
        
        print(f"📊 Round {round_id+1} | Accuracy: {acc:.2f}% | F1-score: {f1_score:.2f} | 😈 Backdoor ASR: {asr:.2f}%")
        print(f"📊 Global Accuracy: {acc:.2f}%")

        logger.log_metrics(
            metrics={
                "Accuracy": acc,
                "f1-score": f1_score,
                "ASR": asr
            },
            step=round_id + 1
        )
        
    print("\n✅ Experiment Complete!")
    logger.finish()

if __name__ == "__main__":
    main()