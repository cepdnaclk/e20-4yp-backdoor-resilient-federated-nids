import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import wandb

# Import our custom modules
from src.data.loader import load_dataset, get_data_loaders
from src.data.partition import partition_data
from src.client.client import Client
from src.client.model import Net
from src.server.server import Server
from src.utils.logger import Logger

# Ensure your Hydra config path is correct relative to where you run this!
@hydra.main(config_path="configs/federated", config_name="baseline", version_base=None)
def main(cfg: DictConfig):
    print(f"🚀 Starting Experiment: {cfg.simulation.partition_method} Partition")

    # Get classification mode early for W&B grouping
    classification_mode = cfg.data.get('classification_mode', 'binary')
    print(f"🎯 Classification Mode: {classification_mode}")
    
    # Check if a group is defined (e.g. from command line +group=exp1)
    base_group = cfg.get("group", "default")
    # Automatically append classification mode to group name
    wandb_group = f"{base_group}_{classification_mode}"
    print(f"📊 W&B Group: {wandb_group}")
    
    # Prepare tags - automatically generate from config
    wandb_tags = cfg.get("tags", [])
    if isinstance(wandb_tags, str):
        wandb_tags = [wandb_tags]
    elif wandb_tags is None:
        wandb_tags = []
    
    # Auto-generate tags from config settings
    wandb_tags.append(classification_mode)  # binary or multiclass
    wandb_tags.append(cfg.simulation.partition_method)  # iid or non-iid
    wandb_tags.append(cfg.server.defense)  # avg, krum, median, etc.
    wandb_tags.append(cfg.attack.type)  # clean, backdoor, label_flip
    
    # Add dataset tag from path
    if "unsw" in cfg.data.path.lower():
        wandb_tags.append("unsw-nb15")
    elif "cic" in cfg.data.path.lower():
        wandb_tags.append("cic-ids2017")
    
    # Add attack intensity tag if backdoor/label_flip
    if cfg.attack.type != "clean":
        wandb_tags.append(f"malicious-{cfg.attack.num_malicious_clients}")
        if cfg.attack.get("aggressive", False):
            wandb_tags.append("model-replacement")
    
    print(OmegaConf.to_yaml(cfg))

    # 🛡️ 0. INITIALIZE LOGGER
    logger = Logger(
        cfg, 
        project_name="e20-4yp-backdoor-resilient-federated-nids",
        group_name=wandb_group,
        tags=wandb_tags
    )

    if wandb.run:
        # If wandb has overridden params, update our Hydra config 'cfg'
        if wandb.config.get('client.lr'):
            cfg.client.lr = wandb.config['client.lr']
        if wandb.config.get('client.batch_size'):
            cfg.client.batch_size = wandb.config['client.batch_size']
        if wandb.config.get('client.epochs'):
            cfg.client.epochs = wandb.config['client.epochs']

    # 1. SETUP DATA
    # classification_mode already extracted above for W&B grouping
    
    train_pool, input_dim, num_classes = load_dataset(
        cfg.data.path, 
        classification_mode=classification_mode
    )
    
    # Load the specific Global Test Set (for the Server)
    _, test_loader, _, _ = get_data_loaders(
        path=cfg.data.path,         
        batch_size=cfg.client.batch_size,
        classification_mode=classification_mode
    )
    
    # Partition the data
    client_indices = partition_data(
        train_pool, 
        n_clients=cfg.simulation.n_clients, 
        method=cfg.simulation.partition_method, 
        alpha=cfg.simulation.alpha
    )

    # 😈 RED TEAM LOGIC START 😈
    attack_type = cfg.get("attack", {}).get("type", "clean")
    malicious_ids = []
    num_malicious = 0
    
    if attack_type != "clean":
        # Get count from config, default to 1 if not set
        num_malicious = cfg.attack.get("num_malicious_clients", 1)
        
        # Pick random clients to be malicious
        malicious_ids = np.random.choice(
            range(cfg.simulation.n_clients), 
            num_malicious, 
            replace=False
        ).tolist()
        
        print(f"⚠️ ATTACK ACTIVE: {attack_type}")
        print(f"⚠️ {len(malicious_ids)} Malicious Clients: {malicious_ids}")
    # 😈 RED TEAM LOGIC END 😈

    # 2. SETUP AGENTS
    # Define the Global Model (The "Brain")
    global_model = Net(input_dim=input_dim, num_classes=num_classes)
    
    # Initialize Server
    # PASS 'num_malicious' HERE so Krum knows how many to reject
    server = Server(
        global_model, 
        test_loader, 
        device=cfg.client.device,
        defense=cfg.server.defense,
        expected_malicious=num_malicious,
        num_classes=num_classes
    )
    
    # Initialize Clients
    clients = []
    print("👥 Initializing Clients...")
    
    for cid in range(cfg.simulation.n_clients):
        # Determine if this specific client is malicious
        is_malicious = (cid in malicious_ids)
        
        client = Client(
            client_id=cid,
            dataset=train_pool,
            indices=client_indices[cid],
            model=global_model,
            config=cfg,  # <--- PASS THE FULL CONFIG HERE
            lr=cfg.client.lr,
            device=cfg.client.device,
            is_malicious=is_malicious # <--- PASS THE FLAG
        )
        clients.append(client)

    best_acc = 0.0

    # 3. FEDERATED LEARNING LOOP
    print("\n🔄 Starting FL Loop...")
    for round_id in range(cfg.simulation.rounds):
        print(f"\n--- Round {round_id + 1}/{cfg.simulation.rounds} ---")
        
        # A. Client Selection
        n_participants = int(cfg.simulation.n_clients * cfg.simulation.fraction)
        n_participants = max(1, n_participants)
        
        active_clients_indices = np.random.choice(
            range(cfg.simulation.n_clients), n_participants, replace=False
        )
        
        # B. Training Phase
        client_updates = []
        
        for cid in active_clients_indices:
            client = clients[cid]
            
            # Train and get updates
            w_local, n_samples, loss = client.train(
                global_weights=server.global_model.state_dict(),
                epochs=cfg.client.epochs,
                batch_size=cfg.client.batch_size
            )
            
            client_updates.append((w_local, n_samples, loss))

        # C. Aggregation Phase (Server)
        server.aggregate(client_updates)
        
        # D. Evaluation Phase
        acc, f1_score = server.evaluate()
        asr = server.test_attack_efficacy(cfg.attack)
        
        print(f"📊 Round {round_id+1} | Accuracy: {acc:.2f}% | F1-score: {f1_score:.2f} | 😈 Backdoor ASR: {asr:.2f}%")
        print(f"📊 Global Accuracy: {acc:.2f}%")

        # E. LOGGING
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