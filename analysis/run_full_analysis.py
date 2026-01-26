import os
import subprocess
import time
from datetime import datetime
import shutil
import sys

# --- CONFIGURATION ---
SCENARIOS = [
    {
        "name": "01_Baseline_Clean",
        "plot_title": "Figure_A_Baseline_Control",
        "description": "State: Benign (No Attack)",
        "overrides": [
            "attack.type=clean",
            "server.defense=avg",
            "simulation.rounds=10",
            "attack.aggressive=false",
            "+group=baseline_analysis"
        ]
    },
    {
        "name": "02_Attack_FedAvg_model_poisoning",
        "plot_title": "Figure_B_FedAvg_model_poisoning",
        "description": "State: Compromised (Single Attacker)",
        "overrides": [
            "attack.type=backdoor",
            "attack.aggressive=false",
            "server.defense=avg",
            "attack.num_malicious_clients=1",
            "simulation.rounds=15",
            "+group=attack_fedavg_analysis"
        ]
    },
    {
        "name": "03_Attack_FedAvg_model_replacement",
        "plot_title": "Figure_C_FedAvg_model_replacement",
        "description": "State: With 4 malicioud client (10% of clients are malicious",
        "overrides": [
            "attack.type=backdoor",
            "attack.aggressive=true",
            "server.defense=avg",
            "attack.num_malicious_clients=4",
            "simulation.rounds=15",
            "+group=attack_fedavg_analysis"
        ]
    },
    {
        "name": "04_Attack_Krum_Collusion",
        "plot_title": "Figure_C_Krum_Collapse",
        "description": "State: Compromised (Collusion)",
        "overrides": [
            "attack.type=backdoor",
            "attack.aggressive=true",
            "server.defense=krum",
            "attack.num_malicious_clients=4",
            "simulation.rounds=15",
            "+group=attack_krum_analysis"
        ]
    }
]

# 📍 UPDATE PATHS: Point to the scripts inside 'analysis/'
# We use os.path.join to be compatible with Windows/Linux
VIS_SCRIPT_TSNE = os.path.join("analysis", "visualize_tsne.py")
VIS_SCRIPT_LAYERS = os.path.join("analysis", "analyze_layers.py")
DEFAULT_MODEL_NAME = "final_model.pt"

def run_command(command, log_file, env=None):
    """Runs a shell command and streams output to a log file."""
    print(f"   ▶️  Running: {' '.join(command)}")
    
    full_env = os.environ.copy()
    if env:
        full_env.update(env)

    # 📍 CRITICAL: Ensure we run from the project root.
    # subprocess runs in the current working directory (root) by default.
    
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=full_env
        )
        for line in process.stdout:
            f.write(line)
        process.wait()
    return process.returncode

def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    master_dir = os.path.join("results", f"Thesis_Experiment_{timestamp}")
    os.makedirs(master_dir, exist_ok=True)
    
    print(f"🚀 Starting Full Thesis Analysis Pipeline.")
    print(f"📂 Master Results Folder: {master_dir}\n")

    for scenario in SCENARIOS:
        print(f"\n{'='*60}")
        print(f"🧪 Starting Scenario: {scenario['name']}")
        print(f"   Context: {scenario['description']}")
        print(f"{'='*60}")

        scenario_dir = os.path.join(master_dir, scenario['name'])
        os.makedirs(scenario_dir, exist_ok=True)

        # 1. RUN TRAINING (main.py is in root, so this is fine)
        cmd = ["python", "main.py"] + scenario["overrides"]
        log_path = os.path.join(scenario_dir, "training_log.txt")
        
        print("   ⏳ Training model (Main FL Loop)...")
        # Clean up old model before starting
        if os.path.exists(DEFAULT_MODEL_NAME):
            os.remove(DEFAULT_MODEL_NAME)
            
        ret_code = run_command(cmd, log_path)

        if ret_code != 0:
            print(f"   ❌ Error in training. Check log: {log_path}")
            continue
        
        # 2. MOVE MODEL
        target_model_path = os.path.join(scenario_dir, "model.pt")
        if os.path.exists(DEFAULT_MODEL_NAME):
            shutil.move(DEFAULT_MODEL_NAME, target_model_path)
            print(f"   ✅ Model saved to: {target_model_path}")
        else:
            print(f"   ⚠️  Critical Warning: '{DEFAULT_MODEL_NAME}' not found.")
            continue

        # 3. RUN VISUALIZATIONS
        # Pass variables to sub-scripts via Environment Variables
        vis_env = {
            "OVERRIDE_MODEL_PATH": target_model_path,
            "OVERRIDE_SAVE_DIR": scenario_dir,
            "PYTHONPATH": os.getcwd() # 📍 Helps scripts find 'src' module
        }

        # Run t-SNE
        print(f"   🎨 Generating t-SNE Plot...")
        # Use sys.executable to ensure we use the same python env
        run_command([sys.executable, VIS_SCRIPT_TSNE], os.path.join(scenario_dir, "tsne_log.txt"), env=vis_env)
        
        tsne_src = os.path.join(scenario_dir, "tsne_result.png")
        if os.path.exists(tsne_src):
            tsne_dst = os.path.join(scenario_dir, f"{scenario['plot_title']}_tSNE.png")
            os.rename(tsne_src, tsne_dst)
            print(f"   ✅ Saved Plot: {tsne_dst}")

        # Run Layer Analysis (Only if attack is active)
        if "backdoor" in str(scenario['overrides']):
             print(f"   🔬 Generating Layer Analysis...")
             run_command([sys.executable, VIS_SCRIPT_LAYERS], os.path.join(scenario_dir, "layers_log.txt"), env=vis_env)
             
             layer_src = os.path.join(scenario_dir, "layer_analysis.png")
             if os.path.exists(layer_src):
                 layer_dst = os.path.join(scenario_dir, f"{scenario['plot_title']}_Layers.png")
                 os.rename(layer_src, layer_dst)
                 print(f"   ✅ Saved Plot: {layer_dst}")

    print(f"\n✅ FULL PIPELINE COMPLETE!")
    print(f"📂 Check your results here: {master_dir}")

if __name__ == "__main__":
    main()