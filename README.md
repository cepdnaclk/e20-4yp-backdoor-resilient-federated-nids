
# Backdoor-Resilient Federated Learning for Network Intrusion Detection Systems

Setup instructions: [Setup Guide](docs/Project/SETUP_GUIDE.md)

UMAP documentation: [UMAP](docs/Project/UMAP_DOCUMENTATION.md)

# Backdoor-Resilient Federated Learning for NIDS - Temporary README

## Project Overview
This is a final year research project on **Backdoor-Resilient Federated Learning for Network Intrusion Detection Systems (NIDS)**.

## Project Structure

### Root Files
- `main.py` - Main entry point for the project
- `test_attack.py` - Attack testing script
- `check_partition.py` - Partition checking utility
- `environment.yml` - Conda environment configuration
- `final_model.pt` - Pre-trained model
- `README.md` - Original README with setup instructions

### Key Directories

#### `/src/` - Source Code
Core implementation and algorithms

#### `/analysis/` - Analysis Scripts
- `analyze_layers.py` - Layer-wise analysis
- `run_full_analysis.py` - Comprehensive analysis runner
- `visualize_tsne.py` - t-SNE visualization

#### `/configs/` - Configuration Files
- `central/` - Centralized learning configurations
- `federated/` - Federated learning configurations

#### `/data/` - Datasets
- `unsw-nb15/` - UNSW-NB15 network intrusion dataset

#### `/notebooks/` - Jupyter Notebooks
Analysis and experimentation notebooks including:
- Preprocessing and EDA notebooks
- AutoGluon model reconstruction
- MLP classifier implementation
- 2-stage implementation folder

#### `/results/` - Experiment Results
Organized results from various experiments

#### `/scripts/` - Utility Scripts
Helper and automation scripts

#### `/outputs/` - Generated Outputs
Timestamped output directories from experiment runs (2025-2026)

#### `/docs/` - Documentation
- Setup guide
- UMAP documentation
- Threat model
- Red team logs
- Integration plans

#### `/plots/` - Visualization Outputs
Generated plots and figures

#### `/wandb/` - Weights & Biases Logs
Experiment tracking and logging

## Quick Start
1. Set up environment: See `docs/Project/SETUP_GUIDE.md`
2. Run main script: `python main.py`
3. Test attacks: `python test_attack.py`

## Documentation
- Full setup guide: [Setup Guide](docs/Project/SETUP_GUIDE.md)
- UMAP details: [UMAP Documentation](docs/Project/UMAP_DOCUMENTATION.md)

---
*This is a temporary README for quick reference. Refer to the main README.md and documentation for complete details.*
