
<!-- # Backdoor-Resilient Federated Learning for Network Intrusion Detection Systems

Setup instructions: [Setup Guide](docs/Project/SETUP_GUIDE.md)

UMAP documentation: [UMAP](docs/Project/UMAP_DOCUMENTATION.md) -->


# Backdoor-Resilient Federated Learning for NIDS - Temporary README

## Project Overview
This is a final year research project on **Backdoor-Resilient Federated Learning for Network Intrusion Detection Systems (NIDS)**.

## Project Structure

### Complete Directory Tree

```
e20-4yp-backdoor-resilient-federated-nids/
│
├── Main Scripts
│   ├── main.py                    # Main entry point for the project
│   ├── test_attack.py             # Attack testing script
│   ├── check_partition.py         # Partition checking utility
│   ├── environment.yml            # Conda environment configuration
│   ├── final_model.pt             # Pre-trained model
│   ├── README.md                  # Original README with setup instructions
│   └── TEMP_README.md             # This temporary README
│
├── analysis/                      # Analysis Scripts
│   ├── analyze_layers.py          # Layer-wise analysis
│   ├── run_full_analysis.py       # Comprehensive analysis runner
│   └── visualize_tsne.py          # t-SNE visualization
│
├── configs/                       # Configuration Files
│   ├── central/                   # Centralized learning configurations
│   └── federated/                 # Federated learning configurations
│
├── data/                          # Datasets
│   └── unsw-nb15/                 # UNSW-NB15 network intrusion dataset
│
├── docs/                          # Documentation
│   ├── _config.yml
│   ├── README.md                  # Documentation index
│   ├── MANIFEST.md                # Project manifest
│   ├── Threat_Model.md            # Security threat model
│   ├── RedTeamLog.md              # Red team testing logs
│   ├── Binary_Multiclass_Switching.md
│   ├── Centralized_Optimization_Report.md
│   ├── CIC_UNSW_NB15_Integration_Plan.md
│   ├── Project/                   # Project-specific docs
│   │   ├── SETUP_GUIDE.md         # Detailed setup instructions
│   │   └── UMAP_DOCUMENTATION.md  # UMAP methodology documentation
│   ├── images/                    # Documentation images
│   └── data/                      # Data documentation
│
├── notebooks/                     # Jupyter Notebooks
│   ├── 01_unsw_nb15_preprocessing_sanity.ipynb
│   ├── 02_unsw_nb15_umap_model_embeddings.ipynb
│   ├── AutoGluon_Model_Reconstruction.ipynb
│   ├── data_distribution.ipynb
│   ├── EDA_UNSW_big.ipynb
│   ├── EDA_UNSW.ipynb
│   ├── figure.ipynb
│   ├── unsw_autogluon.ipynb
│   ├── unsw_MLP_Classifier.ipynb
│   └── 2-stage-implementation/    # Two-stage implementation notebooks
│
├── outputs/                       # Generated Outputs
│   ├── 2025-12-20/                # Timestamped experiment outputs
│   ├── 2025-12-24/
│   ├── 2025-12-25/
│   ├── 2025-12-29/
│   ├── 2026-01-04/
│   ├── 2026-01-16/
│   ├── 2026-01-19/
│   ├── 2026-01-26/
│   ├── 2026-02-14/
│   └── 2026-03-02/
│
├── plots/                         # Visualization Outputs
│   └── Generated plots and figures from experiments
│
├── results/                       # Experiment Results
│   └── Organized results from various experiments
│
├── scripts/                       # Utility Scripts
│   └── Helper and automation scripts
│
├── src/                           # Source Code
│   └── Core implementation and algorithms
│
└── wandb/                         # Weights & Biases Logs
    └── Experiment tracking and logging
```

### Key Directories Description

#### `/src/` - Source Code
Core implementation and algorithms for the federated learning NIDS system

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
- Setup guide (SETUP_GUIDE.md)
- UMAP documentation (UMAP_DOCUMENTATION.md)
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

