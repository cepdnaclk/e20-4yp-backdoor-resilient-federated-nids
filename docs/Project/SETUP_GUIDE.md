# 🛠 Project Setup Guide

This document guides to set up the development and experiment environment for the research.
Please read this **before starting development or experiments**.

---

## 📌 Prerequisites

Ensure the following are available on your machine or server (Ampere/Tesla):

- Linux (Ubuntu recommended)
- Conda / Miniconda installed
- NVIDIA GPU + CUDA-compatible drivers (for GPU training)
- Internet access (for initial setup & W&B login)

Check conda:
```
conda --version
```

##  Clone the Repository
git clone <REPO_URL>
cd e20-4yp-Federated-Privacy-Aware-Network-Anomaly-Detection

## Create the Conda Environment

This project uses a shared conda environment defined in environment.yml.

⚠️ Do NOT use the base environment.

```
conda env create -f environment.yml
```
This creates an environment named:fl-nids

## Activate the Environment
```
conda activate fl-nids
```
Expected:: (fl-nids) user@machine:~

## Login to Weights & Biases (One-Time Setup)

This project uses **Weights & Biases (W&B)** to track experiments, log metrics, and visualize results.

Run the following command:
```
wandb login
```
Paste your W&B API key when prompted

### Verify W&B Authentication (Optional)
```
wandb status
```
## Executing a W&B Sweep agent

This section describes how to execute a Weights & Biases (W&B) sweep for the first time and how to proceed after modifying the sweep configuration.

If configs/sweep.yaml is running for the first time or modified, the existing sweep cannot be updated. 
A new sweep must be created.
```
wandb sweep configs/sweep.yaml
```
This generates a new Sweep ID reflecting the updated configuration.

Run the sweep agent using the new Sweep ID
```
wandb agent fyp-group8/e20-4yp-backdoor-resilient-federated-nids/<NEW_SWEEP_ID> --count 20
```
fyp-group8 : W&B entity (team or user)

e20-4yp-backdoor-resilient-federated-nids : W&B project name

<SWEEP_ID> : ID returned when the sweep was created

--count 20 : number of sweep runs (parameter configurations) to execute



