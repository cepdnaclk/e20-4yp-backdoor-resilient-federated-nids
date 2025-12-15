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

## WandB (Optional) — Setup and Usage
This project supports experiment tracking with Weights & Biases (W&B). The helper in `src/utils/logger.py` will automatically load a `.env` file in the repo root (if present) so you can store your `WANDB_API_KEY` there securely.

1) Install & login (This is not required since you already installed wnadb library in the fl-nids environment):
```
pip install wandb
```

2) Or provide the API key via a `.env` file (already ignored by `.gitignore`):
Create a file named `.env` in the project root with a single line:
```
WANDB_API_KEY=your_api_key_here
```
The `src/utils/logger.py` helper auto-loads `.env` so you won't need to run `wandb login` each time. Do NOT commit this file — it is included in `.gitignore` by default.

3) Run the demo script (from the project root) to validate logging:
```
python -m scripts.wandb_demo

```
- Use `--online` to force online mode (requires login or `WANDB_API_KEY`).
- Omitting `--online` will run in `offline` mode and still create a local run you can inspect.

4) Verify results
```
wandb whoami
# then visit: https://wandb.ai/<your-username>/fl-nids-demo
```

Notes
- You can also set the API key as a conda env var (persisted to the environment):
```
conda env config vars set WANDB_API_KEY=your_api_key_here
conda activate fl-nids
```
- Do not hard-code keys in source files. Keep them in secrets or the `.env` file and never commit them.
- If needed, set `WANDB_MODE=online` or `WANDB_MODE=offline` to override mode detection.
