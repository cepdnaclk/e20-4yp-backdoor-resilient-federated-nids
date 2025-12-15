"""W&B logging helper utilities.

Usage:
    from src.utils.logger import init_wandb, log_metrics, finish_wandb

    init_wandb(project="fl-nids-demo", name="run-1")
    for step in range(100):
        log_metrics({"Training Loss": some_loss, "Test Accuracy": some_acc}, step=step)
    finish_wandb()

If `WANDB_API_KEY` is not set, this helper will fall back to offline mode.
"""
from __future__ import annotations

import os
from typing import Optional, Dict, Any

try:
    import wandb
except Exception:  # pragma: no cover - import-time fallback if wandb isn't installed
    wandb = None


def _load_dotenv_if_exists(dotenv_path: str = ".env") -> None:
    """Simple `.env` loader: sets any KEY=VALUE pairs into os.environ if not already defined.

    This avoids adding an external dependency and helps picks up `WANDB_API_KEY` placed in a `.env` file.
    """
    if not os.path.exists(dotenv_path):
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                # don't overwrite existing env vars
                if key and os.environ.get(key) is None:
                    os.environ[key] = val
    except Exception:
        # Best-effort loader; ignore errors and continue
        return


def _detect_mode() -> str:
    # Try to load a .env file in the repo root (if user placed their WANDB_API_KEY there)
    _load_dotenv_if_exists()

    # If user has set a WANDB API key, prefer online; otherwise offline.
    if os.environ.get("WANDB_API_KEY"):
        return "online"
    # Allow user override via env var (WANDB_MODE) or use offline
    return os.environ.get("WANDB_MODE", "offline")


def init_wandb(project: str, entity: Optional[str] = None, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None, mode: Optional[str] = None):
    """Initialize a W&B run.

    - `project`: project name to record runs under
    - `entity`: wandb entity/team (optional)
    - `name`: run name (optional)
    - `config`: dict of hyperparameters
    - `mode`: 'online' or 'offline' (auto-detected by default)

    This function will auto-load a `.env` file if present (see project root).
    """
    if wandb is None:
        raise RuntimeError("wandb is not installed in the environment. Install it with `pip install wandb`.")

    # Ensure .env is loaded right before using the env var
    _load_dotenv_if_exists()

    final_mode = mode or _detect_mode()
    # initialize
    wandb.init(project=project, entity=entity, name=name, config=config or {}, mode=final_mode)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log metrics to the current W&B run."""
    if wandb is None:
        raise RuntimeError("wandb is not installed in the environment. Install it with `pip install wandb`.")

    if step is not None:
        wandb.log(metrics, step=step)
    else:
        wandb.log(metrics)


def finish_wandb() -> None:
    """Finish the current W&B run."""
    if wandb is None:
        return
    try:
        wandb.finish()
    except Exception:
        # ignore finish errors
        pass
