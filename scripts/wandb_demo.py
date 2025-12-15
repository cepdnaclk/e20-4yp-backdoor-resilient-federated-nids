"""Demo script that logs "Training Loss" and "Test Accuracy" to W&B using the helper.

Usage:
    python scripts/wandb_demo.py --project fl-nids-demo --name demo-run --steps 50

If you haven't logged in to W&B, the script will run in offline mode and still produce a local run.
To push runs to wandb.ai, run `wandb login` or set `WANDB_API_KEY` in your environment.
"""
import argparse
import math
import time
from src.utils.logger import init_wandb, log_metrics, finish_wandb


def run_demo(project: str, name: str, steps: int, delay: float = 0.1, online: bool | None = None):
    init_wandb(project=project, name=name, mode=("offline" if online else None))

    print(f"Starting demo run: project={project} name={name} steps={steps}")

    for step in range(steps):
        # synthetic metrics: decaying loss and increasing accuracy with noise
        loss = math.exp(-step / (steps / 5.0)) + (0.02 * (0.5 - math.sin(step)))
        acc = (1.0 - math.exp(-step / (steps / 6.0))) * 100.0

        # Log the metrics expected by the acceptance criteria
        log_metrics({"Training Loss": float(loss), "Test Accuracy": float(acc)}, step=step)

        # minimal delay so W&B can show the stream if online
        time.sleep(delay)

    finish_wandb()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="fl-nids-demo")
    parser.add_argument("--name", default="demo-run")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--online", action="store_true", help="Force online mode (requires wandb login)")

    args = parser.parse_args()
    run_demo(args.project, args.name, args.steps, delay=args.delay, online=args.online)


if __name__ == "__main__":
    main()
