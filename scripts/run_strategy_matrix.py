#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import itertools
import os
import re
import subprocess
import time
from pathlib import Path

ROUND_METRIC_RE = re.compile(
    r"Round\s+(?P<round>\d+)\s*\|\s*Accuracy:\s*(?P<acc>[0-9.]+)%\s*\|\s*F1-score:\s*(?P<f1>[0-9.]+)\s*\|\s*😈 Backdoor ASR:\s*(?P<asr>[0-9.]+)%"
)


DEFAULT_ATTACKS = ["clean", "backdoor", "label_flip"]
DEFAULT_DEFENSES = [
    "avg",
    "median",
    "trimmed_mean",
    "krum",
    "multi_krum",
    "flame",
    "adaptive_clipping",
    "sentinel",
]
DEFAULT_PARTITIONS = ["iid", "dirichlet", "pathological"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full FL attack-defense-partition matrix.")
    parser.add_argument("--project-root", default=".", help="Path to repository root")
    parser.add_argument("--main-script", default="main.py", help="Entrypoint script")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--defenses", nargs="+", default=DEFAULT_DEFENSES)
    parser.add_argument("--partitions", nargs="+", default=DEFAULT_PARTITIONS)
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha")
    parser.add_argument("--group-prefix", default="matrix_fullscale", help="Hydra +group value prefix")
    parser.add_argument("--output-dir", default="results/matrix_runs", help="Output directory")
    parser.add_argument("--device", default=None, help="Override client.device if provided")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--max-runs", type=int, default=None, help="Optional cap for debugging")
    parser.add_argument(
        "--resume-dir",
        default=None,
        help="Existing matrix run directory under output-dir to resume from (e.g. 2026-03-13_21-58-05)",
    )
    return parser.parse_args()


def run_one(project_root: Path, main_script: str, combo: dict, wandb_mode: str) -> dict:
    attack = combo["attack"]
    defense = combo["defense"]
    partition = combo["partition"]
    seed = combo["seed"]
    alpha = combo["alpha"]
    group = combo["group"]
    device = combo.get("device")

    cmd = [
        "python",
        main_script,
        f"+group={group}",
        f"attack.type={attack}",
        f"server.defense={defense}",
        f"simulation.partition_method={partition}",
        f"simulation.random_seed={seed}",
    ]

    if partition == "dirichlet":
        cmd.append(f"simulation.alpha={alpha}")

    if device:
        cmd.append(f"client.device={device}")

    env = os.environ.copy()
    env["WANDB_MODE"] = wandb_mode

    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
    )
    duration = time.time() - started

    stdout = proc.stdout or ""
    stderr = proc.stderr or ""

    final_round = None
    final_acc = None
    final_f1 = None
    final_asr = None

    for match in ROUND_METRIC_RE.finditer(stdout):
        final_round = int(match.group("round"))
        final_acc = float(match.group("acc"))
        final_f1 = float(match.group("f1"))
        final_asr = float(match.group("asr"))

    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "attack": attack,
        "defense": defense,
        "partition": partition,
        "seed": seed,
        "alpha": alpha if partition == "dirichlet" else "",
        "group": group,
        "return_code": proc.returncode,
        "duration_sec": round(duration, 2),
        "final_round": final_round if final_round is not None else "",
        "accuracy": final_acc if final_acc is not None else "",
        "f1": final_f1 if final_f1 is not None else "",
        "asr": final_asr if final_asr is not None else "",
        "stderr_tail": stderr.strip()[-1000:],
    }

    return row


def summarize(rows: list[dict], out_dir: Path) -> None:
    ok_rows = [r for r in rows if r["return_code"] == 0 and r["accuracy"] != ""]

    summary_csv = out_dir / "summary_by_attack_defense_partition.csv"
    fields = [
        "attack",
        "defense",
        "partition",
        "runs",
        "failed_runs",
        "mean_accuracy",
        "std_accuracy",
        "mean_f1",
        "std_f1",
        "mean_asr",
        "std_asr",
        "mean_duration_sec",
    ]

    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["attack"], row["defense"], row["partition"])
        grouped.setdefault(key, []).append(row)

    def _mean_std(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        return mean, var ** 0.5

    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for (attack, defense, partition), group_rows in sorted(grouped.items()):
            succ = [r for r in group_rows if r["return_code"] == 0 and r["accuracy"] != ""]
            fails = len(group_rows) - len(succ)

            acc_vals = [float(r["accuracy"]) for r in succ]
            f1_vals = [float(r["f1"]) for r in succ]
            asr_vals = [float(r["asr"]) for r in succ]
            dur_vals = [float(r["duration_sec"]) for r in succ]

            m_acc, s_acc = _mean_std(acc_vals)
            m_f1, s_f1 = _mean_std(f1_vals)
            m_asr, s_asr = _mean_std(asr_vals)
            m_dur, _ = _mean_std(dur_vals)

            writer.writerow(
                {
                    "attack": attack,
                    "defense": defense,
                    "partition": partition,
                    "runs": len(group_rows),
                    "failed_runs": fails,
                    "mean_accuracy": "" if succ == [] else round(m_acc, 4),
                    "std_accuracy": "" if succ == [] else round(s_acc, 4),
                    "mean_f1": "" if succ == [] else round(m_f1, 6),
                    "std_f1": "" if succ == [] else round(s_f1, 6),
                    "mean_asr": "" if succ == [] else round(m_asr, 4),
                    "std_asr": "" if succ == [] else round(s_asr, 4),
                    "mean_duration_sec": "" if succ == [] else round(m_dur, 2),
                }
            )

    # Save top-level high-level tables
    defense_view_csv = out_dir / "summary_by_defense.csv"
    attack_view_csv = out_dir / "summary_by_attack.csv"

    with defense_view_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["defense", "runs", "failed", "mean_accuracy", "mean_f1", "mean_asr"])
        by_def: dict[str, list[dict]] = {}
        for r in ok_rows:
            by_def.setdefault(r["defense"], []).append(r)
        for defense, def_rows in sorted(by_def.items()):
            writer.writerow(
                [
                    defense,
                    len(def_rows),
                    0,
                    round(sum(float(x["accuracy"]) for x in def_rows) / len(def_rows), 4),
                    round(sum(float(x["f1"]) for x in def_rows) / len(def_rows), 6),
                    round(sum(float(x["asr"]) for x in def_rows) / len(def_rows), 4),
                ]
            )

    with attack_view_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attack", "runs", "failed", "mean_accuracy", "mean_f1", "mean_asr"])
        by_attack: dict[str, list[dict]] = {}
        for r in ok_rows:
            by_attack.setdefault(r["attack"], []).append(r)
        for attack, atk_rows in sorted(by_attack.items()):
            writer.writerow(
                [
                    attack,
                    len(atk_rows),
                    0,
                    round(sum(float(x["accuracy"]) for x in atk_rows) / len(atk_rows), 4),
                    round(sum(float(x["f1"]) for x in atk_rows) / len(atk_rows), 6),
                    round(sum(float(x["asr"]) for x in atk_rows) / len(atk_rows), 4),
                ]
            )


def main() -> None:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    run_dir_name = args.resume_dir if args.resume_dir else dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = (project_root / args.output_dir / run_dir_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_combos = list(
        itertools.product(args.attacks, args.defenses, args.partitions, args.seeds)
    )
    completed_keys = set()

    raw_csv = out_dir / "run_results.csv"
    fields = [
        "timestamp",
        "attack",
        "defense",
        "partition",
        "seed",
        "alpha",
        "group",
        "return_code",
        "duration_sec",
        "final_round",
        "accuracy",
        "f1",
        "asr",
        "stderr_tail",
    ]

    existing_rows: list[dict] = []
    if raw_csv.exists() and raw_csv.stat().st_size > 0:
        with raw_csv.open("r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        for row in existing_rows:
            if str(row.get("return_code", "")) == "0":
                completed_keys.add(
                    (
                        row.get("attack"),
                        row.get("defense"),
                        row.get("partition"),
                        int(row.get("seed", 0)),
                    )
                )

    combos = [
        combo
        for combo in all_combos
        if (combo[0], combo[1], combo[2], combo[3]) not in completed_keys
    ]

    if args.max_runs is not None:
        combos = combos[: args.max_runs]

    total = len(combos)
    print(f"Total runs: {len(all_combos)} | Completed: {len(completed_keys)} | Pending: {total}")

    all_rows: list[dict] = list(existing_rows)

    write_mode = "a" if raw_csv.exists() and raw_csv.stat().st_size > 0 else "w"
    with raw_csv.open(write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_mode == "w":
            writer.writeheader()

        for idx, (attack, defense, partition, seed) in enumerate(combos, start=1):
            group = f"{args.group_prefix}_{attack}_{defense}_{partition}_s{seed}"
            combo = {
                "attack": attack,
                "defense": defense,
                "partition": partition,
                "seed": seed,
                "alpha": args.alpha,
                "group": group,
                "device": args.device,
            }

            print(f"[{idx}/{total}] attack={attack} defense={defense} partition={partition} seed={seed}")
            row = run_one(project_root, args.main_script, combo, args.wandb_mode)
            all_rows.append(row)
            writer.writerow(row)
            f.flush()

            status = "OK" if row["return_code"] == 0 else f"FAIL({row['return_code']})"
            print(
                f"    -> {status} | acc={row['accuracy']} f1={row['f1']} asr={row['asr']} | {row['duration_sec']}s"
            )

    summarize(all_rows, out_dir)

    total_fail = sum(1 for r in all_rows if r["return_code"] != 0)
    print("\nMatrix run complete")
    print(f"Output dir: {out_dir}")
    print(f"Failures: {total_fail}/{len(all_rows)}")


if __name__ == "__main__":
    main()
