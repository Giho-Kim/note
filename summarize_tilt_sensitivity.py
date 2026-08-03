"""Summarize the tilt_temperature x tilt_candidate_multiplier sensitivity
sweep launched by run_tilt_sensitivity.sh. Reads each run's wandb summary
(final logged metrics) and pivots eval/task_reward_iqm into a
domain -> multiplier -> temperature table.

Usage: python summarize_tilt_sensitivity.py
"""
import json
from pathlib import Path

import yaml

WANDB_DIR = Path(__file__).parent / "wandb"


def load_runs():
    rows = []
    for run_dir in sorted(WANDB_DIR.glob("run-*")):
        files_dir = run_dir / "files"
        config_path = files_dir / "config.yaml"
        summary_path = files_dir / "wandb-summary.json"
        if not config_path.exists() or not summary_path.exists():
            continue
        config = yaml.safe_load(config_path.read_text())
        run_name = config.get("run_name", {}).get("value")
        if not run_name or not str(run_name).startswith("tiltsens_"):
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "run_name": run_name,
                "domain": config.get("domain_name", {}).get("value"),
                "temperature": config.get("tilt_temperature", {}).get("value"),
                "multiplier": config.get("tilt_candidate_multiplier", {}).get("value"),
                "refresh_interval": config.get("tilt_refresh_interval", {}).get("value"),
                "task_reward_iqm": summary.get("eval/task_reward_iqm"),
            }
        )
    return rows


def main():
    rows = load_runs()
    if not rows:
        print("No tiltsens_* runs found yet.")
        return

    domains = sorted({r["domain"] for r in rows})
    for domain in domains:
        print(f"\n=== {domain} ===")
        drows = [r for r in rows if r["domain"] == domain]
        multipliers = sorted({r["multiplier"] for r in drows})
        temperatures = sorted({r["temperature"] for r in drows})
        header = "temperature".ljust(12) + "".join(f"mult={m}".rjust(12) for m in multipliers)
        print(header)
        for t in temperatures:
            line = str(t).ljust(12)
            for m in multipliers:
                match = [r for r in drows if r["temperature"] == t and r["multiplier"] == m]
                val = match[0]["task_reward_iqm"] if match and match[0]["task_reward_iqm"] is not None else None
                line += (f"{val:.2f}".rjust(12) if val is not None else "pending".rjust(12))
            print(line)


if __name__ == "__main__":
    main()
