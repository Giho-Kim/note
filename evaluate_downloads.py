"""
Re-evaluates checkpoints saved under downloads/ (e.g. downloads/FB/..., downloads/D-LEVER/...)
on their zero-shot task reward, using the same eval procedure as OfflineRLWorkspace.eval.

Each run directory is expected to contain a config.json (used to reconstruct the
environment/tasks the checkpoint was trained with) plus one or more <seed>.pt files,
each a fully pickled agent (torch.save(agent, ...)).

Usage:
    python evaluate_downloads.py
    python evaluate_downloads.py --downloads_dir downloads/D-LEVER --eval_rollouts 20
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from tqdm import tqdm

from agents.fb.agent import FB
from agents.fb.replay_buffer import FBReplayBuffer
from custom_dmc_tasks.point_mass_maze import GOALS as POINT_MASS_MAZE_GOALS
from rewards import RewardFunctionConstructor
from utils import BASE_DIR, set_seed_everywhere

# Extra zero-shot tasks to evaluate in addition to each run's original
# config["eval_tasks"], keyed by domain_name. FB/D-LEVER are task-agnostic at
# eval time, so any task with a reward_function under rewards/<domain>/ can be
# scored without retraining.
EXTRA_EVAL_TASKS = {
    "walker": ["walk_backward", "run_backward", "flip_backward"],
}


def resolve_eval_tasks(config: dict) -> List[str]:
    tasks = list(config["eval_tasks"])
    for extra_task in EXTRA_EVAL_TASKS.get(config["domain_name"], []):
        if extra_task not in tasks:
            tasks.append(extra_task)
    return tasks


def find_runs(downloads_dir: Path) -> Iterator[Tuple[Path, dict, List[Path]]]:
    for config_path in sorted(downloads_dir.glob("**/config.json")):
        if config_path.stat().st_size == 0:
            print(f"[skip] {config_path} is empty.")
            continue
        run_dir = config_path.parent
        checkpoints = sorted(run_dir.glob("*.pt"))
        if not checkpoints:
            continue
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        yield run_dir, config, checkpoints


def load_agent(checkpoint_path: Path, device: torch.device) -> FB:
    agent = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent.to(device)
    agent._device = device  # pylint: disable=protected-access
    agent.eval()
    return agent


def build_z_source(config: dict, reward_constructor: RewardFunctionConstructor, device: torch.device) -> dict:
    if config["domain_name"] == "point_mass_maze":
        goal_states = {
            task: torch.tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
            for task, goal in POINT_MASS_MAZE_GOALS.items()
            if task in config["eval_tasks"]
        }
        return {"goal_states": goal_states}

    dataset_path = (
        BASE_DIR
        / "datasets"
        / config["domain_name"]
        / config["exploration_algorithm"]
        / "dataset.npz"
    )
    replay_buffer = FBReplayBuffer(
        reward_constructor=reward_constructor,
        dataset_path=dataset_path,
        transitions=config["dataset_transitions"],
        relabel=False,
        task=None,
        device=device,
        discount=config["discount"],
        action_condition=config.get("action_condition"),
    )
    observations_z, rewards_z = replay_buffer.sample_task_inference_transitions(
        inference_steps=config["z_inference_steps"]
    )
    return {"observations_z": observations_z, "rewards_z": rewards_z}


def infer_zs(agent: FB, config: dict, z_source: dict) -> Dict[str, np.ndarray]:
    zs = {}
    if config["domain_name"] == "point_mass_maze":
        for task, goal_state in z_source["goal_states"].items():
            zs[task] = agent.infer_z(goal_state)
    else:
        for task, rewards in z_source["rewards_z"].items():
            zs[task] = agent.infer_z(z_source["observations_z"], rewards)
    return zs


def bootstrap_iqm_ci(
    rewards: np.ndarray,
    bootstrap_indices: np.ndarray,
    ci: float,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap CI for the IQM (25% trimmed mean) of `rewards`.

    `bootstrap_indices` has shape [n_bootstrap, len(rewards)] and is shared
    across tasks within a checkpoint so the OVERALL score can be bootstrapped
    by averaging the same resample across tasks (paired by rollout index),
    rather than by pretending per-task resamples are independent.
    """
    boot_iqms = stats.trim_mean(rewards[bootstrap_indices], 0.25, axis=1)
    alpha = (1 - ci) / 2
    ci_low, ci_high = np.quantile(boot_iqms, [alpha, 1 - alpha])
    return float(ci_low), float(ci_high), boot_iqms


def eval_agent(
    agent: FB,
    env,
    reward_functions: dict,
    zs: Dict[str, np.ndarray],
    tasks: List[str],
    eval_rollouts: int,
    eval_std: float,
    n_bootstrap: int,
    ci: float,
    rng: np.random.Generator,
) -> Tuple[Dict[str, dict], Tuple[float, float]]:
    agent.std_dev_schedule = eval_std
    eval_rewards = {task: [] for task in tasks}

    for _ in tqdm(range(eval_rollouts), leave=False, desc="rollouts"):
        for task in tasks:
            task_reward = 0.0
            timestep = env.reset()
            while not timestep.last():
                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=zs[task],
                    step=None,
                    sample=False,
                )
                timestep = env.step(action)
                task_reward += reward_functions[task](env.physics)
            eval_rewards[task].append(task_reward)

    bootstrap_indices = rng.integers(0, eval_rollouts, size=(n_bootstrap, eval_rollouts))

    metrics = {}
    boot_task_iqms = []
    for task, rewards in eval_rewards.items():
        rewards = np.asarray(rewards, dtype=np.float32)
        ci_low, ci_high, boot_iqms = bootstrap_iqm_ci(rewards, bootstrap_indices, ci)
        metrics[task] = {
            "iqm": float(stats.trim_mean(rewards, 0.25)),
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
        boot_task_iqms.append(boot_iqms)

    boot_overall = np.mean(np.stack(boot_task_iqms, axis=0), axis=0)
    alpha = (1 - ci) / 2
    overall_ci = tuple(
        float(v) for v in np.quantile(boot_overall, [alpha, 1 - alpha])
    )

    return metrics, overall_ci


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads_dir", type=Path, default=BASE_DIR / "downloads")
    parser.add_argument(
        "--eval_rollouts", type=int, default=None, help="Overrides config eval_rollouts."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Overrides config seed for env construction."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to <downloads_dir>/eval_results.csv.",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples used for each task's IQM CI.",
    )
    parser.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence interval width, e.g. 0.95 for a 95% CI.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=0,
        help="Seed for the bootstrap resampling RNG (independent of --seed).",
    )
    args = parser.parse_args()
    output_path = args.output or (args.downloads_dir / "eval_results.csv")

    device = torch.device(args.device)
    bootstrap_rng = np.random.default_rng(args.bootstrap_seed)
    rows = []

    for run_dir, config, checkpoints in find_runs(args.downloads_dir):
        seed = args.seed if args.seed is not None else config["seed"]
        set_seed_everywhere(seed)

        eval_rollouts = args.eval_rollouts or config.get("eval_rollouts", 10)
        eval_std = config.get("std_dev_eval", 0.05)
        eval_tasks = resolve_eval_tasks(config)

        reward_constructor = RewardFunctionConstructor(
            domain_name=config["domain_name"],
            task_names=eval_tasks,
            seed=seed,
            device=device,
        )
        z_source = build_z_source(config, reward_constructor, device)

        group = run_dir.relative_to(args.downloads_dir).as_posix()
        print(f"\n=== {group} ({config['domain_name']}, tilt={config.get('tilt')}) ===")

        for checkpoint_path in checkpoints:
            agent = load_agent(checkpoint_path, device)
            zs = infer_zs(agent, config, z_source)
            metrics, overall_ci = eval_agent(
                agent=agent,
                env=reward_constructor._env,
                reward_functions=reward_constructor.reward_functions,
                zs=zs,
                tasks=eval_tasks,
                eval_rollouts=eval_rollouts,
                eval_std=eval_std,
                n_bootstrap=args.n_bootstrap,
                ci=args.ci,
                rng=bootstrap_rng,
            )

            overall_iqm = float(np.mean([m["iqm"] for m in metrics.values()]))
            print(
                f"  {checkpoint_path.name}: task_reward_iqm={overall_iqm:.3f} "
                f"[{overall_ci[0]:.3f}, {overall_ci[1]:.3f}]"
            )

            for task, m in metrics.items():
                print(
                    f"    {task}: iqm={m['iqm']:.3f} [{m['ci_low']:.3f}, {m['ci_high']:.3f}] "
                    f"mean={m['mean']:.3f} std={m['std']:.3f}"
                )
                rows.append(
                    {
                        "run": group,
                        "domain_name": config["domain_name"],
                        "algorithm": config["algorithm"],
                        "tilt": config.get("tilt", False),
                        "checkpoint": checkpoint_path.name,
                        "seed": config.get("seed"),
                        "task": task,
                        "iqm": m["iqm"],
                        "ci_low": m["ci_low"],
                        "ci_high": m["ci_high"],
                        "mean": m["mean"],
                        "std": m["std"],
                        "eval_rollouts": eval_rollouts,
                    }
                )
            rows.append(
                {
                    "run": group,
                    "domain_name": config["domain_name"],
                    "algorithm": config["algorithm"],
                    "tilt": config.get("tilt", False),
                    "checkpoint": checkpoint_path.name,
                    "seed": config.get("seed"),
                    "task": "OVERALL",
                    "iqm": overall_iqm,
                    "ci_low": overall_ci[0],
                    "ci_high": overall_ci[1],
                    "mean": float(np.mean([m["mean"] for m in metrics.values()])),
                    "std": float(np.mean([m["std"] for m in metrics.values()])),
                    "eval_rollouts": eval_rollouts,
                }
            )

            del agent
            if device.type == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
