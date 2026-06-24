"""Roll out point-mass agents and plot eval trajectories."""

import argparse
import importlib
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors as mpl_colors
import numpy as np
import torch
from scipy import stats

from custom_dmc_tasks.point_mass_maze import GOALS
from agents.fb.agent import FB
from metamotivo.nn_models import eval_mode
from metamotivo.agents.td_jepa.agent import TDJEPAAgent
from rewards import RewardFunctionConstructor
from utils import set_seed_everywhere


DEFAULT_TASKS = (
    "reach_top_left",
    "reach_top_right",
    "reach_bottom_left",
    "reach_bottom_right",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("td_jepa", "fb"), default="td_jepa")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("agents/td_jepa/saved_models/stellar-universe-90/180000"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("trajectory_plots"))
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--output-prefix", type=str, default="stellar-universe-90")
    return parser.parse_args()


def select_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device_arg


def load_agent(agent_kind: str, model_path: Path, device: str):
    if agent_kind == "td_jepa":
        return TDJEPAAgent.load(str(model_path), device=device)
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location=device)


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def infer_goal_z(agent, goal: np.ndarray, device: str, agent_kind: str):
    goal_tensor = torch.as_tensor(goal, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if agent_kind == "fb":
            return agent.infer_z(goal_tensor)
        return agent._model.project_z(agent._model.psi(goal_tensor))


def rollout_task(agent, env, reward_fn, z, device: str, agent_kind: str):
    timestep = env.reset()
    observations = [np.asarray(timestep.observation["observations"], dtype=np.float32)]
    actions = []
    rewards = []

    while not timestep.last():
        obs = torch.as_tensor(
            timestep.observation["observations"],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        with torch.no_grad():
            if agent_kind == "fb":
                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=z,
                    step=None,
                    sample=False,
                )
            else:
                action = agent.act(obs=obs, z=z, mean=True).detach().cpu().numpy()[0]

        timestep = env.step(action)
        reward = float(np.asarray(reward_fn(env.physics), dtype=np.float32).reshape(-1)[0])
        observations.append(
            np.asarray(timestep.observation["observations"], dtype=np.float32)
        )
        actions.append(np.asarray(action, dtype=np.float32))
        rewards.append(reward)

    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "return": float(np.sum(rewards)),
    }


def compute_fb_leverage(agent: FB, rollout: dict, z, device: str) -> None:
    if agent.tilt is None:
        rollout["leverage_scores"] = np.asarray([], dtype=np.float32)
        rollout["leverage_score"] = np.nan
        return

    observations = rollout["observations"][:-1]
    actions = rollout["actions"]
    if len(actions) == 0:
        rollout["leverage_scores"] = np.asarray([], dtype=np.float32)
        rollout["leverage_score"] = np.nan
        return

    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act = torch.as_tensor(actions, dtype=torch.float32, device=device)
    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, -1)
    z_tensor = z_tensor.expand(obs.shape[0], -1)

    with torch.no_grad():
        target_f1, target_f2 = agent.FB.forward_representation_target(
            observation=obs,
            z=z_tensor,
            action=act,
        )
        features = 0.5 * (target_f1 + target_f2)
        query = z_tensor if agent._tilting_by_z else features

        gram = agent.tilt.gram.to(device=query.device, dtype=query.dtype)
        if gram.shape[0] != query.shape[-1]:
            raise ValueError(
                f"FB Gram dimension {gram.shape[0]} does not match "
                f"leverage query dimension {query.shape[-1]}."
            )

        trace_g = torch.trace(gram)
        lam = max(
            agent._tilt_ridge_alpha * trace_g.item() / gram.shape[0],
            agent._tilt_ridge_min,
        )
        identity = torch.eye(query.shape[-1], device=query.device, dtype=query.dtype)
        ginv = torch.linalg.pinv(gram + lam * identity)
        scores = torch.sum((query @ ginv) * query, dim=1)

    leverage_scores = scores.detach().cpu().numpy().astype(np.float32)
    rollout["leverage_scores"] = leverage_scores
    rollout["leverage_score"] = float(leverage_scores.mean())


def compute_tdjepa_leverage(agent: TDJEPAAgent, rollout: dict, z, device: str) -> None:
    if agent.tilt is None:
        rollout["leverage_scores"] = np.asarray([], dtype=np.float32)
        rollout["leverage_score"] = np.nan
        return

    observations = rollout["observations"][:-1]
    actions = rollout["actions"]
    if len(actions) == 0:
        rollout["leverage_scores"] = np.asarray([], dtype=np.float32)
        rollout["leverage_score"] = np.nan
        return

    model = agent._model
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    act = torch.as_tensor(actions, dtype=torch.float32, device=device)
    z_tensor = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, -1)
    z_tensor = z_tensor.expand(obs.shape[0], -1)

    with torch.no_grad(), eval_mode(model._obs_normalizer):
        obs = model._obs_normalizer(obs)
    with torch.no_grad():
        phi_obs = model._phi_rgb_encoder(model._augmentator(obs))
        phi_enc = model._target_phi_mlp_encoder(phi_obs)
        target_phi_predictors = model._target_phi_predictor(phi_enc, z_tensor, act)
        if target_phi_predictors.ndim == 3:
            features = target_phi_predictors.mean(dim=0)
        else:
            features = target_phi_predictors
        query = z_tensor if agent.cfg.train.tilting_by_z else features

        gram = agent.tilt.gram.to(device=query.device, dtype=query.dtype)
        if gram.shape[0] != query.shape[-1]:
            raise ValueError(
                f"TD-JEPA Gram dimension {gram.shape[0]} does not match "
                f"leverage query dimension {query.shape[-1]}."
            )

        trace_g = torch.trace(gram)
        lam = max(
            agent.cfg.train.tilt_ridge_alpha * trace_g.item() / gram.shape[0],
            agent.cfg.train.tilt_ridge_min,
        )
        identity = torch.eye(query.shape[-1], device=query.device, dtype=query.dtype)
        ginv = torch.linalg.pinv(gram + lam * identity)
        scores = torch.sum((query @ ginv) * query, dim=1)

    leverage_scores = scores.detach().cpu().numpy().astype(np.float32)
    rollout["leverage_scores"] = leverage_scores
    rollout["leverage_score"] = float(leverage_scores.mean())


def draw_maze(ax):
    ax.set_xlim(-0.31, 0.31)
    ax.set_ylim(-0.31, 0.31)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linewidth=0.4, alpha=0.25)

    walls = [
        (-0.30, -0.30, 0.60, 0.012),
        (-0.30, 0.288, 0.60, 0.012),
        (-0.30, -0.30, 0.012, 0.60),
        (0.288, -0.30, 0.012, 0.60),
        (-0.18, -0.02, 0.36, 0.04),
        (-0.02, -0.18, 0.04, 0.36),
    ]
    for x, y, width, height in walls:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                width,
                height,
                facecolor="black",
                edgecolor="black",
                alpha=0.18,
                linewidth=0.0,
            )
        )


def _finite_leverage_scores(results) -> np.ndarray:
    scores = [
        rollout.get("leverage_score", np.nan)
        for task_rollouts in results.values()
        for rollout in task_rollouts
    ]
    scores = np.asarray(scores, dtype=np.float32)
    return scores[np.isfinite(scores)]


def plot_trajectories(results, output_path: Path, title: str):
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
    fallback_colors = plt.get_cmap("tab10")
    leverage_scores = _finite_leverage_scores(results)
    leverage_norm = None
    leverage_cmap = cm.get_cmap("viridis")
    if leverage_scores.size > 0:
        vmin = float(leverage_scores.min())
        vmax = float(leverage_scores.max())
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-6
        leverage_norm = mpl_colors.Normalize(vmin=vmin, vmax=vmax)

    for ax, (task, task_rollouts) in zip(axes.flat, results.items()):
        draw_maze(ax)
        goal = GOALS[task][:2]
        ax.scatter(goal[0], goal[1], marker="*", s=180, color="crimson", label="goal")

        for idx, rollout in enumerate(task_rollouts):
            xy = rollout["observations"][:, :2]
            leverage_score = rollout.get("leverage_score", np.nan)
            if leverage_norm is not None and np.isfinite(leverage_score):
                color = leverage_cmap(leverage_norm(leverage_score))
            else:
                color = fallback_colors(idx % 10)
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=color,
                alpha=0.82,
                linewidth=1.4,
            )
            ax.scatter(xy[0, 0], xy[0, 1], color=color, s=12, alpha=0.75)

        returns = np.asarray([r["return"] for r in task_rollouts], dtype=np.float32)
        iqm = stats.trim_mean(returns, 0.25)
        title_parts = [f"{task}", f"IQM={iqm:.3f}, mean={returns.mean():.3f}"]
        task_leverage = np.asarray(
            [r.get("leverage_score", np.nan) for r in task_rollouts], dtype=np.float32
        )
        task_leverage = task_leverage[np.isfinite(task_leverage)]
        if task_leverage.size > 0:
            title_parts.append(f"lev_mean={task_leverage.mean():.3f}")
        ax.set_title("\n".join(title_parts))

    fig.suptitle(title, fontsize=14)
    if leverage_norm is not None:
        sm = cm.ScalarMappable(norm=leverage_norm, cmap=leverage_cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.82, label="leverage score")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def run_trajectory_plot(
    agent,
    output_dir: Path,
    rollouts: int,
    seed: int,
    device: str,
    tasks=DEFAULT_TASKS,
    output_prefix: str = "td_jepa",
    title: str = "TD-JEPA eval trajectories",
    agent_kind: str = "td_jepa",
):
    if rollouts <= 0:
        raise ValueError("--rollouts must be positive.")

    tasks = list(tasks)
    rng_state = capture_rng_state()
    try:
        set_seed_everywhere(seed)

        reward_constructor = RewardFunctionConstructor(
            domain_name="point_mass_maze",
            task_names=tasks,
            seed=seed,
            device=torch.device(device),
        )
        reward_functions = {
            task: importlib.import_module(
                f"rewards.point_mass_maze.{task}"
            ).reward_function
            for task in tasks
        }

        model = agent._model if agent_kind == "td_jepa" else agent
        was_training = model.training
        model.eval()

        try:
            zs = {
                task: infer_goal_z(agent, GOALS[task], device, agent_kind)
                for task in tasks
            }
            results = {}
            for task in tasks:
                task_rollouts = []
                for _ in range(rollouts):
                    rollout = rollout_task(
                        agent=agent,
                        env=reward_constructor._env,
                        reward_fn=reward_functions[task],
                        z=zs[task],
                        device=device,
                        agent_kind=agent_kind,
                    )
                    if agent_kind == "fb":
                        compute_fb_leverage(agent, rollout, zs[task], device)
                    elif agent_kind == "td_jepa":
                        compute_tdjepa_leverage(agent, rollout, zs[task], device)
                    task_rollouts.append(rollout)
                results[task] = task_rollouts
        finally:
            model.train(was_training)

        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f"{output_prefix}_trajectories.png"
        plot_trajectories(results, plot_path, title=title)

        npz_path = output_dir / f"{output_prefix}_trajectories.npz"
        np.savez(
            npz_path,
            tasks=np.asarray(tasks),
            returns={
                task: np.asarray([rollout["return"] for rollout in task_rollouts])
                for task, task_rollouts in results.items()
            },
            observations={
                task: [rollout["observations"] for rollout in task_rollouts]
                for task, task_rollouts in results.items()
            },
            actions={
                task: [rollout["actions"] for rollout in task_rollouts]
                for task, task_rollouts in results.items()
            },
            rewards={
                task: [rollout["rewards"] for rollout in task_rollouts]
                for task, task_rollouts in results.items()
            },
            leverage_scores={
                task: [
                    rollout.get("leverage_scores", np.asarray([], dtype=np.float32))
                    for rollout in task_rollouts
                ]
                for task, task_rollouts in results.items()
            },
            trajectory_leverage={
                task: np.asarray(
                    [rollout.get("leverage_score", np.nan) for rollout in task_rollouts],
                    dtype=np.float32,
                )
                for task, task_rollouts in results.items()
            },
        )

        return plot_path, npz_path, results
    finally:
        restore_rng_state(rng_state)


def main():
    args = parse_args()

    device = select_device(args.device)
    agent = load_agent(args.agent, args.model_path, device)
    plot_path, npz_path, results = run_trajectory_plot(
        agent=agent,
        output_dir=args.output_dir,
        rollouts=args.rollouts,
        seed=args.seed,
        device=device,
        tasks=args.tasks,
        output_prefix=args.output_prefix,
        title=f"{args.agent.upper()} {args.output_prefix} eval trajectories",
        agent_kind=args.agent,
    )

    print(f"saved plot: {plot_path}")
    print(f"saved rollout data: {npz_path}")
    for task, task_rollouts in results.items():
        returns = np.asarray([rollout["return"] for rollout in task_rollouts])
        print(
            f"{task}: iqm={stats.trim_mean(returns, 0.25):.6f}, "
            f"mean={returns.mean():.6f}, returns={np.array2string(returns, precision=3)}"
        )


if __name__ == "__main__":
    main()
