"""Module that creates workspaces for training/evaling various agents."""

import os

os.environ.setdefault("WANDB_CONSOLE", "off")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_DISABLE_GIT", "true")
os.environ.setdefault("WANDB_SILENT", "true")

import wandb
import torch
import shutil
from os import makedirs
from loguru import logger
from tqdm import tqdm
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional
from scipy import stats

from rewards import RewardFunctionConstructor
from custom_dmc_tasks.point_mass_maze import GOALS as point_mass_maze_goals

from agents.base import AbstractWorkspace
from agents.fb.agent import FB
from agents.fb.replay_buffer import FBReplayBuffer, OnlineFBReplayBuffer

from agents.cql.agent import CQL
from agents.base import OfflineReplayBuffer

from agents.cfb.agent import CFB
from agents.gciql.agent import GCIQL

from agents.sf.agent import SF
from agents.td_jepa.agent import TDJEPA
from agents.base import D4RLReplayBuffer


def _configure_wandb_run(run) -> None:
    if run is None:
        return
    run.define_metric("step")
    run.define_metric("train/*", step_metric="step")
    run.define_metric("eval/*", step_metric="step")
    run.define_metric("collection/*", step_metric="step")
    run.define_metric("tilt/*", step_metric="step")


def _log_wandb(run, metrics: Dict[str, float], step: int) -> None:
    if not metrics:
        return
    metrics = {**metrics, "step": step}
    run.log(metrics, step=step)


class OfflineRLWorkspace(AbstractWorkspace):
    """
    Trains/evals/rollouts an offline RL agent given
    """
    COLLECTION_TILT_TEMPERATURE = 5.0
    TILT_VERBOSE_INTERVAL = 20000
    SAVE_EVERY_INTERVAL = 100000

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        device: torch.device,
        z_inference_steps: Optional[int] = None,  # FB only
        train_std: Optional[float] = None,  # FB only
        eval_std: Optional[float] = None,  # FB only
        collection_interval: int = 0,
        collection_episodes: int = 0,
        verbose: bool = False,
        save_every: bool = False,
    ):
        super().__init__(
            env=reward_constructor._env,
            reward_functions=reward_constructor.reward_functions,
        )

        self.eval_frequency = eval_frequency  # how frequently to eval
        self.eval_rollouts = eval_rollouts  # how many rollouts per eval step
        self.model_dir = model_dir
        self.learning_steps = learning_steps
        self.z_inference_steps = z_inference_steps
        self.train_std = train_std
        self.eval_std = eval_std
        self.observations_z = None
        self.rewards_z = None
        self.wandb_logging = wandb_logging
        self.domain_name = reward_constructor.domain_name
        self.device = device
        self.collection_interval = collection_interval
        self.collection_episodes = collection_episodes
        self.verbose = verbose
        self.save_every = save_every  # save a checkpoint at every eval step, not just the best
        self._tilt_header_printed = False

    def train(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        tasks: List[str],
        agent_config: Dict,
        replay_buffer: Union[OfflineReplayBuffer, FBReplayBuffer],
    ) -> None:
        """
        Trains an offline RL algorithm on one task.
        """
        run = None
        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name],
                reinit=True,
                settings=wandb.Settings(console="off", _disable_stats=True, silent=True),
            )
            _configure_wandb_run(run)
            model_path = self.model_dir / run.name
            makedirs(str(model_path))

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        logger.info(f"Training {agent.name}.")
        best_mean_task_reward = -np.inf
        best_model_path = None

        # sample set transitions for z inference
        if isinstance(agent, (FB, SF, GCIQL, TDJEPA)):
            if self.domain_name == "point_mass_maze":
                self.goal_states = {}
                for task, goal_state in point_mass_maze_goals.items():
                    self.goal_states[task] = torch.tensor(
                        goal_state, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            else:
                (
                    self.observations_z,
                    self.rewards_z,
                ) = replay_buffer.sample_task_inference_transitions(
                    inference_steps=self.z_inference_steps
                )

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)
            train_metrics = agent.update(batch=batch, step=i)

            if self.verbose and i % self.TILT_VERBOSE_INTERVAL == 0:
                self._log_tilt_health(agent=agent, step=i)

            eval_metrics = {}
            collection_metrics = {}
            tilt_metrics = {}

            if i % self.eval_frequency == 0:
                tilt_metrics = self._eval_task_tilt_score_metrics(
                    agent=agent,
                    observations=batch.observations,
                    timesteps=batch.timesteps,
                    tasks=tasks,
                    step=i,
                )
                eval_metrics = self.eval(agent=agent, tasks=tasks)
                self._maybe_plot_point_mass_trajectories(
                    agent=agent,
                    tasks=tasks,
                    step=i,
                    run_name=model_path.name,
                    wandb_run=run,
                )

                if self.save_every and i % self.SAVE_EVERY_INTERVAL == 0:
                    checkpoint_dir = model_path / "checkpoints"
                    agent._name = i  # pylint: disable=protected-access
                    checkpoint_path = agent.save(checkpoint_dir)
                    if self.wandb_logging:
                        run.save(
                            checkpoint_path.as_posix(), base_path=model_path.as_posix()
                        )

                if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                    logger.info(
                        f"New max IQM task reward: {best_mean_task_reward:.3f} -> "
                        f"{eval_metrics['eval/task_reward_iqm']:.3f}."
                        f" Saving model."
                    )

                    # delete current best model
                    if best_model_path is not None:
                        if best_model_path.is_dir():
                            shutil.rmtree(best_model_path)
                        else:
                            best_model_path.unlink(missing_ok=True)

                    agent._name = i  # pylint: disable=protected-access
                    # save locally
                    best_model_path = agent.save(model_path)

                    best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                agent.train()

            if (
                self.collection_interval > 0
                and self.collection_episodes > 0
                and i > 0
                and i % self.collection_interval == 0
            ):
                collection_metrics = self.collect_training_episodes(
                    agent=agent,
                    tasks=tasks,
                    replay_buffer=replay_buffer,
                    step=i,
                )

            metrics = {
                **train_metrics,
                **eval_metrics,
                **collection_metrics,
                **tilt_metrics,
            }

            if self.wandb_logging and (
                i % self.eval_frequency == 0 or bool(collection_metrics)
            ):
                _log_wandb(run, metrics, i)

        if self.wandb_logging:
            # save to wandb_logging
            run.save(best_model_path.as_posix(), base_path=model_path.as_posix())
            run.finish()

        # delete local models
        shutil.rmtree(model_path)

    def eval(
        self,
        agent: Union[CQL, FB, CFB, SF, TDJEPA],
        tasks: List[str],
    ) -> Dict[str, float]:
        """
        Performs eval rollouts.
        Args:
            agent: agent to evaluate
            tasks: tasks to evaluate on
        Returns:
            metrics: dict of metrics
        """

        if isinstance(agent, (FB, SF, GCIQL, TDJEPA)):
            zs = {}
            if self.domain_name == "point_mass_maze":
                for task, goal_state in self.goal_states.items():
                    zs[task] = agent.infer_z(goal_state)
            else:
                for task, rewards in self.rewards_z.items():
                    zs[task] = agent.infer_z(self.observations_z, rewards)

            agent.std_dev_schedule = self.eval_std

        logger.info("Performing eval rollouts.")
        eval_rewards = {}
        agent.eval()
        for _ in tqdm(range(self.eval_rollouts)):

            for task in tasks:
                task_rewards = 0.0

                timestep = self.env.reset()
                while not timestep.last():
                    if isinstance(agent, (FB, GCIQL, TDJEPA)):
                        action, _ = agent.act(
                            timestep.observation["observations"],
                            task=zs[task],
                            step=None,
                            sample=False,
                        )

                    elif isinstance(agent, SF):
                        if self.domain_name != "point_mass_maze":
                            z = zs[task]
                        # calculate z at every step
                        else:
                            z = agent.infer_z_from_goal(
                                observation=timestep.observation["observations"],
                                goal_state=self.goal_states[task],
                            )
                        action, _ = agent.act(
                            timestep.observation["observations"],
                            task=z,
                            step=None,
                            sample=False,
                        )

                    else:
                        action = agent.act(
                            timestep.observation["observations"],
                            sample=False,
                            step=None,
                        )
                    timestep = self.env.step(action)
                    task_rewards += self.reward_functions[task](self.env.physics)

                if task not in eval_rewards:
                    eval_rewards[task] = []
                eval_rewards[task].append(task_rewards)

        # average over rollouts for metrics
        metrics = {}
        mean_task_performance = 0.0
        for task, rewards in eval_rewards.items():
            rewards = np.asarray(rewards, dtype=np.float32)
            mean_task_reward = stats.trim_mean(rewards, 0.25)  # IQM
            metrics[f"eval/{task}/episode_reward_iqm"] = mean_task_reward
            mean_task_performance += mean_task_reward

        # log mean task performance
        metrics["eval/task_reward_iqm"] = mean_task_performance / len(tasks)
        eval_parts = [
            f"{key}={value:.4g}"
            for key, value in metrics.items()
            if key.startswith("eval/")
        ]
        print(
            "[eval returns] " + ", ".join(eval_parts),
            flush=True,
        )

        if hasattr(agent, "std_dev_schedule") and self.train_std is not None:
            agent.std_dev_schedule = self.train_std

        return metrics

    @staticmethod
    def _extract_observation(timestep) -> np.ndarray:
        observation = timestep.observation
        if isinstance(observation, dict):
            observation = observation["observations"]
        return np.asarray(observation, dtype=np.float32)

    @staticmethod
    def _pack_scalar(value: float) -> np.ndarray:
        if value is None:
            value = 0.0
        return np.asarray([value], dtype=np.float32)

    def _pack_task_rewards(self) -> np.ndarray:
        rewards = []
        for reward_fn in self.reward_functions.values():
            reward = np.asarray(reward_fn(self.env.physics), dtype=np.float32)
            rewards.append(float(reward.reshape(-1)[0]))
        return np.asarray(rewards, dtype=np.float32)

    def _pack_collection_reward(
        self,
        timestep_reward: float,
        reward_dim: int,
    ) -> np.ndarray:
        if reward_dim == 1:
            return self._pack_scalar(timestep_reward)

        task_rewards = self._pack_task_rewards()
        if task_rewards.shape[0] == reward_dim:
            return task_rewards
        if task_rewards.shape[0] > reward_dim:
            return task_rewards[:reward_dim]
        raise ValueError(
            f"Cannot pack collection reward with dim {task_rewards.shape[0]} "
            f"for replay reward dim {reward_dim}."
        )

    def _current_physics(self) -> np.ndarray:
        return np.asarray(self.env.physics.state())

    def _rollout_collection_episode(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        condition: Optional[np.ndarray],
        step: int,
        reward_dim: int,
    ) -> Dict[str, np.ndarray]:
        timestep = self.env.reset()
        action_spec = self.env.action_spec()
        episode = {
            "observation": [self._extract_observation(timestep)],
            "action": [np.zeros(action_spec.shape, dtype=np.float32)],
            "reward": [self._pack_collection_reward(timestep.reward, reward_dim)],
            "discount": [self._pack_scalar(timestep.discount)],
            "physics": [self._current_physics()],
        }

        while not timestep.last():
            observation = self._extract_observation(timestep)

            if isinstance(agent, (FB, GCIQL, SF, TDJEPA)):
                action, _ = agent.act(
                    observation,
                    task=condition,
                    step=step,
                    sample=True,
                )
            else:
                action = agent.act(
                    observation=observation,
                    sample=True,
                    step=step,
                )

            timestep = self.env.step(action)
            episode["observation"].append(self._extract_observation(timestep))
            episode["action"].append(np.asarray(action, dtype=np.float32))
            episode["reward"].append(
                self._pack_collection_reward(timestep.reward, reward_dim)
            )
            episode["discount"].append(self._pack_scalar(timestep.discount))
            episode["physics"].append(self._current_physics())

        return {
            "observation": np.asarray(episode["observation"], dtype=np.float32),
            "action": np.asarray(episode["action"], dtype=np.float32),
            "reward": np.asarray(episode["reward"], dtype=np.float32),
            "discount": np.asarray(episode["discount"], dtype=np.float32),
            "physics": np.asarray(episode["physics"]),
        }

    def _collection_reward_names(self, reward_dim: int) -> List[str]:
        task_names = list(self.reward_functions.keys())
        if len(task_names) == reward_dim:
            return task_names
        return [f"reward_{i}" for i in range(reward_dim)]

    def _print_collection_reward_stats(
        self,
        episodes: List[Dict[str, np.ndarray]],
        step: int,
    ) -> None:
        transition_rewards = [
            np.asarray(episode["reward"][1:], dtype=np.float32)
            for episode in episodes
            if episode["reward"].shape[0] > 1
        ]
        if not transition_rewards:
            print(
                f"[collection rewards] step={step} no transitions collected",
                flush=True,
            )
            return

        rewards = np.concatenate(transition_rewards, axis=0)
        if rewards.ndim == 1:
            rewards = rewards[:, None]

        episode_returns = np.asarray(
            [reward.sum(axis=0) for reward in transition_rewards],
            dtype=np.float32,
        )
        episode_max_rewards = np.asarray(
            [reward.max(axis=0) for reward in transition_rewards],
            dtype=np.float32,
        )
        reward_names = self._collection_reward_names(rewards.shape[-1])

        print(
            f"[collection rewards] step={step} episodes={len(transition_rewards)} "
            f"transitions={rewards.shape[0]}",
            flush=True,
        )
        for idx, name in enumerate(reward_names):
            print(
                f"  {name}: "
                f"mean_step={rewards[:, idx].mean():.4f}, "
                f"mean_return={episode_returns[:, idx].mean():.2f}, "
                f"mean_ep_max={episode_max_rewards[:, idx].mean():.4f}, "
                f"max={rewards[:, idx].max():.4f}, "
                f"positive_frac={(rewards[:, idx] > 0).mean():.4f}",
                flush=True,
            )

    def _eval_task_tilt_score_metrics(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        observations: torch.Tensor,
        timesteps: torch.Tensor,
        tasks: List[str],
        step: int,
    ) -> Dict[str, float]:
        if (
            self.domain_name != "point_mass_maze"
            or not isinstance(agent, FB)
            or agent.tilt is None
            or not hasattr(self, "goal_states")
        ):
            return {}

        init_timesteps = timesteps.to(device=observations.device, dtype=observations.dtype)
        init_weights = torch.pow(agent.tilt.init_geom_ratio, init_timesteps)
        init_weights = init_weights / init_weights.sum()
        obs_idx = torch.multinomial(
            init_weights, num_samples=observations.shape[0], replacement=True
        )
        observations = observations[obs_idx]

        metrics = {}
        score_parts = []
        for task in tasks:
            if task not in self.goal_states:
                continue
            z = agent.infer_z(self.goal_states[task])
            z = torch.as_tensor(
                z,
                dtype=observations.dtype,
                device=observations.device,
            ).unsqueeze(0)
            z = z.expand(observations.shape[0], -1)
            scores, _ = agent.score_and_features(
                observations=observations,
                z=z,
                step=step,
            )
            score_mean = float(scores.mean().detach().cpu())
            score_std = float(scores.std(unbiased=False).detach().cpu())
            metrics[f"tilt/score_mean/{task}"] = score_mean
            metrics[f"tilt/score_std/{task}"] = score_std
            score_parts.append(f"{task}={score_mean:.4f}+/-{score_std:.4f}")

        if score_parts:
            print(
                f"[tilt eval-task scores] step={step} "
                f"geom_ratio={agent.tilt.init_geom_ratio} "
                + ", ".join(score_parts),
                flush=True,
            )

        return metrics

    @torch.no_grad()
    def _log_tilt_health(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        step: int,
    ) -> None:
        """Prints internal tilt diagnostics so `--verbose` runs can confirm the
        latent selector is healthy (well-conditioned Gram, stable feature scale,
        diverse z pool). Only meaningful once tilting is active."""
        if not isinstance(agent, (FB, TDJEPA)) or agent.tilt is None:
            return
        if isinstance(agent, FB):
            tilt_start_step = agent._tilt_start_step  # noqa: SLF001  pylint: disable=protected-access
            tilt_ridge_alpha = agent._tilt_ridge_alpha  # noqa: SLF001  pylint: disable=protected-access
            tilt_ridge_min = agent._tilt_ridge_min  # noqa: SLF001  pylint: disable=protected-access
        else:
            tilt_start_step = agent.cfg.train.tilt_start_step
            tilt_ridge_alpha = agent.cfg.train.tilt_ridge_alpha
            tilt_ridge_min = agent.cfg.train.tilt_ridge_min
        if step < tilt_start_step:
            return

        tilt = agent.tilt
        gram = tilt.gram
        dim = gram.shape[0]

        # --- Gram conditioning + the effective ridge that score_and_features uses ---
        try:
            gram_eig = torch.linalg.eigvalsh(gram)
            eig_min = float(gram_eig[0])
            eig_max = float(gram_eig[-1])
            cond = eig_max / max(eig_min, 1e-12)
        except Exception:  # pylint: disable=broad-except
            eig_min = eig_max = cond = float("nan")
        trace_g = float(torch.trace(gram))
        alpha_lam = tilt_ridge_alpha * trace_g / dim
        lam = max(alpha_lam, tilt_ridge_min)
        ridge_clamped = alpha_lam < tilt_ridge_min

        # --- feature scale (RMS EMA that keeps the Gram from overflowing) ---
        feat_scale = float(tilt.feature_scale())

        # --- z pool diversity: participation ratio (effective rank in [1, D]) and
        #     mean |cosine| between pooled z's. Collapse => PR->1, |cos|->1. ---
        z = tilt.z
        try:
            cov = (z.T @ z) / z.shape[0]
            z_eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
            part_ratio = float((z_eig.sum() ** 2) / (z_eig.pow(2).sum() + 1e-12))
        except Exception:  # pylint: disable=broad-except
            part_ratio = float("nan")
        z_norm = torch.nn.functional.normalize(z, dim=1)
        cos = z_norm @ z_norm.T
        off = ~torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        mean_abs_cos = float(cos[off].abs().mean())

        # --- selection probability spread from the most recent refresh(), vs.
        #     the uniform baseline 1/n_candidates a non-tilted selector would give ---
        prob_min = tilt.last_prob_min
        prob_max = tilt.last_prob_max
        n_candidates = tilt.candidate_multiplier * z.shape[0]
        prob_uniform = 1.0 / n_candidates

        columns = [
            ("step", f"{step}"),
            ("temp", f"{tilt.temperature:.3f}"),
            ("f_scale", f"{feat_scale:.3e}"),
            ("eig_min", f"{eig_min:.3e}"),
            ("eig_max", f"{eig_max:.3e}"),
            ("cond", f"{cond:.2e}"),
            ("trace", f"{trace_g:.3e}"),
            ("lam", f"{lam:.3e}"),
            ("clamp", f"{ridge_clamped}"),
            ("eff_rk", f"{part_ratio:.2f}/{dim}"),
            ("mcos", f"{mean_abs_cos:.3f}"),
            ("p_min", f"{prob_min:.3e}"),
            ("p_max", f"{prob_max:.3e}"),
            ("p_unif", f"{prob_uniform:.3e}"),
        ]
        widths = [max(len(name), len(value)) for name, value in columns]

        if not self._tilt_header_printed:
            header = "  ".join(
                name.rjust(width) for (name, _), width in zip(columns, widths)
            )
            print(f"[tilt health]\n{header}", flush=True)
            self._tilt_header_printed = True

        row = "  ".join(
            value.rjust(width) for (_, value), width in zip(columns, widths)
        )
        print(row, flush=True)

    def _refresh_collection_tilt(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        replay_buffer: Union[OfflineReplayBuffer, FBReplayBuffer],
        step: int,
    ) -> None:
        if not isinstance(agent, FB) or agent.tilt is None:
            return

        previous_temperature = agent.tilt.temperature
        batch = replay_buffer.sample(agent.batch_size)
        try:
            agent.tilt.temperature = self.COLLECTION_TILT_TEMPERATURE
            agent.tilt.refresh(
                init_features=batch.observations,
                init_timesteps=batch.timesteps,
                sample_z=lambda size: agent.sample_z(size=size),
                score_fn=lambda observations, z_candidates: agent.score_and_features(
                    observations=observations,
                    z=z_candidates,
                    step=step,
                ),
            )
        finally:
            agent.tilt.temperature = previous_temperature

    def _sample_training_condition(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        replay_buffer: Union[OfflineReplayBuffer, FBReplayBuffer],
        step: int,
    ) -> Optional[np.ndarray]:
        if isinstance(agent, FB):
            if agent.tilt is not None:
                idx = torch.randint(
                    agent.tilt.z.shape[0],
                    (1,),
                    device=agent.tilt.z.device,
                )
                z = agent.tilt.z[idx[0]]
                return z.detach().cpu().numpy()
            batch = replay_buffer.sample(agent.batch_size)
            z = agent.sample_mixed_z(train_goal=batch.observations)[0]
            return z.detach().cpu().numpy()

        if isinstance(agent, SF):
            batch = replay_buffer.sample(agent.batch_size)
            z = agent.sample_mixed_z(next_observations=batch.next_observations)[0]
            return z.detach().cpu().numpy()

        if isinstance(agent, TDJEPA):
            if agent.agent.tilt is not None:
                idx = torch.randint(
                    agent.agent.tilt.z.shape[0],
                    (1,),
                    device=agent.agent.tilt.z.device,
                )
                z = agent.agent.tilt.z[idx[0]]
                return z.detach().cpu().numpy()
            z = agent.sample_z(size=1)[0]
            return z.detach().cpu().numpy()

        if isinstance(agent, GCIQL):
            batch = replay_buffer.sample(1)
            goal = batch.gciql_goals[0]
            return goal.detach().cpu().numpy()

        return None

    def collect_training_episodes(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        tasks: List[str],
        replay_buffer: Union[OfflineReplayBuffer, FBReplayBuffer],
        step: int,
    ) -> Dict[str, float]:
        logger.info(
            f"Collecting {self.collection_episodes} episode(s) at training step {step}."
        )

        agent.eval()
        if hasattr(agent, "std_dev_schedule") and self.train_std is not None:
            agent.std_dev_schedule = self.train_std
        self._refresh_collection_tilt(
            agent=agent,
            replay_buffer=replay_buffer,
            step=step,
        )
        reward_dim = int(replay_buffer.storage["rewards"].shape[-1])
        episodes = []
        for _ in range(self.collection_episodes):
            condition = self._sample_training_condition(
                agent=agent,
                replay_buffer=replay_buffer,
                step=step,
            )
            episodes.append(
                self._rollout_collection_episode(
                    agent=agent,
                    condition=condition,
                    step=step,
                    reward_dim=reward_dim,
                )
            )

        self._print_collection_reward_stats(episodes=episodes, step=step)
        transitions_added = replay_buffer.add_episodes(episodes)

        if (
            isinstance(agent, (FB, SF, GCIQL, TDJEPA))
            and self.domain_name != "point_mass_maze"
        ):
            (
                self.observations_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps
            )

        agent.train()

        return {
            "collection/episodes": float(len(episodes)),
            "collection/transitions": float(transitions_added),
            "collection/buffer_size": float(len(replay_buffer.storage["observations"])),
        }

    def _maybe_plot_point_mass_trajectories(
        self,
        agent: Union[CQL, FB, CFB, GCIQL, SF, TDJEPA],
        tasks: List[str],
        step: int,
        run_name: str,
        wandb_run=None,
    ) -> None:
        if self.domain_name != "point_mass_maze":
            return

        if isinstance(agent, TDJEPA):
            plot_agent = agent.agent
            agent_kind = "td_jepa"
            agent_label = "TD-JEPA"
        elif isinstance(agent, FB):
            if agent.tilt is None:
                print(
                    "[fb trajectories] skipped: FB tilt/gram is unavailable; "
                    "run with --tilt to enable leverage-colored plots.",
                    flush=True,
                )
                return
            plot_agent = agent
            agent_kind = "fb"
            agent_label = "FB"
        else:
            return

        from plot_td_jepa_trajectories import run_trajectory_plot

        device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        output_dir = Path("trajectory_plots") / run_name
        output_prefix = f"step_{step:08d}"
        plot_path, npz_path, _ = run_trajectory_plot(
            agent=plot_agent,
            output_dir=output_dir,
            rollouts=self.eval_rollouts,
            seed=step,
            device=device,
            tasks=tasks,
            output_prefix=output_prefix,
            title=f"{agent_label} {run_name} step {step} eval trajectories",
            agent_kind=agent_kind,
        )
        print(
            f"[{agent_kind} trajectories] step={step} saved plot={plot_path} "
            f"data={npz_path}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval/{agent_kind}_trajectories": wandb.Image(str(plot_path)),
                    "step": step,
                },
                step=step,
            )


class FinetuningWorkspace(OfflineRLWorkspace):
    """
    Finetunes FB or CFB on one task.
    """

    def __init__(
        self,
        reward_constructor: RewardFunctionConstructor,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        online: bool,
        critic_tuning: bool,
        device: torch.device,
        z_inference_steps: Optional[int] = None,  # FB only
        train_std: Optional[float] = None,  # FB only
        eval_std: Optional[float] = None,  # FB only
    ):
        super().__init__(
            reward_constructor=reward_constructor,
            learning_steps=learning_steps,
            model_dir=model_dir,
            eval_frequency=eval_frequency,
            eval_rollouts=eval_rollouts,
            wandb_logging=wandb_logging,
            device=device,
            z_inference_steps=z_inference_steps,
            train_std=train_std,
            eval_std=eval_std,
        )

        self.online = online
        self.critic_tuning = critic_tuning

    def train(
        self,
        agent: Union[FB, CFB],
        tasks: List[str],
        agent_config: Dict,
        replay_buffer: Union[FBReplayBuffer, OnlineFBReplayBuffer],
        episodes: int = None,
    ) -> None:

        assert len(tasks) == 1

        if self.online:
            self.tune_online(
                agent=agent,
                task=tasks,
                agent_config=agent_config,
                replay_buffer=replay_buffer,
                episodes=episodes,
            )

        else:
            self.tune_offline(
                agent=agent,
                task=tasks,
                agent_config=agent_config,
                replay_buffer=replay_buffer,
            )

    def tune_offline(
        self,
        agent: Union[FB, CFB],
        task: List[str],
        agent_config: Dict,
        replay_buffer: FBReplayBuffer,
    ) -> None:
        """
        Finetunes FB or CFB on one task offline, without online interaction.
        Args:
            agent: agent to finetune
            task: task to finetune on
            agent_config: agent config
            replay_buffer: replay buffer for z sampling
        """

        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name, "finetuning"],
                reinit=True,
                settings=wandb.Settings(console="off", _disable_stats=True, silent=True),
            )
            _configure_wandb_run(run)

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        # get observations and rewards for task inference
        if self.domain_name == "point_mass_maze":
            self.goal_states = {}

            goal_state = point_mass_maze_goals[task[0]]
            self.goal_states[task[0]] = torch.tensor(
                goal_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            (
                self.observations_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps
            )

        best_mean_task_reward = -np.inf

        # get initial eval metrics
        logger.info("Getting init performance.")
        eval_metrics = self.eval(agent=agent, tasks=task)
        init_performance = eval_metrics["eval/task_reward_iqm"]

        logger.info(f"Finetuning {agent.name} on {self.domain_name}-{task[0]}.")

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)

            # infer z for task
            if self.domain_name == "point_mass_maze":
                z = agent.infer_z(self.goal_states[task[0]])
            else:
                z = agent.infer_z(self.observations_z, self.rewards_z[task[0]])

            z_batch = torch.tile(
                torch.as_tensor(z, dtype=torch.float32, device=self.device),
                (agent.batch_size, 1),
            )  # repeat z for batch size

            if self.critic_tuning:
                fb_metrics = agent.update_fb(
                    observations=batch.observations,
                    next_observations=batch.next_observations,
                    actions=batch.actions,
                    discounts=batch.discounts,
                    zs=z_batch,
                    step=i,
                )
                actor_metrics = agent.update_actor(
                    observation=batch.observations, z=z_batch, step=i
                )

                agent.soft_update_params(
                    network=agent.FB.forward_representation,
                    target_network=agent.FB.forward_representation_target,
                    tau=agent._tau,  # pylint: disable=protected-access
                )
                agent.soft_update_params(
                    network=agent.FB.backward_representation,
                    target_network=agent.FB.backward_representation_target,
                    tau=agent._tau,  # pylint: disable=protected-access
                )
                if agent.name in ("VCalFB", "MCalFB"):
                    agent.soft_update_params(
                        network=agent.FB.forward_mu,
                        target_network=agent.FB.forward_mu_target,
                        tau=agent._tau,  # pylint: disable=protected-access
                    )

                train_metrics = {**fb_metrics, **actor_metrics}

            else:
                train_metrics = agent.update_actor(
                    observation=batch.observations, z=z_batch, step=i
                )

            eval_metrics = {}

            if i % self.eval_frequency == 0:
                eval_metrics = self.eval(agent=agent, tasks=task)
                eval_metrics["eval/init_performance"] = init_performance

                if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                    logger.info(
                        f"Finetuned performance:"
                        f"{eval_metrics['eval/task_reward_iqm']:.1f} |"
                        f" Init performance:"
                        f"{eval_metrics['eval/init_performance']:.1f}"
                    )

                    best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                _log_wandb(run, metrics, i)

        if self.wandb_logging:
            # save to wandb_logging
            run.finish()

    def tune_online(
        self,
        agent: Union[FB, CFB],
        task: List[str],
        agent_config: Dict,
        replay_buffer: OnlineFBReplayBuffer,
        episodes: int,
    ) -> None:
        """
        Finetunes FB or CFB on one task using online data.
        Args:
            agent: agent to finetune
            task: task to finetune on
            agent_config: agent config
            replay_buffer: replay buffer for z sampling
            episodes: number of episodes to finetune for
        """

        if self.wandb_logging:
            run = wandb.init(
                config=agent_config,
                tags=[agent.name, "finetuning"],
                reinit=True,
                settings=wandb.Settings(console="off", _disable_stats=True, silent=True),
            )
            _configure_wandb_run(run)

        else:
            date = datetime.today().strftime("Y-%m-%d-%H-%M-%S")
            model_path = self.model_dir / f"local-run-{date}"
            makedirs(str(model_path))

        # get observations and rewards for task inference
        if self.domain_name == "point_mass_maze":
            self.goal_states = {}

            goal_state = point_mass_maze_goals[task[0]]
            self.goal_states[task[0]] = torch.tensor(
                goal_state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
        else:
            (
                self.observations_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps
            )

        # get initial eval metrics
        logger.info("Getting init performance.")
        eval_metrics = self.eval(agent=agent, tasks=task)
        init_performance = eval_metrics["eval/task_reward_iqm"]
        best_mean_task_reward = -np.inf

        logger.info(f"Online finetuning {agent.name} on {self.domain_name}-{task[0]}.")
        j = 0
        for i in tqdm(range(episodes)):

            # interact with env
            timestep = self.env.reset()
            while not timestep.last():

                # infer z for task
                if self.domain_name == "point_mass_maze":
                    z = agent.infer_z(self.goal_states[task[0]])
                else:
                    z = agent.infer_z(self.observations_z, self.rewards_z[task[0]])

                action, _ = agent.act(
                    timestep.observation["observations"],
                    task=z,
                    step=None,
                    sample=True,
                )

                observation = timestep.observation["observations"]
                timestep = self.env.step(action)
                reward = self.reward_functions[task[0]](self.env.physics)
                done = timestep.last()
                j += 1

                replay_buffer.add(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=timestep.observation["observations"],
                    done=done,
                )

                # start learning once batch size is reached
                if j >= agent.batch_size:
                    batch = replay_buffer.sample(agent.batch_size)

                    z_batch = torch.tile(
                        torch.as_tensor(z, dtype=torch.float32, device=self.device),
                        (agent.batch_size, 1),
                    )  # repeat z for batch size

                    if self.critic_tuning:
                        fb_metrics = agent.update_fb(
                            observations=batch.observations,
                            next_observations=batch.next_observations,
                            actions=batch.actions,
                            discounts=batch.discounts,
                            zs=z_batch,
                            step=i,
                        )
                        actor_metrics = agent.update_actor(
                            observation=batch.observations, z=z_batch, step=i
                        )

                        agent.soft_update_params(
                            network=agent.FB.forward_representation,
                            target_network=agent.FB.forward_representation_target,
                            tau=agent._tau,  # pylint: disable=protected-access
                        )
                        agent.soft_update_params(
                            network=agent.FB.backward_representation,
                            target_network=agent.FB.backward_representation_target,
                            tau=agent._tau,  # pylint: disable=protected-access
                        )

                        if agent.name in ("VCalFB", "MCalFB"):
                            agent.soft_update_params(
                                network=agent.FB.forward_mu,
                                target_network=agent.FB.forward_mu_target,
                                tau=agent._tau,  # pylint: disable=protected-access
                            )

                        train_metrics = {**fb_metrics, **actor_metrics}

                    else:
                        train_metrics = agent.update_actor(
                            observation=batch.observations, z=z_batch, step=i
                        )
                else:
                    train_metrics = {}

                if j % self.eval_frequency == 0:
                    eval_metrics = self.eval(agent=agent, tasks=task)
                    eval_metrics["eval/init_performance"] = init_performance

                    if eval_metrics["eval/task_reward_iqm"] > best_mean_task_reward:
                        logger.info(
                            f"Finetuned performance:"
                            f"{eval_metrics['eval/task_reward_iqm']:.1f} |"
                            f" Init performance:"
                            f"{eval_metrics['eval/init_performance']:.1f}"
                        )

                        best_mean_task_reward = eval_metrics["eval/task_reward_iqm"]

                    agent.train()
                else:
                    eval_metrics = {}

                metrics = {**train_metrics, **eval_metrics}

                if self.wandb_logging:
                    _log_wandb(run, metrics, j)

        if self.wandb_logging:
            # save to wandb_logging
            run.finish()


class D4RLWorkspace:
    """
    Workspace for training agents on D4RL tasks.
    """

    def __init__(
        self,
        env,
        domain_name: str,
        learning_steps: int,
        model_dir: Path,
        eval_frequency: int,
        eval_rollouts: int,
        wandb_logging: bool,
        device: torch.device,
        wandb_project: str,
        wandb_entity: str,
        z_inference_steps: Optional[int] = None,  # FB only
    ):
        self.env = env
        self.domain_name = domain_name
        self.learning_steps = learning_steps
        self.model_dir = model_dir
        self.eval_frequency = eval_frequency
        self.eval_rollouts = eval_rollouts
        self.wandb_logging = wandb_logging
        self.device = device
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.z_inference_steps = z_inference_steps
        self.ref_max_score = {
            "walker": 4592.3,
            "cheetah": 12135.0,
        }
        self.ref_min_score = {
            "cheetah": -280.178953,
            "walker": 1.629008,
        }

    def train(
        self,
        agent: Union[FB, CFB, SF, TDJEPA],
        agent_config: Dict,
        replay_buffer: D4RLReplayBuffer,
    ) -> None:

        if self.wandb_logging:
            run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project,
                config=agent_config,
                tags=[agent.name, "D4RL"],
                reinit=True,
                settings=wandb.Settings(console="off", _disable_stats=True, silent=True),
            )
            _configure_wandb_run(run)

        logger.info(f"Training {agent.name}.")
        best_mean_task_reward = -np.inf

        # sample set transitions for z inference
        if isinstance(agent, (FB, SF, TDJEPA)):
            (
                self.goals_z,
                self.rewards_z,
            ) = replay_buffer.sample_task_inference_transitions(
                inference_steps=self.z_inference_steps,
            )

        for i in tqdm(range(self.learning_steps + 1)):

            batch = replay_buffer.sample(agent.batch_size)
            train_metrics = agent.update(batch=batch, step=i)

            eval_metrics = {}

            if i % self.eval_frequency == 0:
                eval_metrics = self.eval(agent=agent)

                if eval_metrics["eval/score"] > best_mean_task_reward:
                    new_best_mean_task_reward = eval_metrics["eval/score"]
                    logger.info(
                        f"New max IQM task reward: {best_mean_task_reward:.3f} -> "
                        f"{new_best_mean_task_reward:.3f}."
                    )

                    best_mean_task_reward = new_best_mean_task_reward

                agent.train()

            metrics = {**train_metrics, **eval_metrics}

            if self.wandb_logging:
                _log_wandb(run, metrics, i)

        if self.wandb_logging:
            run.finish()

    def eval(self, agent: Union[FB, CFB, SF, TDJEPA]):
        """
        Evals agent.
        """

        logger.info(f"Evaluating {agent.name}.")

        if isinstance(agent, (FB, SF, TDJEPA)):
            z = agent.infer_z(self.goals_z, self.rewards_z)

        eval_rewards = np.zeros(self.eval_rollouts)

        for i in tqdm(range(self.eval_rollouts), desc="eval rollouts"):

            observation = self.env.reset()
            terminated = False
            rollout_reward = 0.0

            while not terminated:
                if isinstance(agent, (FB, SF, TDJEPA)):
                    action, _ = agent.act(
                        observation=observation, task=z, sample=False, step=None
                    )
                else:
                    action = agent.act(observation=observation, sample=False, step=None)
                observation, reward, terminated, _ = self.env.step(action)
                rollout_reward += reward

            eval_rewards[i] = rollout_reward

        eval_rewards = self._get_normalised_score(eval_rewards)
        metrics = {"eval/score": float(stats.trim_mean(eval_rewards, 0.25))}

        return metrics

    def _get_normalised_score(self, score: np.ndarray):
        return (
            (score - self.ref_min_score[self.domain_name])
            / (
                self.ref_max_score[self.domain_name]
                - self.ref_min_score[self.domain_name]
            )
            * 100
        )
