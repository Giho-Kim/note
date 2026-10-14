# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import typing as tp

import numpy as np
import pydantic
import torch
import tqdm
from scipy.stats import trim_mean

from ogbench_impl.base import BaseConfig
from ogbench_impl.envs.ogbench import OGBenchEnvConfig
from ogbench_impl.envs.utils.rollout import rollout
from ogbench_impl.evaluations.base import extract_model
from ogbench_impl.nn_models import eval_mode


def summarize_returns(rewards, successes):
    rewards = np.asarray(rewards, dtype=np.float64)
    return {
        "reward_mean": float(rewards.mean()),
        "reward_iqm": float(trim_mean(rewards, proportiontocut=0.25)),
        "reward_std": float(rewards.std()),
        "success_mean": float(np.mean(successes)),
        "num_episodes": len(rewards),
    }


class OGBenchRewardEvalConfig(BaseConfig):
    name: tp.Literal["ogbench_reward_eval"] = "ogbench_reward_eval"
    name_in_logs: str = "reward"
    env: OGBenchEnvConfig

    tasks: list[str] = pydantic.Field(default_factory=lambda: [])
    num_episodes: int = 100

    num_inference_samples: int = 50_000
    shift_reward: float = 0.0
    disable_tqdm: bool = True

    def build(self):
        return OGBenchRewardEvaluation(self)


class OGBenchRewardEvaluation:
    def __init__(self, config: OGBenchRewardEvalConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, replay_buffer, logger, summary_logger=None, **kwargs):
        wandb_dict = {}
        eval_metrics = {}
        summaries = {}
        model = extract_model(agent_or_model)

        pbar = tqdm.tqdm(self.cfg.tasks, leave=False, disable=self.cfg.disable_tqdm)
        for task in pbar:
            pbar.set_description(f"task {task}")
            eval_env, _ = self.cfg.env.model_copy(update={"task": task}).build()
            pbar.set_description(f"task {task} (inference)")
            ctx, relabel_metrics = self._reward_inference(agent_or_model, task, replay_buffer)
            print(relabel_metrics)
            pbar.set_description(f"task {task} (rollout)")
            ctx = [None] if ctx is None else ctx
            with torch.no_grad(), eval_mode(model):
                st, infos = rollout(
                    eval_env,
                    agent=model,
                    num_episodes=self.cfg.num_episodes,
                    ctx=ctx,
                )  # return statistics and episodes
            st["success"] = [any([step.get("success", False) for step in info]) for info in infos]
            print(task, {k: np.mean(v) for k, v in st.items()})
            eval_metrics[task] = st
            summary = summarize_returns(st["reward"], st["success"])
            summaries[task] = summary
            print(task, summary)
            wandb_dict[f"{task}/reward"] = np.mean(st["reward"])
            wandb_dict[f"{task}/reward_mean"] = summary["reward_mean"]
            wandb_dict[f"{task}/reward_iqm"] = summary["reward_iqm"]
            wandb_dict[f"{task}/success"] = np.mean(st["success"])
            wandb_dict[f"{task}/reward#std"] = np.std(st["reward"])
            for k, v in relabel_metrics.items():
                wandb_dict[f"{task}/{k}"] = v
            eval_env.close()
            del eval_env

        rewards = np.concatenate([el["reward"] for el in eval_metrics.values()])
        wandb_dict["eval/reward"] = np.mean(rewards)
        wandb_dict["eval/reward#std"] = np.std(rewards)
        successes = np.concatenate([el["success"] for el in eval_metrics.values()])
        wandb_dict["eval/success"] = np.mean(successes)
        overall = summarize_returns(rewards, successes)
        summaries["overall"] = overall
        wandb_dict["eval/reward_mean"] = overall["reward_mean"]
        wandb_dict["eval/reward_iqm"] = overall["reward_iqm"]
        print(f"[eval summary] step={timestep}", overall)
        if summary_logger is not None:
            for task, summary in summaries.items():
                summary_logger.log({"task": task, "timestep": timestep, **summary})

        # log reward results
        if logger is not None:
            for k, v in eval_metrics.items():
                # task and timestamp needs to be repeated length of metrics times so the logger accepts it
                random_key = list(v.keys())[0]
                n = len(v[random_key])
                v["task"] = [k] * n
                v["timestep"] = [timestep] * n
                v["reward_mean"] = [summaries[k]["reward_mean"]] * n
                v["reward_iqm"] = [summaries[k]["reward_iqm"]] * n
                logger.log(v)

        return eval_metrics, wandb_dict

    def _reward_inference(self, agent_or_model, task, replay_buffer) -> torch.Tensor:
        model = extract_model(agent_or_model)
        if not hasattr(model, "reward_inference"):
            return torch.zeros((1, 1)), {}
        num_samples = self.cfg.num_inference_samples
        batch = replay_buffer["train"].sample(num_samples)
        rewards = self.cfg.env.get_relabel_fn(task)(batch["next"]["physics"].cpu().numpy(), batch["action"].cpu().numpy())
        rewards += self.cfg.shift_reward

        non_zero_rewards = np.count_nonzero(rewards.ravel())
        relabel_metrics = {
            "relabel_reward#mean": np.mean(rewards),
            "relabel_reward#nonzero": non_zero_rewards,
            "relabel_reward#zero": rewards.size - non_zero_rewards,
            "relabel_reward#num_samples": rewards.size,
        }
        z = agent_or_model._model.reward_inference(
            next_obs=batch["next"]["observation"].to(agent_or_model.device),
            reward=torch.tensor(rewards, dtype=torch.float32, device=agent_or_model.device),
        )
        if getattr(agent_or_model, "tilt", None) is not None:
            with torch.no_grad(), eval_mode(model):
                relabel_metrics.update(agent_or_model.eval_task_tilt_scores(batch, z))
        return z.reshape(1, -1), relabel_metrics
