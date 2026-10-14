# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MUJOCO_GL"] = "egl"  # for headless rendering

import torch

torch.set_float32_matmul_precision("high")
torch._inductor.config.autotune_local_cache = False

import json
import time
import typing as tp
from pathlib import Path
from typing import Dict, List

import gymnasium
import numpy as np
import pydantic
import torch
import tyro
import wandb

from ogbench_impl.agents import Agent
from ogbench_impl.base import BaseConfig
from ogbench_impl.data_loading.ogbench import OGBenchDataConfig
from ogbench_impl.envs.ogbench import OGBenchEnvConfig
from ogbench_impl.evaluations.ogbench import OGBenchRewardEvalConfig
from ogbench_impl.misc.loggers import CSVLogger
from ogbench_impl.utils import EveryNStepsChecker, get_local_workdir, set_seed_everywhere

TRAIN_LOG_FILENAME = "train_log.txt"

CHECKPOINT_DIR_NAME = "checkpoint"


Env = OGBenchEnvConfig
DataLoading = OGBenchDataConfig

# Stackoverflow #70914419
Evaluation = tp.Annotated[
    OGBenchRewardEvalConfig,
    pydantic.Field(discriminator="name"),
]


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: Agent = pydantic.Field(discriminator="name")

    env: Env = pydantic.Field(discriminator="name")
    data: DataLoading = pydantic.Field(discriminator="name")
    relabel_dataset: bool = False

    work_dir: str = pydantic.Field(default_factory=lambda: get_local_workdir("train_ogbench"))

    seed: int = 0
    log_every_updates: int = 10_000
    num_train_steps: int = 3_000_000
    checkpoint_every_steps: int = 250_000

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None

    # misc
    buffer_device: str | None = None  # if None, use the agent's device

    # eval
    # If you want to add more available evaluations, Update "Evaluations" type above
    evaluations: Dict[str, Evaluation] | List[Evaluation] = pydantic.Field(default_factory=lambda: [])

    eval_every_steps: int = 250_000

    tags: dict = pydantic.Field(default_factory=lambda: {})
    tilt_resume_overrides: dict = pydantic.Field(default_factory=dict)
    verbose: bool = False

    def model_post_init(self, context):
        if self.relabel_dataset:
            if not isinstance(self.env, OGBenchEnvConfig):
                raise ValueError("Relabeling is only supported for DMC and OGBench environments")

    def build(self):
        return Workspace(self)


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    checkpoint_time = 0
    if checkpoint_dir.exists():
        # read train status
        with (checkpoint_dir / "train_status.json").open("r") as f:
            train_status = json.load(f)
        checkpoint_time = train_status["time"]
        if checkpoint_time > cfg.num_train_steps:
            raise ValueError(
                f"Checkpoint is at step {checkpoint_time}, beyond requested "
                f"num_train_steps={cfg.num_train_steps}. Increase the limit or use another work directory."
            )

        print(f"Loading the agent at time {checkpoint_time}")
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
        updates = {**cfg.tilt_resume_overrides, "learning_steps": cfg.num_train_steps}
        # Validate CLI overrides against the restored agent config.
        train_cfg = type(agent.cfg.train)(**{**agent.cfg.train.model_dump(), **updates})
        reset_tilt = agent.cfg.train.tilting_by_z != train_cfg.tilting_by_z
        agent.cfg = agent.cfg.model_copy(update={"train": train_cfg})
        if not train_cfg.tilt:
            agent.tilt = None
        elif agent.tilt is None or reset_tilt:
            agent.setup_tilt()
        else:
            for field, key in (("beta", "tilt_beta"), ("candidate_multiplier", "tilt_candidate_multiplier"),
                               ("init_geom_ratio", "tilt_init_geom_ratio"), ("uniform_mix", "tilt_uniform_mix"),
                               ("linear", "tilt_linear")):
                setattr(agent.tilt, field, getattr(train_cfg, key))
        if "lr_actor" in cfg.tilt_resume_overrides:
            for group in agent.actor_optimizer.param_groups:
                group["lr"] = train_cfg.lr_actor
        cfg = cfg.model_copy(update={"agent": agent.cfg})
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, cfg, checkpoint_time


def init_wandb(cfg: TrainConfig):
    exp_name = "ogbench-offline"
    wandb_name = exp_name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=wandb_name, config=wandb_config, dir="./_wandb")


class Workspace:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg

        sample_env, _ = cfg.env.build()
        self.obs_space = sample_env.observation_space
        assert isinstance(self.obs_space, gymnasium.spaces.Box), "Only Box observation spaces are supported"

        self.action_space = sample_env.action_space
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported"
        self.action_dim = self.action_space.shape[0]
        sample_env.close()

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME)

        set_seed_everywhere(self.cfg.seed)

        self.agent, self.cfg, self._checkpoint_time = create_agent_or_load_checkpoint(
            self.work_dir,
            self.cfg,
            agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim),
        )
        self.agent._model.train()

        if isinstance(self.cfg.evaluations, list):
            self.evaluations = {eval_cfg.name_in_logs: eval_cfg.build() for eval_cfg in self.cfg.evaluations}
        elif isinstance(self.cfg.evaluations, dict):
            self.evaluations = {name: eval_cfg.build() for name, eval_cfg in self.cfg.evaluations.items()}
        self.evaluate = len(self.evaluations) > 0
        self.eval_loggers = {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name, eval_cfg in self.evaluations.items()}
        self.eval_summary_loggers = {
            name: CSVLogger(filename=self.work_dir / f"{name}_summary.csv")
            for name in self.evaluations
        }

        if self.cfg.use_wandb:
            init_wandb(self.cfg)

        with (self.work_dir / "config.json").open("w") as f:
            f.write(self.cfg.model_dump_json(indent=4))

    def train(self):
        self.start_time = time.time()
        self.train_offline()

    def train_offline(self) -> None:
        buffer_device = self.agent.device if self.cfg.buffer_device is None else self.cfg.buffer_device
        relabel_fn = self.cfg.env.get_relabel_fn(self.cfg.env.task) if self.cfg.relabel_dataset else None
        replay_buffer = self.cfg.data.build(buffer_device, self.cfg.agent.train.batch_size, self.cfg.env.frame_stack, relabel_fn)
        print(replay_buffer["train"])

        total_metrics = None
        fps_start_time = time.time()
        checkpoint_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.checkpoint_every_steps)
        eval_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.eval_every_steps)
        log_time_checker = EveryNStepsChecker(self._checkpoint_time, self.cfg.log_every_updates)

        for t in range(self._checkpoint_time, int(self.cfg.num_train_steps)):
            if (t != self._checkpoint_time) and checkpoint_time_checker.check(t):
                checkpoint_time_checker.update_last_step(t)
                self.save(t, replay_buffer)

            if self.evaluate and eval_time_checker.check(t):
                eval_time_checker.update_last_step(t)
                self.eval(t, replay_buffer=replay_buffer)

            metrics = self.agent.update(replay_buffer, t)

            # we need to copy tensors returned by a cudagraph module
            if total_metrics is None:
                total_metrics = {k: metrics[k].clone() for k in metrics.keys()}
            else:
                total_metrics = {k: total_metrics[k] + metrics[k] for k in metrics.keys()}

            if log_time_checker.check(t):
                log_time_checker.update_last_step(t)
                m_dict = {}
                for k in sorted(list(total_metrics.keys())):
                    tmp = total_metrics[k] / (1 if t == 0 else self.cfg.log_every_updates)
                    m_dict[k] = np.round(tmp.mean().item(), 6)
                m_dict["duration"] = time.time() - self.start_time
                m_dict["step"] = t
                m_dict["FPS"] = (1 if t == 0 else self.cfg.log_every_updates) / (time.time() - fps_start_time)
                if self.cfg.use_wandb:
                    wandb.log(
                        {f"train/{k}": v for k, v in m_dict.items()},
                        step=t,
                    )
                print(m_dict)
                self.train_logger.log(m_dict)
                if self.cfg.verbose:
                    self.agent.log_tilt_health(t)
                total_metrics = None
                fps_start_time = time.time()
        self.save(int(self.cfg.num_train_steps), replay_buffer)
        if self.cfg.use_wandb:
            wandb.finish()
        return

    def eval(self, t, replay_buffer):
        print(f"Starting evaluation at time {t}")
        evaluation_results = {}

        self.agent._model.train(False)

        # This will contain the results, mapping evaluation.cfg.name --> dict of metrics
        evaluation_results = {}
        for evaluation_name in self.evaluations:
            evaluation = self.evaluations[evaluation_name]
            logger = self.eval_loggers[evaluation_name]

            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=t,
                agent_or_model=self.agent,
                replay_buffer=replay_buffer,
                logger=logger,
                summary_logger=self.eval_summary_loggers[evaluation_name],
            )
            # For wandb dict, put it on wandb
            if self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        self.agent._model.train()

        return evaluation_results

    def save(self, time: int, replay_buffer: Dict[str, tp.Any]) -> None:
        print(f"Checkpointing at time {time}")
        self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))
        with (self.work_dir / CHECKPOINT_DIR_NAME / "train_status.json").open("w+") as f:
            json.dump({"time": time}, f, indent=4)


if __name__ == "__main__":
    # This is the bare minimum CLI interface to launch experiments, but ideally you should
    # launch your experiments from Python code (e.g., see under "scripts")
    workspace = tyro.cli(Workspace)
    workspace.train()
