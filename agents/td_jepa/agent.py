"""ZSRL wrapper for the vendored TD-JEPA agent."""
from pathlib import Path
from typing import Dict, Optional, Tuple

import gymnasium
import numpy as np
import torch

from agents.base import AbstractAgent, Batch
from metamotivo.agents.td_jepa.agent import (
    TDJEPAAgent as MetaTDJEPAAgent,
    TDJEPAAgentConfig,
    TDJEPAAgentTrainConfig,
)
from metamotivo.agents.td_jepa.model import TDJEPAModelConfig


class _SingleBatchReplayBuffer:
    """Adapter that exposes one ZSRL batch via the MetaMotivo replay API."""

    def __init__(self, batch: Batch):
        not_dones = batch.not_dones.bool()
        if not_dones.ndim == 1:
            not_dones = not_dones.unsqueeze(-1)
        elif not_dones.shape[-1] != 1:
            not_dones = not_dones.all(dim=-1, keepdim=True)
        terminated = ~not_dones
        self._batch = {
            "observation": batch.observations,
            "action": batch.actions,
            "reward": batch.rewards,
            "next": {
                "observation": batch.next_observations,
                "terminated": terminated,
            },
        }

    def __getitem__(self, key: str) -> "_SingleBatchReplayBuffer":
        if key != "train":
            raise KeyError(key)
        return self

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        if self._batch["observation"].shape[0] != batch_size:
            raise ValueError(
                f"TD-JEPA expects a batch of size {batch_size}, got {self._batch['observation'].shape[0]}."
            )
        return self._batch


class TDJEPA(AbstractAgent):
    """TD-JEPA agent exposed through the same interface as the other ZSRL agents."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        device: torch.device,
        name: str,
        batch_size: int,
        discount: float,
        lr_predictor: float,
        lr_phi: float,
        lr_psi: float,
        lr_actor: float,
        weight_decay: float,
        encoder_target_tau: float,
        predictor_target_tau: float,
        phi_ortho_coef: float,
        psi_ortho_coef: float,
        train_goal_ratio: float,
        predictor_pessimism_penalty: float,
        actor_pessimism_penalty: float,
        stddev_clip: float,
        bc_coeff: float,
        log_eigvals: bool,
        scale_train_goals: bool,
        learning_steps: int,
        tilt: bool,
        tilting_by_z: bool,
        tilt_beta: float,
        tilt_temperature: float,
        tilt_temperature_start: float,
        tilt_temperature_end: float,
        tilt_candidate_multiplier: int,
        tilt_init_geom_ratio: float,
        tilt_ridge_alpha: float,
        tilt_ridge_min: float,
        tilt_start_step: int,
        actor_std: float,
        actor_use_full_encoder: bool,
        symmetric: bool,
        compile: bool,
        phi_dim: int,
        psi_dim: int,
        norm_z: bool,
        rgb_encoder_name: str,
        augmentator_name: str,
        phi_predictor_hidden_dim: int,
        phi_predictor_hidden_layers: int,
        phi_predictor_embedding_layers: int,
        phi_predictor_num_parallel: int,
        psi_predictor_hidden_dim: int,
        psi_predictor_hidden_layers: int,
        psi_predictor_embedding_layers: int,
        psi_predictor_num_parallel: int,
        phi_mlp_hidden_dim: int,
        phi_mlp_hidden_layers: int,
        phi_mlp_norm: bool,
        psi_mlp_hidden_dim: int,
        psi_mlp_hidden_layers: int,
        psi_mlp_norm: bool,
        actor_hidden_dim: int,
        actor_hidden_layers: int,
        actor_embedding_layers: int,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )
        self.batch_size = batch_size
        self.device = device
        self._z_dimension = phi_dim if symmetric else psi_dim
        model_device = device.type if device.type in {"cpu", "cuda"} else "cpu"
        self._obs_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_length,),
            dtype=np.float32,
        )

        model_cfg = TDJEPAModelConfig(
            device=model_device,
            actor_std=actor_std,
            actor_use_full_encoder=actor_use_full_encoder,
            symmetric=symmetric,
            archi={
                "phi_dim": phi_dim,
                "psi_dim": psi_dim,
                "norm_z": norm_z,
                "rgb_encoder": {"name": rgb_encoder_name},
                "augmentator": {"name": augmentator_name},
                "phi_predictor": {
                    "name": "ForwardArchi",
                    "hidden_dim": phi_predictor_hidden_dim,
                    "hidden_layers": phi_predictor_hidden_layers,
                    "embedding_layers": phi_predictor_embedding_layers,
                    "num_parallel": phi_predictor_num_parallel,
                },
                "psi_predictor": {
                    "name": "ForwardArchi",
                    "hidden_dim": psi_predictor_hidden_dim,
                    "hidden_layers": psi_predictor_hidden_layers,
                    "embedding_layers": psi_predictor_embedding_layers,
                    "num_parallel": psi_predictor_num_parallel,
                },
                "phi_mlp_encoder": {
                    "name": "BackwardArchi",
                    "hidden_dim": phi_mlp_hidden_dim,
                    "hidden_layers": phi_mlp_hidden_layers,
                    "norm": phi_mlp_norm,
                },
                "psi_mlp_encoder": {
                    "name": "BackwardArchi",
                    "hidden_dim": psi_mlp_hidden_dim,
                    "hidden_layers": psi_mlp_hidden_layers,
                    "norm": psi_mlp_norm,
                },
                "actor": {
                    "name": "simple",
                    "hidden_dim": actor_hidden_dim,
                    "hidden_layers": actor_hidden_layers,
                    "embedding_layers": actor_embedding_layers,
                },
            },
        )
        train_cfg = TDJEPAAgentTrainConfig(
            lr_predictor=lr_predictor,
            lr_phi=lr_phi,
            lr_psi=lr_psi,
            lr_actor=lr_actor,
            weight_decay=weight_decay,
            encoder_target_tau=encoder_target_tau,
            predictor_target_tau=predictor_target_tau,
            phi_ortho_coef=phi_ortho_coef,
            psi_ortho_coef=psi_ortho_coef,
            train_goal_ratio=train_goal_ratio,
            predictor_pessimism_penalty=predictor_pessimism_penalty,
            actor_pessimism_penalty=actor_pessimism_penalty,
            stddev_clip=stddev_clip,
            batch_size=batch_size,
            discount=discount,
            bc_coeff=bc_coeff,
            log_eigvals=log_eigvals,
            scale_train_goals=scale_train_goals,
            tilt=tilt,
            tilting_by_z=tilting_by_z,
            learning_steps=learning_steps,
            tilt_beta=tilt_beta,
            tilt_temperature=tilt_temperature,
            tilt_temperature_start=tilt_temperature_start,
            tilt_temperature_end=tilt_temperature_end,
            tilt_candidate_multiplier=tilt_candidate_multiplier,
            tilt_init_geom_ratio=tilt_init_geom_ratio,
            tilt_ridge_alpha=tilt_ridge_alpha,
            tilt_ridge_min=tilt_ridge_min,
            tilt_start_step=tilt_start_step,
        )
        cfg = TDJEPAAgentConfig(model=model_cfg, train=train_cfg, compile=compile)
        self.agent = MetaTDJEPAAgent(obs_space=self._obs_space, action_dim=action_length, cfg=cfg)

    def act(
        self,
        observation: np.ndarray,
        task: np.ndarray,
        step: Optional[int],
        sample: bool = False,
    ) -> Tuple[np.ndarray, None]:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        z = torch.as_tensor(task, dtype=torch.float32, device=self.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        action = self.agent.act(obs=obs, z=z, mean=not sample)
        return action.detach().cpu().numpy()[0], None

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        init_obs = batch.observations.detach().cpu().numpy()
        init_steps = None if batch.timesteps is None else batch.timesteps.detach().cpu().numpy()
        if self.agent._tilt_active(step):  # pylint: disable=protected-access
            progress = min(max(step, 0) / self.agent.cfg.train.learning_steps, 1.0)
            self.agent.tilt.temperature = self.agent.cfg.train.tilt_temperature_start + progress * (
                self.agent.cfg.train.tilt_temperature_end - self.agent.cfg.train.tilt_temperature_start
            )
        metrics = self.agent.update(
            _SingleBatchReplayBuffer(batch),
            step=step,
            init_obs=init_obs,
            init_steps=init_steps,
        )
        return {key: float(value.detach().cpu()) for key, value in metrics.items()}

    def sample_z(self, size: int) -> torch.Tensor:
        return self.agent._model.sample_z(size=size, device=self.device)

    def infer_z(
        self, observations: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        with torch.no_grad():
            obs = observations.to(self.device)
            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            if rewards is None:
                z = self.agent._model.project_z(self.agent._model.psi(obs))
            else:
                z = self.agent._model.reward_inference(obs, rewards.to(self.device))
        return z.squeeze(0).detach().cpu().numpy()

    def save(self, dir_path: Path) -> Path:
        output_path = dir_path / str(self.name)
        self.agent.save(str(output_path))
        return output_path

    def load(self, filepath: Path):
        load_device = self.device.type if self.device.type in {"cpu", "cuda"} else "cpu"
        loaded = MetaTDJEPAAgent.load(str(filepath), device=load_device)
        self.agent = loaded
        return self
