"""Module defining the Forward-Backward Agent."""

import logging
import math
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import numpy as np

logger = logging.getLogger(__name__)

from agents.fb.models import ForwardBackwardRepresentation, ActorModel
from agents.base import AbstractAgent, Batch, AbstractGaussianActor
from agents.utils import schedule
from metamotivo.agents.tilt import TiltLatentSelector


class FB(AbstractAgent):
    """Forward-Backward Agent."""

    def __init__(
        self,
        observation_length: int,
        action_length: int,
        preprocessor_hidden_dimension: int,
        preprocessor_output_dimension: int,
        preprocessor_hidden_layers: int,
        preprocessor_activation: str,
        z_dimension: int,
        forward_hidden_dimension: int,
        forward_hidden_layers: int,
        forward_number_of_features: int,
        backward_hidden_dimension: int,
        backward_hidden_layers: int,
        actor_hidden_dimension: int,
        actor_hidden_layers: int,
        forward_activation: str,
        backward_activation: str,
        actor_activation: str,
        actor_learning_rate: float,
        critic_learning_rate: float,
        learning_rate_coefficient: float,
        orthonormalisation_coefficient: float,
        discount: float,
        batch_size: int,
        z_mix_ratio: float,
        gaussian_actor: bool,
        std_dev_clip: float,
        std_dev_schedule: str,
        tau: float,
        learning_steps: int,
        tilt: bool,
        tilt_start_step: int,
        tilting_by_z: bool,
        tilt_beta: float,
        tilt_temperature: float,
        tilt_temperature_start: float,
        tilt_temperature_end: float,
        tilt_candidate_multiplier: int,
        tilt_init_geom_ratio: float,
        tilt_ridge_alpha: float,
        tilt_ridge_min: float,
        device: torch.device,
        name: str,
        tilt_goal: bool = False,
    ):
        super().__init__(
            observation_length=observation_length,
            action_length=action_length,
            name=name,
        )

        self.FB = ForwardBackwardRepresentation(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            number_of_features=forward_number_of_features,
            z_dimension=z_dimension,
            forward_hidden_dimension=forward_hidden_dimension,
            forward_hidden_layers=forward_hidden_layers,
            backward_hidden_dimension=backward_hidden_dimension,
            backward_hidden_layers=backward_hidden_layers,
            forward_activation=forward_activation,
            backward_activation=backward_activation,
            orthonormalisation_coefficient=orthonormalisation_coefficient,
            discount=discount,
            device=device,
        )

        self.actor = ActorModel(
            observation_length=observation_length,
            action_length=action_length,
            preprocessor_hidden_dimension=preprocessor_hidden_dimension,
            preprocessor_feature_space_dimension=preprocessor_output_dimension,
            preprocessor_hidden_layers=preprocessor_hidden_layers,
            preprocessor_activation=preprocessor_activation,
            z_dimension=z_dimension,
            number_of_features=forward_number_of_features,
            actor_hidden_dimension=actor_hidden_dimension,
            actor_hidden_layers=actor_hidden_layers,
            actor_activation=actor_activation,
            gaussian_actor=gaussian_actor,
            std_dev_clip=std_dev_clip,
            device=device,
        )

        self.encoder = torch.nn.Identity()
        self.augmentation = torch.nn.Identity()

        # load weights into target networks
        self.FB.forward_representation_target.load_state_dict(
            self.FB.forward_representation.state_dict()
        )
        self.FB.backward_representation_target.load_state_dict(
            self.FB.backward_representation.state_dict()
        )

        # optimisers
        self.FB_optimizer = torch.optim.Adam(
            [
                {"params": self.FB.forward_representation.parameters()},
                {
                    "params": self.FB.backward_representation.parameters(),
                    "lr": critic_learning_rate * learning_rate_coefficient,
                },
            ],
            lr=critic_learning_rate,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )

        self._device = device
        self.batch_size = batch_size
        self._z_mix_ratio = z_mix_ratio
        self._tau = tau
        self._z_dimension = z_dimension
        self._learning_steps = max(1, learning_steps)
        self._tilt_temperature_start = tilt_temperature_start
        self._tilt_temperature_end = tilt_temperature_end
        self._tilting_by_z = tilting_by_z
        self._tilt_ridge_alpha = tilt_ridge_alpha
        self._tilt_ridge_min = tilt_ridge_min
        self._tilt_start_step = tilt_start_step
        self._tilt_goal = tilt_goal
        self.std_dev_schedule = std_dev_schedule
        self.tilt = None
        if tilt:
            self.tilt = TiltLatentSelector(
                z=self.sample_z(size=self.batch_size),
                beta=tilt_beta,
                temperature=tilt_temperature,
                candidate_multiplier=tilt_candidate_multiplier,
                init_geom_ratio=tilt_init_geom_ratio,
            )

    def _tilt_temperature(self, step: int) -> float:
        progress = min(max(step, 0) / self._learning_steps, 1.0)
        return self._tilt_temperature_start + progress * (
            self._tilt_temperature_end - self._tilt_temperature_start
        )

    @torch.no_grad()
    def act(
        self,
        observation: Dict[str, np.ndarray],
        task: np.array,
        step: int,
        sample: bool = False,
    ) -> Tuple[np.array, float]:
        """
        Used at test time to perform zero-shot rollouts.
        Takes observation array from environment, encodes, and selects
        action from actor.
        Args:
            observation: observation array of shape [observation_length]
            task: task array of shape [z_dimension]
            step: current step in env
            sample: whether to sample action from actor distribution
        Returns:
            action: action array of shape [action_length]
            std_dev: current actor standard deviation
        """

        observation = torch.as_tensor(
            observation, dtype=torch.float32, device=self._device
        ).unsqueeze(0)
        h = self.encoder(observation)
        z = torch.as_tensor(task, dtype=torch.float32, device=self._device).unsqueeze(0)

        # get action from actor
        std_dev = schedule(self.std_dev_schedule, step)
        action, _ = self.actor(h, z, std_dev, sample=sample)

        return action.detach().cpu().numpy()[0], std_dev

    def update(self, batch: Batch, step: int) -> Dict[str, float]:
        """
        Updates agent's networks given a batch_size sample from the replay buffer.
        Args:
            batch: memory buffer containing transitions
            step: no. of steps taken in the environment
        Returns:
            metrics: dictionary of metrics for logging
        """

        perm = torch.randperm(self.batch_size)
        backward_input = batch.observations[perm]
        if self.tilt is not None and step >= self._tilt_start_step:
            self.tilt.temperature = self._tilt_temperature(step)
            self.tilt.refresh(
                init_features=batch.observations,
                init_timesteps=batch.timesteps,
                sample_z=lambda size: self.sample_z(size=size),
                score_fn=lambda observations, z_candidates: self.score_and_features(
                    observations=observations,
                    z=z_candidates,
                    step=step,
                ),
            )
        zs = self.sample_mixed_z(
            train_goal=backward_input,
            init_observations=batch.observations,
            init_timesteps=batch.timesteps,
            step=step,
        )
        actor_zs = zs.clone().requires_grad_(True)
        actor_observations = batch.observations.clone().requires_grad_(True)

        # update forward and backward models
        fb_metrics = self.update_fb(
            observations=batch.observations,
            next_observations=batch.next_observations,
            actions=batch.actions,
            discounts=batch.discounts,
            zs=zs,
            step=step,
        )

        # update actor
        actor_metrics = self.update_actor(
            observation=actor_observations, z=actor_zs, step=step
        )

        # update target networks for forwards and backwards models
        self.soft_update_params(
            network=self.FB.forward_representation,
            target_network=self.FB.forward_representation_target,
            tau=self._tau,
        )
        self.soft_update_params(
            network=self.FB.backward_representation,
            target_network=self.FB.backward_representation_target,
            tau=self._tau,
        )

        metrics = {
            **fb_metrics,
            **actor_metrics,
        }

        return metrics

    def _tilt_active(self, step: int) -> bool:
        return self.tilt is not None and step >= self._tilt_start_step

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: Optional[torch.Tensor] = None,
        init_observations: Optional[torch.Tensor] = None,
        init_timesteps: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        if not self._tilt_active(step):
            zs = self.sample_z(size=self.batch_size)
        else:
            zs = self.tilt.z.clone()

        if train_goal is not None:
            mix_indices = np.where(np.random.rand(self.batch_size) < self._z_mix_ratio)[0]
            if len(mix_indices) > 0:
                if self._tilt_active(step) and self._tilt_goal:
                    if init_observations is None or init_timesteps is None:
                        raise ValueError(
                            "tilt_goal requires initial observations and timesteps."
                        )
                    mix_zs = self.sample_tilted_goal_z(
                        train_goal=train_goal,
                        init_observations=init_observations,
                        init_timesteps=init_timesteps,
                        size=len(mix_indices),
                        step=step,
                    )
                else:
                    mix_zs = self.FB.backward_representation(
                        train_goal[mix_indices]
                    ).detach()
                    mix_zs = math.sqrt(
                        self._z_dimension
                    ) * torch.nn.functional.normalize(mix_zs, dim=1)
                zs[mix_indices] = mix_zs

        return zs

    @torch.no_grad()
    def sample_tilted_goal_z(
        self,
        train_goal: torch.Tensor,
        init_observations: torch.Tensor,
        init_timesteps: torch.Tensor,
        size: int,
        step: int,
    ) -> torch.Tensor:
        """Tilt goal latents using independently sampled initial-state observations."""
        candidate_size = self.tilt.candidate_multiplier * size
        goal_idx = torch.randint(
            0, train_goal.shape[0], (candidate_size,), device=train_goal.device
        )
        goal_candidates = train_goal[goal_idx]
        z_candidates = self.FB.backward_representation(goal_candidates).detach()
        z_candidates = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(
            z_candidates, dim=1
        )

        score_observations = self.tilt.sample_init_features(
            init_features=init_observations,
            init_timesteps=init_timesteps,
            num_samples=candidate_size,
        )
        candidate_score, _ = self.score_and_features(
            observations=score_observations,
            z=z_candidates,
            step=step,
        )
        logits = candidate_score / self.tilt.temperature
        logits = logits - logits.max()
        prob = torch.softmax(logits, dim=0)
        selected_idx = torch.multinomial(prob, num_samples=size, replacement=False)
        return z_candidates[selected_idx]

    @torch.no_grad()
    def score_and_features(
        self,
        observations: torch.Tensor,
        z: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        std = schedule(self.std_dev_schedule, step)
        actions, _ = self.actor(observations, z, std, sample=True)
        target_f1, target_f2 = self.FB.forward_representation_target(
            observation=observations,
            z=z,
            action=actions,
        )
        features = 0.5 * (target_f1 + target_f2)

        # Cap each candidate's feature norm at the p99 of this batch so a
        # handful of oversized-norm candidates can't dominate the leverage
        # score (and the Gram it feeds into); candidates below the cap are
        # left untouched, only the top outliers are pulled down to it.
        feature_norms = features.norm(dim=-1)
        norm_hi = torch.quantile(feature_norms, 0.99)
        clipped_norms = torch.clamp(feature_norms, max=norm_hi)
        features = features * (clipped_norms / feature_norms).unsqueeze(-1)

        trace_g = torch.trace(self.tilt.gram)
        alpha_lam = self._tilt_ridge_alpha * trace_g.item() / self.tilt.gram.shape[0]
        lam = max(alpha_lam, self._tilt_ridge_min)
        if alpha_lam < self._tilt_ridge_min:
            logger.warning(
                "FB.score_and_features: tilt_ridge_min activated "
                "(alpha_lam=%.4e < ridge_min=%.4e). gram may be degenerate.",
                alpha_lam,
                self._tilt_ridge_min,
            )
        identity = torch.eye(
            features.shape[-1], device=features.device, dtype=features.dtype
        )
        ginv = torch.linalg.pinv(self.tilt.gram + lam * identity)
        # The Gram is built from features divided by tilt.feature_scale(), so the
        # query must use the same scale to keep the leverage score consistent.
        query = z if self._tilting_by_z else features / self.tilt.feature_scale()
        projected = query @ ginv
        score = torch.sum(projected * query, dim=1)
        return score, features

    def update_fb(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """
        Updates the forward-backward representation network.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """

        total_loss, metrics, _, _, _, _, _, _, _, _ = self._update_fb_inner(
            observations, actions, next_observations, discounts, zs, step
        )

        self.FB_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        for param in self.FB.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.FB_optimizer.step()

        return metrics

    def _update_fb_inner(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        next_observations: torch.Tensor,
        discounts: torch.Tensor,
        zs: torch.Tensor,
        step: int,
    ):
        """
        Loss computation common to FB and all child classes. All equation references
        are to the appendix of the FB paper (Touati et. al (2022)).
        The loss contains two components:
            1. Forward-backward representation loss: a Bellman update on the successor
                measure (equation 24, Appendix B)
            2. Orthonormalisation loss: constrains backward function such that the
                measure of state s from state s = 1 (equation 26, Appendix B)
            Note: Q loss (Equation 9) is not implemented.
        Args:
            observations: observation tensor of shape [batch_size, observation_length]
            actions: action tensor of shape [batch_size, action_length]
            next_observations: next observation tensor of
                                shape [batch_size, observation_length]
            discounts: discount tensor of shape [batch_size, 1]
            zs: policy tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            total_loss: total loss for FB
            metrics: dictionary of metrics for logging
            F1: forward embedding no. 1
            F2: forward embedding no. 2
            B_next: backward embedding
            M1_next: successor measure no. 1
            M2_next: successor measure no. 2
            target_B: backward embedding from target network
            off_diagonal: off-diagonal masking array
            actor_std_dev: current actor standard deviation
        """

        with torch.no_grad():
            actor_std_dev = schedule(self.std_dev_schedule, step)
            next_actions, _ = self.actor(
                next_observations, zs, actor_std_dev, sample=True
            )

            target_F1, target_F2 = self.FB.forward_representation_target(
                observation=next_observations, z=zs, action=next_actions
            )
            target_B = self.FB.backward_representation_target(
                observation=next_observations
            )
            target_M1 = torch.einsum(
                "sd, td -> st", target_F1, target_B
            )  # [batch_size, batch_size]
            target_M2 = torch.einsum(
                "sd, td -> st", target_F2, target_B
            )  # [batch_size, batch_size]
            target_M = torch.min(target_M1, target_M2)

        # --- Forward-backward representation loss ---
        F1, F2 = self.FB.forward_representation(observations, actions, zs)
        B_next = self.FB.backward_representation(next_observations)

        M1_next = torch.einsum("sd, td -> st", F1, B_next)
        M2_next = torch.einsum("sd, td -> st", F2, B_next)

        # diagonal/off diagonal method mentioned here:
        # https://github.com/facebookresearch/controllable_agent/issues/4
        I = torch.eye(*M1_next.size(), device=self._device)  # next state = s_{t+1}
        off_diagonal = ~I.bool()  # future states =/= s_{t+1}

        fb_off_diag_loss = 0.5 * sum(
            (M - discounts * target_M)[off_diagonal].pow(2).mean()
            for M in [M1_next, M2_next]
        )

        fb_diag_loss = -sum(M.diag().mean() for M in [M1_next, M2_next])

        fb_loss = fb_diag_loss + fb_off_diag_loss

        # --- orthonormalisation loss ---
        covariance = torch.matmul(B_next, B_next.T)
        ortho_loss_diag = -2 * covariance.diag().mean()
        ortho_loss_off_diag = covariance[off_diagonal].pow(2).mean()
        ortho_loss = self.FB.orthonormalisation_coefficient * (
            ortho_loss_diag + ortho_loss_off_diag
        )

        total_loss = fb_loss + ortho_loss

        metrics = {
            "train/forward_backward_total_loss": total_loss,
            "train/forward_backward_fb_loss": fb_loss,
            "train/forward_backward_fb_diag_loss": fb_diag_loss,
            "train/forward_backward_fb_off_diag_loss": fb_off_diag_loss,
            "train/ortho_diag_loss": ortho_loss_diag,
            "train/ortho_off_diag_loss": ortho_loss_off_diag,
            "train/target_M": target_M.mean().item(),
            "train/M": M1_next.mean().item(),
            "train/F": F1.mean().item(),
            "train/B": B_next.mean().item(),
        }

        return total_loss, metrics, \
               F1, F2, B_next, M1_next, M2_next, target_B, off_diagonal, actor_std_dev

    def update_actor(
        self, observation: torch.Tensor, z: torch.Tensor, step: int
    ) -> Dict[str, float]:
        """
        Computes the actor loss.
        Args:
            observation: tensor of shape [batch_size, observation_length]
            z: tensor of shape [batch_size, z_dimension]
            step: current training step
        Returns:
            metrics: dictionary of metrics for logging
        """
        std = schedule(self.std_dev_schedule, step)
        action, action_dist = self.actor(observation, z, std, sample=True)

        # with torch.no_grad():
        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=action
        )

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        # update actor towards action that maximise Q (minimise -Q)
        actor_loss = -Q

        if (
            type(self.actor.actor)  # pylint: disable=unidiomatic-typecheck
            == AbstractGaussianActor
        ):
            # add an entropy regularisation term
            log_prob = action_dist.log_prob(action).sum(-1)
            actor_loss += 0.1 * log_prob  # NOTE: currently hand-coded weight!
            mean_log_prob = log_prob.mean().item()
        else:
            mean_log_prob = 0.0

        actor_loss = actor_loss.mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        for param in self.actor.parameters():
            param.grad.data.clamp_(-1, 1)
        self.actor_optimizer.step()

        metrics = {
            "train/actor_loss": actor_loss.item(),
            "train/actor_Q": Q.mean().item(),
            "train/actor_log_prob": mean_log_prob,
        }

        return metrics

    def load(self, filepath: Path):
        """Loads model."""
        pass

    def sample_z(self, size: int) -> torch.Tensor:
        """Samples z in the sphere of radius sqrt(D)."""
        gaussian_random_variable = torch.randn(
            size, self._z_dimension, dtype=torch.float32, device=self._device
        )
        gaussian_random_variable = torch.nn.functional.normalize(
            gaussian_random_variable, dim=1
        )
        z = math.sqrt(self._z_dimension) * gaussian_random_variable

        return z

    def infer_z(
        self, observations: torch.Tensor, rewards: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Infers z from observations and rewards. Follows page 17 from:
        https://arxiv.org/pdf/2209.14935.pdf. If no rewards are passed, we
        assume we're inferring z from a goal-reaching task, and thus set z
        directly from the goal state.
        Args:
            observations: tensor of shape [inference_steps, observation_length]
            rewards (optional): tensor of shape [inference_steps, 1]
        Returns:
            z: tensor of shape [z_dimension,]
        """

        with torch.no_grad():
            z = self.FB.backward_representation(observations)

        if rewards is not None:
            z = torch.matmul(rewards.T, z) / rewards.shape[0]  # reward-weighted average

        z = math.sqrt(self._z_dimension) * torch.nn.functional.normalize(z, dim=1)

        z = z.squeeze().cpu().numpy()

        return z

    def predict_q(
        self, observation: torch.Tensor, z: torch.Tensor, action: torch.Tensor
    ):
        """
        Predicts the value of a state-action pair w.r.t. a task. Used as a utility
        function for downstream analysis.
        Args:
            observation: tensor of shape [N, observation_length]
            z: tensor of shape [N, z_dimension]
            action: tensor of shape [N, action_length]
        Returns:
            Qs
        """
        F1, F2 = self.FB.forward_representation(
            observation=observation, z=z, action=action
        )

        # get Qs from F and z
        Q1 = torch.einsum("sd, sd -> s", F1, z)
        Q2 = torch.einsum("sd, sd -> s", F2, z)
        Q = torch.min(Q1, Q2)

        return Q

    @staticmethod
    def soft_update_params(
        network: torch.nn.Sequential, target_network: torch.nn.Sequential, tau: float
    ) -> None:
        """
        Soft updates the target network parameters via Polyak averaging.
        Args:
            network: Online network.
            target_network: Target network.
            tau: Interpolation parameter.
        """

        for param, target_param in zip(
            network.parameters(), target_network.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
