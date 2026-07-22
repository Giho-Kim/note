"""Reusable latent selection utilities for tilt-style agents."""

import logging
from dataclasses import dataclass
from typing import Callable, Tuple

import torch

logger = logging.getLogger(__name__)


ScoreFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
SampleZFn = Callable[[int], torch.Tensor]


@dataclass
class TiltLatentSelector:
    """Maintains and refreshes a latent pool using a task-coverage score."""

    z: torch.Tensor
    beta: float = 0.995
    temperature: float = 20.0
    candidate_multiplier: int = 10
    init_geom_ratio: float = 0.9

    def __post_init__(self) -> None:
        dim = self.z.shape[-1]
        self.gram = torch.eye(dim, device=self.z.device, dtype=self.z.dtype)
        self.running_mean = torch.zeros(dim, device=self.z.device, dtype=self.z.dtype)
        # Running mean-square of the (unbounded) forward features, used as a
        # single global scale so the Gram matrix stays well-conditioned even if
        # the FB feature norm diverges. Relative magnitudes across candidates are
        # preserved because every feature is divided by the *same* scalar.
        # float64 so squaring a finite-but-huge feature (which would overflow
        # float32 at ~3.4e38) does not turn the scale into inf.
        self.feat_ms = torch.ones((), device=self.z.device, dtype=torch.float64)
        self._refresh_count = 0
        self.last_prob_min = float("nan")
        self.last_prob_max = float("nan")

    def feature_scale(self) -> torch.Tensor:
        """Global scale (RMS) applied to forward features before the Gram."""
        return (torch.sqrt(self.feat_ms) + 1e-8).to(self.z.dtype)

    @torch.no_grad()
    def sample_init_features(
        self,
        init_features: torch.Tensor,
        init_timesteps: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample initial-state features from the selector's geometric distribution."""
        init_timesteps = init_timesteps.to(
            device=init_features.device, dtype=init_features.dtype
        )
        init_weights = torch.pow(self.init_geom_ratio, init_timesteps)
        init_weights = init_weights / init_weights.sum()
        obs_idx = torch.multinomial(
            init_weights, num_samples=num_samples, replacement=True
        )
        return init_features[obs_idx]

    @torch.no_grad()
    def refresh(
        self,
        init_features: torch.Tensor,
        init_timesteps: torch.Tensor,
        sample_z: SampleZFn,
        score_fn: ScoreFn,
    ) -> torch.Tensor:
        n = self.z.shape[0]
        n_candidates = self.candidate_multiplier * n
        z_candidates = sample_z(n_candidates)

        feature_candidates = self.sample_init_features(
            init_features=init_features,
            init_timesteps=init_timesteps,
            num_samples=n_candidates,
        )

        candidate_score, feature_stats = score_fn(feature_candidates, z_candidates)

        self._refresh_count += 1

        logits = candidate_score / self.temperature
        logits = logits - logits.max()
        prob = torch.softmax(logits, dim=0)
        self.last_prob_min = float(prob.min())
        self.last_prob_max = float(prob.max())
        selected_idx = torch.multinomial(prob, num_samples=n, replacement=False)

        # Update the running feature scale (EMA of per-element mean-square), then
        # normalise features by that single scalar before forming the Gram. This
        # bounds the Gram's magnitude (no float32 overflow) while preserving the
        # relative sizes of candidate features. score_and_features divides its
        # query by the same scale so the leverage score stays consistent.
        self.feat_ms.mul_(self.beta).add_(
            (1 - self.beta) * (feature_stats.double() ** 2).mean().detach()
        )
        f_tilde = feature_stats / self.feature_scale()


        if self._refresh_count % 10000 == 0:
            logger.warning(
                "TiltLatentSelector.refresh: diag feat_max=%.4e feat_finite=%s ",
                feature_stats.abs().max().item(),
                feature_stats.abs().min().item(),
            )

        # obs candidates are already sampled proportionally to init_weights, so
        # the Gram matrix is a plain (uniform) average over them. Re-weighting by
        # candidate_init_weights here would apply init_geom_ratio twice.
        gram_batch = f_tilde.T @ f_tilde / f_tilde.shape[0]
        self.gram.mul_(self.beta).add_((1 - self.beta) * gram_batch)

        self.z = z_candidates[selected_idx]
        return self.z
