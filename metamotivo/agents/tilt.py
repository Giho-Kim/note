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

        init_timesteps = init_timesteps.to(
            device=init_features.device, dtype=init_features.dtype
        )
        init_weights = torch.pow(self.init_geom_ratio, init_timesteps)
        init_weights = init_weights / init_weights.sum()

        obs_idx = torch.multinomial(
            init_weights, num_samples=n_candidates, replacement=True
        )
        feature_candidates = init_features[obs_idx]
        candidate_init_weights = init_weights[obs_idx]

        candidate_score, feature_stats = score_fn(feature_candidates, z_candidates)

        if not feature_stats.isfinite().all() or not candidate_score.isfinite().all():
            logger.warning(
                "TiltLatentSelector.refresh: non-finite values detected in "
                "feature_stats (isfinite=%s) or candidate_score (isfinite=%s).",
                feature_stats.isfinite().all().item(),
                candidate_score.isfinite().all().item(),
            )

        logits = candidate_score / self.temperature
        logits = logits - logits.max()
        prob = torch.softmax(logits, dim=0)
        selected_idx = torch.multinomial(prob, num_samples=n, replacement=False)

        candidate_weights = candidate_init_weights / candidate_init_weights.sum()
        weighted_features = feature_stats * candidate_weights.unsqueeze(-1)
        gram_batch = feature_stats.T @ weighted_features
        self.gram.mul_(self.beta).add_((1 - self.beta) * gram_batch)

        min_eig = torch.linalg.eigvalsh(self.gram).min().item()
        if min_eig < 0.1:
            logger.warning(
                "TiltLatentSelector.refresh: gram matrix degenerate "
                "(min_eigenvalue=%.4e).",
                min_eig,
            )
        self.z = z_candidates[selected_idx]
        return self.z
