"""Reusable latent selection utilities for tilt-style agents."""

import logging
import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


ScoreFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
ScoreFromFeaturesFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
SampleZFn = Callable[[int], torch.Tensor]


def sample_init_indices(
    init_geom_ratio: Optional[float],
    init_timesteps: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Indices into init_timesteps to sample num_samples initial states from.

    init_geom_ratio=None (the default) samples uniformly at random -- no bias
    toward early-episode states. Passing a ratio in (0, 1) instead reweights
    samples by ratio**timestep (only set this explicitly if you want that
    early-episode bias)."""
    if init_geom_ratio is None:
        return torch.randint(
            0, init_timesteps.shape[0], (num_samples,), device=init_timesteps.device
        )
    init_weights = torch.pow(init_geom_ratio, init_timesteps.to(torch.float32))
    init_weights = init_weights / init_weights.sum()
    return torch.multinomial(init_weights, num_samples=num_samples, replacement=True)


@dataclass
class TiltLatentSelector:
    """Maintains and refreshes a latent pool using a task-coverage score."""

    z: torch.Tensor
    beta: float = 0.995
    temperature: float = 20.0
    candidate_multiplier: int = 10
    init_geom_ratio: Optional[float] = None
    states_per_candidate: int = 2

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
        # Starts at an arbitrary 1.0 since no data has been seen yet at
        # construction time; update_gram seeds it from the first real batch's
        # own scale instead of EMA-blending into it (see _feat_ms_seeded),
        # so there's no multi-hundred-step warmup where this is badly wrong.
        self.feat_ms = torch.ones((), device=self.z.device, dtype=torch.float64)
        self._feat_ms_seeded = False
        self._refresh_count = 0
        self.last_prob_min = float("nan")
        self.last_prob_max = float("nan")

    def feature_scale(self) -> torch.Tensor:
        """Global scale (RMS) applied to forward features before the Gram."""
        return (torch.sqrt(self.feat_ms) + 1e-8).to(self.z.dtype)

    def normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        """Divide by feature_scale() (the drifting global RMS) then tanh-compress
        each row's own norm around sqrt(D) -- the expected norm of a "typical"
        row once feature_scale() has made every dimension ~unit variance. Needs
        no new tunable threshold: sqrt(D) falls straight out of feature_scale()'s
        own definition. Unlike feature_scale() alone (a single scalar shared by
        every row, which provably cannot change the Gram's eigenvalue ratios --
        uniform rescaling preserves conditioning exactly), this compresses each
        row by its OWN amount, so genuine outlier candidates get pulled toward
        the pack instead of dominating the Gram, while typical-magnitude rows
        pass through nearly unchanged (tanh is ~linear near 0)."""
        scaled = features / self.feature_scale()
        typical_norm = math.sqrt(features.shape[-1])
        norm = scaled.norm(dim=-1, keepdim=True)
        compressed = typical_norm * torch.tanh(norm / typical_norm)
        return scaled * (compressed / (norm + 1e-12))

    @torch.no_grad()
    def sample_init_features(
        self,
        init_features: torch.Tensor,
        init_timesteps: torch.Tensor,
        num_samples: int,
    ) -> torch.Tensor:
        """Sample initial-state features (uniform unless init_geom_ratio is set)."""
        init_timesteps = init_timesteps.to(device=init_features.device)
        obs_idx = sample_init_indices(self.init_geom_ratio, init_timesteps, num_samples)
        return init_features[obs_idx]

    @torch.no_grad()
    def refresh(
        self,
        init_features: torch.Tensor,
        init_timesteps: torch.Tensor,
        sample_z: SampleZFn,
        score_fn: ScoreFn,
        score_from_features_fn: Optional[ScoreFromFeaturesFn] = None,
        update_gram: bool = True,
        return_features: bool = False,
        num_samples: Optional[int] = None,
    ):
        n = self.z.shape[0] if num_samples is None else num_samples
        if n <= 0:
            raise ValueError("TiltLatentSelector.refresh requires num_samples > 0.")
        n_candidates = self.candidate_multiplier * n
        z_candidates = sample_z(n_candidates)

        # Pair each candidate z with `states_per_candidate` independently sampled
        # states, compute a feature per (z, state), and average those features
        # per candidate -- cuts noise in both the feature going into the Gram/
        # feat_ms update AND (via score_from_features_fn, recomputed on the
        # averaged feature with the current Ginv/ridge) the leverage score used
        # for selection. We deliberately do NOT average the k raw per-state
        # scores themselves: score = query^T Ginv query is a convex quadratic
        # form, so mean(score_i) over states is a biased (Jensen's-inequality-
        # inflated) estimate of "the score of the average feature" -- scoring
        # the averaged feature directly is the correct quantity.
        k = max(1, self.states_per_candidate)
        z_repeated = z_candidates.repeat_interleave(k, dim=0)
        feature_candidates = self.sample_init_features(
            init_features=init_features,
            init_timesteps=init_timesteps,
            num_samples=n_candidates * k,
        )

        score_repeated, feat_repeated = score_fn(feature_candidates, z_repeated)
        feature_stats = feat_repeated.view(n_candidates, k, -1).mean(dim=1)
        if k == 1:
            candidate_score = score_repeated
        elif score_from_features_fn is not None:
            candidate_score = score_from_features_fn(feature_stats, z_candidates)
        else:
            # Fallback if the caller didn't wire score_from_features_fn: mean of
            # the per-state scores (the biased estimate described above).
            candidate_score = score_repeated.view(n_candidates, k).mean(dim=1)

        self._refresh_count += 1

        logits = candidate_score / self.temperature
        logits = logits - logits.max()
        prob = torch.softmax(logits, dim=0)
        self.last_prob_min = float(prob.min())
        self.last_prob_max = float(prob.max())
        selected_idx = torch.multinomial(prob, num_samples=n, replacement=False)

        if self._refresh_count % 10000 == 0:
            logger.warning(
                "TiltLatentSelector.refresh: diag feat_max=%.4e feat_finite=%s ",
                feature_stats.abs().max().item(),
                feature_stats.abs().min().item(),
            )

        self.z = z_candidates[selected_idx]
        if update_gram:
            self.update_gram(((feature_stats, 1.0),))
        if return_features:
            return self.z, feature_stats
        return self.z

    @torch.no_grad()
    def update_gram(
        self,
        weighted_feature_batches: Sequence[Tuple[torch.Tensor, float]],
    ) -> None:
        """Update the Gram once from a weighted mixture of feature distributions."""
        batches = [
            (features, float(weight))
            for features, weight in weighted_feature_batches
            if features.numel() > 0 and weight > 0
        ]
        if not batches:
            return

        total_weight = sum(weight for _, weight in batches)
        batch_ms = sum(
            weight * (features.double() ** 2).mean().detach()
            for features, weight in batches
        ) / total_weight
        if not self._feat_ms_seeded:
            # First real batch: hard-set instead of EMA-blending into the
            # arbitrary 1.0 init, which would otherwise take hundreds of steps
            # (beta=0.995 -> N_eff~200) to catch up to the true feature scale --
            # badly miscalibrating feature_scale()/the Gram during that window.
            self.feat_ms = batch_ms.clone()
            self._feat_ms_seeded = True
        else:
            self.feat_ms.mul_(self.beta).add_((1 - self.beta) * batch_ms)

        normalized = [(self.normalized_features(features), weight)
                      for features, weight in batches]
        gram_batch = sum(
            weight * (nf.T @ nf) / nf.shape[0] for nf, weight in normalized
        ) / total_weight
        self.gram.mul_(self.beta).add_((1 - self.beta) * gram_batch)
