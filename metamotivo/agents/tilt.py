"""Reusable latent selection utilities for tilt-style agents."""

import logging
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


def weighted_select(
    candidate_score: torch.Tensor,
    temperature: float,
    n: int,
    uniform_mix: float = 0.5,
    normalize: bool = False,
) -> Tuple[torch.Tensor, float, float]:
    """Softmax(candidate_score / temperature) then multinomial-without-
    replacement selection of n indices. Factored out of refresh() so a cached
    candidate pool's scores can be cheaply RE-selected from on later calls
    (a fresh draw at the current temperature, no new forward passes) instead
    of only ever selecting once right after scoring -- see
    TiltLatentSelector.resample().

    normalize z-scores candidate_score (subtract mean, divide by std) before
    dividing by temperature, so a fixed temperature keeps the same effective
    sharpness regardless of upstream feature-scale drift inflating the raw
    score's spread over training -- otherwise a fixed temperature effectively
    gets sharper (more collapse-prone) as that spread grows.

    uniform_mix blends in a uniform component: prob = uniform_mix/n_total +
    (1-uniform_mix)*softmax. Temperature alone can't guarantee a floor/cap on
    selection probability -- if candidate_score's own spread drifts (e.g. from
    upstream feature-scale drift), even a large fixed temperature can still
    end up fully collapsed onto one candidate (softmax_i -> 1). The uniform
    mix caps this structurally regardless of how extreme the scores get:
    p_i >= uniform_mix/n_total (no candidate fully starved) and
    p_i <= uniform_mix/n_total + (1-uniform_mix) (no candidate can take more
    than (1-uniform_mix) of the selection mass, however degenerate the
    scores)."""
    n_total = candidate_score.shape[0]
    n = min(n, n_total)
    if normalize:
        candidate_score = (candidate_score - candidate_score.mean()) / (
            candidate_score.std() + 1e-8
        )
    logits = candidate_score / temperature
    logits = logits - logits.max()
    softmax_prob = torch.softmax(logits, dim=0)
    prob = uniform_mix / n_total + (1 - uniform_mix) * softmax_prob
    idx = torch.multinomial(prob, num_samples=n, replacement=False)
    return idx, float(prob.min()), float(prob.max())


@dataclass
class TiltLatentSelector:
    """Maintains and refreshes a latent pool using a task-coverage score."""

    z: torch.Tensor
    beta: float = 0.995
    temperature: float = 20.0
    candidate_multiplier: int = 10
    init_geom_ratio: Optional[float] = None
    states_per_candidate: int = 2
    uniform_mix: float = 0.5
    normalize: bool = False

    def __post_init__(self) -> None:
        dim = self.z.shape[-1]
        self.gram = torch.eye(dim, device=self.z.device, dtype=self.z.dtype)
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
        # Cache of the last refresh()'s (oversampled candidate z, its score)
        # pool, so resample() can cheaply re-select from it between refreshes
        # without paying for new forward passes. goal_candidate_{z,score} is
        # the analogous cache for goal-conditioned z selection (populated by
        # FB/TDJEPA's sample_goal_z_candidates, which owns that scoring since
        # it isn't part of this class's own sphere-z pool).
        self._candidate_z: Optional[torch.Tensor] = None
        self._candidate_score: Optional[torch.Tensor] = None
        self.goal_candidate_z: Optional[torch.Tensor] = None
        self.goal_candidate_score: Optional[torch.Tensor] = None

    def feature_scale(self) -> torch.Tensor:
        """Global scale (RMS) applied to forward features before the Gram."""
        return (torch.sqrt(self.feat_ms) + 1e-8).to(self.z.dtype)

    def normalized_features(self, features: torch.Tensor) -> torch.Tensor:
        """Divide by feature_scale() (the drifting global RMS). A single scalar
        shared by every row, so it keeps Gram's trace at ~z_dimension but
        provably cannot change the Gram's eigenvalue ratios (conditioning) --
        that's a property of the underlying features, not something this
        rescaling step can fix."""
        return features / self.feature_scale()

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

        selected_idx, self.last_prob_min, self.last_prob_max = weighted_select(
            candidate_score, self.temperature, n, self.uniform_mix, self.normalize
        )
        self.z = z_candidates[selected_idx]
        self._candidate_z = z_candidates
        self._candidate_score = candidate_score
        if update_gram:
            self.update_gram(((feature_stats, 1.0),))
        if return_features:
            return self.z, feature_stats
        return self.z

    @torch.no_grad()
    def resample(self, n: Optional[int] = None) -> torch.Tensor:
        """Cheap re-selection from the LAST refresh()'s cached (candidate z,
        score) pool: a fresh softmax(temperature)+multinomial draw, no new
        forward passes. Use between refresh() calls (e.g. under a
        tilt_refresh_interval) to keep every step's z leverage-informed
        without paying refresh()'s oversampled-candidate scoring cost every
        step -- the tradeoff is the pool itself goes stale until the next
        refresh()."""
        if self._candidate_z is None:
            raise RuntimeError("resample() called before any refresh().")
        n = self.z.shape[0] if n is None else n
        selected_idx, self.last_prob_min, self.last_prob_max = weighted_select(
            self._candidate_score, self.temperature, n, self.uniform_mix, self.normalize
        )
        self.z = self._candidate_z[selected_idx]
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
