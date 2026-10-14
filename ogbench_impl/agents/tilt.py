"""Shared tilt latent selection for the OGBench FB and TD-JEPA agents.

Selection methods are ported from note's TD-JEPA adapter; the selector itself
is shared with the DMC path to retain the same weighting and normalization.
"""
from pathlib import Path
from typing import Optional

import torch
from pydantic import Field

from ..base import BaseConfig
from metamotivo.agents.tilt import TiltLatentSelector, weighted_select


class TiltTrainConfig(BaseConfig):
    tilt: bool = False
    tilting_by_z: bool = False
    tilt_beta: float = Field(default=0.995, ge=0, lt=1)
    tilt_temperature: float = Field(default=20.0, gt=0)
    tilt_temperature_start: float = Field(default=20.0, gt=0)
    tilt_temperature_end: float = Field(default=1.0, gt=0)
    tilt_candidate_multiplier: int = Field(default=10, ge=1)
    tilt_init_geom_ratio: Optional[float] = Field(default=None, gt=0, le=1)
    tilt_ridge_alpha: float = Field(default=1e-2, ge=0)
    tilt_ridge_min: float = Field(default=1e-8, gt=0)
    tilt_start_step: int = Field(default=0, ge=0)
    tilt_refresh_interval: int = Field(default=5, ge=1)
    tilt_uniform_mix: float = Field(default=0.1, ge=0, le=1)
    tilt_goal: bool = False
    tilt_linear: bool = False
    tilt_free_compete: bool = False
    learning_steps: int = Field(default=1_000_000, ge=0)


class TiltAgentMixin:
    _tilt_warmup = False

    def setup_tilt(self):
        self.tilt = None
        if not self.cfg.train.tilt:
            return
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            cfg = self.cfg.train
            self.tilt = TiltLatentSelector(
                z=self._model.sample_z(cfg.batch_size, device=self.device),
                beta=cfg.tilt_beta, temperature=cfg.tilt_temperature,
                candidate_multiplier=cfg.tilt_candidate_multiplier,
                init_geom_ratio=cfg.tilt_init_geom_ratio,
                uniform_mix=cfg.tilt_uniform_mix, linear=cfg.tilt_linear,
            )
        finally:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

    def _tilt_score_this_step(self, step):
        if self.tilt is None:
            return False
        if self._tilt_warmup:
            return step % self.cfg.train.tilt_refresh_interval == 0
        return self._tilt_active(step) and (step - self.cfg.train.tilt_start_step) % self.cfg.train.tilt_refresh_interval == 0

    @torch.no_grad()
    def sample_tilt_batch(self, train_goal, init_features, timesteps, step):
        cfg = self.cfg.train
        self.tilt.temperature = cfg.tilt_temperature_start + min(max(step, 0) / max(1, cfg.learning_steps), 1) * (cfg.tilt_temperature_end - cfg.tilt_temperature_start)
        if timesteps is None:
            raise ValueError("Tilt requires episode timesteps in the OGBench replay batch.")
        timesteps = timesteps.to(device=init_features.device).reshape(-1)
        goal_mask = torch.rand(cfg.batch_size, device=self.device) < cfg.train_goal_ratio
        sphere_count = int((~goal_mask).sum())
        goal_count = int(goal_mask.sum())
        active = self._tilt_active(step)
        scoring = self._tilt_score_this_step(step)
        if active:
            # A changing mixture can need more rows than a cached pool contains.
            if sphere_count and (self.tilt._candidate_z is None or len(self.tilt._candidate_z) < sphere_count):
                scoring = True
            if cfg.tilt_goal and goal_count and (self.tilt.goal_candidate_z is None or len(self.tilt.goal_candidate_z) < goal_count):
                scoring = True
        sphere_features = None
        if sphere_count and scoring:
            _, sphere_features = self.tilt.refresh(
                init_features=init_features.detach(), init_timesteps=timesteps,
                sample_z=lambda n: self._model.sample_z(n, device=self.device),
                score_fn=lambda features, zs: self.score_and_features(features, zs),
                score_from_features_fn=self.score_from_features,
                update_gram=False, return_features=True, num_samples=sphere_count,
            )
        elif sphere_count and active:
            self.tilt.resample(n=sphere_count)
        zs, goal_features, goal_fraction = self.sample_mixed_z(
            train_goal=train_goal, init_features=init_features.detach(),
            init_timesteps=timesteps, step=step, goal_mask=goal_mask,
            return_goal_features=True, score=scoring,
        )
        # FB warms its Gram before selection activates; TD-JEPA starts at activation.
        if not active and self._tilt_warmup and scoring and goal_count:
            _, goal_features = self.sample_goal_z_candidates(
                train_goal, init_features.detach(), timesteps, goal_count,
                tilt_selection=False, return_features=True,
            )
        if active and cfg.tilt_free_compete and sphere_count and goal_count and cfg.tilt_goal:
            if self.tilt._candidate_z is not None and self.tilt.goal_candidate_z is not None:
                zs = self._tilt_free_compete_select()
        gram_batches = []
        if sphere_features is not None:
            gram_batches.append((sphere_features, 1 - goal_fraction))
        if goal_features is not None:
            gram_batches.append((goal_features, goal_fraction))
        if gram_batches:
            self.tilt.update_gram(gram_batches)
        self._tilt_last_scoring = scoring
        return zs.clone()

    def tilt_metrics(self, step):
        if self.tilt is None:
            return {}
        values = {
            "tilt/active": float(self._tilt_active(step)),
            "tilt/scored": float(self._tilt_last_scoring),
            "tilt/temperature": self.tilt.temperature,
            "tilt/gram_trace": self.tilt.gram.trace(),
            "tilt/feature_scale": self.tilt.feature_scale(),
            "tilt/prob_min": self.tilt.last_prob_min,
            "tilt/prob_max": self.tilt.last_prob_max,
        }
        return {key: torch.as_tensor(value, device=self.device).detach() for key, value in values.items()}

    def save_tilt(self, output_folder):
        if self.tilt is not None:
            torch.save(vars(self.tilt), Path(output_folder) / "tilt.pth")

    def load_tilt(self, folder):
        path = Path(folder) / "tilt.pth"
        if self.tilt is not None and path.exists():
            state = torch.load(path, map_location=self.device, weights_only=True)
            for key, value in state.items():
                setattr(self.tilt, key, value)

    @torch.no_grad()
    def log_tilt_health(self, step):
        if not self._tilt_active(step):
            return
        eig = torch.linalg.eigvalsh(self.tilt.gram)
        ridge = max(self.cfg.train.tilt_ridge_alpha * float(eig.sum()) / len(eig), self.cfg.train.tilt_ridge_min)
        z = self.tilt.z
        zeig = torch.linalg.eigvalsh(z.T @ z / max(1, len(z))).clamp_min(0)
        rank = float(zeig.sum().square() / (zeig.square().sum() + 1e-12))
        scores = self.tilt.goal_candidate_score
        if scores is None:
            scores = self.tilt._candidate_score
        print(
            f"[tilt health] step={step} gram_cond={float(eig[-1] / eig[0].clamp_min(1e-12)):.3g} "
            f"ridge={ridge:.3g} feature_scale={float(self.tilt.feature_scale()):.3g} "
            f"z_rank={rank:.3g} prob=[{self.tilt.last_prob_min:.3g}, {self.tilt.last_prob_max:.3g}] "
            f"score_mean={float(scores.mean()) if scores is not None else float('nan'):.3g}",
            flush=True,
        )

    @torch.no_grad()
    def eval_task_tilt_scores(self, batch, z):
        model = self._model
        obs = model._normalize(batch["observation"].to(self.device))
        encoder = model._fw_encoder if hasattr(model, "_fw_encoder") else model._phi_rgb_encoder
        features = encoder(obs)
        features = self.tilt.sample_init_features(
            features, batch["timesteps"].reshape(-1), min(len(features), self.cfg.train.batch_size),
        )
        scores, _ = self.score_and_features(features, z.reshape(1, -1).expand(len(features), -1))
        return {"tilt_score_mean": float(scores.mean()), "tilt_score_std": float(scores.std(unbiased=False))}

    def _tilt_active(self, step: Optional[int]) -> bool:
        if self.tilt is None:
            return False
        if step is None:
            return True
        return step >= self.cfg.train.tilt_start_step

    @torch.no_grad()
    def _tilt_free_compete_select(self) -> torch.Tensor:
        """See FB._tilt_free_compete_select: pool the sphere and goal
        candidate pools (same shared Gram/score scale, so directly
        comparable) and jointly weighted_select batch_size z's from the
        union -- whichever pool actually carries more leverage wins more of
        the batch, instead of the fixed train_goal_ratio split used for the
        Gram's own bookkeeping. Requires both caches to be populated
        (checked by the caller)."""
        combined_z = torch.cat([self.tilt._candidate_z, self.tilt.goal_candidate_z], dim=0)
        combined_score = torch.cat(
            [self.tilt._candidate_score, self.tilt.goal_candidate_score], dim=0
        )
        idx, self.tilt.last_prob_min, self.tilt.last_prob_max = weighted_select(
            combined_score, self.tilt.temperature, self.cfg.train.batch_size,
            self.tilt.uniform_mix, self.tilt.linear,
        )
        return combined_z[idx]

    @torch.no_grad()
    def sample_mixed_z(
        self,
        train_goal: Optional[torch.Tensor] = None,
        init_features: Optional[torch.Tensor] = None,
        init_timesteps: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        return_goal_features: bool = False,
        goal_mask: Optional[torch.Tensor] = None,
        score: bool = True,
        *args,
        **kwargs,
    ):
        # samples a batch from the z distribution used to update the networks
        if goal_mask is None:
            goal_mask = (
                torch.rand(self.cfg.train.batch_size, device=self.device)
                < self.cfg.train.train_goal_ratio
            )
        sphere_mask = ~goal_mask
        sphere_count = int(sphere_mask.sum().item())
        if not self._tilt_active(step):
            z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        else:
            if sphere_count > 0 and self.tilt.z.shape[0] != sphere_count:
                raise ValueError(
                    "Sphere tilt pool size does not match the non-goal batch size."
                )
            z = torch.empty(
                (self.cfg.train.batch_size, self.tilt.z.shape[-1]),
                device=self.tilt.z.device,
                dtype=self.tilt.z.dtype,
            )
            if sphere_count > 0:
                z[sphere_mask] = self.tilt.z
        goal_features = None
        goal_fraction = float(goal_mask.float().mean())
        if train_goal is not None:
            if goal_mask.any() and self._tilt_active(step):
                if init_features is None or init_timesteps is None:
                    raise ValueError(
                        "Goal Gram update requires initial-state features and timesteps."
                    )
                z = z.clone()
                selected_z, goal_features = self.sample_goal_z_candidates(
                    train_goal=train_goal,
                    init_features=init_features,
                    init_timesteps=init_timesteps,
                    size=int(goal_mask.sum().item()),
                    tilt_selection=self.cfg.train.tilt_goal,
                    return_features=True,
                    score=score,
                )
                z[goal_mask] = selected_z
            else:
                # torch.where returns a new tensor, so tilt.z is never mutated.
                perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
                goals = self.project_train_goals(train_goal[perm])
                z = torch.where(goal_mask.unsqueeze(-1), goals, z)
        if return_goal_features:
            return z, goal_features, goal_fraction
        return z

    @torch.no_grad()
    def sample_goal_z_candidates(
        self,
        train_goal: torch.Tensor,
        init_features: torch.Tensor,
        init_timesteps: torch.Tensor,
        size: int,
        tilt_selection: bool,
        return_features: bool = False,
        score: bool = True,
    ):
        """Build a goal candidate pool and optionally select from it by tilt score.

        score=False skips the expensive oversampled-candidate leverage scoring
        entirely: instead of scoring a fresh pool, cheaply RE-selects (fresh
        softmax+multinomial, no new forward passes) from the last score=True
        call's cached candidate pool (TiltLatentSelector.goal_candidate_z/
        goal_candidate_score) at the current temperature. Falls back to a
        plain random pick from real batch observations if there's no cache yet
        (first tilt-active step) or tilt_selection is off (no scoring signal
        to reuse). Either way returns no features, so the caller's Gram/
        feat_ms update is skipped for this call."""
        if not score:
            if tilt_selection and self.tilt.goal_candidate_z is not None:
                idx, self.tilt.last_prob_min, self.tilt.last_prob_max = weighted_select(
                    self.tilt.goal_candidate_score, self.tilt.temperature, size,
                    self.tilt.uniform_mix, self.tilt.linear,
                )
                z = self.tilt.goal_candidate_z[idx]
            else:
                goal_idx = torch.randint(
                    0, train_goal.shape[0], (size,), device=train_goal.device
                )
                z = self.project_train_goals(train_goal[goal_idx])
            return (z, None) if return_features else z

        candidate_size = self.cfg.train.tilt_candidate_multiplier * size
        goal_idx = torch.randint(
            0, train_goal.shape[0], (candidate_size,), device=train_goal.device
        )
        z_candidates = self.project_train_goals(train_goal[goal_idx])

        # Same states_per_candidate averaging as TiltLatentSelector.refresh: score
        # each candidate goal z against several independently sampled states and
        # average the FEATURE (not the raw scores -- see refresh's docstring for
        # why averaging scores directly would be Jensen-inequality-biased), then
        # rescore the averaged feature for the actual selection.
        k = max(1, self.tilt.states_per_candidate)
        z_repeated = z_candidates.repeat_interleave(k, dim=0)
        score_features = self.tilt.sample_init_features(
            init_features=init_features,
            init_timesteps=init_timesteps,
            num_samples=candidate_size * k,
        )
        score_repeated, feat_repeated = self.score_and_features(
            init_features=score_features,
            z=z_repeated,
        )
        goal_features = feat_repeated.view(candidate_size, k, -1).mean(dim=1)
        candidate_score = (
            score_repeated
            if k == 1
            else self.score_from_features(goal_features, z_candidates)
        )
        if tilt_selection:
            selected_idx, self.tilt.last_prob_min, self.tilt.last_prob_max = (
                weighted_select(
                    candidate_score, self.tilt.temperature, size, self.tilt.uniform_mix,
                    self.tilt.linear,
                )
            )
            self.tilt.goal_candidate_z = z_candidates
            self.tilt.goal_candidate_score = candidate_score
        else:
            selected_idx = torch.randperm(candidate_size, device=train_goal.device)[
                :size
            ]
        selected_z = z_candidates[selected_idx]
        if return_features:
            return selected_z, goal_features
        return selected_z

    def score_from_features(self, v_metric, z):
        """Leverage score of an already-computed (possibly multi-state-averaged)
        feature -- split out of score_and_grad so callers that already have a
        feature (e.g. tilt.refresh scoring an average over several states per
        candidate) don't have to rerun the encoder/predictor to rescore it."""
        with torch.no_grad():
            trace_G = torch.trace(self.tilt.gram)
            lam = torch.maximum(
                self.cfg.train.tilt_ridge_alpha * trace_G / self.tilt.gram.shape[0],
                torch.tensor(
                    self.cfg.train.tilt_ridge_min,
                    device=self.tilt.gram.device,
                    dtype=self.tilt.gram.dtype,
                ),
            )
            identity = torch.eye(v_metric.shape[-1], device=v_metric.device, dtype=v_metric.dtype)
            ginv = torch.linalg.pinv(self.tilt.gram + lam * identity)

        # The Gram is built from tilt.normalized_features(v_metric), so the
        # query must go through the same transform to keep the leverage score
        # consistent.
        query = z if self.cfg.train.tilting_by_z else self.tilt.normalized_features(v_metric)
        qg = query @ ginv
        # /n_eff puts the quadratic form back on the classical finite-sample
        # leverage scale (see TiltLatentSelector.n_eff) so temperature is
        # interpretable against an O(1) score instead of a beta-dependent one.
        return torch.sum(qg * query, dim=1) / self.tilt.n_eff
