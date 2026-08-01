"""
Basis Trajectory Distribution (BTD) for FB.

Builds a Gaussian mixture over normalized discounted-feature summaries of
dataset subtrajectories:

    phi(s) = (E[B(s)B(s)^T])^{-1} B(s)                 (whitened backward features)
    psi_tau = sum_t discount^t * phi(s_t)               (per subtrajectory)
    z_tau = psi_tau / ||psi_tau||
    GMM.fit({z_tau})

The fitted GMM is then used in place of Unif(S^{d-1}) to sample z for
actor-only fine-tuning (see OfflineRLWorkspace.train_btd).
"""

import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from agents.fb.agent import FB


def _fit_whitening_matrix(
    agent: FB, observations: torch.Tensor, ridge: float, batch_size: int = 4096
) -> torch.Tensor:
    """Estimates (E[B(s)B(s)^T])^{-1} from a sample of states."""
    device = agent._device  # pylint: disable=protected-access
    z_dimension = agent._z_dimension  # pylint: disable=protected-access
    gram = torch.zeros(z_dimension, z_dimension, dtype=torch.float64, device=device)
    n = observations.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            b = agent.FB.backward_representation(
                observations[start : start + batch_size].to(device)
            ).double()
            gram += b.T @ b
    gram /= n
    gram += ridge * torch.eye(z_dimension, dtype=torch.float64, device=device)
    return torch.linalg.inv(gram).float()


def _compute_phi(
    agent: FB, observations: torch.Tensor, whitening_matrix: torch.Tensor, batch_size: int = 4096
) -> torch.Tensor:
    """phi(s) = (E[B(s)B(s)^T])^{-1} B(s)."""
    device = agent._device  # pylint: disable=protected-access
    outputs = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            b = agent.FB.backward_representation(
                observations[start : start + batch_size].to(device)
            )
            outputs.append(b @ whitening_matrix.T)
    return torch.cat(outputs, dim=0)


def _sample_subtrajectory_observations(
    dataset_path: Path,
    n_subtrajectories: int,
    min_len: int,
    max_len: int,
    seed: int,
    dataset_transitions: int = None,
) -> List[torch.Tensor]:
    """Reads raw episodes directly from dataset_path (bypassing the
    flattened/subsampled OfflineReplayBuffer, which discards episode
    boundaries) and samples n_subtrajectories contiguous observation windows,
    each of random length in [min_len, max_len].

    If dataset_transitions is given, only the first
    dataset_transitions // episode_length episodes are considered, so the GMM
    is built from the same transitions budget as the actor's replay buffer
    (episode_length taken from the first episode in the file)."""
    rng = np.random.default_rng(seed)
    episodes = dict(np.load(dataset_path, allow_pickle=True))
    episode_list = [episode.item() for episode in episodes.values()]

    if dataset_transitions is not None:
        episode_length = len(episode_list[0]["observation"])
        n_episodes = max(1, dataset_transitions // episode_length)
        episode_list = episode_list[:n_episodes]

    episode_lengths = np.array([len(ep["observation"]) for ep in episode_list])

    # sample episodes proportional to length so short episodes aren't
    # over-represented relative to how much data they actually contain
    episode_idxs = rng.choice(
        len(episode_list),
        size=n_subtrajectories,
        p=episode_lengths / episode_lengths.sum(),
    )

    subtrajectories = []
    for episode_idx in episode_idxs:
        obs = episode_list[episode_idx]["observation"]
        length = min(int(rng.integers(min_len, max_len + 1)), len(obs))
        start = int(rng.integers(0, len(obs) - length + 1))
        subtrajectories.append(
            torch.as_tensor(obs[start : start + length], dtype=torch.float32)
        )
    return subtrajectories


def build_btd_gmm(
    agent: FB,
    dataset_path: Path,
    n_subtrajectories: int,
    min_len: int,
    max_len: int,
    gmm_components: int,
    whitening_ridge: float,
    seed: int,
    dataset_transitions: int = None,
) -> Tuple[GaussianMixture, torch.Tensor]:
    """Runs the full BTD build step (Phase 1's phi(s) fit + GMM fit) and
    returns the fitted (frozen) GMM together with the whitening matrix that
    defines phi(s) = whitening_matrix @ B(s) -- the same fixed phi used by
    Phase 2's critic retraining (see FB.update_critic_btd)."""
    subtrajectories = _sample_subtrajectory_observations(
        dataset_path=dataset_path,
        n_subtrajectories=n_subtrajectories,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
        dataset_transitions=dataset_transitions,
    )

    all_observations = torch.cat(subtrajectories, dim=0)
    whitening_matrix = _fit_whitening_matrix(agent, all_observations, ridge=whitening_ridge)
    phi = _compute_phi(agent, all_observations, whitening_matrix)

    discount = agent.FB._discount  # pylint: disable=protected-access
    z_taus = []
    offset = 0
    for subtraj in subtrajectories:
        length = subtraj.shape[0]
        phi_tau = phi[offset : offset + length]
        offset += length
        discounts = discount ** torch.arange(
            length, dtype=torch.float32, device=phi_tau.device
        )
        psi_tau = (discounts.unsqueeze(-1) * phi_tau).sum(dim=0)
        norm = psi_tau.norm()
        if norm > 0:
            z_taus.append((psi_tau / norm).cpu().numpy())

    z_taus = np.stack(z_taus, axis=0)
    gmm = GaussianMixture(n_components=gmm_components, random_state=seed)
    gmm.fit(z_taus)
    return gmm, whitening_matrix


class GMMZSampler:
    """Samples z ~ BTD-GMM, renormalized onto the sqrt(z_dimension)-radius
    sphere that FB's forward/backward representations were trained on (GMM
    samples aren't sphere-constrained, unlike FB.sample_z's Unif(S^{d-1}))."""

    def __init__(self, gmm: GaussianMixture, z_dimension: int, device: torch.device):
        self.gmm = gmm
        self.z_dimension = z_dimension
        self.device = device
        # Precomputed once so sample() never has to touch gmm.sample() --
        # sklearn's GaussianMixture.sample() re-derives its per-component
        # cholesky factors from precisions_cholesky_ on every call, which at
        # real z_dimensions (~100) costs ~150ms/call vs ~0.4ms/call here, a
        # >100x difference that otherwise shows up as run time depending on
        # whether novelty_weighted happens to be on or off.
        self._chol = np.stack(
            [np.linalg.cholesky(cov) for cov in gmm.covariances_]
        )
        self._component_probs = gmm.weights_
        inverse_weights = 1.0 / gmm.weights_
        self._inverse_component_probs = inverse_weights / inverse_weights.sum()

    def sample(self, size: int, novelty_weighted: bool = False) -> torch.Tensor:
        probs = self._inverse_component_probs if novelty_weighted else self._component_probs
        component_idx = np.random.choice(self.gmm.n_components, size=size, p=probs)
        noise = np.random.standard_normal((size, self.z_dimension))
        samples = self.gmm.means_[component_idx] + np.einsum(
            "nij,nj->ni", self._chol[component_idx], noise
        )
        z = torch.as_tensor(samples, dtype=torch.float32, device=self.device)
        return math.sqrt(self.z_dimension) * torch.nn.functional.normalize(z, dim=-1)
