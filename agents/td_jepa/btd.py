"""
Basis Trajectory Distribution (BTD) for TD-JEPA.

TD-JEPA analog of agents/fb/btd.py: psi(s) plays FB's B role (the fixed
representation phi_btd(s) is built from, frozen throughout BTD Phase 2)
and phi plays FB's F role (the network Phase 2 reinitializes and retrains).

Unlike FB (which whitens B(s) via (E[BB^T])^{-1}B(s)), TD-JEPA uses raw
psi(s) directly -- no whitening matrix is fit and no per-state
normalization is applied. Only the discounted per-subtrajectory sum gets
L2-normalized:

    phi_btd(s) = psi(s)                                     (raw psi, unnormalized)
    psi_tau = sum_t discount^t * phi_btd(s_t)               (per subtrajectory)
    z_tau = psi_tau / ||psi_tau||                            (L2-normalized here)
    GMM.fit({z_tau})

The fitted GMM is then used in place of Unif(S^{d-1}) to sample z for BTD
Phase 2 (see OfflineRLWorkspace.train_btd). Reuses agents.fb.btd's
GMMZSampler (algorithm-agnostic) and _sample_subtrajectory_observations
(reads raw episodes off disk, doesn't touch the agent).
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from agents.fb.btd import _sample_subtrajectory_observations
from agents.td_jepa.agent import TDJEPA


def _compute_phi(agent: TDJEPA, observations: torch.Tensor, batch_size: int = 4096) -> torch.Tensor:
    """phi_btd(s) = psi(s), unnormalized -- only the discounted per-subtrajectory
    sum (psi_tau, see build_btd_gmm) gets L2-normalized, not each state."""
    device = agent.device
    outputs = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            psi = agent.agent._model.psi(  # pylint: disable=protected-access
                observations[start : start + batch_size].to(device)
            )
            outputs.append(psi)
    return torch.cat(outputs, dim=0)


def build_btd_gmm(
    agent: TDJEPA,
    dataset_path: Path,
    n_subtrajectories: int,
    min_len: int,
    max_len: int,
    gmm_components: int,
    seed: int,
    dataset_transitions: int = None,
) -> Tuple[GaussianMixture, Optional[torch.Tensor]]:
    """Runs the full BTD build step (Phase 1's phi_btd(s) fit + GMM fit) and
    returns the fitted (frozen) GMM. The second element of the tuple mirrors
    agents.fb.btd.build_btd_gmm's (gmm, whitening_matrix) signature -- always
    None here, since TD-JEPA's phi_btd(s) is plain L2-normalized psi(s), not
    a whitened linear map (see TDJEPAAgent.update_critic_btd)."""
    subtrajectories = _sample_subtrajectory_observations(
        dataset_path=dataset_path,
        n_subtrajectories=n_subtrajectories,
        min_len=min_len,
        max_len=max_len,
        seed=seed,
        dataset_transitions=dataset_transitions,
    )

    all_observations = torch.cat(subtrajectories, dim=0)
    phi = _compute_phi(agent, all_observations)

    discount = agent.agent.cfg.train.discount
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
    return gmm, None
