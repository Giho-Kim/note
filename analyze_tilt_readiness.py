"""When is the tilt leverage score actually meaningful?

The tilt score is

    q(z) = phi^T (G + lam I)^-1 phi / n_eff,
    phi  = normalized_features( 0.5*(F1_target, F2_target)(s, z, pi(s,z)) )

so it depends on the composition actor -> forward_representation_target, and G
is an EMA of those same features with window n_eff = 1/(1-beta) steps.

Two things have to hold for q to mean "under-covered task direction" rather
than noise:

  D1 (temporal)  the feature map must be quasi-static over G's EMA window.
                 F_target is itself an EMA of F_online with window ~1/tau, so
                 ||F_online - F_target|| / ||F_target|| measures exactly how
                 far the feature map moves over that window, for free.

  D2 (sampling)  q must be reproducible. refresh() draws states to build the
                 features; run the whole scoring twice on the same candidate
                 z's with independent state draws and correlate. Low
                 correlation => refresh() would pick different z's each time,
                 i.e. the selection is noise, whatever the temperature.

Usage:
    python analyze_tilt_readiness.py                 # all known checkpoints
    python analyze_tilt_readiness.py walker.pickle   # specific ones
"""

import glob
import math
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

BASE = Path(__file__).parent
N_CANDIDATES = 512      # candidate z's scored per replica
K_STATES = 2            # states_per_candidate, matches TiltLatentSelector
GRAM_POOL = 4096        # candidates used to build the Gram
SEED = 0


def force_cpu(agent):
    agent.to("cpu")
    for m in agent.modules():
        if hasattr(m, "device"):
            m.device = torch.device("cpu")
    agent._device = torch.device("cpu")
    agent.eval()
    return agent


def opt_step(agent):
    steps = {
        int(s["step"].item()) if hasattr(s.get("step"), "item") else int(s.get("step", -1))
        for s in agent.FB_optimizer.state.values()
        if "step" in s
    }
    return max(steps) if steps else -1


def load_observations(dataset_path, n_obs, rng):
    d = np.load(dataset_path, allow_pickle=True)
    keys = list(d.files)
    rng.shuffle(keys)
    obs, taken = [], 0
    for k in keys:
        ep = d[k].item()["observation"]
        idx = rng.choice(len(ep), size=min(400, len(ep)), replace=False)
        obs.append(ep[idx])
        taken += len(idx)
        if taken >= n_obs:
            break
    return torch.as_tensor(np.concatenate(obs)[:n_obs], dtype=torch.float32)


def sample_z(agent, n, gen):
    g = torch.randn(n, agent._z_dimension, generator=gen, dtype=torch.float32)
    return math.sqrt(agent._z_dimension) * torch.nn.functional.normalize(g, dim=1)


@torch.no_grad()
def features_for(agent, obs, z, std, target=True):
    """0.5*(F1+F2) for (obs, z, actor(obs,z)) -- the exact tilt feature."""
    actions, _ = agent.actor(obs, z, std, sample=False)
    net = agent.FB.forward_representation_target if target else agent.FB.forward_representation
    f1, f2 = net(observation=obs, z=z, action=actions)
    return 0.5 * (f1 + f2)


@torch.no_grad()
def drift_online_vs_target(agent, obs, gen, std):
    """D1: relative movement of the feature map over ~1/tau steps."""
    z = sample_z(agent, obs.shape[0], gen)
    f_tgt = features_for(agent, obs, z, std, target=True)
    f_on = features_for(agent, obs, z, std, target=False)
    return ((f_on - f_tgt).norm(dim=1) / (f_tgt.norm(dim=1) + 1e-12)).mean().item()


@torch.no_grad()
def replica_scores(agent, obs_pool, z_cand, gen, std, ridge_alpha, ridge_min, beta):
    """One full independent replica of refresh()'s scoring of z_cand."""
    n_eff = 1.0 / (1.0 - beta)

    def k_averaged(z):
        n = z.shape[0]
        z_rep = z.repeat_interleave(K_STATES, dim=0)
        idx = torch.randint(0, obs_pool.shape[0], (n * K_STATES,), generator=gen)
        feats = features_for(agent, obs_pool[idx], z_rep, std, target=True)
        return feats.view(n, K_STATES, -1).mean(dim=1)

    # Gram, as update_gram() builds it: normalize by the features' own RMS,
    # then mean outer product (the EMA's fixed point).
    gram_feats = k_averaged(sample_z(agent, GRAM_POOL, gen))
    scale = torch.sqrt((gram_feats.double() ** 2).mean()).float() + 1e-8
    nf = gram_feats / scale
    gram = nf.T @ nf / nf.shape[0]

    lam = max(ridge_alpha * torch.trace(gram).item() / gram.shape[0], ridge_min)
    ginv = torch.linalg.pinv(gram + lam * torch.eye(gram.shape[0]))

    query = k_averaged(z_cand) / scale
    return (torch.sum(query @ ginv * query, dim=1) / n_eff), gram


def analyse(ckpt_path, dataset_path, label):
    agent = force_cpu(torch.load(ckpt_path, map_location="cpu", weights_only=False))
    step = opt_step(agent)
    std = float(agent.std_dev_schedule) if isinstance(agent.std_dev_schedule, (int, float)) else 0.2

    beta = getattr(agent.tilt, "beta", 0.99) if agent.tilt is not None else 0.99
    ridge_alpha = agent._tilt_ridge_alpha
    ridge_min = agent._tilt_ridge_min

    rng = np.random.default_rng(SEED)
    gen = torch.Generator().manual_seed(SEED)
    obs = load_observations(dataset_path, 20000, rng)

    d1 = drift_online_vs_target(agent, obs[:2048], gen, std)

    z_cand = sample_z(agent, N_CANDIDATES, gen)
    half = obs.shape[0] // 2
    s_a, gram_a = replica_scores(agent, obs[:half], z_cand, gen, std, ridge_alpha, ridge_min, beta)
    s_b, _ = replica_scores(agent, obs[half:], z_cand, gen, std, ridge_alpha, ridge_min, beta)

    a, b = s_a.numpy(), s_b.numpy()
    pearson = float(np.corrcoef(a, b)[0, 1])
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    spearman = float(np.corrcoef(ra, rb)[0, 1])

    eig = torch.linalg.eigvalsh(gram_a).clamp(min=0)
    cond = float(eig.max() / (eig.min() + 1e-12))
    gram_eff_rank = float((eig.sum() ** 2 / (eig.pow(2).sum() + 1e-12)))

    return dict(
        label=label, step=step, drift=d1, pearson=pearson, spearman=spearman,
        score_cv=float(a.std() / (a.mean() + 1e-12)), gram_cond=cond,
        gram_eff_rank=gram_eff_rank, z_dim=agent._z_dimension,
    )


def main():
    targets = []
    for name, domain in [("walker", "walker"), ("quad", "quadruped"),
                         ("jaco", "jaco"), ("maze", "point_mass_maze")]:
        p = BASE / "checkpoints/fb_rnd_10k" / f"{name}.pickle"
        if p.exists():
            targets.append((p, BASE / f"datasets/{domain}/rnd/dataset.npz", f"fb_rnd_10k/{name}"))

    for p in sorted(glob.glob(str(BASE / "agents/fb/saved_models/checkpoints/walker_rnd_*/best.pickle"))):
        targets.append((Path(p), BASE / "datasets/walker/rnd/dataset.npz", Path(p).parent.name))

    if len(sys.argv) > 1:
        targets = [t for t in targets if any(a in str(t[0]) for a in sys.argv[1:])]

    rows = []
    for ckpt, ds, label in targets:
        if not ds.exists():
            print(f"skip {label}: no dataset at {ds}")
            continue
        try:
            rows.append(analyse(ckpt, ds, label))
            r = rows[-1]
            print(f"{r['label']:42s} step={r['step']:>7} drift={r['drift']:.4f} "
                  f"spearman={r['spearman']:+.3f} pearson={r['pearson']:+.3f} "
                  f"score_cv={r['score_cv']:.3f} gram_cond={r['gram_cond']:.3g}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{label}: FAILED {type(exc).__name__}: {exc}", flush=True)

    if rows:
        print("\n" + "=" * 100)
        print("D1 drift  = ||F_online - F_target|| / ||F_target||  (feature movement over the Gram's window)")
        print("spearman  = rank correlation of two independent replicas of refresh()'s scoring")
        print("            low => refresh() would pick different z's each time => selection is noise")
        rows.sort(key=lambda r: r["step"])
        print(f"\n{'checkpoint':42s} {'step':>8} {'drift':>8} {'spearman':>9} {'score_cv':>9}")
        for r in rows:
            print(f"{r['label']:42s} {r['step']:>8} {r['drift']:>8.4f} "
                  f"{r['spearman']:>+9.3f} {r['score_cv']:>9.3f}")


if __name__ == "__main__":
    main()
