"""Why is the tilt leverage score unreliable early in training?

The score of a candidate is computed from
    phi = 0.5*(F1_target + F2_target)(s, z, pi(s,z))
averaged over states_per_candidate=2 states drawn at random. So phi carries
two sources of variation:

    Var_z  variation across candidate z's holding the state fixed  -> SIGNAL
    Var_s  variation across states holding z fixed                 -> NOISE
           (k=2 states only averages this down by sqrt(2))

If Var_z << Var_s the candidates are indistinguishable except through which
states happened to be drawn, so refresh() ranks them by state-sampling noise
and picks essentially at random. This measures that ratio directly.

Usage: python analyze_tilt_z_signal.py
"""

import glob
import math
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

BASE = Path(__file__).parent
N_Z = 128
N_S = 128
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
        for s in agent.FB_optimizer.state.values() if "step" in s
    }
    return max(steps) if steps else -1


def load_observations(dataset_path, n_obs, rng):
    d = np.load(dataset_path, allow_pickle=True)
    keys = list(d.files)
    rng.shuffle(keys)
    obs = []
    for k in keys:
        ep = d[k].item()["observation"]
        obs.append(ep[rng.choice(len(ep), size=min(400, len(ep)), replace=False)])
        if sum(len(o) for o in obs) >= n_obs:
            break
    return torch.as_tensor(np.concatenate(obs)[:n_obs], dtype=torch.float32)


@torch.no_grad()
def z_vs_state_variance(agent, obs, std, gen):
    """Full N_S x N_Z grid of phi, then decompose its variance."""
    g = torch.randn(N_Z, agent._z_dimension, generator=gen)
    zs = math.sqrt(agent._z_dimension) * torch.nn.functional.normalize(g, dim=1)
    states = obs[torch.randint(0, obs.shape[0], (N_S,), generator=gen)]

    grid = []
    for i in range(N_S):
        s_rep = states[i : i + 1].expand(N_Z, -1)
        actions, _ = agent.actor(s_rep, zs, std, sample=False)
        f1, f2 = agent.FB.forward_representation_target(
            observation=s_rep, z=zs, action=actions
        )
        grid.append(0.5 * (f1 + f2))
    phi = torch.stack(grid)                       # [N_S, N_Z, D]
    phi = phi / (phi.pow(2).mean().sqrt() + 1e-12)  # same global RMS scaling as the Gram

    grand = phi.mean(dim=(0, 1))
    z_means = phi.mean(dim=0)                     # [N_Z, D]  average over states
    s_means = phi.mean(dim=1)                     # [N_S, D]  average over z

    var_z = (z_means - grand).pow(2).sum(dim=1).mean().item()
    var_s = (s_means - grand).pow(2).sum(dim=1).mean().item()
    total = (phi - grand).pow(2).sum(dim=2).mean().item()
    return var_z, var_s, total


def main():
    targets = []
    for name, domain in [("walker", "walker"), ("quad", "quadruped"),
                         ("jaco", "jaco"), ("maze", "point_mass_maze")]:
        p = BASE / "checkpoints/fb_rnd_10k" / f"{name}.pickle"
        if p.exists():
            targets.append((p, BASE / f"datasets/{domain}/rnd/dataset.npz", f"fb_rnd_10k/{name}"))
    for p in sorted(glob.glob(str(BASE / "agents/fb/saved_models/checkpoints/walker_rnd_*/best.pickle"))):
        targets.append((Path(p), BASE / "datasets/walker/rnd/dataset.npz", Path(p).parent.name))

    rows = []
    for ckpt, ds, label in targets:
        if not ds.exists():
            continue
        agent = force_cpu(torch.load(ckpt, map_location="cpu", weights_only=False))
        std = float(agent.std_dev_schedule) if isinstance(agent.std_dev_schedule, (int, float)) else 0.2
        rng = np.random.default_rng(SEED)
        gen = torch.Generator().manual_seed(SEED)
        obs = load_observations(ds, 4000, rng)
        var_z, var_s, total = z_vs_state_variance(agent, obs, std, gen)
        rows.append((label, opt_step(agent), var_z, var_s, var_z / (var_s + 1e-12),
                     var_z / (total + 1e-12)))
        print(f"{label:42s} step={rows[-1][1]:>7} var_z={var_z:.4g} var_s={var_s:.4g} "
              f"z/s={rows[-1][4]:.4f} z_share={rows[-1][5]*100:.2f}%", flush=True)

    rows.sort(key=lambda r: r[1])
    print("\n" + "=" * 92)
    print("var_z   = feature variance across candidate z's  (the signal tilt ranks on)")
    print("var_s   = feature variance across states         (the noise k=2 states barely averages out)")
    print(f"\n{'checkpoint':42s} {'step':>8} {'z/s ratio':>11} {'z share of var':>16}")
    for label, step, _, _, ratio, share in rows:
        print(f"{label:42s} {step:>8} {ratio:>11.4f} {share*100:>15.2f}%")


if __name__ == "__main__":
    main()
