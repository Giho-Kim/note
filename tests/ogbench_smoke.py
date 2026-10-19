"""Small integration check: real environment, raw conversion, training/eval/resume.

Run with .venv-ogbench/bin/python -m tests.ogbench_smoke.
Uses temporary data and tiny networks; does not download the full benchmark.
"""

import copy
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import pandas as pd
import torch

from main_ogbench import build_config
from ogbench_impl.envs.ogbench import OGBenchEnvConfig
from ogbench_impl.extract import extract
from ogbench_impl.train import TrainConfig
from ogbench_impl.evaluations.ogbench import summarize_returns
from metamotivo.agents.tilt import weighted_select


def run(device="cpu", tilt=False, compile=False):
    torch.set_num_threads(1)
    summary = summarize_returns([0, 1, 2, 3, 4, 5, 6, 100], [False] * 8)
    assert summary["reward_mean"] == 15.125 and summary["reward_iqm"] == 3.5
    if tilt:
        idx, pmin, pmax = weighted_select(torch.zeros(4, device=device), 1.0, 4, linear=True)
        assert len(idx.unique()) == 4 and pmin == pmax == 0.25
    domain = "antmaze-medium-navigate-v0"
    with tempfile.TemporaryDirectory(prefix="note-ogbench-smoke-") as tmp:
        root = Path(tmp)
        env, _ = OGBenchEnvConfig(
            domain=domain, task="antmaze-medium-navigate-singletask-task1-v0"
        ).build()
        observations, actions, physics, terminals = [], [], [], []
        try:
            for seed in (1, 2):
                obs, _ = env.reset(seed=seed)
                for i in range(17):
                    action = np.full(env.action_space.shape, 0.05, dtype=np.float32)
                    observations.append(obs.copy())
                    actions.append(action.copy())
                    physics.append(env.unwrapped.data.qpos.copy())
                    terminals.append(i == 16)
                    obs, _, terminated, truncated, _ = env.step(action)
                    assert not (terminated or truncated)
        finally:
            env.close()
        np.savez(
            root / f"{domain}.npz", observations=np.asarray(observations),
            actions=np.asarray(actions), qpos=np.asarray(physics),
            terminals=np.asarray(terminals),
        )
        extract(domain, str(root / "data"), str(root))
        np.savez(
            root / f"visual-{domain}.npz",
            observations=np.random.default_rng(1).integers(0, 256, (len(observations), 64, 64, 3), dtype=np.uint8),
            actions=np.asarray(actions), qpos=np.asarray(physics), terminals=np.asarray(terminals),
        )
        extract(f"visual-{domain}", str(root / "data"), str(root))
        episode = np.load(next((root / "data" / domain / "buffer").glob("*.npz")))
        assert len(episode["observation"]) == 17
        np.testing.assert_array_equal(episode["action"][1:], np.asarray(actions)[:16])

        for algorithm in ("fb", "td_jepa"):
            for obs_type in ("state", "pixels"):
                args = SimpleNamespace(
                    algorithm=algorithm, domain=domain, obs_type=obs_type,
                    data_path=root / "data", seed=3917, device=device, compile=compile,
                    batch_size=4, learning_steps=6 if tilt else 2, work_dir=root / f"{algorithm}_{obs_type}",
                    buffer_device="cpu", eval_frequency=100, checkpoint_frequency=100,
                    log_frequency=1, eval_rollouts=1, inference_samples=8,
                    load_n_episodes=2, num_workers=0, no_eval=obs_type == "pixels",
                    wandb=False, wandb_project="note-ogbench", wandb_group=None,
                    bc_coeff=None, ortho_coef=None, lr_backward=None,
                    tilt=True if tilt else None, tilt_start_step=1 if tilt else None,
                    tilt_refresh_interval=2 if tilt else None,
                    tilt_candidate_multiplier=4 if tilt else None,
                    verbose=tilt,
                )
                cfg = build_config(args)
                TrainConfig(**cfg)  # Both state/pixel launcher configs validate.
                arch = cfg["agent"]["model"]["archi"]
                for value in arch.values():
                    if isinstance(value, dict) and "hidden_dim" in value:
                        value["hidden_dim"] = 16
                for key in ("L_dim", "z_dim", "phi_dim", "psi_dim"):
                    if key in arch:
                        arch[key] = 4
                # One real task, one episode, real reward inference and success logging.
                if cfg["evaluations"]:
                    cfg["evaluations"][0]["tasks"] = cfg["evaluations"][0]["tasks"][:1]
                workspace = TrainConfig(**cfg).build()
                workspace.train()
                checkpoint = args.work_dir / "checkpoint"
                assert (checkpoint / "model" / "model.safetensors").is_file()
                assert json.loads((checkpoint / "train_status.json").read_text())["time"] == args.learning_steps
                if tilt:
                    assert (checkpoint / "tilt.pth").is_file()
                    selector = workspace.agent.tilt
                    assert selector._feat_ms_seeded and selector._refresh_count > 0
                    assert torch.isfinite(selector.gram).all()
                    assert selector._candidate_z is not None
                    tilt_state = copy.deepcopy(vars(selector))
                    # Coverage of boundary mixtures, both scoring bases and weighting modes.
                    sample = torch.zeros((4, *workspace.agent.obs_space.shape), device=device)
                    model = workspace.agent._model
                    with torch.no_grad():
                        inputs = (model._fw_encoder if algorithm == "fb" else model._phi_rgb_encoder)(model._normalize(sample))
                        goal_inputs = (model._bw_encoder if algorithm == "fb" else model._psi_rgb_encoder)(model._normalize(sample))
                    zs = model.sample_z(4, device=device)
                    score1, feature1 = workspace.agent.score_and_features(inputs, zs)
                    score2, feature2 = workspace.agent.score_and_features(inputs, zs)
                    assert torch.equal(score1, score2) and torch.equal(feature1, feature2)
                    for ratio, by_z, linear, goal_tilt in ((0., False, False, False), (1., True, True, True), (0.5, True, False, False)):
                        probe_cfg = workspace.agent.cfg.model_copy(update={"compile": False, "train": workspace.agent.cfg.train.model_copy(update={
                            "train_goal_ratio": ratio, "tilting_by_z": by_z,
                            "tilt_linear": linear, "tilt_goal": goal_tilt, "tilt_start_step": 0,
                        })})
                        probe = probe_cfg.build(workspace.agent.obs_space, workspace.agent.action_dim)
                        for step in (0, 1, 2):
                            z_batch = probe.sample_tilt_batch(goal_inputs, inputs, torch.arange(4), step)
                            assert z_batch.shape == (4, 4) and torch.isfinite(z_batch).all()
                        assert probe.tilt._feat_ms_seeded
                if obs_type == "state":
                    assert (args.work_dir / "reward.csv").is_file()
                    summary_rows = pd.read_csv(args.work_dir / "reward_summary.csv")
                    assert len(summary_rows) == 4
                    assert "overall" in set(summary_rows["task"])
                    assert set(summary_rows["timestep"]) == {0, args.learning_steps}
                    assert summary_rows[["reward_mean", "reward_iqm"]].notna().all().all()
                params = copy.deepcopy(workspace.agent._model.state_dict())
                cfg["num_train_steps"] = args.learning_steps + 1
                resumed = TrainConfig(**cfg).build()
                assert resumed._checkpoint_time == args.learning_steps
                for name, tensor in params.items():
                    assert torch.equal(tensor, resumed.agent._model.state_dict()[name]), name
                if tilt:
                    for key, value in tilt_state.items():
                        restored = vars(resumed.agent.tilt)[key]
                        if isinstance(value, torch.Tensor):
                            assert torch.equal(value, restored), key
                    assert resumed.agent.tilt._refresh_count == tilt_state["_refresh_count"]
                resumed.train()
                assert json.loads((checkpoint / "train_status.json").read_text())["time"] == args.learning_steps + 1
                if tilt:
                    cfg["tilt_resume_overrides"] = {"tilt": False}
                    disabled = TrainConfig(**cfg).build()
                    assert disabled.agent.tilt is None
                    cfg["tilt_resume_overrides"] = {"tilt": True, "tilting_by_z": True, "lr_actor": 7e-5}
                    changed = TrainConfig(**cfg).build()
                    assert changed.agent.tilt is not None and changed.agent.tilt._refresh_count == 0
                    assert all(group["lr"] == 7e-5 for group in changed.agent.actor_optimizer.param_groups)
                print(f"PASS {algorithm}/{obs_type}: conversion, update, checkpoint and resume")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tilt", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    run(args.device, args.tilt, args.compile)
