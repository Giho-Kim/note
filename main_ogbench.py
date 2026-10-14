"""OGBench download and offline training using the upstream flow-BC agents."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("algorithm", choices=("fb", "td_jepa"))
    parser.add_argument("domain", help="e.g. antmaze-medium-navigate-v0 or cube-single-play-v0")
    parser.add_argument("--obs-type", choices=("state", "pixels"), default="state")
    parser.add_argument("--data-path", type=Path, default=ROOT / "datasets" / "ogbench")
    parser.add_argument("--dataset-path", type=Path, help="Directory of original .npz datasets; otherwise download with OGBench")
    parser.add_argument("--download", action="store_true", help="Download/convert the selected dataset before training")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--seed", type=int, default=3917)
    parser.add_argument("--learning-steps", "--learning_steps", type=int, default=1_000_000)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=256)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--buffer-device", choices=("cpu", "cuda"))
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-frequency", "--eval_frequency", type=int, default=250_000)
    parser.add_argument("--eval-rollouts", "--eval_rollouts", type=int, default=10)
    parser.add_argument("--inference-samples", "--z_inference_steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-frequency", type=int, default=250_000)
    parser.add_argument("--log-frequency", type=int, default=10_000)
    parser.add_argument("--load-n-episodes", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--bc-coeff", type=float)
    parser.add_argument("--ortho-coef", type=float)
    parser.add_argument("--lr-backward", type=float)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="note-ogbench")
    parser.add_argument("--wandb-group")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved config without importing dependencies or downloading")
    parser.add_argument("--tilt", dest="tilt", action="store_true", default=None)
    parser.add_argument("--no-tilt", dest="tilt", action="store_false")
    parser.add_argument("--tilting_by_z", "--tilting-by-z", action="store_true", default=None)
    for key, dtype in (
        ("tilt_beta", float), ("tilt_temperature", float),
        ("tilt_temperature_start", float), ("tilt_temperature_end", float),
        ("tilt_candidate_multiplier", int), ("tilt_init_geom_ratio", float),
        ("tilt_ridge_alpha", float), ("tilt_ridge_min", float),
        ("tilt_start_step", int), ("tilt_refresh_interval", int),
        ("tilt_uniform_mix", float),
    ):
        parser.add_argument("--" + key, "--" + key.replace("_", "-"), type=dtype)
    for key in ("tilt_goal", "tilt_linear", "tilt_free_compete"):
        parser.add_argument("--" + key, "--" + key.replace("_", "-"), dest=key, action="store_true", default=None)
        parser.add_argument("--no-" + key.replace("_", "-"), dest=key, action="store_false")
    parser.add_argument("--z_mix_ratio", "--z-mix-ratio", "--train-goal-ratio", type=float)
    parser.add_argument("--actor_learning_rate", "--actor-learning-rate", type=float)
    parser.add_argument("--verbose", action="store_true", help="Print tilt health metrics at each training log interval")
    args = parser.parse_args()
    for key in ("batch_size", "eval_frequency", "eval_rollouts", "inference_samples", "checkpoint_frequency", "log_frequency", "load_n_episodes"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.learning_steps < 0 or args.num_workers < 0:
        parser.error("learning steps and number of workers must be non-negative")
    return args


def build_config(args):
    mode = "proprio" if args.obs_type == "state" else "pixel"
    with (ROOT / "configs" / "ogbench" / f"{args.algorithm}_{mode}.json").open() as f:
        cfg = json.load(f)
    # Select the first regularization/lr combination in the upstream sweep.
    antmaze = args.domain.startswith("antmaze-")
    cfg["agent"]["train"]["bc_coeff"] = args.bc_coeff if args.bc_coeff is not None else (0.3 if antmaze else 3.0)
    ortho = args.ortho_coef if args.ortho_coef is not None else (100 if args.algorithm == "fb" else (0.1 if antmaze else 1.0))
    cfg["agent"]["train"]["ortho_coef" if args.algorithm == "fb" else "psi_ortho_coef"] = float(ortho)
    cfg["agent"]["train"]["lr_b" if args.algorithm == "fb" else "lr_psi"] = args.lr_backward if args.lr_backward is not None else 1e-4
    tasks = [args.domain[:-2] + f"singletask-task{i}-" + args.domain[-2:] for i in range(1, 6)]
    cfg["data"].update(domain=args.domain, dataset_root=str(args.data_path.resolve()), load_n_episodes=args.load_n_episodes, num_workers=args.num_workers)
    cfg["env"].update(domain=args.domain, task=tasks[0], seed=args.seed)
    cfg["agent"]["model"]["device"] = args.device
    cfg["agent"]["compile"] = args.compile
    cfg["agent"]["train"]["batch_size"] = args.batch_size
    tilt_enabled = bool(getattr(args, "tilt", False))
    tilt_defaults = {
        "tilt": tilt_enabled, "tilting_by_z": False,
        "tilt_beta": 0.995, "tilt_temperature": 20.0,
        "tilt_temperature_start": 20.0, "tilt_temperature_end": 1.0,
        "tilt_candidate_multiplier": 10, "tilt_init_geom_ratio": 0.99 if tilt_enabled else None,
        "tilt_ridge_alpha": 1e-2, "tilt_ridge_min": 1e-8,
        "tilt_start_step": 200_000 if tilt_enabled else 0, "tilt_refresh_interval": 5,
        "tilt_uniform_mix": 0.1, "tilt_goal": tilt_enabled,
        "tilt_linear": tilt_enabled, "tilt_free_compete": tilt_enabled,
    }
    explicit = {}
    for key, default in tilt_defaults.items():
        value = getattr(args, key, None)
        tilt_defaults[key] = default if value is None else value
        if value is not None:
            explicit[key] = value
    temperature = getattr(args, "tilt_temperature", None)
    if temperature is not None and all(getattr(args, key, None) is None for key in ("tilt_temperature_start", "tilt_temperature_end")):
        for key in ("tilt_temperature_start", "tilt_temperature_end"):
            tilt_defaults[key] = explicit[key] = temperature
    if getattr(args, "tilt", None) is not None:
        # Explicitly enabling/disabling tilt on resume applies this invocation's settings.
        explicit.update(tilt_defaults)
    cfg["agent"]["train"].update(tilt_defaults, learning_steps=args.learning_steps)
    ratio = getattr(args, "z_mix_ratio", None)
    if ratio is not None:
        if not 0 <= ratio <= 1:
            raise ValueError("z_mix_ratio must be between 0 and 1")
        cfg["agent"]["train"]["train_goal_ratio"] = explicit["train_goal_ratio"] = ratio
    actor_lr = getattr(args, "actor_learning_rate", None)
    if actor_lr is not None or tilt_enabled:
        cfg["agent"]["train"]["lr_actor"] = actor_lr if actor_lr is not None else 3e-5
        explicit["lr_actor"] = cfg["agent"]["train"]["lr_actor"]
    cfg.update(
        seed=args.seed, num_train_steps=args.learning_steps,
        work_dir=str((args.work_dir or ROOT / "results" / "ogbench" / f"{args.algorithm}_{args.domain}_{args.obs_type}_seed{args.seed}{'_tilt' if tilt_enabled else ''}").resolve()),
        buffer_device=args.buffer_device, eval_every_steps=args.eval_frequency,
        checkpoint_every_steps=args.checkpoint_frequency, log_every_updates=args.log_frequency,
        use_wandb=args.wandb, wandb_pname=args.wandb_project, wandb_gname=args.wandb_group,
        tilt_resume_overrides=explicit, verbose=getattr(args, "verbose", False),
        evaluations=[] if args.no_eval else [{
            "name": "ogbench_reward_eval", "shift_reward": 1.0,
            "env": dict(cfg["env"]), "tasks": tasks,
            "num_episodes": args.eval_rollouts, "num_inference_samples": args.inference_samples,
        }],
    )
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    os.environ.setdefault("MUJOCO_GL", "egl")
    from ogbench_impl.train import TrainConfig

    config = TrainConfig(**cfg)
    if args.download or args.download_only:
        from ogbench_impl.extract import extract

        domain = ("visual-" if args.obs_type == "pixels" else "") + args.domain
        extract(domain, str(args.data_path), None if args.dataset_path is None else str(args.dataset_path))
    if args.download_only:
        return
    config.build().train()


if __name__ == "__main__":
    main()
