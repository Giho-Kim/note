"""
Two-phase Basis Trajectory Distribution (BTD) training, self-contained (does
not depend on main_exorl.py or a pretrained checkpoint). Supports FB and
TD-JEPA (--algorithm fb / td_jepa), mirroring main_exorl.py's algorithm
selection.

Phase 1 -- representation training from scratch: standard joint objective
(OfflineRLWorkspace.train, same loop main_exorl.py uses), baseline
z ~ Unif(S^{d-1}).
  - fb:      trains F, B (agents/fb/agent.py).
  - td_jepa: trains phi, psi (metamotivo/agents/td_jepa/agent.py) -- psi
             plays FB's B role, phi plays FB's F role.
At the end, fits phi_btd(s) = (E[B(s)B(s)^T])^-1 @ B(s) (fb) or
(E[psi(s)psi(s)^T])^-1 @ psi(s) (td_jepa) from the trained backward-role
network, and a GMM over normalized discounted phi_btd(s) summaries of dataset
subtrajectories (build_btd_gmm). Phase 1's forward-role network is then
discarded.

Phase 2 -- policy training: the forward-role network (F / phi) is
reinitialized from scratch and retrained via an SF-style Bellman residual
against the fixed phi_btd(s) reward (agent.update_critic_btd), while the
actor is trained on z drawn from the Phase-1 GMM instead of Unif(S^{d-1})
(agent.update_actor). The backward-role network (B / psi) stays frozen
throughout Phase 2.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
import torch

from agents.fb.agent import FB
from agents.fb.btd import GMMZSampler, build_btd_gmm as build_btd_gmm_fb
from agents.fb.replay_buffer import FBReplayBuffer
from agents.td_jepa.agent import TDJEPA
from agents.td_jepa.btd import build_btd_gmm as build_btd_gmm_td_jepa
from agents.workspaces import OfflineRLWorkspace
from rewards import RewardFunctionConstructor
from utils import BASE_DIR, set_seed_everywhere

parser = ArgumentParser()
parser.add_argument("algorithm", type=str, choices=("fb", "td_jepa"))
parser.add_argument("domain_name", type=str)
parser.add_argument("exploration_algorithm", type=str)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--dataset_transitions", type=int, default=100000)

# --- Phase 1 hyperparameters (fb-only CLI overrides; td_jepa uses its
# config.yaml as-is, matching main_exorl.py's level of exposure for it) ---
parser.add_argument("--phase1_checkpoint_dir", type=str, default=None)
# If set, loads this checkpoint instead of running Phase 1 training at all
# (fb: a final.pickle from a previous --phase1_checkpoint_dir save; td_jepa:
# a directory saved the same way). Only Phase 1's GMM build + Phase 2 run.
parser.add_argument("--phase1_checkpoint_path", type=str, default=None)
parser.add_argument("--phase1_learning_steps", type=int, default=2000000)

# --- Phase 2 checkpointing (mirrors phase1_checkpoint_dir's semantics: a
# permanent local final-state copy, uploaded to wandb once at the end) plus
# a permanent local best-so-far checkpoint, also uploaded to wandb whenever
# it's replaced. Without these, Phase 2 only has a transient copy that's
# deleted at the end of train_btd -- with --wandb_logging False that means
# nothing from Phase 2 persists at all. ---
parser.add_argument("--phase2_checkpoint_dir", type=str, default=None)
parser.add_argument("--phase2_best_checkpoint_dir", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--critic_learning_rate", type=float, default=None)
parser.add_argument("--actor_learning_rate", type=float, default=None)
parser.add_argument("--tau", type=float, default=None)
parser.add_argument("--orthonormalisation_coefficient", type=float, default=None)
parser.add_argument("--discount", type=float, default=0.99)
# BTD-specific default. For FB this also controls Phase 1; in Phase 2 it is
# shared by FB and TD-JEPA and controls the fraction of goal-conditioned z's.
# The remaining Phase-2 z's come from the BTD GMM without tilt, or from the
# same tilted sphere sampling used by main_exorl.py when tilt is active.
parser.add_argument("--z_mix_ratio", type=float, default=0.0)
# Phase 1 only builds the representation for build_btd_gmm -- eval there is
# against tasks meant for the (not-yet-built) Phase 2 policy, so it's wasted
# cost by default. Pass --phase1_eval to turn it back on.
parser.add_argument("--phase1_eval", action="store_true")

# --- Phase 1 -> BTD GMM build ---
parser.add_argument("--n_subtrajectories", type=int, default=10000)
parser.add_argument("--subtraj_min_len", type=int, default=50)
parser.add_argument("--subtraj_max_len", type=int, default=100)
parser.add_argument("--gmm_components", type=int, default=20)
parser.add_argument("--btd_whitening_ridge", type=float, default=1e-6)
parser.add_argument("--novelty", action="store_true")

# --- Phase 2 (critic retrain + actor) hyperparameters ---
parser.add_argument("--btd_critic_learning_rate", type=float, default=1e-4)
parser.add_argument("--phase2_learning_steps", type=int, default=1000000)
parser.add_argument("--policy_freq", type=int, default=2)
parser.add_argument("--tasks_per_batch", type=int, default=32)
parser.add_argument("--transitions_per_task", type=int, default=64)
parser.add_argument("--resume_step", type=int, default=0)
phase2_forward_init_group = parser.add_mutually_exclusive_group()
phase2_forward_init_group.add_argument(
    "--reinit_phase2_forward",
    dest="reinit_phase2_forward",
    action="store_true",
    help="Reinitialize F/phi before Phase 2.",
)
phase2_forward_init_group.add_argument(
    "--keep_phase2_forward",
    dest="reinit_phase2_forward",
    action="store_false",
    help="Keep the Phase-1 F/phi weights for Phase 2 (default).",
)
phase2_actor_init_group = parser.add_mutually_exclusive_group()
phase2_actor_init_group.add_argument(
    "--reinit_phase2_actor",
    dest="reinit_phase2_actor",
    action="store_true",
    help="Reinitialize the actor before Phase 2.",
)
phase2_actor_init_group.add_argument(
    "--keep_phase2_actor",
    dest="reinit_phase2_actor",
    action="store_false",
    help="Keep the Phase-1 actor weights for Phase 2 (default).",
)
parser.set_defaults(reinit_phase2_forward=False, reinit_phase2_actor=False)

# --- Phase 2 tilt: the same sphere/goal candidate sampling as main_exorl.py.
# The BTD GMM remains the non-goal source only while tilt is inactive. ---
parser.add_argument("--tilt", action="store_true")
parser.add_argument("--tilt_goal", action="store_true")
parser.add_argument("--tilting_by_z", action="store_true")
parser.add_argument("--tilt_beta", type=float)
parser.add_argument("--tilt_temperature", type=float)
parser.add_argument("--tilt_temperature_start", type=float)
parser.add_argument("--tilt_temperature_end", type=float)
parser.add_argument("--tilt_candidate_multiplier", type=int)
parser.add_argument("--tilt_init_geom_ratio", type=float)
parser.add_argument("--tilt_ridge_alpha", type=float)
parser.add_argument("--tilt_ridge_min", type=float)
parser.add_argument("--tilt_start_step", type=int)
parser.add_argument("--tilt_refresh_interval", type=int)
parser.add_argument("--tilt_uniform_mix", type=float)
parser.add_argument("--tilt_linear", action="store_true")

# --- shared ---
parser.add_argument("--eval_frequency", type=int, default=20000)
parser.add_argument("--eval_rollouts", type=int, default=10)
parser.add_argument("--eval_std", type=str, default="0.05")
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--save_every", action="store_true")
args = parser.parse_args()

if args.wandb_logging == "True":
    args.wandb_logging = True
elif args.wandb_logging == "False":
    args.wandb_logging = False
else:
    raise ValueError("wandb_logging must be either True or False")

if not 0.0 <= args.z_mix_ratio <= 1.0:
    raise ValueError("z_mix_ratio must be between 0 and 1.")

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)

set_seed_everywhere(args.seed)

working_dir = Path.cwd()
model_dir = working_dir / "agents" / args.algorithm / "saved_models"
dataset_path = (
    BASE_DIR / "datasets" / args.domain_name / args.exploration_algorithm / "dataset.npz"
)

with open(working_dir / "agents" / args.algorithm / "config.yaml", "rb") as f:
    config = yaml.safe_load(f)

reward_constructor = RewardFunctionConstructor(
    domain_name=args.domain_name,
    task_names=args.eval_tasks,
    seed=args.seed,
    device=device,
)

if args.domain_name == "jaco":
    observation_length = reward_constructor._env.observation_spec().shape[0]  # pylint: disable=protected-access
    action_length = reward_constructor._env.action_spec().shape[0]  # pylint: disable=protected-access
else:
    observation_length = reward_constructor._env.observation_spec()["observations"].shape[0]  # pylint: disable=protected-access
    action_length = reward_constructor._env.action_spec().shape[0]  # pylint: disable=protected-access

if args.algorithm == "fb":
    # CLI overrides for the Phase 1 hyperparameters config.yaml doesn't
    # otherwise expose (mirrors the --arg-if-not-None pattern main_exorl.py
    # uses for its tilt overrides).
    for key in (
        "batch_size",
        "critic_learning_rate",
        "actor_learning_rate",
        "tau",
        "orthonormalisation_coefficient",
        "discount",
        "z_mix_ratio",
    ):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    # matches main_exorl.py's point_mass_maze special-case (applied last, so
    # it wins even over an explicit --discount).
    if args.domain_name == "point_mass_maze":
        config["discount"] = 0.99
        config["z_dimension"] = 100

    agent = FB(
        observation_length=observation_length,
        action_length=action_length,
        preprocessor_hidden_dimension=config["preprocessor_hidden_dimension"],
        preprocessor_output_dimension=config["preprocessor_output_dimension"],
        preprocessor_hidden_layers=config["preprocessor_hidden_layers"],
        forward_hidden_dimension=config["forward_hidden_dimension"],
        forward_hidden_layers=config["forward_hidden_layers"],
        forward_number_of_features=config["forward_number_of_features"],
        backward_hidden_dimension=config["backward_hidden_dimension"],
        backward_hidden_layers=config["backward_hidden_layers"],
        actor_hidden_dimension=config["actor_hidden_dimension"],
        actor_hidden_layers=config["actor_hidden_layers"],
        preprocessor_activation=config["preprocessor_activation"],
        forward_activation=config["forward_activation"],
        backward_activation=config["backward_activation"],
        actor_activation=config["actor_activation"],
        z_dimension=config["z_dimension"],
        critic_learning_rate=config["critic_learning_rate"],
        actor_learning_rate=config["actor_learning_rate"],
        learning_rate_coefficient=config["learning_rate_coefficient"],
        orthonormalisation_coefficient=config["orthonormalisation_coefficient"],
        discount=config["discount"],
        batch_size=config["batch_size"],
        z_mix_ratio=config["z_mix_ratio"],
        gaussian_actor=config["gaussian_actor"],
        std_dev_clip=config["std_dev_clip"],
        std_dev_schedule=config["std_dev_schedule"],
        tau=config["tau"],
        learning_steps=args.phase1_learning_steps,
        tilt=False,
        tilt_start_step=0,
        tilting_by_z=False,
        tilt_beta=config["tilt_beta"],
        tilt_temperature=config["tilt_temperature"],
        tilt_temperature_start=config["tilt_temperature_start"],
        tilt_temperature_end=config["tilt_temperature_end"],
        tilt_candidate_multiplier=config["tilt_candidate_multiplier"],
        tilt_init_geom_ratio=None,
        tilt_ridge_alpha=config["tilt_ridge_alpha"],
        tilt_ridge_min=config["tilt_ridge_min"],
        device=device,
        name="fb",
    )
    if args.phase1_checkpoint_path is not None:
        print(f"Loading Phase 1 checkpoint from {args.phase1_checkpoint_path}, skipping Phase 1 training...")
        agent = torch.load(args.phase1_checkpoint_path, map_location=device, weights_only=False)
        agent.to(device)
        agent._device = device  # pylint: disable=protected-access
    discount_for_replay_buffer = agent.FB._discount  # pylint: disable=protected-access
    train_std = config["std_dev_schedule"]

else:  # td_jepa
    if args.domain_name == "point_mass_maze":
        config["discount"] = 0.99

    # agents/td_jepa/config.yaml doesn't define these (main_exorl.py fills
    # the same defaults in for the td_jepa branch there).
    config.setdefault("tilt_init_geom_ratio", None)
    config.setdefault("tilt_start_step", 0)
    config.setdefault("tilt_refresh_interval", 1)

    agent = TDJEPA(
        observation_length=observation_length,
        action_length=action_length,
        device=device,
        name="td_jepa",
        batch_size=config["batch_size"],
        discount=config["discount"],
        lr_predictor=config["lr_predictor"],
        lr_phi=config["lr_phi"],
        lr_psi=config["lr_psi"],
        lr_actor=config["lr_actor"],
        weight_decay=config["weight_decay"],
        encoder_target_tau=config["encoder_target_tau"],
        predictor_target_tau=config["predictor_target_tau"],
        phi_ortho_coef=config["phi_ortho_coef"],
        psi_ortho_coef=config["psi_ortho_coef"],
        train_goal_ratio=config["train_goal_ratio"],
        predictor_pessimism_penalty=config["predictor_pessimism_penalty"],
        actor_pessimism_penalty=config["actor_pessimism_penalty"],
        stddev_clip=config["stddev_clip"],
        bc_coeff=config["bc_coeff"],
        log_eigvals=config["log_eigvals"],
        scale_train_goals=config["scale_train_goals"],
        learning_steps=args.phase1_learning_steps,
        tilt=False,
        tilting_by_z=False,
        tilt_beta=config["tilt_beta"],
        tilt_temperature=config["tilt_temperature"],
        tilt_temperature_start=config["tilt_temperature_start"],
        tilt_temperature_end=config["tilt_temperature_end"],
        tilt_candidate_multiplier=config["tilt_candidate_multiplier"],
        tilt_init_geom_ratio=config["tilt_init_geom_ratio"],
        tilt_ridge_alpha=config["tilt_ridge_alpha"],
        tilt_ridge_min=config["tilt_ridge_min"],
        tilt_start_step=config["tilt_start_step"],
        tilt_goal=False,
        tilt_refresh_interval=config["tilt_refresh_interval"],
        actor_std=config["actor_std"],
        actor_use_full_encoder=config["actor_use_full_encoder"],
        symmetric=config["symmetric"],
        compile=config["compile"],
        phi_dim=config["phi_dim"],
        psi_dim=config["psi_dim"],
        norm_z=config["norm_z"],
        rgb_encoder_name=config["rgb_encoder_name"],
        augmentator_name=config["augmentator_name"],
        phi_predictor_hidden_dim=config["phi_predictor_hidden_dim"],
        phi_predictor_hidden_layers=config["phi_predictor_hidden_layers"],
        phi_predictor_embedding_layers=config["phi_predictor_embedding_layers"],
        phi_predictor_num_parallel=config["phi_predictor_num_parallel"],
        psi_predictor_hidden_dim=config["psi_predictor_hidden_dim"],
        psi_predictor_hidden_layers=config["psi_predictor_hidden_layers"],
        psi_predictor_embedding_layers=config["psi_predictor_embedding_layers"],
        psi_predictor_num_parallel=config["psi_predictor_num_parallel"],
        phi_mlp_hidden_dim=config["phi_mlp_hidden_dim"],
        phi_mlp_hidden_layers=config["phi_mlp_hidden_layers"],
        phi_mlp_norm=config["phi_mlp_norm"],
        psi_mlp_hidden_dim=config["psi_mlp_hidden_dim"],
        psi_mlp_hidden_layers=config["psi_mlp_hidden_layers"],
        psi_mlp_norm=config["psi_mlp_norm"],
        actor_hidden_dim=config["actor_hidden_dim"],
        actor_hidden_layers=config["actor_hidden_layers"],
        actor_embedding_layers=config["actor_embedding_layers"],
    )
    if agent.agent.cfg.model.symmetric:
        raise ValueError(
            "BTD requires an asymmetric TD-JEPA model (separate phi/psi "
            "networks) -- psi must be a distinct, freezable network. Set "
            "symmetric: false in agents/td_jepa/config.yaml."
        )
    if args.phase1_checkpoint_path is not None:
        print(f"Loading Phase 1 checkpoint from {args.phase1_checkpoint_path}, skipping Phase 1 training...")
        agent.load(args.phase1_checkpoint_path)
    discount_for_replay_buffer = agent.agent.cfg.train.discount
    train_std = None

replay_buffer = FBReplayBuffer(
    reward_constructor=reward_constructor,
    dataset_path=dataset_path,
    transitions=args.dataset_transitions,
    relabel=False,
    task=None,
    device=device,
    discount=discount_for_replay_buffer,
    action_condition=None,
)

workspace = OfflineRLWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=args.phase1_learning_steps,
    model_dir=model_dir,
    eval_frequency=args.eval_frequency,
    eval_rollouts=args.eval_rollouts,
    z_inference_steps=args.z_inference_steps,
    train_std=train_std,
    eval_std=args.eval_std,
    wandb_logging=args.wandb_logging,
    device=device,
    collection_interval=0,
    collection_episodes=0,
    verbose=args.verbose,
    save_every=args.save_every,
)

if __name__ == "__main__":

    agent_config = vars(args).copy()
    agent_config["algorithm"] = args.algorithm

    # --- Phase 1: joint representation training from scratch (forward-role +
    # backward-role network + actor; baseline z ~ Unif(S^{d-1}), no tilt) ---
    # extra_checkpoint_dir gives a permanent local copy of the FINAL agent
    # state (before Phase 2 discards the forward-role network) and uploads it
    # to this Phase 1 wandb run before that run closes -- the backward-role
    # network must not exist only in this process's memory.
    if args.phase1_checkpoint_path is None:
        phase1_checkpoint_dir = Path(
            args.phase1_checkpoint_dir
            or model_dir / "btd_phase1" / f"{args.domain_name}_{args.exploration_algorithm}_seed{args.seed}"
        )
        print(f"Phase 1 ({args.algorithm}): training from scratch for {args.phase1_learning_steps} steps...")
        workspace.train(
            agent=agent,
            tasks=args.eval_tasks,
            agent_config=agent_config,
            replay_buffer=replay_buffer,
            start_step=0,
            extra_checkpoint_dir=phase1_checkpoint_dir,
            eval_enabled=args.phase1_eval,
        )
    # train()'s own checkpointing loop (best-eval AND the final-state save
    # above) overwrites agent._name -- reset it before it's reused as the
    # wandb tag for Phase 2's own wandb.init(tags=[agent.name, "btd"]), and
    # capture the trained std schedule before eval() overwrites it with
    # eval_std (see OfflineRLWorkspace.eval; a no-op for td_jepa, which has
    # no std_dev_schedule attribute).
    agent._name = f"{args.algorithm}-btd"  # pylint: disable=protected-access
    if hasattr(agent, "std_dev_schedule"):
        train_std = agent.std_dev_schedule
    agent.train()

    # --- Phase 1 -> BTD GMM: fit phi_btd(s) (fb: whitened B(s); td_jepa:
    # L2-normalized psi(s), no whitening) + GMM from the just-trained
    # representation, then discard the forward-role network ---
    print(f"Building BTD GMM from {args.n_subtrajectories} subtrajectories...")
    if args.algorithm == "fb":
        gmm, whitening_matrix = build_btd_gmm_fb(
            agent=agent,
            dataset_path=dataset_path,
            n_subtrajectories=args.n_subtrajectories,
            min_len=args.subtraj_min_len,
            max_len=args.subtraj_max_len,
            gmm_components=args.gmm_components,
            whitening_ridge=args.btd_whitening_ridge,
            seed=args.seed,
            whitening_observations=replay_buffer.storage["observations"],
            dataset_transitions=args.dataset_transitions,
        )
    else:
        gmm, whitening_matrix = build_btd_gmm_td_jepa(
            agent=agent,
            dataset_path=dataset_path,
            n_subtrajectories=args.n_subtrajectories,
            min_len=args.subtraj_min_len,
            max_len=args.subtraj_max_len,
            gmm_components=args.gmm_components,
            seed=args.seed,
            dataset_transitions=args.dataset_transitions,
        )
    z_sampler = GMMZSampler(
        gmm=gmm, z_dimension=agent._z_dimension, device=device  # pylint: disable=protected-access
    )

    # --- Phase 2 setup: optionally reinitialize the forward-role network and
    # actor, then freeze the backward-role network (the only one phi_btd(s)
    # needs) so it is never touched again. Optimizers are recreated even when
    # weights are kept, so the Phase-2 learning rates always take effect. ---
    agent.reinit_forward_representation(
        learning_rate=args.btd_critic_learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        reinitialize_forward=args.reinit_phase2_forward,
        reinitialize_actor=args.reinit_phase2_actor,
    )
    print(
        "Phase 2 initialization: "
        f"forward={'reinitialized' if args.reinit_phase2_forward else 'kept'}, "
        f"actor={'reinitialized' if args.reinit_phase2_actor else 'kept'}"
    )
    if args.algorithm == "fb":
        frozen_modules = (agent.FB.backward_representation, agent.FB.backward_representation_target)
    else:
        frozen_modules = (
            agent.agent._model._psi_rgb_encoder,  # pylint: disable=protected-access
            agent.agent._model._psi_mlp_encoder,  # pylint: disable=protected-access
            agent.agent._model._target_psi_mlp_encoder,  # pylint: disable=protected-access
        )
    for frozen_module in frozen_modules:
        for param in frozen_module.parameters():
            param.requires_grad_(False)

    workspace.learning_steps = args.phase2_learning_steps
    workspace.train_std = train_std
    # _tilt_temperature(step) anneals over the Phase 1 step count captured at
    # construction; Phase 2 needs its own count.
    if args.algorithm == "fb":
        agent._learning_steps = max(1, args.phase2_learning_steps)  # pylint: disable=protected-access
    else:
        agent.agent.cfg = agent.agent.cfg.model_copy(
            update={"train": agent.agent.cfg.train.model_copy(
                update={"learning_steps": max(1, args.phase2_learning_steps)}
            )}
        )

    if args.tilt:
        tilt_temperature_start = args.tilt_temperature_start
        tilt_temperature_end = args.tilt_temperature_end
        if args.tilt_temperature is not None:
            if tilt_temperature_start is None:
                tilt_temperature_start = args.tilt_temperature
            if tilt_temperature_end is None:
                tilt_temperature_end = args.tilt_temperature
        agent.enable_tilt(
            tilt_beta=args.tilt_beta if args.tilt_beta is not None else config["tilt_beta"],
            tilt_temperature=(
                args.tilt_temperature if args.tilt_temperature is not None else config["tilt_temperature"]
            ),
            tilt_temperature_start=(
                tilt_temperature_start if tilt_temperature_start is not None else config["tilt_temperature_start"]
            ),
            tilt_temperature_end=(
                tilt_temperature_end if tilt_temperature_end is not None else config["tilt_temperature_end"]
            ),
            tilt_candidate_multiplier=(
                args.tilt_candidate_multiplier
                if args.tilt_candidate_multiplier is not None
                else config["tilt_candidate_multiplier"]
            ),
            tilt_init_geom_ratio=args.tilt_init_geom_ratio,
            tilt_ridge_alpha=(
                args.tilt_ridge_alpha if args.tilt_ridge_alpha is not None else config["tilt_ridge_alpha"]
            ),
            tilt_ridge_min=(
                args.tilt_ridge_min if args.tilt_ridge_min is not None else config["tilt_ridge_min"]
            ),
            tilt_start_step=args.tilt_start_step if args.tilt_start_step is not None else 0,
            tilting_by_z=args.tilting_by_z,
            tilt_goal=args.tilt_goal,
            tilt_refresh_interval=(
                args.tilt_refresh_interval if args.tilt_refresh_interval is not None else 1
            ),
            tilt_uniform_mix=(
                args.tilt_uniform_mix if args.tilt_uniform_mix is not None else 0.5
            ),
            tilt_linear=args.tilt_linear,
        )

    phase2_checkpoint_dir = Path(
        args.phase2_checkpoint_dir
        or model_dir / "btd_phase2" / f"{args.domain_name}_{args.exploration_algorithm}_seed{args.seed}"
    )
    phase2_best_checkpoint_dir = Path(
        args.phase2_best_checkpoint_dir
        or model_dir / "btd_phase2_best" / f"{args.domain_name}_{args.exploration_algorithm}_seed{args.seed}"
    )

    print(f"Phase 2: critic (forward-role) + actor training for {args.phase2_learning_steps} steps...")
    workspace.train_btd(
        agent=agent,
        tasks=args.eval_tasks,
        agent_config=agent_config,
        replay_buffer=replay_buffer,
        z_sampler=z_sampler,
        tasks_per_batch=args.tasks_per_batch,
        transitions_per_task=args.transitions_per_task,
        start_step=args.resume_step,
        novelty_weighted=args.novelty,
        whitening_matrix=whitening_matrix,
        z_mix_ratio=args.z_mix_ratio,
        policy_freq=args.policy_freq,
        extra_checkpoint_dir=phase2_checkpoint_dir,
        best_checkpoint_dir=phase2_best_checkpoint_dir,
    )
