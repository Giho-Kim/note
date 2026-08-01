"""
Two-phase Basis Trajectory Distribution (BTD) training, self-contained (does
not depend on main_exorl.py or a pretrained checkpoint):

Phase 1 -- FB training (F, B) from scratch: standard FB objective
(OfflineRLWorkspace.train, same loop main_exorl.py uses), baseline
z ~ Unif(S^{d-1}). At the end, fits phi(s) = (E[B(s)B(s)^T])^-1 @ B(s) from
the trained B and a GMM over normalized discounted phi(s) summaries of
dataset subtrajectories (build_btd_gmm). Phase 1's F is then discarded.

Phase 2 -- policy training: F is reinitialized from scratch and retrained via
an SF-style Bellman residual against the fixed phi(s) reward
(agent.update_critic_btd), while the actor is trained on z drawn from the
Phase-1 GMM instead of Unif(S^{d-1}) (agent.update_actor, unchanged). B stays
frozen throughout Phase 2.
"""

from argparse import ArgumentParser
from pathlib import Path

import yaml
import torch

from agents.fb.agent import FB
from agents.fb.btd import GMMZSampler, build_btd_gmm
from agents.fb.replay_buffer import FBReplayBuffer
from agents.workspaces import OfflineRLWorkspace
from rewards import RewardFunctionConstructor
from utils import BASE_DIR, set_seed_everywhere

parser = ArgumentParser()
parser.add_argument("domain_name", type=str)
parser.add_argument("exploration_algorithm", type=str)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--dataset_transitions", type=int, default=100000)

# --- Phase 1 (FB training) hyperparameters ---
parser.add_argument("--phase1_checkpoint_dir", type=str, default=None)
parser.add_argument("--phase1_learning_steps", type=int, default=2000000)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--critic_learning_rate", type=float, default=None)
parser.add_argument("--actor_learning_rate", type=float, default=None)
parser.add_argument("--tau", type=float, default=None)
parser.add_argument("--orthonormalisation_coefficient", type=float, default=None)
parser.add_argument("--discount", type=float, default=0.99)

# --- Phase 1 -> BTD GMM build ---
parser.add_argument("--n_subtrajectories", type=int, default=10000)
parser.add_argument("--subtraj_min_len", type=int, default=5)
parser.add_argument("--subtraj_max_len", type=int, default=50)
parser.add_argument("--gmm_components", type=int, default=20)
parser.add_argument("--btd_whitening_ridge", type=float, default=1e-6)
parser.add_argument("--novelty", action="store_true")

# --- Phase 2 (critic retrain + actor) hyperparameters ---
parser.add_argument("--btd_critic_learning_rate", type=float, default=1e-4)
parser.add_argument("--phase2_learning_steps", type=int, default=1000000)
parser.add_argument("--tasks_per_batch", type=int, default=32)
parser.add_argument("--transitions_per_task", type=int, default=64)
parser.add_argument("--resume_step", type=int, default=0)

# --- Phase 2 tilt (leverage-score-weighted selection among the BTD GMM's
# z candidates, scored against Phase 2's reinitialized/retrained F -- same
# CLI surface as main_exorl.py's --tilt) ---
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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_built() else "cpu")
)

set_seed_everywhere(args.seed)

working_dir = Path.cwd()
model_dir = working_dir / "agents" / "fb" / "saved_models"
dataset_path = (
    BASE_DIR / "datasets" / args.domain_name / args.exploration_algorithm / "dataset.npz"
)

with open(working_dir / "agents" / "fb" / "config.yaml", "rb") as f:
    config = yaml.safe_load(f)

# CLI overrides for the Phase 1 hyperparameters the config.yaml doesn't
# otherwise expose (mirrors the --arg-if-not-None pattern main_exorl.py uses
# for its tilt overrides).
for key in (
    "batch_size",
    "critic_learning_rate",
    "actor_learning_rate",
    "tau",
    "orthonormalisation_coefficient",
    "discount",
):
    value = getattr(args, key)
    if value is not None:
        config[key] = value

# matches main_exorl.py's point_mass_maze special-case (applied last, same as
# there, so it wins even over an explicit --discount/--z_dimension).
if args.domain_name == "point_mass_maze":
    config["discount"] = 0.99
    config["z_dimension"] = 100

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

replay_buffer = FBReplayBuffer(
    reward_constructor=reward_constructor,
    dataset_path=dataset_path,
    transitions=args.dataset_transitions,
    relabel=False,
    task=None,
    device=device,
    discount=agent.FB._discount,  # pylint: disable=protected-access
    action_condition=None,
)

workspace = OfflineRLWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=args.phase1_learning_steps,
    model_dir=model_dir,
    eval_frequency=args.eval_frequency,
    eval_rollouts=args.eval_rollouts,
    z_inference_steps=args.z_inference_steps,
    train_std=config["std_dev_schedule"],
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
    agent_config["algorithm"] = "fb"

    # --- Phase 1: FB objective training from scratch (F, B, actor jointly;
    # baseline z ~ Unif(S^{d-1}), no tilt) ---
    # extra_checkpoint_dir gives a permanent local copy of the FINAL agent
    # state (F, B before Phase 2 discards F) and uploads it to this Phase 1
    # wandb run before that run closes -- B must not exist only in this
    # process's memory.
    phase1_checkpoint_dir = Path(
        args.phase1_checkpoint_dir
        or model_dir / "btd_phase1" / f"{args.domain_name}_{args.exploration_algorithm}_seed{args.seed}"
    )
    print(f"Phase 1: training FB from scratch for {args.phase1_learning_steps} steps...")
    workspace.train(
        agent=agent,
        tasks=args.eval_tasks,
        agent_config=agent_config,
        replay_buffer=replay_buffer,
        start_step=0,
        extra_checkpoint_dir=phase1_checkpoint_dir,
    )
    # train()'s own checkpointing loop (best-eval AND the final-state save
    # above) overwrites agent._name -- reset it before it's reused as the
    # wandb tag for Phase 2's own wandb.init(tags=[agent.name, "btd"]), and
    # capture the trained std schedule before eval() overwrites it with
    # eval_std (see OfflineRLWorkspace.eval).
    agent._name = "fb-btd"  # pylint: disable=protected-access
    train_std = agent.std_dev_schedule
    agent.train()

    # --- Phase 1 -> BTD GMM: fit phi(s) (whitened B) + GMM from the just
    # trained representation, then discard F ---
    print(f"Building BTD GMM from {args.n_subtrajectories} subtrajectories...")
    gmm, whitening_matrix = build_btd_gmm(
        agent=agent,
        dataset_path=dataset_path,
        n_subtrajectories=args.n_subtrajectories,
        min_len=args.subtraj_min_len,
        max_len=args.subtraj_max_len,
        gmm_components=args.gmm_components,
        whitening_ridge=args.btd_whitening_ridge,
        seed=args.seed,
        dataset_transitions=args.dataset_transitions,
    )
    z_sampler = GMMZSampler(
        gmm=gmm, z_dimension=agent._z_dimension, device=device  # pylint: disable=protected-access
    )

    # --- Phase 2 setup: discard Phase 1's F, reinitialize it, and freeze B
    # (the only network phi(s) needs) so it's never touched again ---
    agent.reinit_forward_representation(learning_rate=args.btd_critic_learning_rate)
    for frozen_module in (agent.FB.backward_representation, agent.FB.backward_representation_target):
        for param in frozen_module.parameters():
            param.requires_grad_(False)

    workspace.learning_steps = args.phase2_learning_steps
    workspace.train_std = train_std
    # _tilt_temperature(step) anneals over agent._learning_steps -- that was
    # set to phase1_learning_steps at construction; Phase 2 needs its own count.
    agent._learning_steps = max(1, args.phase2_learning_steps)  # pylint: disable=protected-access

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

    print(f"Phase 2: critic (F) + actor training for {args.phase2_learning_steps} steps...")
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
    )
