"""
Loads a pretrained FB checkpoint, freezes F and B, builds a Basis Trajectory
Distribution (BTD) -- a Gaussian mixture fit on normalized discounted-feature
summaries of dataset subtrajectories -- and fine-tunes only the actor with z
drawn from that GMM instead of Unif(S^{d-1}).
"""

from argparse import ArgumentParser
from pathlib import Path

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
parser.add_argument("--checkpoint_path", type=str, required=True)
parser.add_argument("--eval_tasks", nargs="+", required=True)
parser.add_argument("--dataset_transitions", type=int, default=100000)
parser.add_argument("--n_subtrajectories", type=int, default=2000)
parser.add_argument("--subtraj_min_len", type=int, default=5)
parser.add_argument("--subtraj_max_len", type=int, default=50)
parser.add_argument("--gmm_components", type=int, default=20)
parser.add_argument("--btd_whitening_ridge", type=float, default=1e-6)
parser.add_argument("--learning_steps", type=int, default=200000)
parser.add_argument("--tasks_per_batch", type=int, default=32)
parser.add_argument("--transitions_per_task", type=int, default=64)
parser.add_argument("--eval_frequency", type=int, default=20000)
parser.add_argument("--eval_rollouts", type=int, default=10)
parser.add_argument("--eval_std", type=str, default="0.05")
parser.add_argument("--z_inference_steps", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--wandb_logging", type=str, default="True")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--save_every", action="store_true")
parser.add_argument("--resume_step", type=int, default=0)
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

checkpoint_path = Path(args.checkpoint_path)
print(f"Loading agent checkpoint from {checkpoint_path}")
agent = torch.load(checkpoint_path, map_location=device, weights_only=False)
if not isinstance(agent, FB):
    raise TypeError(f"main_btd.py only supports FB checkpoints, got {type(agent)}.")
agent.to(device)
agent._device = device  # pylint: disable=protected-access
# checkpoint saves overwrite agent._name with the (int) step number just
# before pickling (see OfflineRLWorkspace.train), which then breaks
# wandb.init(tags=[agent.name, "btd"]) on load -- restore a proper name.
agent._name = "fb-btd"  # pylint: disable=protected-access
agent.train()

# capture the checkpoint's trained std schedule before eval() ever overwrites
# agent.std_dev_schedule with eval_std (see OfflineRLWorkspace.eval)
train_std = agent.std_dev_schedule

# freeze F and B -- only the actor is trained on the BTD z distribution
for frozen_module in (
    agent.FB.forward_representation,
    agent.FB.backward_representation,
    agent.FB.forward_representation_target,
    agent.FB.backward_representation_target,
):
    for param in frozen_module.parameters():
        param.requires_grad_(False)

reward_constructor = RewardFunctionConstructor(
    domain_name=args.domain_name,
    task_names=args.eval_tasks,
    seed=args.seed,
    device=device,
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

print(f"Building BTD GMM from {args.n_subtrajectories} subtrajectories...")
gmm = build_btd_gmm(
    agent=agent,
    dataset_path=dataset_path,
    n_subtrajectories=args.n_subtrajectories,
    min_len=args.subtraj_min_len,
    max_len=args.subtraj_max_len,
    gmm_components=args.gmm_components,
    whitening_ridge=args.btd_whitening_ridge,
    seed=args.seed,
)
z_sampler = GMMZSampler(
    gmm=gmm, z_dimension=agent._z_dimension, device=device  # pylint: disable=protected-access
)

workspace = OfflineRLWorkspace(
    reward_constructor=reward_constructor,
    learning_steps=args.learning_steps,
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
    agent_config["algorithm"] = "fb-btd"

    workspace.train_btd(
        agent=agent,
        tasks=args.eval_tasks,
        agent_config=agent_config,
        replay_buffer=replay_buffer,
        z_sampler=z_sampler,
        tasks_per_batch=args.tasks_per_batch,
        transitions_per_task=args.transitions_per_task,
        start_step=args.resume_step,
    )
