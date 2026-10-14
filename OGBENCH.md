# OGBench in note

FB and TD-JEPA use the reference implementation from `../td_jepa`, vendored
locally in `ogbench_impl`. Both use flow matching behavior cloning and actor
distillation during offline training. No sibling repository is required at
runtime. This is a separate entrypoint from `main_exorl.py`. Both OGBench
agents also support note's tilt latent selection for state and pixel inputs.

## Environment

Use Python 3.11 or 3.12 in a separate environment because the legacy DMC requirements
pin older MuJoCo and PyTorch versions:

```bash
uv venv --python 3.11 .venv-ogbench
uv pip install --python .venv-ogbench/bin/python -r requirements-ogbench.txt
```

Install a PyTorch CUDA build compatible with your machine's NVIDIA driver.
For this workspace, `.venv-ogbench` has been prepared using the existing
Python 3.12 `mujoco_playground` environment's CUDA 12.6 PyTorch build.

## Download and prepare data

```bash
.venv-ogbench/bin/python main_ogbench.py fb antmaze-medium-navigate-v0 --download-only
```

OGBench downloads its original dataset into its own default cache. The
reference extractor splits the training dataset into episode NPZs under
`datasets/ogbench/<domain>/buffer/`, preserving observations, shifted actions,
physics (qpos plus button states when present), and the reference discounts.
Use `--data-path /path/to/episodes` to change the converted-data destination.
Use `--dataset-path /path/to/raw_npzs` to convert an already downloaded dataset.
The same converted dataset is shared by FB and TD-JEPA.

For visual datasets, add `--obs-type pixels`; this downloads the `visual-`
dataset and uses the upstream RGB encoder, augmentation, and frame stacking.

## Train and evaluate

```bash
.venv-ogbench/bin/python main_ogbench.py fb antmaze-medium-navigate-v0 --download
.venv-ogbench/bin/python main_ogbench.py td_jepa antmaze-medium-navigate-v0
.venv-ogbench/bin/python main_ogbench.py td_jepa cube-single-play-v0 --download
```

Supported task families include AntMaze, Cube, Scene, and Puzzle. Evaluation
uses the five single tasks for the domain, reward relabeling with a +1 shift,
10,000 inference samples, and 10 rollouts per task, matching the upstream
launchers. Both agents use `bc_coeff=0.3` for AntMaze and `3.0` otherwise, and
10 flow integration steps. Training defaults to one million updates.

```bash
# Pixel training
.venv-ogbench/bin/python main_ogbench.py fb cube-single-play-v0 --obs-type pixels --download
# Inspect config without installing dependencies or downloading
python main_ogbench.py td_jepa antmaze-medium-navigate-v0 --dry-run
# CPU / short diagnostic run; keep the default batch size for full experiments
.venv-ogbench/bin/python main_ogbench.py fb antmaze-medium-navigate-v0 --device cpu --no-compile --learning-steps 10 --batch-size 16 --no-eval
```

Use `--bc-coeff`, `--ortho-coef`, and `--lr-backward` to select other upstream
sweep settings. `--wandb` enables logging (`--wandb-project`, `--wandb-group`).
See `--help` for training, evaluation, buffer-device, and episode-limit options.

## Outputs and resume

The default run directory is
`results/ogbench/<algorithm>_<domain>_<obs-type>_seed<seed>/` (with a `_tilt`
suffix when `--tilt` is enabled). It contains
`config.json`, `train_log.txt`, `reward.csv` when evaluation runs, and
`checkpoint/` with config, optimizer states, model safetensors, and
`train_status.json`. A final checkpoint is always written, including for short
runs. Reusing the same work directory automatically resumes from its checkpoint;
choose another `--work-dir` for a fresh run. Resume with the same model and
training settings; the checkpoint restores the saved agent configuration.

Evaluation also writes `reward_summary.csv`: one row per task and an `overall`
row per evaluation step, with `reward_mean`, `reward_iqm`, `reward_std`,
`success_mean`, and `num_episodes`. Overall statistics pool all task episodes.
IQM uses the same `scipy.stats.trim_mean(..., 0.25)` convention as note's DMC
evaluation (drop the lowest/highest quarter, rounding the number dropped down).
Task mean/IQM are also included in the episode CSV and logged to WandB.

## Tilt

The same underscore-style tilt options from `main_exorl.py` are accepted,
along with equivalent hyphen-style aliases:

```bash
.venv-ogbench/bin/python main_ogbench.py fb scene-play-v0 --tilt --tilt_start_step 20000 --verbose
.venv-ogbench/bin/python main_ogbench.py td_jepa puzzle-3x3-play-v0 --tilt --tilt_start_step 20000 --verbose
```

Supported controls: `--tilt_goal`, `--tilting_by_z`, `--tilt_beta`,
`--tilt_temperature`, `--tilt_temperature_start`, `--tilt_temperature_end`,
`--tilt_candidate_multiplier`, `--tilt_init_geom_ratio`, `--tilt_ridge_alpha`,
`--tilt_ridge_min`, `--tilt_start_step`, `--tilt_refresh_interval`,
`--tilt_uniform_mix`, `--tilt_linear`, `--tilt_free_compete`, and
`--z_mix_ratio` (OGBench's `train_goal_ratio`). `--actor_learning_rate` is also
accepted. `--no-tilt-goal`, `--no-tilt-linear`, and `--no-tilt-free-compete`
disable the corresponding switches. `--no-tilt` disables tilt explicitly.

With `--tilt`, defaults match note's non-Jaco tilt defaults: goal selection,
linear weighting and free competition enabled; candidate multiplier 10;
geometric ratio 0.99; start step 200,000; refresh interval 5; uniform mix 0.1;
ridge alpha 0.01/minimum 1e-8; and actor learning rate 3e-5. Beta is 0.995;
temperature runs from 20 to 1. Setting `--tilt_temperature` alone fixes both
temperature endpoints; linear weighting ignores temperature. Goal ratio
defaults to the upstream 0.5.

The sphere/goal pools, normalized-feature Gram EMA, initial-state geometric
sampling, cached resampling, and free competition use the same selector as
DMC. FB warms its Gram before activation, as in note; TD-JEPA starts at
activation. Because the OGBench actor is noise-conditioned, scoring uses zero
noise to give deterministic action scores. Flow BC training and evaluation
retain their original stochastic behavior.

`checkpoint/tilt.pth` saves the Gram, feature scale and both candidate caches.
On resume these are restored along with model/optimizers. Explicit tilt CLI
options apply to the resumed run; omitted options preserve saved settings.
Changing the scoring basis (`tilting_by_z`) resets the selector's Gram/caches.
To resume an existing baseline with tilt, pass its `--work-dir` explicitly;
the default tilt run directory is separate.

Training logs and WandB contain `tilt/*` health metrics; `--verbose` also prints
Gram conditioning, ridge, feature scale, pool rank and score statistics.
Evaluation reports task-conditioned tilt score means and standard deviations.

Integration checks (tiny networks and temporary datasets):

```bash
.venv-ogbench/bin/python -m tests.ogbench_smoke --device cuda --tilt
.venv-ogbench/bin/python -m tests.ogbench_smoke --device cpu --tilt
```

The OGBench backend follows the reference directory checkpoint format rather
than the DMC pickle format. It saves the latest/final state, not a separate best
checkpoint. It does not currently save trajectory plots.
