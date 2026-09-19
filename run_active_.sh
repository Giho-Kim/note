#!/usr/bin/env bash
set -euo pipefail

export WANDB_PROJECT=ICLR_ACTIVE

for seed in 45 46; do
  python main_exorl.py fb walker rnd --eval_tasks stand walk run flip --dataset_transitions 100000 --collection_interval 5000 --collection_episodes 50 --tilt --tilt_ridge_min 1e-8 --tilt_candidate_multiplier 10 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.99 --tilt_ridge_alpha 1e-2 --tilt_linear --tilt_uniform_mix 0.1 --tilt_goal --tilt_free_compete --actor_learning_rate 3e-5 --tilt_start_step 20000 --seed "$seed"
done

for seed in 42 43 44 45 46; do
  python main_exorl.py fb quadruped rnd --eval_tasks stand jump roll roll_fast escape --dataset_transitions 100000 --collection_interval 5000 --collection_episodes 50 --tilt --tilt_ridge_min 1e-8 --tilt_candidate_multiplier 10 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.99 --tilt_ridge_alpha 1e-2 --tilt_linear --tilt_uniform_mix 0.1 --tilt_goal --tilt_free_compete --actor_learning_rate 3e-5 --tilt_start_step 20000 --seed "$seed"
done
