#!/bin/bash
# Tilt temperature x candidate_multiplier sensitivity sweep, one axis each,
# tilt_refresh_interval fixed at 5. 4 domains x 4 temperatures x 2 multipliers
# = 32 short (100k-step) runs, run with limited concurrency to share the GPU
# with whatever else is training. Base hyperparameters per domain are copied
# from that domain's most recent full (1M-step) tilt run in wandb/, with only
# seed/learning_steps/tilt_temperature/tilt_candidate_multiplier overridden.
export WANDB_PROJECT="${WANDB_PROJECT:-RANDOM}"
PYTHON_BIN="${PYTHON_BIN:-/home/core/anaconda3/envs/zsrl/bin/python}"

CONCURRENCY="${1:-3}"
LEARNING_STEPS=100000
SEED=42
TEMPERATURES="0.5 1 2 5"
MULTIPLIERS="5 10"

JOBS_FILE=$(mktemp)
LOG_DIR=~/D-LEVER/logs/tiltsens

base_walker=(fb walker random --eval_tasks stand walk run flip --z_mix_ratio 0.5
  --actor_learning_rate 5e-05 --dataset_transitions 10000000 --tilt
  --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_init_geom_ratio 0.98
  --tilt_ridge_alpha 1e-2 --tilt_beta 0.99)

base_quadruped=(fb quadruped rnd --eval_tasks stand jump roll roll_fast escape --z_mix_ratio 0.5
  --actor_learning_rate 3e-05 --dataset_transitions 100000 --tilt
  --tilt_ridge_min 1e-8 --tilt_start_step 20000 --tilt_init_geom_ratio 0.98
  --tilt_ridge_alpha 1e-2 --tilt_beta 0.99 --tilt_linear --tilt_uniform_mix 0.5)

base_jaco=(fb jaco random --eval_tasks reach_top_left reach_top_right reach_bottom_left reach_bottom_right --z_mix_ratio 0.5
  --actor_learning_rate 1e-4 --dataset_transitions 10000000 --tilt
  --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_init_geom_ratio 0.98
  --tilt_ridge_alpha 1e-2 --tilt_beta 0.99)

base_point_mass_maze=(fb point_mass_maze rnd --eval_tasks reach_top_left reach_top_right reach_bottom_left reach_bottom_right --z_mix_ratio 0.5
  --actor_learning_rate 3e-05 --dataset_transitions 100000 --tilt
  --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_init_geom_ratio 0.99
  --tilt_ridge_alpha 0.1 --tilt_beta 0.99)

for domain in walker quadruped jaco point_mass_maze; do
  base_var="base_${domain}[@]"
  base_args=("${!base_var}")
  for temp in $TEMPERATURES; do
    for mult in $MULTIPLIERS; do
      run_name="tiltsens_${domain}_t${temp}_m${mult}"
      log_file="${LOG_DIR}/${run_name}.log"
      cmd="${PYTHON_BIN} main_exorl.py ${base_args[*]} --tilt_temperature ${temp} --tilt_candidate_multiplier ${mult} --tilt_refresh_interval 5 --learning_steps ${LEARNING_STEPS} --seed ${SEED} --run_name ${run_name} > '${log_file}' 2>&1"
      echo "$cmd" >> "$JOBS_FILE"
    done
  done
done

echo "Launching $(wc -l < "$JOBS_FILE") runs with concurrency=${CONCURRENCY}. Logs in ${LOG_DIR}/"
cd ~/D-LEVER
xargs -P "$CONCURRENCY" -I CMD -d '\n' bash -c CMD < "$JOBS_FILE"
rm -f "$JOBS_FILE"
