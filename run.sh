#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
for seed in ${SEEDS:-42 43 44 45 46}; do
  python main_exorl.py td_jepa jaco rnd \
    --eval_tasks reach_top_left reach_top_right reach_bottom_left reach_bottom_right \
    --tilt --seed "$seed" --dataset_transitions 6350000 "$@"
done
