export WANDB_PROJECT=RANDO
export WANDB_PROJECT=RANDOM
python main_exorl.py fb walker random  --eval_tasks stand walk run flip  --z_mix_ratio 0.5  --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.98 --tilt_temperature_start 20 --tilt_temperature_end 20 --tilt_ridge_alpha 1e-2  --learning_steps 1000000 --seed 42 --dataset_transitions 10000000 --tilt_beta 0.99 --actor_learning_rate 1e-4
python main_exorl.py fb walker random  --eval_tasks stand walk run flip  --z_mix_ratio 0.5  --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.98 --tilt_temperature_start 20 --tilt_temperature_end 20 --tilt_ridge_alpha 1e-2  --learning_steps 1000000 --seed 43 --dataset_transitions 10000000 --tilt_beta 0.99 --actor_learning_rate 1e-4
python main_exorl.py fb walker random  --eval_tasks stand walk run flip  --z_mix_ratio 0.5  --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8 --tilt_start_step 100 --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.98 --tilt_temperature_start 20 --tilt_temperature_end 20 --tilt_ridge_alpha 1e-2  --learning_steps 1000000 --seed 44 --dataset_transitions 10000000 --tilt_beta 0.99 --actor_learning_rate 1e-4


