export WANDB_PROJECT=RND
python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_linear --tilt_uniform_mix 0.5 --tilt_goal  --seed 42
python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_linear --tilt_uniform_mix 0.5 --tilt_goal  --seed 43
python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip --z_mix_ratio 0.5 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_linear --tilt_uniform_mix 0.5 --tilt_goal  --seed 44

