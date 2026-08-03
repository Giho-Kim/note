export WANDB_PROJECT=SENSITIVITY
python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip  --learning_steps 200000 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_beta 0.99 --tilt_temperature_start 10.  --tilt_temperature_end 0.   --checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --seed 42

python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip  --learning_steps 200000 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_beta 0.99 --tilt_temperature_start 0.  --tilt_temperature_end 10.   --checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --seed 43

python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip  --learning_steps 200000 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_beta 0.99 --tilt_temperature_start 10.  --tilt_temperature_end 10.   --checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --seed 44

python main_exorl.py fb walker rnd  --eval_tasks stand walk run flip  --learning_steps 200000 --tilt --tilt_ridge_min 1e-8  --tilt_candidate_multiplier 5 --tilt_refresh_interval 5 --tilt_init_geom_ratio 0.9 --tilt_ridge_alpha 1e-2 --tilt_beta 0.99 --tilt_temperature_start 10.  --tilt_temperature_end 10.   --checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --seed 45

