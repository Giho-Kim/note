export WANDB_PROJECT=REVISION
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.0 --phase2_learning_steps 500000 --phase1_checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --reinit_phase2_forward --reinit_phase2_actor --seed 42
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.0 --phase2_learning_steps 500000 --phase1_checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --reinit_phase2_forward --reinit_phase2_actor --seed 43
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.0 --phase2_learning_steps 500000 --phase1_checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --reinit_phase2_forward --reinit_phase2_actor --seed 44
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.0 --phase2_learning_steps 500000 --phase1_checkpoint_path ./checkpoints/fb_rnd_10k/walker.pickle --reinit_phase2_forward --reinit_phase2_actor --seed 45
