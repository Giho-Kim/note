export WANDB_PROJECT=REVISION
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.5 --phase2_learning_steps 200000 --phase1_checkpoint_dir ./checkpoints/fb_rnd_10k/walker.pickle --seed 42
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.5 --phase2_learning_steps 200000 --phase1_checkpoint_dir ./checkpoints/fb_rnd_10k/walker.pickle --seed 43
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.5 --phase2_learning_steps 200000 --phase1_checkpoint_dir ./checkpoints/fb_rnd_10k/walker.pickle --seed 44
python main_btd.py fb walker rnd  --eval_tasks stand walk run flip  --z_mix_ratio 0.5 --phase2_learning_steps 200000 --phase1_checkpoint_dir ./checkpoints/fb_rnd_10k/walker.pickle --seed 45

