export WANDB_PROJECT=RANDOM
python main_exorl.py fb walker random --eval_tasks stand walk run flip --z_mix_ratio 0.5 --seed 43 --dataset_transitions 10000000 --save_every

