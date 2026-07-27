export WANDB_PROJECT=hyperparameter
python main_btd.py walker rnd --eval_tasks stand walk run flip --checkpoint_path checkpoints/280000.pickle --n_subtrajectories 100000 --subtraj_min_len 5 --subtraj_max_len 50 --gmm_components 20 --learning_steps 200000 --tasks_per_batch 32 --transitions_per_task 64 --eval_frequency 20000 --eval_rollouts 10 --seed 42 --verbose
