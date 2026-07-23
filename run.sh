export WANDB_PROJECT=mcfb
python main_exorl.py mcfb jaco rnd --eval_tasks reach_top_left reach_top_right reach_bottom_left reach_bottom_right --z_mix_ratio 0.5 --seed 44
python main_exorl.py mcfb point_mass_maze rnd --eval_tasks reach_top_left reach_top_right reach_bottom_left reach_bottom_right  --z_mix_ratio 0.5 --seed 44
