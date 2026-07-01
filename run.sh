while true; do
    python main_exorl.py vcfb quadruped rnd  --eval_tasks stand jump roll roll_fast escape --seed 42 --z_mix_ratio 0.5
    python main_exorl.py vcfb quadruped rnd  --eval_tasks stand jump roll roll_fast escape --seed 43 --z_mix_ratio 0.5
    python main_exorl.py vcfb quadruped rnd  --eval_tasks stand jump roll roll_fast escape --seed 44 --z_mix_ratio 0.5
    python main_exorl.py vcfb quadruped rnd  --eval_tasks stand jump roll roll_fast escape --seed 45 --z_mix_ratio 0.5
    sleep 1
done