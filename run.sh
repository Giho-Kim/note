while true; do
    python main_exorl.py vcfb walker rnd  --eval_tasks stand walk run flip --seed 42 --z_mix_ratio 0.5
    python main_exorl.py vcfb walker rnd  --eval_tasks stand walk run flip --seed 43 --z_mix_ratio 0.5
    python main_exorl.py vcfb walker rnd  --eval_tasks stand walk run flip --seed 44 --z_mix_ratio 0.5
    python main_exorl.py vcfb walker rnd  --eval_tasks stand walk run flip --seed 45 --z_mix_ratio 0.5
    python main_exorl.py vcfb walker rnd  --eval_tasks stand walk run flip --seed 46 --z_mix_ratio 0.5
    sleep 1
done