# OGBench implementation

Vendored from `../td_jepa` (TD-JEPA / Meta Motivo), commit
`840a745455a124f04a58ae7ee31d7d5054e381f5`, under the accompanying
CC BY-NC 4.0 LICENSE. Only the FB/TD-JEPA flow-BC agents and their OGBench
dependencies are included. Absolute `metamotivo` imports are renamed to
`ogbench_impl` to keep note's existing DMC implementation independent.

The four configs under `configs/ogbench` reproduce the upstream proprio/pixel
launcher BASE_CFG dictionaries. `main_ogbench.py` selects the first upstream
sweep combination by default; it exposes BC coefficient, backward-network
learning rate, and orthogonality coefficient for the remaining combinations.

Local changes: OGBench-only agent/config registry; sorted data loading with a
missing-data error; environment cleanup; a final checkpoint; and an exclusive
training-step bound so train_status records the next update on resume; restored
agent config on resume; optimizer device mapping; and initialized BC metrics
when `bc_coeff=0`.

This backend reproduces upstream baseline training and optionally integrates
note's shared tilt selector (sphere/goal selection, Gram EMA, cached
resampling, free competition and scoring). A separate `tilt.pth` persists the
selector state. Flow actors use zero noise for deterministic scoring.
Upstream directory checkpoints include model,
config, optimizer state, and train_status, and are distinct from note's DMC
pickle checkpoints.
