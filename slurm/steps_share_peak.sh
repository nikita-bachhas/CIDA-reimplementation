# Share-peak stop on the six Office Table 3 tasks, for slurm/steps.sbatch (which defines `run`).
#   sbatch slurm/steps.sbatch slurm/steps_share_peak.sh
# Source models are trained first where missing (train_source.py skips finished ones).
SEEDS="0 1 2 3 4"
W="num_workers=6"
for t in a2d a2w d2a d2w w2a w2d; do
    run python train_source.py --config configs/office_$t.yaml --set $W
    for s in $SEEDS; do
        run python adapt.py --config configs/office_$t.yaml --seed "$s" --set $W adapt.stop=share_peak adapt.tag=sharestop
    done
done
