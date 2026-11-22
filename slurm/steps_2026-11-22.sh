# Steps 1 and 2 of the 22 November plan, for slurm/steps.sbatch (which defines `run`).
#   sbatch slurm/steps.sbatch slurm/steps_2026-11-22.sh
# Office Table 3, seeds 0-4. Source models for the five existing tasks are reused.
SEEDS="0 1 2 3 4"
W="num_workers=6"
STOP="adapt.stop=share_peak adapt.tag=sharestop"

# Step 1a. Share-peak stop on the five tasks that shaped the rule (D->A first: it is the determinism check).
for t in d2a w2a a2w d2w w2d; do
    for s in $SEEDS; do run python adapt.py --config configs/office_$t.yaml --seed "$s" --set $W $STOP; done
done

# Step 1b. A->D, the held-out task: source model and 60-epoch Main runs, then share-peak stop.
run python train_source.py --config configs/office_a2d.yaml --set $W
for s in $SEEDS; do run python adapt.py --config configs/office_a2d.yaml --seed "$s" --set $W; done
for s in $SEEDS; do run python adapt.py --config configs/office_a2d.yaml --seed "$s" --set $W $STOP; done

# Step 1c. LP after training on the stopped models and on the new A->D Main runs. posthoc.py always
# rewrites posthoc.json, so each call is skipped once the last run it writes has one.
[ -f runs/office_a2d/adapt_1shot_abl-none_sharestop_seed4/posthoc.json ] ||
    run python scripts/posthoc.py runs/office_d2a runs/office_w2a runs/office_a2w runs/office_d2w runs/office_w2d \
        runs/office_a2d --runs adapt_1shot_abl-none_sharestop --num_workers 6
[ -f runs/office_a2d/adapt_1shot_abl-none_seed4/posthoc.json ] ||
    run python scripts/posthoc.py runs/office_a2d --num_workers 6

# Step 2. LP inside training on the Amazon targets, all 60 epochs. Primary: balanced. Secondary: no balancing.
for t in d2a w2a; do
    for v in "adapt.pseudo_label=lp adapt.tag=lp" "adapt.pseudo_label=lp adapt.lp.balance=false adapt.tag=lpnobal"; do
        # shellcheck disable=SC2086
        for s in $SEEDS; do run python adapt.py --config configs/office_$t.yaml --seed "$s" --set $W $v; done
    done
done
