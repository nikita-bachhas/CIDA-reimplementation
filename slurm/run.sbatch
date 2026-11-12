#!/bin/bash
# One job = one experiment: source training (skipped if already done), then the
# adaptation seeds one after another (each skipped if already finished).
# Follows the CCDS GPU TC2 user guide: QoS "normal" = 1 GPU, <=10 CPUs, <=30G, <=6 h,
# so experiments run one job at a time instead of as parallel arrays.
#
#   sbatch slurm/run.sbatch configs/digits_m2u.yaml
#   sbatch slurm/run.sbatch configs/digits_m2u.yaml adapt.epochs=40          # overrides: key=value
#   sbatch --export=ALL,STAGES=adapt slurm/run.sbatch configs/office_a2d.yaml adapt.ablation=a
#   sbatch --export=ALL,SEEDS="0 1" slurm/run.sbatch configs/digits_m2u.yaml
#
# If the 6 h limit is near, the job requeues itself and resumes from the latest checkpoint.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=05:55:00
#SBATCH --job-name=cida
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

# Conda setup as in the cluster guide (activate before `set -u`: conda's scripts use unset variables).
module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
set -euo pipefail

CONFIG=$1; shift
SEEDS=${SEEDS:-"0 1 2 3 4"}
STAGES=${STAGES:-"source adapt"}
echo "job ${SLURM_JOB_ID:-local} on $(hostname): $CONFIG | stages: $STAGES | seeds: $SEEDS | overrides: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

PID=""
trap 'echo "6 h limit near: requeueing; work resumes from checkpoints"; [ -n "$PID" ] && kill -TERM "$PID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
run() { "$@" & PID=$!; wait "$PID"; PID=""; }

if [[ " $STAGES " == *" source "* ]]; then
    run python train_source.py --config "$CONFIG" --set num_workers=6 "$@"
fi
if [[ " $STAGES " == *" adapt "* ]]; then
    for s in $SEEDS; do
        run python adapt.py --config "$CONFIG" --seed "$s" --set num_workers=6 "$@"
    done
fi
echo "all done: python scripts/aggregate.py runs/"
