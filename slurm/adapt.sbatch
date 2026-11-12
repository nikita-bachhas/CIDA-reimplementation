#!/bin/bash
# Stage 2 (adaptation), one array task per one-shot seed.
# Usage:  sbatch --array=0-4 slurm/adapt.sbatch configs/digits_m2u.yaml [extra --set overrides]
#SBATCH --job-name=cida-adapt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:55:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud

# conda's activate scripts reference unset variables, so activate before `set -u`.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh && conda activate cida
set -euo pipefail
CONFIG=$1; shift || true
SEED=${SLURM_ARRAY_TASK_ID:-0}
trap 'echo "time limit near: requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

python adapt.py --config "$CONFIG" --seed "$SEED" --set num_workers=8 "$@" &
wait $!
