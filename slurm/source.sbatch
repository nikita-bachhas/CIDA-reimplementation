#!/bin/bash
# Stage 1 (source training). Usage:  sbatch slurm/source.sbatch configs/digits_m2u.yaml [extra --set overrides]
# Configured for the CCDS-TC2 cluster (partition MGPU-TC2, conda env "cida").
#SBATCH --job-name=cida-src
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:55:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=MGPU-TC2
#SBATCH --account=stud

# conda's activate scripts reference unset variables, so activate before `set -u`.
source /apps/anaconda3-2025.06-0/etc/profile.d/conda.sh && conda activate cida
set -euo pipefail
CONFIG=$1; shift || true
# If the 6 h limit is near, requeue: the script resumes from the latest checkpoint.
trap 'echo "time limit near: requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

python train_source.py --config "$CONFIG" --set num_workers=8 "$@" &
wait $!
