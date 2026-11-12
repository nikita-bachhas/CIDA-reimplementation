#!/bin/bash
# Stage 1 (source training). Usage:  sbatch slurm/source.sbatch configs/digits_m2u.yaml [extra --set overrides]
# Fill in partition / account / module lines for your cluster.
#SBATCH --job-name=cida-src
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:55:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%x-%j.out
##SBATCH --partition=CHANGE_ME
##SBATCH --account=CHANGE_ME

set -euo pipefail
CONFIG=$1; shift || true
# If the 6 h limit is near, requeue: the script resumes from the latest checkpoint.
trap 'echo "time limit near: requeueing"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1

# source ~/miniconda3/etc/profile.d/conda.sh && conda activate cida   # adapt to your env
python train_source.py --config "$CONFIG" --set num_workers=8 "$@" &
wait $!
