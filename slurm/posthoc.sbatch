#!/bin/bash
# Prediction-step variants (gate / bias) on finished runs (no training; a few minutes).
#   sbatch slurm/posthoc.sbatch runs/digits_m2u runs/digits_u2m runs/digits_s2m
#   -> logs/cida-posthoc_<jobid>.out, plus <run>/posthoc.json
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cida-posthoc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
set -euo pipefail
python scripts/posthoc.py "$@" --num_workers 3
