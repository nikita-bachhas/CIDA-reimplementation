#!/bin/bash
# Run an analysis script on a GPU node (the guide forbids running code on the head node).
#   sbatch slurm/py.sbatch scripts/feature_probe.py runs/office_a2d
#   -> logs/cida-py_<jobid>.out
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cida-py
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
set -euo pipefail
python "$@" --num_workers 3
