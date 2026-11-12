#!/bin/bash
# Source training, then 5 one-shot adaptation seeds once it succeeds.
# Overrides apply to BOTH stages (e.g. out_dir=..., src_train.*, adapt.*):
#   bash slurm/pipeline.sh configs/digits_m2u.yaml
#   bash slurm/pipeline.sh configs/digits_m2u.yaml adapt.epochs=40
# For adaptation-only variants that reuse an existing source model, submit adapt.sbatch directly:
#   sbatch --array=0-4 slurm/adapt.sbatch configs/office_a2d.yaml adapt.ablation=a
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=$1; shift || true
mkdir -p logs
SRC=$(sbatch --parsable slurm/source.sbatch "$CONFIG" "$@")
ADA=$(sbatch --parsable --dependency=afterok:"$SRC" --array=0-4 slurm/adapt.sbatch "$CONFIG" "$@")
echo "source job $SRC -> adapt array $ADA ($CONFIG $*)"
