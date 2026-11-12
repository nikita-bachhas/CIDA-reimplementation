#!/bin/bash
# Source training, then 5 one-shot adaptation seeds once it succeeds.
# Usage:  bash slurm/pipeline.sh configs/office_a2d.yaml [extra --set overrides for adaptation]
set -euo pipefail
CONFIG=$1; shift || true
mkdir -p logs
SRC=$(sbatch --parsable slurm/source.sbatch "$CONFIG")
ADA=$(sbatch --parsable --dependency=afterok:"$SRC" --array=0-4 slurm/adapt.sbatch "$CONFIG" "$@")
echo "source job $SRC -> adapt array $ADA"
