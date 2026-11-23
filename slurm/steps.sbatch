#!/bin/bash
# A multi-step plan in one job. QoS "normal" allows only 2 submitted jobs per user, so a long plan
# cannot be queued as a chain of dependent jobs. Instead, this job sources a bash steps file that
# calls `run <command>` for every step, in order, from the repository root.
#
#   sbatch slurm/steps.sbatch slurm/steps_share_peak.sh
#
# If the 6 h limit is near, the job requeues itself and sources the steps file again from the top.
# adapt.py and train_source.py skip finished runs and resume unfinished ones from their checkpoints,
# so a steps file must guard any step that is not idempotent, e.g. `[ -f <its output> ] || run ...`.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=05:55:00
#SBATCH --job-name=cida-steps
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@180

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
set -euo pipefail

STEPS_FILE=${1:?usage: sbatch slurm/steps.sbatch <steps file>}
[ -f "$STEPS_FILE" ] || { echo "no such steps file: $STEPS_FILE" >&2; exit 1; }
echo "job ${SLURM_JOB_ID:-local} on $(hostname), restart ${SLURM_RESTART_COUNT:-0}: $STEPS_FILE"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

PID=""
trap 'echo "6 h limit near: requeueing; the steps file restarts from the top and skips finished work"; [ -n "$PID" ] && kill -TERM "$PID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
run() {  # run one step; on failure say which one, with its exit code, in both .out and .err
    local rc=0
    echo "--- $(date '+%F %T') $*"
    "$@" & PID=$!; wait "$PID" || rc=$?; PID=""
    if [ "$rc" -ne 0 ]; then
        echo "FAILED with exit code $rc at $(date '+%F %T'): $*" | tee /dev/stderr
        [ "$rc" -gt 128 ] && echo "(exit code > 128: killed by signal $((rc - 128)); 9 = SIGKILL, often out of memory)" | tee /dev/stderr
        exit "$rc"
    fi
}

# shellcheck source=/dev/null
source "$STEPS_FILE"
echo "all steps done at $(date '+%F %T')"
