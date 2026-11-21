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
#   sbatch --export=ALL,STAGES=adapt,ABLATIONS="b c nola1" slurm/run.sbatch configs/digits_m2u.yaml adapt.epochs=15
#     (ABLATIONS loops over adapt.ablation values; seeds run inside each one)
#   sbatch --export=ALL,STAGES=adapt,VARIANTS="adapt.la1_samples=confident+adapt.tag=la1conf adapt.la1_temp=100+adapt.tag=la1t100" \
#          slurm/run.sbatch configs/digits_m2u.yaml
#     (VARIANTS: space-separated variants, each a "+"-joined list of overrides; sbatch --export splits on commas)
#   sbatch --export=ALL,CONFIGS="configs/digits_m2u.yaml configs/office_a2d.yaml" slurm/run.sbatch preset=vanilla
#     (CONFIGS: several experiments one after another in one job; all arguments are then overrides)
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

# Reuse an existing uv/venv interpreter when supplied; otherwise retain the original setup.
# export CIDA_PYTHON=/absolute/path/to/existing/project/.venv/bin/python
if [ -n "${CIDA_PYTHON:-}" ]; then
    [ -x "$CIDA_PYTHON" ] || { echo "Python interpreter is not executable: $CIDA_PYTHON" >&2; exit 1; }
else
    # Activate before `set -u`: Conda's scripts use unset variables.
    module load anaconda/25.5.1
    eval "$(conda shell.bash hook)"
    conda activate cida
fi
set -euo pipefail
PYTHON_BIN=${CIDA_PYTHON:-python}

if [ -n "${CONFIGS:-}" ]; then read -r -a CONFIG_LIST <<< "$CONFIGS"; else CONFIG_LIST=("$1"); shift; fi
for c in "${CONFIG_LIST[@]}"; do [ -f "$c" ] || { echo "no such config: $c" >&2; exit 1; }; done
SEEDS=${SEEDS:-"0 1 2 3 4"}
STAGES=${STAGES:-"source adapt"}
ABLATIONS=${ABLATIONS:-""}
VARIANTS=${VARIANTS:-""}
echo "job ${SLURM_JOB_ID:-local} on $(hostname): ${CONFIG_LIST[*]} | stages: $STAGES | ablations: ${ABLATIONS:-config default} | variants: ${VARIANTS:-none} | seeds: $SEEDS | overrides: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

PID=""
trap 'echo "6 h limit near: requeueing; work resumes from checkpoints"; [ -n "$PID" ] && kill -TERM "$PID"; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
run() {  # run one step; on failure say which one, with its exit code, in both .out and .err
    local rc=0
    "$@" & PID=$!; wait "$PID" || rc=$?; PID=""
    if [ "$rc" -ne 0 ]; then
        echo "FAILED with exit code $rc at $(date '+%F %T'): $*" | tee /dev/stderr
        [ "$rc" -gt 128 ] && echo "(exit code > 128: killed by signal $((rc - 128)); 9 = SIGKILL, often out of memory)" | tee /dev/stderr
        exit "$rc"
    fi
}

for CONFIG in "${CONFIG_LIST[@]}"; do
echo "=== $CONFIG"
if [[ " $STAGES " == *" source "* ]]; then
    run "$PYTHON_BIN" train_source.py --config "$CONFIG" --set num_workers=6 "$@"
fi
if [[ " $STAGES " == *" adapt "* ]]; then
    for v in ${VARIANTS:-__default__}; do
        var=(); [ "$v" != "__default__" ] && IFS=+ read -r -a var <<< "$v"
        for a in ${ABLATIONS:-__default__}; do
            abl=(); [ "$a" != "__default__" ] && abl=("adapt.ablation=$a")
            for s in $SEEDS; do
                run "$PYTHON_BIN" adapt.py --config "$CONFIG" --seed "$s" --set num_workers=6 "$@" \
                    ${var[@]+"${var[@]}"} ${abl[@]+"${abl[@]}"}
            done
        done
    done
fi
done
echo "all done: python scripts/aggregate.py runs/   (runs_vanilla/ for preset=vanilla)"
