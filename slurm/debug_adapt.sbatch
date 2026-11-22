#!/bin/bash
# Run one adaptation seed and print the full traceback of whatever stops it, also on stdout,
# including a SystemExit raised inside a library. Writes to runs_debug/, not runs/.
#   sbatch slurm/debug_adapt.sbatch configs/office20_d2a.yaml runs/office20_d2a/source/final.pt
#   -> logs/cida-dbg_<jobid>.out
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --job-name=cida-dbg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
CONFIG=$1; SRC=$2; shift 2
NAME=$(basename "$CONFIG" .yaml)
echo "host $(hostname) | python $(python -V 2>&1) | $(df -h ~ | tail -1) | home $(du -sh ~ 2>/dev/null | cut -f1)"
rc=0
python -X faulthandler -u -c '
import sys, traceback
import adapt
try:
    adapt.main(sys.argv[1:])
except BaseException as e:
    print(f"CAUGHT {type(e).__name__}: {e!r}", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    raise
' --config "$CONFIG" --seed 0 --source_ckpt "$SRC" --force \
  --set num_workers=6 out_dir=runs_debug/$NAME adapt.epochs=1 "$@" || rc=$?
echo "python exit code: $rc"
echo "home after: $(du -sh ~ 2>/dev/null | cut -f1)"
