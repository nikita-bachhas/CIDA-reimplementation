#!/bin/bash
# One-off environment check on a GPU compute node (the guide forbids running code or
# nvidia-smi on the head node, and advises sbatch over srun).
#   sbatch slurm/check.sbatch   ->  logs/cida-check_<jobid>.out
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --account=stud
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=cida-check
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda/25.5.1
eval "$(conda shell.bash hook)"
conda activate cida
set -euo pipefail

echo "== node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== python: $(which python)"
python - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| CUDA build", torch.version.cuda)
print("cuda available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
print("matmul on GPU ok:", (x @ x).isfinite().all().item())
EOF
echo "== tests"
python -m pytest -q
echo "== CHECK PASSED"
