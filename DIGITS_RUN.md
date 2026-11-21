# Siddharth's MNIST -> USPS runs

Reuse your existing uv-created environment; no Conda or package reinstall is needed.
Run these exports in the cluster shell before the commands below:

```bash
export CIDA_OLD=/home/msai/siddhart022/CIDA_Class-Incremental-Domain-Adaptation
export CIDA_PYTHON="$CIDA_OLD/.venv/bin/python"
export CIDA_DATA="$CIDA_OLD/data"
"$CIDA_PYTHON" -c 'import torch; print(torch.__version__)'
```

Commands below run from the NEW repository root. `slurm/run.sbatch` inherits
`CIDA_PYTHON` and uses it for both stages, bypassing its original Conda setup. Existing
`digits.slurm` belongs to your old codebase; use the new job script for this checkout.
The target is USPS: shared digits 0-4 and private digits 5-9. Target training uses the
train split; final accuracy is measured on the test split. One-shot means five labelled
target images total, one for each private digit.

## 1. Improved baseline with random supports

```bash
mkdir -p logs
sbatch slurm/run.sbatch configs/digits_m2u_improved.yaml "data_root=$CIDA_DATA"
```

This trains one source model (seed 0, one warm-up plus ten epochs), then five target
models (30 epochs each). Actual target RNG seeds are 0-4. Support seed k is also k.
Batch size is 64; Adam uses lr 1e-4. The improved setup enables AdaBN, identity-initialized
residual projections with a linear output, and confident-only La1. The Gaussian losses use
the inherited log-density logits. All five adaptation losses are enabled. Guides use
nearest-guide pseudo-labels; guide rectification and label propagation are off.

## 2. Review support images, then fix one clear image per private digit

```bash
"$CIDA_PYTHON" scripts/choose_digit_supports.py --data-root "$CIDA_DATA"
```

Open `oneshot_review/digits_m2u_support_preview.png`. Choose images by legibility and a
representative digit shape, before looking at their training/test performance. The five
rows are digits 5,6,7,8,9; each column is a support-selection seed. It is fine to choose
different columns for different digits. The numbers below are examples, not recommended picks:

```bash
"$CIDA_PYTHON" scripts/choose_digit_supports.py --data-root "$CIDA_DATA" --picks 0,1,2,3,4
sbatch slurm/run.sbatch configs/digits_m2u_clearshot.yaml
```

If none of the first five candidates is clear, add `--candidates 20` to BOTH preview and
selection commands. Then picks may be 0-19. Candidate seeds select support images; target
training still uses actual RNG seeds 0-4. The picker writes a config; it never edits code.

Both runs reuse `runs/digits_m2u_siddharth/source/final.pt` and write distinct target run
folders. Different support picks get distinct run tags. Repeating the same run skips
completed results. Keep the source checkpoint and environment unchanged between comparisons.

## 3. Read results

```bash
"$CIDA_PYTHON" scripts/aggregate.py runs/
```

Compare random and hand-picked runs separately; aggregate.py groups run variants by tag.
It reports last-epoch ALL/PRIV and class-averaged ALL over the seeds. Shared accuracy is
also saved in each run's `final_metrics.json`. Keep the
five-seed mean and standard deviation; do not select the best seed or best test epoch.
This setup is an improved reproduction, not the paper-as-written vanilla preset. Report
hand-picked supports separately from random one-shot draws. Clear images can help, but
clarity alone does not guarantee a representative guide or high private accuracy.

The unmodified vanilla reference is `configs/digits_m2u.yaml preset=vanilla`; do not append
`preset=vanilla` to the improved config, as that would switch its training settings back.

## Local GPU instead of Slurm

With the same environment active, use these commands if this machine has a CUDA GPU:

```bash
python train_source.py --config configs/digits_m2u_improved.yaml --set data_root=/absolute/path/to/existing/data num_workers=0
for seed in 0 1 2 3 4; do
  python adapt.py --config configs/digits_m2u_improved.yaml --seed "$seed" --set data_root=/absolute/path/to/existing/data num_workers=0
done
```

For fixed supports, replace the config in both commands with the generated
`configs/digits_m2u_clearshot.yaml`. Slurm uses six workers, so local/cluster results may
differ slightly. On the NTU cluster, submit through sbatch rather than training on a login node.

For a local Windows PowerShell terminal, the five-seed loop is:

```powershell
0..4 | ForEach-Object {
  python adapt.py --config configs/digits_m2u_improved.yaml --seed $_ --set "data_root=C:/path/to/existing/data" num_workers=0
}
```

Train the source first with the preceding `train_source.py` command, using your Windows
data path. The selected Python environment must have CUDA-enabled PyTorch installed.
