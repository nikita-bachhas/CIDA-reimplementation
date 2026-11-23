# Siddharth's digit support-selection experiments

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

## Reproduce the reported epoch-15 comparison (all three transfers)

This section specifies Siddharth's random-versus-hand-picked comparison. The
historical Office results and default `1000 + seed` schedule in REPRODUCE.md are
separate experiments. The digit configs below explicitly use training RNG seeds
0-4. The random suite was implemented at commit `5bbcc71`; local integration
commit `d300029` retains that implementation and adds develop's AdaBN singleton
batch guard. That guard only skips a final batch containing one image.

Common configured settings:

| Item | Setting |
|---|---|
| Transfers | MNIST -> USPS; USPS -> MNIST; SVHN -> MNIST |
| Class split | Shared 0-4; target-private 5-9 |
| Target supervision | One target-training image per private digit; five images total |
| Evaluation | Target test split; ALL, SHARED and PRIV from epoch 15 |
| Training seeds | Actual Python/NumPy/PyTorch seeds 0, 1, 2, 3, 4 (`adapt.seed_offset=0`) |
| Source | Seed 0; one warm-up epoch plus ten source-training epochs |
| Adaptation | Thirty epochs; batch size 64; Adam learning rate 1e-4 |
| Model | Trainable LeNet; residual identity-initialised fe/fd with linear outputs |
| Adaptation settings | AdaBN; all five losses; confident La1; top 20% per guide |
| Gaussian scores | Existing log-density logits; no density-softmax change |
| Pseudo-labels | Nearest guide; label propagation and guide rectification disabled |

Random supports change with support seed 0-4. Hand-picked supports stay fixed
while the training seed changes. Candidate-column numbers below are support
sampler seeds, not global dataset indices or training-seed choices. Selection
uses visual appearance, not model/test accuracy.

### Generate the exact selected-support configurations

Run from this repository root after exporting CIDA_PYTHON and CIDA_DATA above.
The candidate count is 20 for every command; picks are ordered by digits 5-9.
Keep the generated YAML files: they record the fixed per-class dataset positions
and the data root used for the run.

```bash
"$CIDA_PYTHON" scripts/choose_digit_supports.py --config configs/digits_m2u_improved.yaml --data-root "$CIDA_DATA" --candidates 20 --picks 0,11,10,11,14
"$CIDA_PYTHON" scripts/choose_digit_supports.py --config configs/digits_u2m.yaml --data-root "$CIDA_DATA" --candidates 20 --picks 12,6,4,14,19
"$CIDA_PYTHON" scripts/choose_digit_supports.py --config configs/digits_s2m.yaml --data-root "$CIDA_DATA" --candidates 20 --picks 12,6,4,14,19
```

The picker writes `configs/digits_m2u_clearshot.yaml`,
`configs/digits_u2m_clearshot.yaml`, and `configs/digits_s2m_clearshot.yaml`.
M->U inherits the task name `digits_m2u` but keeps the improved config's output
directory `runs/digits_m2u_siddharth`.

### Submit the selected-support runs

```bash
mkdir -p logs
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_m2u_clearshot.yaml adapt.la1_samples=confident adapt.tag=clearshot_0-11-10-11-14
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_u2m_clearshot.yaml adapt.la1_samples=confident adapt.tag=clearshot_12-6-4-14-19_la1conf
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_s2m_clearshot.yaml adapt.la1_samples=confident adapt.tag=clearshot_12-6-4-14-19_la1conf
```

These write to `runs/digits_m2u_siddharth`, `runs/digits_u2m`, and
`runs/digits_s2m`. Source training is skipped when its final checkpoint exists.
The random-suite submission commands in section 1 below write separately to
`runs_digits_random_confident/<task>`. Both groups use the common configured
training settings above; the source is shared by adaptation seeds within a task.

### Summarise the recorded results

```bash
"$CIDA_PYTHON" scripts/best_epoch_summary.py runs/ --csv digits_selected_summary.csv
"$CIDA_PYTHON" scripts/best_epoch_summary.py runs_digits_random_confident/ --csv digits_random_summary.csv
```

For this comparison, use only the `EPOCH 15` rows with the exact confident-La1
tags above and `random_la1conf`. The script also prints BEST and FINAL diagnostics;
neither is the reported checkpoint here. Require epoch-15 coverage of 5/5 seeds.
CSV columns beginning `epoch15_` contain the corresponding per-seed measurements.
Epoch 15 is a common reporting checkpoint, not a demonstrated convergence rule.

Recorded epoch-15 scores (percent mean +/- sample standard deviation, n=5):

| Transfer | Supports | ALL | SHARED | PRIV |
|---|---|---|---|---|
| M->U | Random | 71.61 +/- 10.01 | 81.36 +/- 8.47 | 57.49 +/- 14.21 |
| M->U | Selected | 83.73 +/- 1.83 | 89.42 +/- 1.23 | 75.49 +/- 2.84 |
| U->M | Random | 68.30 +/- 4.95 | 81.41 +/- 2.76 | 54.43 +/- 12.74 |
| U->M | Selected | 77.10 +/- 4.65 | 82.86 +/- 1.52 | 71.02 +/- 8.76 |
| S->M | Random | 58.15 +/- 11.31 | 63.37 +/- 11.96 | 52.63 +/- 10.71 |
| S->M | Selected | 60.37 +/- 4.07 | 61.57 +/- 4.25 | 59.09 +/- 4.54 |

Preserve each source `final.pt`, the generated support YAML, checkpoint `cfg` and
`labelled` entries, Python/package versions, and run logs to audit an exact rerun.
The supplied log archive validates the recorded scores but contains no checkpoint
weights, so it cannot establish source-checkpoint identity between the two groups.
Fresh training is a repeat of the configured procedure, not a guarantee of
bitwise-identical numbers. Existing completed outputs are skipped; use a separate
output location when intentionally starting a fresh experiment.

## 1. Improved baseline with random supports

### Automatic restart near the Slurm time limit (all three transfers)

Use the existing environment exports above, then choose one transfer:

```bash
mkdir -p logs
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_m2u.yaml preset=digits_random_confident "data_root=$CIDA_DATA"
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_u2m.yaml preset=digits_random_confident "data_root=$CIDA_DATA"
sbatch --export=ALL,SEEDS="0 1 2 3 4",STAGES="source adapt" slurm/run.sbatch configs/digits_s2m.yaml preset=digits_random_confident "data_root=$CIDA_DATA"
```

Each job trains one source model and adaptation seeds 0-4 sequentially. For each seed,
one target-training image per private digit (5-9) is selected randomly with that seed;
there is no hand-picked support set or label propagation. Actual training RNG seeds
are also 0-4. The source is shared across seeds. The LeNet trains normally; source
training is one warm-up plus ten epochs and adaptation is thirty epochs per seed.
The Gaussian logits retain the existing log-density implementation.

Outputs are isolated under `runs_digits_random_confident/digits_m2u`, `digits_u2m`
or `digits_s2m`. Checkpoints are saved every five minutes and after every epoch.
Three minutes before the 5h55 allocation ends, Slurm signals the batch shell, which
requests a fresh checkpoint from Python, waits for its safe exit, and calls
`scontrol requeue`. The same job then waits for another allocation and resumes:
finished source/seed stages are skipped and the unfinished stage reloads its model,
Adam states and RNG checkpoint. Logs append across allocations.

An unfinished epoch is replayed from its checkpointed weights; minibatch position
and worker-prefetch state are not restored, so this is not bitwise identical to an
uninterrupted run. Requeue needs the cluster to permit owner requeue requests. If
Slurm rejects it, the job prints an error and retains its checkpoint; submitting the
same command resumes manually. A manual cancellation is not automatically resubmitted.

For example, monitor and summarise U->M with:

```bash
squeue -u "$USER"
tail -f logs/cida_JOBID.out
"$CIDA_PYTHON" scripts/aggregate.py runs_digits_random_confident/
```

Do not change the preset, output folder or seed list when resuming an experiment.

### Original random-support configuration

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

## Best-epoch diagnostic summary

For a diagnostic best-epoch summary from existing logs (no retraining):

```bash
"$CIDA_PYTHON" scripts/best_epoch_summary.py runs/ --csv best_epochs.csv
```

This chooses the epoch with highest test ALL accuracy independently for each seed,
then reports ALL, SHARED, PRIV and ALL(cls) from that same epoch, plus fixed epoch-15
and final-epoch results for comparison. All three rows use mean and sample standard
deviation across seeds. Epoch-15 coverage is shown explicitly; missing epoch-15
records are excluded from that row, with no other epoch substituted. Per-seed epoch-15
scores are also included in the optional CSV. `--min-epoch` filters BEST selection
only; it does not remove the fixed epoch-15 comparison.
Epoch 0 is excluded from BEST by default; ties use the earliest epoch. Only
completed runs are included. This is explicitly test-selected diagnostic reporting,
not a validation-selected checkpoint or an unbiased benchmark result. Add
`--select-by priv` only when you explicitly want selection by private test accuracy.
The script summarises logged scores; it cannot restore unsaved epoch weights.

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
