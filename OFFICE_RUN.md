# Siddharth's Office runs

The same workflow as the digit experiments: inspect one-shot images, keep one image
per private class fixed across five training seeds, train with confident La1, and
summarise BEST / EPOCH 15 / FINAL. Choose supports by appearance before examining
their model performance. This is a hand-picked-support experiment, distinct from
the paper's five random support draws.

## Configurations

| New config | Transfer | Shared / private | Output directory |
|---|---|---|---|
| `configs/office20_a2d_siddharth.yaml` (picker default) | Amazon -> DSLR | 20 / 11 | `runs/office20_a2d_siddharth` |
| `configs/office_a2d_siddharth.yaml` | Amazon -> DSLR | 10 / 10 | `runs/office_a2d_siddharth` |

The `office20` split uses alphabetic class indices 0-19 shared and 20-30 private.
The other split uses 0-9 shared and 20-29 private. These are different experiments;
do not combine their results or paper reference rows. Existing transfer configs
cover all six directions among Amazon, DSLR and Webcam; the picker accepts any
`configs/office20_*.yaml` or `configs/office_*.yaml` task config and writes an isolated
`*_siddharth_clearshot.yaml` config for it.

Both new baselines use ResNet-50 pretrained on ImageNet, the existing 256-dimensional
feature head, and Office augmentations. The backbone weights stay frozen in both
stages (`backbone_lr_mult=0`); trainable FC heads and projections still learn.
Backbone BatchNorm running statistics can update, including AdaBN. This retains the
existing Office configuration; digit runs train their entire LeNet.

Adam uses `0.0001`, batch size 32, one source warm-up epoch plus 50 source epochs,
and 60 target epochs. The duration is an implementation choice, not a documented
convergence criterion. Epoch 15 is an additional fixed comparison. Confident samples
are the top 20% per guide. AdaBN and residual projections with linear outputs are
enabled. All five losses are enabled. Gaussian losses use the existing log-density
logits. Label propagation and guide rectification are disabled. Actual adaptation
RNG seeds are 0-4 and source seed is 0. The baselines draw a new random support set
per seed; generated clearshot configs keep the selected images fixed.

## Existing environment and data

Run on TC2 from the same `CIDA-digits` clone. Reuse the existing uv environment:

```bash
cd ~/CIDA-digits
export CIDA_OLD=/home/msai/siddhart022/CIDA_Class-Incremental-Domain-Adaptation
export CIDA_PYTHON="$CIDA_OLD/.venv/bin/python"
export CIDA_DATA="$CIDA_OLD/data"
mkdir -p logs
```

The configured folder layout is:

```
$CIDA_DATA/office31/amazon/images/<class>/<image>.jpg
$CIDA_DATA/office31/dslr/images/<class>/<image>.jpg
$CIDA_DATA/office31/webcam/images/<class>/<image>.jpg
```

Point `CIDA_DATA` at the appropriate root if your Office data lives elsewhere. Office
is read from existing folders; this code does not download it. Office has no separate
validation split here: adaptation/evaluation use the target domain's images, with
labelled supports excluded from the reported evaluation. Other target labels are
used only for reporting. The first model construction may download ImageNet weights
if they are not already cached.

## Select supports and run

Start with the 20-shared / 11-private Amazon -> DSLR setting:

```bash
"$CIDA_PYTHON" scripts/choose_office_supports.py \
  --data-root "$CIDA_DATA" --candidates 20
```

Open `oneshot_review/office20_a2d_siddharth_support_preview.png`. The preview shows
the deterministic center-cropped view used to build guides, with class names and
filenames. There are two blocks of ten columns, labelled support seeds 0-9 and 10-19.
Choose one seed number per class in the printed order. Candidate seeds select images;
they do not change the training seeds. Repeated candidates are possible, especially
for small classes.

The all-zero picks below are an example, not recommended selections. Replace them
with your eleven choices:

```bash
"$CIDA_PYTHON" scripts/choose_office_supports.py \
  --data-root "$CIDA_DATA" --candidates 20 \
  --picks 0,0,0,0,0,0,0,0,0,0,0

sbatch slurm/run.sbatch configs/office20_a2d_siddharth_clearshot.yaml
```

The generated config records exact image filenames, confident La1 and seed offset 0.
The existing Slurm job trains the source once, then adaptation seeds 0-4, using
`CIDA_PYTHON`. Source checkpoints are reused for later support comparisons within
the same experiment. Changed picks get distinct adaptation tags; completed runs skip
automatically and interrupted runs resume. Existing digit runs use different folders.

For the 10-shared / 10-private setting, add
`--config configs/office_a2d_siddharth.yaml` to both picker commands, provide ten picks,
then submit `configs/office_a2d_siddharth_clearshot.yaml`.

For a different transfer, e.g. Amazon -> Webcam, add
`--config configs/office20_a2w.yaml` to both picker commands. Submit the generated
`configs/office20_a2w_siddharth_clearshot.yaml`. Review supports for each target domain;
DSLR image filenames are not a selection for Webcam or Amazon.

## Summary

```bash
"$CIDA_PYTHON" scripts/best_epoch_summary.py runs/office20_a2d_siddharth \
  --csv office20_a2d_summary.csv
```

This reports ALL, SHARED, PRIV and ALL_CLS as five-seed mean +/- sample standard
deviation. BEST is selected using test ALL and is diagnostic; epoch 15 is fixed and
FINAL is epoch 60. Missing epoch-15 records are explicitly reported. Fixed supports
mean the standard deviation reflects training variation rather than different
support draws. None of these choices defines an automatic convergence rule.
