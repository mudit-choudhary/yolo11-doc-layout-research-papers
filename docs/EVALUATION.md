# Evaluation

What the numbers mean, which one to quote, and what the results actually say.

## Reproducing

```bash
python -m doclayout_ft.evaluation.evaluate --split val
python -m doclayout_ft.evaluation.per_class --model yolo11s_doc_layout_imgsz_1024
```

Results land in `reports/` as a CSV and a chart.

## Three splits, three different numbers

The same checkpoint scores differently depending on which held-out set it is
measured against. This is not noise, and it is the first thing to state
alongside any figure.

| Split | What it is | When to use it |
|---|---|---|
| `val` | Validation only, 130 pages | Comparing against training-time metrics. This is what early stopping watched. |
| `test` | Test only, 88 pages | An external claim. Never touched by model selection. |
| `combined` | val + test merged, 218 pages | A tighter estimate from more pages, but no longer clean, since val guided early stopping. |

`yolo11s_doc_layout_imgsz_1024` scores 0.769 on `val` and 0.753 on `combined`.
Neither is wrong. Quoting either without naming the split is.

**Use `val` for comparing models to each other, and `test` for stating how good
the chosen model is.** That separation is the entire point of holding out a
third set: once val has been used to pick a winner, it has been fit to, however
indirectly.

## Why each model is scored at its own image size

The evaluation reads `imgsz` from each run's `args.yaml` and validates at that
size. Scoring a model trained at 1024 with 640-pixel inputs costs accuracy that
has nothing to do with the model, and a table built that way would rank
checkpoints by resolution mismatch. Baselines that were never fine-tuned here
are scored at 640, the size they were originally trained at.

## Reading mAP50 against mAP50-95

**mAP50** counts a prediction correct if it overlaps the ground-truth box by
half. That is a low bar. A box covering roughly the right area passes.

**mAP50-95** averages across overlap thresholds from 0.50 to 0.95, so it only
rewards boxes that agree closely with the label's edges.

The gap between them is diagnostic. A class with high mAP50 and low mAP50-95 is
being *found* but not *bounded*, and the usual cause is inconsistent
annotation rather than model capacity. For chunking, mAP50-95 is the number
that matters, since a loose box takes half the next paragraph with it.

## Results, val split, round_final

All 22 models, scored on the 130-page validation split. Runs ending
`attempt_02` carry the failed augmentation experiment and are included so the
regression is visible rather than merely asserted.

| Model | imgsz | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| `yolo11_doc_layout_v2224_imgsz_1024` | 1024 | 0.881 | 0.910 | 0.939 | **0.7719** |
| **`yolo11s_doc_layout_imgsz_1024`** | 1024 | 0.935 | 0.913 | 0.945 | **0.7694** |
| `yolo11_doc_layout_v222_round03_imgsz_1024` | 1024 | 0.923 | 0.907 | 0.943 | 0.7666 |
| `yolo11n_doc_layout_imgsz_1024` | 1024 | 0.921 | 0.883 | 0.940 | 0.7661 |
| `yolo11_doc_layout_v2224_round03_imgsz_1024` | 1024 | 0.899 | 0.906 | 0.938 | 0.7650 |
| `yolo11_doc_layout_v222_imgsz_1024` | 1024 | 0.887 | 0.888 | 0.926 | 0.7623 |
| `yolo11_doc_layout_v22_imgsz_1024` | 1024 | 0.887 | 0.888 | 0.926 | 0.7623 |
| `yolo11_doc_layout_v2_imgsz_1024` | 1024 | 0.924 | 0.890 | 0.932 | 0.7623 |
| `yolo11s_doc_layout_imgsz_1024_attempt_02` | 1024 | 0.899 | 0.886 | 0.928 | 0.7533 |
| `yolo11s_doc_layout_attempt_02` | 1024 | 0.891 | 0.883 | 0.937 | 0.7522 |
| `yolo11_doc_layout_v222_round03` | 640 | 0.897 | 0.844 | 0.918 | 0.7442 |
| `yolo11_doc_layout_v2224_round03` | 640 | 0.861 | 0.886 | 0.918 | 0.7376 |
| `yolo11_doc_layout_v222_round03_attempt_02` | 1024 | 0.861 | 0.842 | 0.892 | 0.7138 |
| `yolo11_doc_layout_v222_imgsz_1024_attempt_02` | 1024 | 0.860 | 0.845 | 0.893 | 0.7108 |
| `yolo11_doc_layout_v222_round03_imgsz_1024_attempt_02` | 1024 | 0.871 | 0.836 | 0.875 | 0.7070 |
| `yolo11_doc_layout_v2224` | 640 | 0.839 | 0.800 | 0.850 | 0.6538 |
| `yolo11_doc_layout_v22` | 640 | 0.716 | 0.709 | 0.728 | 0.5282 |
| `yolo11_doc_layout_v222` | 640 | 0.716 | 0.709 | 0.728 | 0.5282 |
| `yolo11_doc_layout_v2` | 640 | 0.776 | 0.656 | 0.719 | 0.5175 |
| `yolo11m_doc_layout` (baseline) | 640 | 0.735 | 0.578 | 0.711 | 0.4707 |
| `yolo11n_doc_layout` (baseline) | 640 | 0.695 | 0.562 | 0.671 | 0.4340 |
| `yolo11s_doc_layout` (baseline) | 640 | 0.641 | 0.567 | 0.665 | 0.4318 |

Fine-tuning takes the base checkpoints from roughly 0.43 to roughly 0.77
mAP50-95, a 78% relative gain, on a few hundred annotated pages.

`v22` and `v222` score identically because they are the same weights under two
names, an artefact of the run naming during the sprint. The same holds for
`v22_imgsz_1024` and `v222_imgsz_1024`.

## Best model, per class

`yolo11s_doc_layout_imgsz_1024`, val split, 1894 instances.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Table | 1.000 | 0.928 | 0.995 | **0.985** |
| List-item | 0.947 | 0.987 | 0.991 | 0.926 |
| Text | 0.963 | 0.949 | 0.984 | 0.918 |
| Title | 0.987 | 0.966 | 0.989 | 0.886 |
| Authors | 0.974 | 0.906 | 0.959 | 0.810 |
| Picture | 0.880 | 0.871 | 0.896 | 0.797 |
| Formula | 0.895 | 0.960 | 0.978 | 0.772 |
| Caption | 0.947 | 0.872 | 0.929 | 0.762 |
| Page-header | 0.928 | 0.954 | 0.960 | 0.695 |
| Footnote | 0.868 | 0.680 | 0.859 | 0.678 |
| Section-header | 0.898 | 0.976 | 0.943 | 0.643 |
| **Page-footer** | 0.929 | 0.908 | 0.861 | **0.360** |
| **All** | **0.935** | **0.913** | **0.945** | **0.769** |

`Table` and `Text`, the two classes that matter most for chunking, are at 0.99
and 0.92. That is the result worth taking from this table.

## Findings

### Resolution was the largest lever

Moving from 640 to 1024 added roughly 0.03 mAP50-95 to every checkpoint in the
lineage. Small classes gained most. A page rendered at 2550 pixels wide and
scaled to 640 leaves a page header a handful of pixels tall, below what the
detection head can localise.

### Page-footer is a labelling defect

Precision 0.93 and recall 0.91, against mAP50-95 of 0.36. The model finds
essentially every footer and cannot agree with the labels on where it ends. The
number is identical across every model size and every resolution tried, which
rules out capacity and resolution as causes.

The remaining explanation is the labels. Some `Page-footer` boxes evidently
include the rule above the footer, or surrounding whitespace, and others do
not. The model cannot learn a boundary the annotations do not agree on.

The fix is an annotation audit, not more training.

### Copy-paste augmentation regressed everything

An `attempt_02` sweep raising `copy_paste` and `multi_scale` lost accuracy on
every model it touched, without exception. Each `attempt_02` run against the
checkpoint it was fine-tuned from, on the held-out validation split:

| Starting checkpoint | Base | With attempt_02 | Delta |
|---|---|---|---|
| yolo11s_doc_layout_imgsz_1024 | 0.7694 | 0.7533 | -0.0161 |
| yolo11_doc_layout_v222_round03 | 0.7442 | 0.7138 | -0.0304 |
| yolo11_doc_layout_v222_imgsz_1024 | 0.7623 | 0.7108 | -0.0515 |
| yolo11_doc_layout_v222_round03_imgsz_1024 | 0.7666 | 0.7070 | -0.0596 |

Two causes:

1. Ultralytics' `copy_paste_mode="flip"` mirrors each pasted crop even when
   whole-image `fliplr` is off, reintroducing mirrored text at instance level.
2. Compositing regions between unrelated pages produces layouts that cannot
   occur. Spatial relationships between document regions carry signal, and
   copy-paste destroys it.

A consistent regression across every model is a finding, not noise. Do not use
copy-paste for document layout.

### yolo11s over yolo11n, narrowly

0.769 against 0.766 overall, which is within noise. The case for `s` is
per-class: it is better on `Table` and `Footnote`, the classes that decide
chunk boundaries. The cost is roughly 3.5 hours of training against 40 minutes,
and batch 3 rather than 4.

If throughput matters more than the last point of accuracy, `yolo11n` is the
better trade.

### yolo11m was never fairly assessed

It ran out of memory at batch 5 and batch 2 on a 4 GB card, and only completed
a forward pass at batch 1, where batch-norm statistics are unreliable. Its
baseline number, 0.471, is the highest of the three baselines, so the capacity
may well be worth having. It needs a larger card to find out.

## Known gaps

1. **Box metrics have plateaued around 0.77.** The next meaningful signal is
   task-level: word coverage, leakage, orphan runs and boundary offset measured
   against extracted text positions. That is what actually determines chunk
   quality, and mAP is only a proxy for it.
2. **`test` has not been used for a headline claim.** Everything above is `val`.
   Score on `test` once, at the end, for an external number.
3. **The lineage is entangled.** Several `v2`, `v22`, `v222` checkpoints were
   fine-tuned on top of one another across rounds, including the void
   `round_03`. Their relative ordering in the table is not a clean experiment.
