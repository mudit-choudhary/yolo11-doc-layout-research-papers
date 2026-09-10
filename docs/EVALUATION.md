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

## Latency

Accuracy alone cannot answer which model to run. Measured at batch 1 on the
GTX 1650, each model at its own resolution, excluding image decode:

```bash
python -m doclayout_ft.evaluation.benchmark
```

| Configuration | Params | GFLOPs | Median | FPS |
|---|---|---|---|---|
| yolo11n at 640 | 2.58 M | 6.4 | 9 – 11 ms | 87 – 106 |
| yolo11n at 1024 | 2.58 M | 16.6 | 18 – 19 ms | 54 – 56 |
| yolo11s at 1024 | 9.42 M | 55.4 | 36 – 37 ms | 27 – 28 |

Latency is decided almost entirely by architecture and resolution. Within a
configuration, every variant times the same to within a millisecond, because
they differ only in weights.

Image decode is excluded because it costs the same for every model, roughly
60 to 80 ms for a 2550x3301 JPEG, and would have added a constant that hides
the differences. Warmup inferences are discarded and the GPU is synchronised
around each timed region; without the sync the measurement would be of how fast
Python queues work, not how fast it finishes.

### The trade this exposes

| Model | mAP50-95 | Median | Verdict |
|---|---|---|---|
| `12-yolo11s-1024` | 0.7694 | 37 ms | Accuracy pick. Leads on Table and Footnote. |
| `11-yolo11n-1024` | 0.7661 | 19 ms | **Half the latency for 0.003 mAP.** |
| `05-yolo11n-640` | 0.7376 | 9 ms | Four times the throughput, small classes suffer. |

0.003 mAP50-95 is inside noise. A 2x latency difference is not. For a pipeline
processing pages in bulk, `yolo11n` at 1024 is the better engineering choice,
and the earlier recommendation of `yolo11s` was made without this measurement
in hand.

`yolo11s` keeps a real advantage on `Table` (0.985) and `Footnote`, the two
classes that decide chunk boundaries, so it remains the right pick when
boundary quality matters more than throughput. Both are published.

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

### Older lineages overlap the held-out set, but do not gain from it

Most checkpoints are the product of several successive fine-tuning passes. The
earliest of those used `round_01` and `round_02`, whose `data.yaml` set
`train: images` and `val: images`, so the model validated on its own training
data. Worse, those rounds overlap the current held-out papers:

| Round | Papers | Overlap with round_final val+test |
|---|---|---|
| round_01 | 71 | 10 papers, 7% of held-out |
| round_02 | 257 | 27 papers, 19% of held-out |

Only four runs have a fully clean ancestry, having trained on `round_final`
alone: `11-yolo11n-1024`, `12-yolo11s-1024`, and the two `yolo11s` `augexp`
runs. The other fifteen descend from something that saw part of today's
validation set.

That looks like it should inflate their scores. **It was tested, and it does
not.** Splitting the 130 validation pages into the 40 whose papers an ancestor
had seen and the 90 it had not:

| Model | Lineage | Seen by an ancestor | Never seen | Gap |
|---|---|---|---|---|
| 08-yolo11n-1024 | chained | 0.7937 | 0.7691 | +0.0246 |
| 09-yolo11n-1024 | chained | 0.7756 | 0.7667 | +0.0089 |
| 11-yolo11n-1024 | **clean** | 0.7843 | 0.7647 | +0.0196 |
| 12-yolo11s-1024 | **clean** | 0.7922 | 0.7667 | +0.0255 |

If contamination were driving the gap, the chained models would show a larger
one than the clean models. They do not: the largest gap of the four belongs to
`12-yolo11s-1024`, which has a spotless lineage. The 40 overlapping papers are
simply easier pages for every model.

The most likely reason is that the ancestral exposure was several fine-tuning
generations and one resolution change ago, and subsequent training on
`round_final` overwrote it. Detection also does not memorise the way
classification can.

**So the table above is comparable across lineages.** The reason to prefer a
clean lineage is reproducibility, not score integrity.

On the 90 genuinely-unseen pages the ranking is `08` at 0.7691, then `09` and
`12` tied at 0.7667, then `11` at 0.7647. A spread of 0.004 across four models
is noise. Nothing here separates them on accuracy.

### Why yolo11s is recommended over the checkpoint that scores highest

`08-yolo11n-1024` edges `12-yolo11s-1024` by about 0.002 overall and 0.002 on
unseen pages. That is not a difference.

The recommendation goes to `12-yolo11s-1024` on grounds other than the headline
number:

- **It is one fine-tune from a published base checkpoint.** `08` is three
  passes deep over `round_02` and an early flat dataset that no longer exists in
  its original form. Reproducing `08` is not currently possible; reproducing
  `12` is one command.
- **It leads on the classes that decide chunk boundaries.** `Table` at 0.985 and
  `Footnote` at 0.678, against `yolo11n`'s weaker numbers on both.
- **Its lineage needs no caveat**, which matters for something being published
  for other people to use.

If throughput matters more than any of that, `11-yolo11n-1024` is 0.005 behind
and roughly five times faster to train.

### The attempt_02 sweep regressed everything, but does not say why

Every `attempt_02` run scored below the checkpoint it was fine-tuned from, on
the held-out validation split:

| Starting checkpoint | Base | With attempt_02 | Delta |
|---|---|---|---|
| yolo11s_doc_layout_imgsz_1024 | 0.7694 | 0.7533 | -0.0161 |
| yolo11_doc_layout_v222_round03 | 0.7442 | 0.7138 | -0.0304 |
| yolo11_doc_layout_v222_imgsz_1024 | 0.7623 | 0.7108 | -0.0515 |
| yolo11_doc_layout_v222_round03_imgsz_1024 | 0.7666 | 0.7070 | -0.0596 |

The direction is consistent and the effect is not small. What it cannot do is
name a cause, because **the sweep changed seven settings at once**. Read from
the runs' own `args.yaml`, against their parents:

| Setting | Before | After | Expected effect |
|---|---|---|---|
| `copy_paste` | 0.0 | 0.2 – 0.3 | Suspect |
| `multi_scale` | 0.0 | 0.23 – 0.5 | Suspect |
| `fliplr` | 0.5 | 0.0 | Should **help** a document task |
| `erasing` | 0.4 | 0.0 | Should **help** rare classes |
| `epochs` | 100 – 150 | 210 | Neutral; early stopping governs |
| `patience` | 45 | 60 | Neutral |
| `batch` | 3 – 5 | 2 – 3 | Minor; see below |

Two of those changes were expected to improve things, which makes the result
more interesting rather than less. If disabling horizontal flip and random
erasing genuinely helps a document task, then whatever `copy_paste` and
`multi_scale` cost is **larger** than the measured delta, because it had to
overcome two improvements to land where it did.

The batch reduction is a weak confound. Ultralytics accumulates gradients to a
nominal batch of 64 (`nbs: 64`) regardless of the micro-batch, so the effective
optimisation batch did not change. Batch-norm statistics still come from the
micro-batch, so 2 is noisier than 5, but this is a second-order effect next to
an augmentation change of this size.

Epoch count is not the explanation either. Two of the three cleanest
`attempt_02` runs early-stopped **sooner** than their parents despite a higher
cap and higher patience (73 against 93, and 61 against 101). They plateaued
earlier and lower. The third ran far longer, 189 against 100, and still
finished worse.

So the honest conclusion is narrower than "copy-paste is bad":

**`copy_paste` and `multi_scale`, applied together at these strengths, cost
0.016 to 0.060 mAP50-95 on this task. Which of the two is responsible, or
whether it takes both, was never isolated.**

The mechanistic case against `copy_paste` specifically is still worth stating,
because it is independent of this experiment. The runs recorded
`copy_paste_mode: flip`, which mirrors each pasted instance crop *even when
whole-image `fliplr` is 0*. So these runs reintroduced mirrored text at the
instance level while believing they had turned mirroring off. Separately,
compositing regions from unrelated pages produces layouts that cannot occur,
and the spatial relationships between document regions carry real signal.

That argument is a reason to be suspicious, not a measurement. Isolating it
needs one run changing `copy_paste` alone.

### The fliplr and erasing settings are untested here

Worth being explicit, because the reasoning is seductive. Disabling horizontal
flip on a document task is a well-argued idea: a mirrored page never occurs at
inference. Lowering random erasing to protect rare classes is similarly
plausible.

Neither has been tested in isolation in this project. Both appear only inside
the `attempt_02` bundle, alongside the changes that regressed. **Every model
that scored well here was trained with Ultralytics' defaults, `fliplr=0.5` and
`erasing=0.4`**, including the recommended `yolo11s_doc_layout_imgsz_1024`.

The fine-tuning defaults therefore match the best run rather than the better
argument. Reproducing a published model matters more than acting on untested
reasoning. Both remain flags, and isolating them is cheap: one run each.

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
