# Run inventory

What is actually on disk under `models/` and `FinetunedModels/`, and which of
it is real. Regenerate at any time:

```bash
python -m doclayout_ft.audit            # problems only
python -m doclayout_ft.audit --verbose  # every run
```

Exit code is 0 when everything is healthy, 1 otherwise.

## Why this document exists

A failed run does not look failed. Ultralytics creates the run directory and
writes its dataset plots *before* training starts, so a run that dies in its
first epoch leaves behind a folder containing `args.yaml`, `labels.jpg` and a
few `train_batch` mosaics, and no weights.

Nothing downstream complains. Checkpoint discovery looks for
`weights/best.pt`, does not find one, and moves on. The run is then absent from
every evaluation table and from the publish queue, with nothing anywhere saying
why. Five runs are in that state.

## Current state

| Directory | Complete | Incomplete | Dead | Base checkpoints |
|---|---|---|---|---|
| `FinetunedModels/` | 14 | 0 | 1 | 3 |
| `models/` | 12 | 0 | 4 | 2 |

Seven run names appear in both directories; those are copies of the same run,
not separate models. Across both, there are **17 distinct publishable models**,
plus two exact duplicates that are skipped.

## The five dead runs

None wrote weights. All died on CUDA out-of-memory, and the cause is
recoverable from each one's `args.yaml`.

| Run | Cause |
|---|---|
| `models/yolo11n_doc_layout_attempt_02` | `multi_scale=0.5` at `imgsz=1024`, batch 4 |
| `models/yolo11_doc_layout_v22_attempt_02` | `multi_scale=0.5` at `imgsz=1024`, batch 4 |
| `models/yolo11_doc_layout_v222_attempt_02` | `multi_scale=0.5` at `imgsz=1024`, batch 3 |
| `models/yolo11_doc_layout_v2224_attempt_02` | `multi_scale=0.5` at `imgsz=1024`, batch 3 |
| `FinetunedModels/yolo11m_doc_layout_imgsz_1024` | `yolo11m` at batch 5, `imgsz=1024` |

### The multi_scale trap

Ultralytics computes `max_imgsz = imgsz * (1 + multi_scale)`, verified in
`ultralytics/engine/trainer.py`. So `multi_scale=0.5` at `imgsz=1024` trains on
images up to 1536 pixels, and activation memory grows with area:

| multi_scale | Peak size | Memory vs 1024 | Outcome at batch 3-4 |
|---|---|---|---|
| 0.0 | 1024 | 1.00x | fine |
| 0.23 – 0.25 | 1259 – 1280 | ~1.55x | fine |
| 0.5 | 1536 | 2.25x | **dies in epoch 1** |

This is exactly why the `attempt_02` sweep has gaps. The four runs launched
with `multi_scale=0.5` died; the ones later lowered to 0.23 – 0.25 survived and
are the five `augexp` variants that get published.

### yolo11m

Consistent with the sprint report: `yolo11m` at 20M parameters does not train on
a 4 GB card at 1024 pixels. It OOM'd at batch 5 and batch 2, and completed only
a forward pass at batch 1, where batch-norm statistics come from a single
sample and are meaningless. Its only honest number here is its un-fine-tuned
baseline, 0.4707.

## Nothing needs deleting

The dead directories are harmless. They hold no weights, publish nothing, and
appear in no table. They are also the only surviving record that those runs were
attempted, which is worth keeping given the whole `attempt_02` sweep is
published as a documented negative result.

They do contain `train_batch*.jpg` mosaics of real annotated pages. Those are
never uploaded: the publisher never discovers these directories at all, and
three tests independently keep page mosaics off the Hub.

## One run needed repair

`yolo11_doc_layout_v22` had weights and a full 100-epoch `results.csv` but no
plots, meaning training ended before the final validation pass. Its
`results.png` was reconstructed from its own `results.csv`, and its confusion
matrix and curves regenerated from the saved checkpoint against the
`round_final` held-out split, since the dataset it originally trained on no
longer exists at its recorded path. Validation reproduced 0.5282 mAP50-95,
matching the recorded score exactly. Its published card says all of this.

Both copies of that run, in `models/` and `FinetunedModels/`, now carry
identical plots.
