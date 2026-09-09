# Fine-Tuning Steps

The full path from a folder of PDFs to a published model. Each stage is one
command, and each explains the settings that are not obvious.

Prerequisites are in [SETUP.md](SETUP.md).

---

## Step 1 — Render PDFs to page images

```bash
python -m doclayout_ft.data.pdf_to_images
python -m doclayout_ft.data.pdf_to_images --limit 10 --dry-run   # try it first
```

Reads `PDFs/`, writes one JPEG per page into `images/`, named
`<PaperName>_page_NN.jpg`.

**That naming is load-bearing.** The `_page_NN` suffix is what the splitter
later strips to group pages by paper. Rename these files and the paper-level
split silently degrades into a page-level one.

**Why 300 DPI.** At 150, footnote and page-header text is too soft to place a
tight box around, and the annotations inherit that imprecision. At 600 the disk
cost doubles for no annotation benefit at this page size.

Re-running skips PDFs already rendered, including pages that have since been
moved into `batch_NN/` folders for annotation. Pass `--overwrite` to redo them.

---

## Step 2 — Carve off a batch to annotate

```bash
python -m doclayout_ft.data.make_batches --batch-name batch_06
python -m doclayout_ft.data.make_batches --batch-name batch_06 --dry-run
```

Moves a sample of pages into `images/batch_06/` so the annotation tool can be
pointed at one folder.

**Why page 1 is over-sampled.** The default is 200 first pages against 650
interior pages, far above their natural ratio. `Title` and `Authors` occur on
the first page of a paper and essentially nowhere else, so a uniform sample
would leave both classes too rare to learn. Roughly one part page-1 to three
parts interior keeps them learnable without drowning out the body-text layouts
that dominate real documents.

**The move is destructive on purpose.** A page belongs to exactly one batch, so
it can never be annotated twice under two different batch folders. Use `--copy`
if you would rather leave the pool intact, and accept that risk.

---

## Step 3 — Annotate

Done by hand in [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling),
exported in YOLO horizontal-bounding-box format. Dockerised, so the annotation
environment does not disturb the training one.

Assemble the exports into a round directory:

```
training_dataset/round_final/
├── images/        page images
├── labels/        one <stem>.txt per image
├── classes.txt    the 12 class names, in ID order
└── data.yaml      Ultralytics dataset config
```

`classes.txt` must match `CLASS_NAMES` in `doclayout_ft/config.py` exactly, in
order. YOLO labels reference classes by integer index alone, so a reordering
silently retrains every class onto the wrong label. A test enforces this.

Annotation guidance, including the `Page-footer` tightness problem worth
avoiding, is in [DATASET.md](DATASET.md).

---

## Step 4 — Split by paper

```bash
python -m doclayout_ft.data.split_dataset --round round_final
python -m doclayout_ft.data.split_dataset --round round_final --dry-run
```

Writes `train.txt`, `val.txt` and `test.txt`, and repoints `data.yaml` at them.
Images and labels are not moved.

**Why by paper and not by page.** Pages of one paper share a template: the same
column grid, the same header, the same caption style. Put page 3 in train and
page 7 in val, and the model has effectively seen the validation layout during
training. Validation mAP then measures memorisation.

This was a correction made mid-project. Rounds up to and including `round_03`
pointed `train:` and `val:` at the same image folder, so every metric from that
period is void. `round_final` is the first round split honestly.

**Why absolute paths in the list files.** Ultralytics only prefixes the
dataset's `path:` onto entries beginning with `./`. Any other relative entry is
resolved against the working directory at train time, not the dataset
directory. Absolute paths make the lists correct wherever training is launched
from.

Defaults are 75 / 15 / 10 by paper, seeded at 42 so the split is reproducible.
Test takes the remainder rather than its own rounded count, so no paper is ever
dropped.

---

## Step 5 — Fetch base checkpoints

```bash
python -m doclayout_ft.training.download_base_models --variants n s
```

Fine-tuning does not start from COCO weights. It starts from
[`Armaggheddon/yolo11-document-layout`](https://huggingface.co/Armaggheddon/yolo11-document-layout),
already trained for document layout on DocLayNet-style data. Starting from a
model that already knows what a caption looks like is what makes a few hundred
annotated pages sufficient.

| Variant | Parameters | On a 4 GB card |
|---|---|---|
| `yolo11n` | 2.6 M | Batch 4 at imgsz 1024. About 40 minutes. |
| `yolo11s` | 9.4 M | Batch 3 at imgsz 1024. About 3.5 hours. Best results. |
| `yolo11m` | 20 M | Does not fit. See below. |

`yolo11m` is excluded from the default download. It ran out of memory at batch
5 and batch 2, and only completed a forward pass at batch 1. Batch size 1 makes
batch-norm statistics unreliable, since a single sample gives a meaningless
estimate of activation mean and variance, and no amount of gradient
accumulation fixes that. It needs a larger card to assess fairly.

---

## Step 6 — Fine-tune

```bash
python -m doclayout_ft.training.finetune --list                      # what is available
python -m doclayout_ft.training.finetune --only yolo11s_doc_layout   # one run
python -m doclayout_ft.training.finetune --dry-run                   # the whole sweep
```

Each checkpoint under `models/` becomes its own run, writing to
`models/<name>_<suffix>/`. A run whose output directory exists is skipped,
which makes the sweep restartable: if the fifth of twenty runs dies to an
out-of-memory error, re-running resumes from there.

### Settings that differ from the Ultralytics defaults

**`--imgsz 1024`**, against a default of 640. The largest single accuracy gain
in this project, roughly +0.03 mAP50-95, concentrated in the small classes.
`Footnote`, `Page-header` and `Page-footer` are a few pixels tall once a
2550-pixel-wide page is scaled to 640. This one is measured across the whole
checkpoint lineage and is safe to rely on.

**`--epochs 210 --patience 60`.** A generous cap with early stopping, so a
slow-converging run has room and a converged one stops.

Everything else matches Ultralytics, deliberately. The augmentation defaults
here (`fliplr=0.5`, `erasing=0.4`, `copy_paste=0.0`, `multi_scale=0.0`)
reproduce `yolo11s_doc_layout_imgsz_1024`, the strongest model in the project.

**AMP** is left to Ultralytics. On a GTX 1650 its sanity check fails and
training falls back to FP32, which is why memory is tighter than 4 GB suggests.

### The augmentation settings people are tempted to change

There is a good a priori argument for `--fliplr 0.0` on documents: a mirrored
page never occurs at inference, and mirrored text destroys the left-to-right
structure that distinguishes a caption from a list item. The parallel argument
says `--erasing 0.0`, since erasing can remove the only instance of a rare
class from a page.

Both are plausible. **Neither has been tested here in isolation**, and every
model that scored well used the Ultralytics defaults for them. They are
available as flags and would be a cheap, worthwhile experiment: one run each,
changing nothing else.

### Batch size and memory

Start from these and adjust:

| Model | imgsz | Batch | Peak VRAM |
|---|---|---|---|
| yolo11n | 1024 | 4 | ~3.2 GB |
| yolo11s | 1024 | 3 | ~3.6 GB |
| yolo11n | 640 | 8 | ~2.8 GB |

On a CUDA out-of-memory error, halve `--batch` first. Only drop `--imgsz` if
that fails, since resolution is where the accuracy is. If system RAM rather
than VRAM is the constraint, set `--cache disk` or `--cache false`.

#### `--multi-scale` costs far more memory than it looks

This one silently killed four runs in this project, and the number is not
obvious. Ultralytics computes:

```
max_imgsz = imgsz * (1 + multi_scale)
```

So `--multi-scale 0.5` at `--imgsz 1024` trains on images up to **1536 pixels**.
Activation memory grows with area, so that is **2.25x** the peak of a plain 1024
run. On a 4 GB card it dies in the first epoch, before any weights are written.

Measured on this hardware:

| multi_scale | Peak image size | Memory vs 1024 | Result at batch 3-4 |
|---|---|---|---|
| 0.0 | 1024 | 1.00x | fine |
| 0.23 – 0.25 | 1259 – 1280 | ~1.55x | fine |
| 0.5 | 1536 | 2.25x | **dies in epoch 1** |

The failure is quiet. Ultralytics creates the run directory and writes its
dataset plots before training starts, so a dead run leaves behind a folder that
looks real and contains no weights. Checkpoint discovery finds no `best.pt` and
skips it, so the run is simply absent from every later table with nothing
saying why. Run `python -m doclayout_ft.audit` to see them.

### The augmentation experiment that failed

An `attempt_02` sweep raised `copy_paste` to 0.2 – 0.3 and `multi_scale` to
0.23 – 0.5. It regressed **every model it was applied to**, by 0.016 to 0.060
mAP50-95, without exception.

It also changed five other things at the same time: `fliplr` 0.5 to 0.0,
`erasing` 0.4 to 0.0, more epochs, higher patience, and a smaller batch. So the
sweep proves the bundle is harmful, not which part of it is. Two of those
changes were expected to *help*, which means the cost of whatever did the
damage is larger than the measured delta.

The mechanistic suspicion falls on `copy_paste`, for a reason independent of
the experiment: the runs recorded `copy_paste_mode: flip`, which mirrors each
pasted instance crop *even when whole-image `fliplr` is 0*. These runs
reintroduced mirrored text at the instance level while believing they had
turned mirroring off. Copy-paste also composites regions from unrelated pages
into positions that cannot occur, and in a document, unlike a natural scene,
the spatial relationship between regions carries real signal.

**Practical advice: leave `copy_paste` and `multi_scale` off**, which is the
default and matches every run that scored well. If you want to know which one
matters, change one at a time. See docs/EVALUATION.md.

## Step 7 — Evaluate

```bash
python -m doclayout_ft.evaluation.evaluate --split val
python -m doclayout_ft.evaluation.per_class --model yolo11s_doc_layout_imgsz_1024
```

Writes `reports/evaluation_val.csv` and `.png`. Each model is scored at the
image size recorded in its own `args.yaml`, so the table ranks models rather
than resolution mismatches.

Read [EVALUATION.md](EVALUATION.md) before quoting a number externally. The
same checkpoint scores differently on `val`, `test` and `combined`, and which
one you cite changes what the claim means.

---

## Step 8 — Publish

```bash
python -m doclayout_ft.hub.push_to_hub --list          # the queue
python -m doclayout_ft.hub.push_to_hub --limit 2       # dry run
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes # upload
```

See [PUBLISHING_TO_HF.md](PUBLISHING_TO_HF.md).

---

## Rough timings on the reference machine

| Stage | Time |
|---|---|
| Render 1128 PDFs to ~22,000 pages | ~5 hours |
| Annotate one 850-page batch | Days, by hand |
| Split a round | Under a second |
| Fine-tune yolo11n at 1024 | ~40 minutes |
| Fine-tune yolo11s at 1024 | ~3.5 hours |
| Evaluate 17 models on val | ~4 minutes |
