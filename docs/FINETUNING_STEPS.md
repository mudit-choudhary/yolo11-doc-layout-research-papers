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
2550-pixel-wide page is scaled to 640.

**`--fliplr 0.0`**, against a default of 0.5. Ultralytics mirrors half of all
training images. A mirrored page never occurs at inference, and mirrored text
destroys the left-to-right structure that distinguishes a caption from a list
item. This is free accuracy on any document task.

**`--copy-paste 0.0`**, against an Ultralytics default that is also 0. It is
called out because raising it was tried, deliberately, and it failed. See
below.

**`--erasing 0.0`**, against a default of 0.4. Random erasing can remove the
only instance of a rare class from a page, which for `Authors` or `Footnote`
means training on a page labelled for something no longer visible.

**AMP** is left to Ultralytics. On a GTX 1650 its sanity check fails and
training falls back to FP32, which is why memory is tighter than 4 GB suggests.

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

### The augmentation experiment that failed

An `attempt_02` sweep raised `copy_paste` to 0.2 to 0.3 and `multi_scale` to
0.23 to 0.5, hoping to increase exposure to rare classes. It regressed **every
model it was applied to**, by 0.016 to 0.060 mAP50-95, without exception.

Two reasons, and the first is a trap worth remembering:

1. Ultralytics' default `copy_paste_mode="flip"` mirrors each pasted instance
   crop *even when whole-image `fliplr` is disabled*. Turning off flipping at
   the image level does not turn it off at the instance level, so mirrored text
   came back in through the side door.
2. Copy-paste composites regions from unrelated pages into positions that
   cannot occur. In natural-scene detection, its original setting, a cat may
   legitimately appear anywhere. In a document, the spatial relationship
   between regions carries real signal, and destroying it destroys information
   the model was using.

**Do not use copy-paste for document layout.** Rare-class exposure should come
from more annotated real pages.

---

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
