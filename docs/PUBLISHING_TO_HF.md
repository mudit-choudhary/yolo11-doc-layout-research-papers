# Publishing to the Hugging Face Hub

Everything is published into **one** repository, with each checkpoint in its own
subfolder:

```
darkdwine/yolo11-doc-layout-research-papers/
├── README.md                       <- comparison across all variants
├── 01-yolo11n-640-v2/
├── 02-yolo11n-640-v22/
├── ...
├── 11-yolo11n-1024/
├── 12-yolo11s-1024/                <- recommended
│   ├── README.md                   <- this variant's own card
│   ├── best.pt
│   ├── args.yaml
│   ├── results.csv
│   └── ...curves and confusion matrices...
├── 13-yolo11n-1024-v222-augexp/
└── ...
```

One repository, not one per model. These are variants of a single model family,
so someone comparing them should not have to open twenty pages. The root card
puts the comparison table in front of them on arrival.

Uploads are staged: runs go up oldest first, a couple at a time, so the backlog
is cleared over several weeks rather than in one bulk push.

## Before the first push

Authenticate through the Hugging Face CLI. The script never handles a token
itself, and no token should ever be written into this repository.

```bash
hf auth login          # newer huggingface_hub
huggingface-cli login  # older
```

Generate the evaluation the cards will quote, otherwise they fall back to
training-time numbers and mark themselves as such:

```bash
python -m doclayout_ft.evaluation.evaluate --models-dir FinetunedModels models --split val
```

Then look at the queue:

```bash
python -m doclayout_ft.hub.push_to_hub --list
```

## Publishing

```bash
# Dry run. Prints exactly what would be written and where. Uploads nothing.
python -m doclayout_ft.hub.push_to_hub --limit 2

# The same batch, for real.
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

**Nothing uploads without `--yes`.** Read one dry run before the first real
push. A Hub repository is public by default, and its commit history is not
quietly rewritable.

Run the same command again next week and it picks up the next two.

### Which models are in the queue

By default only `FinetunedModels/`, which holds 14 curated runs. The working
`models/` directory holds 5 more, all of them the `attempt_02` runs that
carry the failed augmentation experiment. To publish those too:

```bash
python -m doclayout_ft.hub.push_to_hub --models-dir FinetunedModels models --list
```

Seven run names exist in both directories. The first directory listed wins, so
keep `FinetunedModels` first.

Publishing the failures is a defensible choice, since a documented negative
result is more useful than an undocumented one, and their cards say plainly
that they should not be deployed. Leaving them out is equally defensible. The
default leaves them out.

### Useful variations

```bash
# One specific model, out of queue order
python -m doclayout_ft.hub.push_to_hub --only yolo11s_doc_layout_imgsz_1024 --yes

# Everything still pending, in one pass
python -m doclayout_ft.hub.push_to_hub --limit 0 --yes

# Private, for a collection not ready to be seen
python -m doclayout_ft.hub.push_to_hub --private --yes

# Re-upload a variant already published, after fixing something
python -m doclayout_ft.hub.push_to_hub --only <run> --include-published --yes

# Rebuild only the root comparison table, uploading no weights
python -m doclayout_ft.hub.push_to_hub --refresh-index --yes

# A different repository entirely
python -m doclayout_ft.hub.push_to_hub --repo-id <namespace>/<name> --yes
```

## What each variant folder contains

| File | Source |
|---|---|
| `README.md` | The generated card for this variant |
| `best.pt` | The run's best checkpoint |
| `args.yaml` | Hyperparameters the run was launched with |
| `results.csv` | Per-epoch metrics |
| `results.png` | Training curves |
| `confusion_matrix*.png` | Confusion matrices |
| `Box*_curve.png` | Precision, recall, F1 and PR curves |
| `labels.jpg` | Class distribution of the training set |

Weights are always uploaded as `best.pt`, whatever the run was called, so every
variant has the same entry point:

```python
hf_hub_download(repo_id="darkdwine/yolo11-doc-layout-research-papers",
                filename="12-yolo11s-1024/best.pt")
```

Only files the run actually produced are uploaded. Older runs that predate some
of Ultralytics' plot outputs simply publish fewer files.

## Subfolder naming

Run directory names grew organically and carry history a stranger cannot read.
`yolo11_doc_layout_v2224_round03_imgsz_1024` says nothing about the
architecture, and `v2224` means nothing outside this repository.

Published names follow `NN-yolo11<size>-<resolution>[-lineage]`:

```
yolo11s_doc_layout_imgsz_1024               ->  12-yolo11s-1024/
yolo11_doc_layout_v2224_round03_imgsz_1024  ->  10-yolo11n-1024-v2224-round03/
yolo11s_doc_layout_attempt_02               ->  17-yolo11s-1024-augexp2/
```

The number is chronological, so the Hub file browser lists the collection in
the order it was built. Then come the two things a consumer actually chooses
on, architecture and training resolution. The lineage token comes last so
provenance survives, and every variant card states the original run name.

Two suffixes carry warnings:

| Suffix | Meaning |
|---|---|
| `round03` | Descends from the dataset round whose train and val splits overlapped. Scores are honest; the lineage is not. |
| `augexp` | The failed augmentation experiment. Published as a negative result, not for use. |

The mapping lives in `PUBLISH_ORDER` in `doclayout_ft/hub/push_to_hub.py`. It
is a hand-written table rather than a derived rule, because most runs were
copied into `FinetunedModels/` in one go and their file timestamps carry no
chronology. Add new runs to the table; anything missing from it falls back to a
plain hyphenated name and sorts to the end of the queue.

## Two runs are not published

`yolo11_doc_layout_v222` and `yolo11_doc_layout_v222_imgsz_1024` produce
checkpoints **numerically identical** to `v22` and `v22_imgsz_1024`
respectively, verified by comparing state dicts tensor by tensor. The files
differ only in metadata, which is why their scores match to four decimal places.

Publishing both would put the same model in the collection twice under two
names. The duplicates are skipped, and named on the card of the run they
duplicate so the omission is visible. That leaves 17 distinct models from 19
run directories.

## The cards

Generated from what each run recorded, not written by hand, so they cannot
drift away from the artefacts they describe.

The **root card** carries the class taxonomy, the comparison table across every
published variant, the dataset composition, a usage snippet, and the
limitations that apply to all of them. It is regenerated on every publish from
the full ledger, so it always describes the whole collection rather than the
last batch uploaded.

Each **variant card** carries that run's own scores, its training
configuration, and how to load it specifically. Variants with a compromised
lineage or a known-bad augmentation setting say so at the top of their own card,
so the warning cannot be missed by someone who arrives at a subfolder directly.

A score taken from a run's final training epoch rather than a held-out
evaluation is marked with a dagger in the comparison table and footnoted, so it
is never presented as comparable with the properly measured rows.

## The ledger

`.hf_publish_ledger.json` at the repository root records each published run's
subfolder, image size, metrics and timestamp. It is git-ignored: it describes
what this machine has uploaded, which is not a fact about the source tree.

It is written after each upload rather than once at the end, so an interrupted
batch does not lose track of what it already pushed. A missing or corrupt ledger
reads as empty, and the worst case is re-uploading a variant, which the Hub
treats as a no-op commit.

The ledger is also what the root card is rebuilt from, which is why it caches
metrics rather than only names. To publish from a second machine, copy the
ledger across first, or the next root card will list only what that machine
uploaded.

## Baselines are never published

The un-fine-tuned checkpoints under `FinetunedModels/base_model/`,
`yolov11s/` and `yolov11m/` are excluded from the queue. They belong to
[`Armaggheddon/yolo11-document-layout`](https://huggingface.co/Armaggheddon/yolo11-document-layout),
not to this project, and republishing someone else's weights under a different
account helps nobody. Every card credits them as the base model.

## Licensing

Cards declare AGPL-3.0, inherited from Ultralytics YOLO. Fine-tuned weights are
derivative works of an AGPL-licensed model. If you intend a different licence,
check what the base checkpoint permits first.
