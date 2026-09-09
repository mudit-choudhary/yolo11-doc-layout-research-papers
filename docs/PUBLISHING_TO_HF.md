# Publishing to the Hugging Face Hub

Publishing here works as a backlog cleared in instalments, not a bulk upload.
Runs go up oldest first, a couple at a time, and a ledger records what has
already gone so the same command next week continues rather than repeats.

## Before the first push

Authenticate through the Hugging Face CLI. The script never handles a token
itself, and no token should ever be written into this repository.

```bash
hf auth login          # newer huggingface_hub
huggingface-cli login  # older
```

Then look at the queue:

```bash
python -m doclayout_ft.hub.push_to_hub --list
```

## Publishing

```bash
# Dry run. Prints exactly what each repository would receive. Uploads nothing.
python -m doclayout_ft.hub.push_to_hub --limit 2

# The same batch, for real.
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

**Nothing uploads without `--yes`.** Read one dry run before the first real
push. A Hub repository is public by default, and its commit history is not
quietly rewritable.

Run the same command again next week and it picks up the next two.

### Useful variations

```bash
# One specific model, out of queue order
python -m doclayout_ft.hub.push_to_hub --only yolo11s_doc_layout_imgsz_1024 --yes

# Private, for something not ready to be seen
python -m doclayout_ft.hub.push_to_hub --only <run> --private --yes

# Re-upload a run already published, after fixing its card
python -m doclayout_ft.hub.push_to_hub --only <run> --include-published --yes

# A different account or an organisation
python -m doclayout_ft.hub.push_to_hub --namespace <org> --limit 2 --yes
```

## What a published repository contains

| File | Source |
|---|---|
| `README.md` | The generated model card |
| `best.pt` | The run's best checkpoint |
| `args.yaml` | Hyperparameters the run was launched with |
| `results.csv` | Per-epoch metrics |
| `results.png` | Training curves |
| `confusion_matrix*.png` | Confusion matrices |
| `Box*_curve.png` | Precision, recall, F1 and PR curves |
| `labels.jpg` | Class distribution of the training set |

Weights are always uploaded as `best.pt`, whatever the run was called, so every
published repository has the same entry point.

Only files the run actually produced are uploaded. Older runs that predate some
of Ultralytics' plot outputs simply publish fewer files.

## Repository naming

Underscores become hyphens, lowercased:

```
yolo11s_doc_layout_imgsz_1024  ->  darkdwine/yolo11s-doc-layout-imgsz-1024
```

The mapping is one-to-one, so a repository name always traces back to the run
that produced it. Override it for a single upload with `--repo-id`.

## The model card

Generated from what the run recorded, not written by hand, so it cannot drift
away from the artefact it describes. It carries the class taxonomy, the results
table, a usage snippet at the right image size, the training configuration, and
an honest limitations section covering the `Page-footer` weakness, the
research-paper-only training distribution, and the absence of reading order.

Metrics come from `reports/evaluation_<split>.csv` when it exists. Run the
evaluation before publishing, otherwise cards fall back to training-time
numbers and say so:

```bash
python -m doclayout_ft.evaluation.evaluate --split val
```

Choose which split the cards quote with `--split`.

## The ledger

`.hf_publish_ledger.json` at the repository root records run name, repository,
URL and timestamp for each publish. It is git-ignored: it describes what this
machine has uploaded, which is not a fact about the source tree.

It is written after each upload rather than once at the end, so an interrupted
batch does not lose track of repositories it already created. A missing or
corrupt ledger reads as empty, and the worst case is re-uploading a run, which
the Hub treats as a no-op commit.

To publish from a second machine, copy the ledger across, or pass
`--only` explicitly.

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
