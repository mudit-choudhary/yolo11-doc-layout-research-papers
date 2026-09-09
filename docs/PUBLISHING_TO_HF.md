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

## You do not create the repository by hand

There is nothing to set up in the browser. The first publish calls
`create_repo(..., exist_ok=True)` before uploading anything, so the repository
appears on its own. Later publishes reuse it.

Two things do have to be right, and a dry run now checks both for you:

- **A write token.** A read-only token authenticates fine and then fails when
  the repository is created. Make one at
  <https://huggingface.co/settings/tokens> with the **write** role.
- **A namespace you own.** `darkdwine/...` works only if that is your username
  or an organisation you can write to. If your Hub username differs, the Hub
  refuses the create, which is safe but avoidable.

The dry run reports both:

```
  preflight: logged in as 'darkdwine', token role 'write'
  preflight: 'darkdwine/yolo11-doc-layout-research-papers' will be created automatically on publish.
```

### One thing to decide before the first publish, not after

`--private` takes effect **only when the repository is created**. Because
`exist_ok=True` leaves an existing repository alone, passing `--private` on a
later run does nothing. If you want to inspect the collection before anyone
else can see it, pass `--private` on the *first* publish:

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --private --yes
```

Flipping it to public afterwards is one setting in the Hub UI. Going the other
way means the content was public in the meantime.

## Before the first push

Authenticate through the Hugging Face CLI. The script never handles a token
itself, and no token should ever be written into this repository.

```bash
hf auth login          # newer huggingface_hub
huggingface-cli login  # older
hf auth whoami         # confirm the username matches the namespace
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

All 17 by default, drawn from `FinetunedModels/` and `models/` together. Seven
run names exist in both; the first directory listed wins, which is why
`FinetunedModels` comes first.

Five of those are the `augexp` runs carrying the failed augmentation
experiment. They are published deliberately: a documented negative result is
more useful than an undocumented one, and both their own cards and the root
table say plainly not to deploy them.

To leave them out instead:

```bash
python -m doclayout_ft.hub.push_to_hub --models-dir FinetunedModels --list
```

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

Published names carry three things and no more: **when** it was trained, **what
architecture**, and **at what resolution**.

```
yolo11s_doc_layout_imgsz_1024               ->  12-yolo11s-1024/
yolo11_doc_layout_v2224_round03_imgsz_1024  ->  10-yolo11n-1024/
yolo11s_doc_layout_attempt_02               ->  17-yolo11s-1024-augexp/
```

An earlier draft also encoded the internal lineage, giving names like
`10-yolo11n-1024-v2224-round03`. That was dropped. The sequence number already
makes every name unique, so the lineage token was doing no disambiguating work,
and `v2224` means nothing to anyone outside this repository. Worse, `round03`
was actively wrong: those runs trained on `round_final`, and the token was a
leftover from an earlier naming plan.

Lineage is real information. It belongs on the variant card, which states the
parent checkpoint in words, and in the Notes column of the root comparison
table, which is where people actually choose.

The one status kept in a name is `augexp`, and it earns its place: a higher
sequence number reads as newer and therefore better, and runs 13 to 17 are
newer and worse. Without the suffix the numbering would mislead.

The mapping lives in `PUBLISH_ORDER` in `doclayout_ft/hub/push_to_hub.py`, with
the Notes text in `RUN_STATUS` beside it. Both are hand-written tables rather
than derived rules, because most runs were copied into `FinetunedModels/` in one
go and their file timestamps carry no chronology. Add new runs to the table;
anything missing falls back to a plain hyphenated name and sorts to the end of
the queue.

## Two runs are not published

`yolo11_doc_layout_v222` and `yolo11_doc_layout_v222_imgsz_1024` produce
checkpoints **numerically identical** to `v22` and `v22_imgsz_1024`
respectively, verified by comparing state dicts tensor by tensor. The files
differ only in metadata, which is why their scores match to four decimal places.

Publishing both would put the same model in the collection twice under two
names. The duplicates are skipped, and named on the card of the run they
duplicate so the omission is visible. That leaves 17 distinct models from 19
run directories.

## Keeping the table narrow

The root table is five columns: Variant, imgsz, Passes, mAP50-95, Notes. It was
seven, with Precision, Recall and mAP50 as well, and full sentences in Notes.
That pushed it past the width of the page, so Hugging Face put it behind a
horizontal scrollbar and clipped the Notes column, hiding the one thing a reader
needs in order to choose.

Precision, recall and mAP50 live on each variant's own card and in its
`results.csv`. The lineage depth that used to be prose is now the numeric
**Passes** column, which costs almost no width. Notes are capped at 32
characters, enforced by a test, and most rows are deliberately blank.

## Editing a note after publishing

The Notes column text lives in `RUN_STATUS` in
`doclayout_ft/hub/push_to_hub.py`. It is editorial, not a record of the publish,
so it is re-read from that table every time the root card is rebuilt. Change the
wording there, then:

```bash
python -m doclayout_ft.hub.push_to_hub --refresh-index --yes
```

That rewrites only the root `README.md`, uploads no weights, and takes seconds.
It covers every variant in the ledger, including ones published weeks earlier.

## The charts

The root card carries two charts, regenerated from `reports/` every time the
card is rebuilt so they can never disagree with the table beside them:

| File | Shows |
|---|---|
| `comparison.png` | Every published variant ranked by mAP50-95 |
| `per-class.png` | The recommended model class by class |

Both come from `doclayout_ft/hub/charts.py`. They are uploaded to the repository
root, which the card references by relative path.

Bars start at zero, so the values cluster. That is the finding rather than a
defect: most of these models are within noise of each other, and cropping the
axis to manufacture separation would misrepresent them.

The recommended model is the only one in colour, with everything else in gray
and the failed experiments hatched. One model is the point of the chart and the
rest are context; painting all seventeen in identity colours would bury the row
a reader actually needs. The hatching rather than a third hue keeps the
distinction visible in greyscale and to colour-blind readers.

Both render on a light background. A card is shown on a page whose theme the
reader controls and a PNG cannot adapt, so an opaque light image is the option
that stays legible either way.

If a chart's input is missing, the publish prints a warning and continues
rather than failing.

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
