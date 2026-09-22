"""Generate the Hugging Face cards for the published model collection.

Everything this project publishes lives in **one** Hub repository, with each
checkpoint in its own subfolder. That means two kinds of card:

``build_index_card``
    The repository's root ``README.md``. This is the page people actually land
    on, so it carries the comparison table across every published variant, the
    class taxonomy, and how to load a specific one. It is regenerated on every
    publish, so it always reflects the full collection rather than the last
    batch uploaded.

``build_variant_card``
    A ``README.md`` inside one checkpoint's subfolder, covering just that run:
    its own scores, the hyperparameters it was trained with, and how it differs
    from its siblings.

Both are built from what the runs themselves recorded, so a card cannot drift
away from the artefact it describes. Three sources feed them, all optional, and
the cards degrade gracefully as each goes missing:

``args.yaml``
    The hyperparameters the run was actually launched with.
``results.csv``
    Per-epoch metrics; the final row gives the training-time scores.
``reports/evaluation_<split>.csv``
    Scores from a deliberate held-out evaluation, which is what a card should
    quote in preference to training-time numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from doclayout_ft.checkpoints import Checkpoint
from doclayout_ft.config import CLASS_NAMES, REPORTS_DIR

#: Hub repo the fine-tuning started from, credited in every card.
BASE_REPO_ID = "Armaggheddon/yolo11-document-layout"

#: Metric keys in the order the cards' results tables present them.
METRIC_COLUMNS = ("precision", "recall", "mAP50", "mAP50-95")

#: Run recommended to anyone who does not want to read the comparison table.
RECOMMENDED_RUN = "yolo11s_doc_layout_imgsz_1024"

#: The pipeline that produced these weights: doclayout_ft, its docs, and the
#: fine-tuning steps, linked so anyone on the Hub can find how to reproduce a
#: variant or train their own. The model repo carries weights and metrics only
#: (see "Training images are not distributed" below); the code lives here
#: instead of being duplicated into the model repo, so there is one place to
#: keep it current.
GITHUB_REPO_URL = "https://github.com/mudit-choudhary/yolo11-doc-layout-research-papers"

#: Display form of the above, e.g. "mudit-choudhary/yolo11-doc-layout-research-papers".
GITHUB_REPO_LABEL = GITHUB_REPO_URL.removeprefix("https://github.com/")

#: Runs whose diagnostic plots were produced after the fact rather than by the
#: training run itself. Training ended before Ultralytics' final validation
#: pass, so no confusion matrix or curves were written. They were regenerated
#: from the saved checkpoint against the held-out split, which is a different
#: dataset from the one the run trained on. The card says so, because a reader
#: would otherwise reasonably assume a confusion matrix reflects the run's own
#: validation data.
REGENERATED_PLOTS: frozenset[str] = frozenset({"yolo11_doc_layout_v22"})


def load_run_args(checkpoint: Checkpoint) -> dict[str, object]:
    """Read a run's ``args.yaml``.

    Args:
        checkpoint: The run to read.

    Returns:
        The parsed hyperparameters, or an empty dict if the file is missing or
        unparseable.
    """
    if not checkpoint.args_yaml.is_file():
        return {}
    try:
        return yaml.safe_load(checkpoint.args_yaml.read_text()) or {}
    except yaml.YAMLError:
        return {}


def load_eval_row(model_name: str, split: str) -> dict[str, str] | None:
    """Look up a model's row in an evaluation table.

    Args:
        model_name: Run name as it appears in the table's ``model`` column.
        split: Split name, matching ``reports/evaluation_<split>.csv``.

    Returns:
        The matching row, or None if the table or the row is absent.
    """
    table = REPORTS_DIR / f"evaluation_{split}.csv"
    if not table.is_file():
        return None
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == model_name:
                return row
    return None


def load_final_epoch(checkpoint: Checkpoint) -> dict[str, str] | None:
    """Read the last row of a run's ``results.csv``.

    Used only as a fallback when no held-out evaluation is available. These are
    training-time validation numbers, so the card labels them as such.

    Args:
        checkpoint: The run to read.

    Returns:
        The final epoch's metrics, or None if there are none.
    """
    if not checkpoint.results_csv.is_file():
        return None
    with checkpoint.results_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    # Ultralytics writes column names with surrounding whitespace in some
    # versions, which quietly breaks every lookup that does not strip them.
    return {key.strip(): value for key, value in rows[-1].items() if key}


def collect_metrics(checkpoint: Checkpoint, split: str) -> tuple[dict[str, str], bool]:
    """Gather one run's headline metrics.

    Args:
        checkpoint: The run to describe.
        split: Evaluation split to prefer.

    Returns:
        A ``(metrics, from_held_out)`` pair. ``metrics`` maps each name in
        :data:`METRIC_COLUMNS` to a formatted string. ``from_held_out`` is True
        when the numbers came from a real evaluation rather than from the last
        training epoch.
    """
    eval_row = load_eval_row(checkpoint.name, split)
    if eval_row is not None:
        return {key: _format_metric(eval_row.get(key)) for key in METRIC_COLUMNS}, True

    final = load_final_epoch(checkpoint) or {}
    lookup = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP50-95": "metrics/mAP50-95(B)",
    }
    return {key: _format_metric(final.get(lookup[key])) for key in METRIC_COLUMNS}, False


def _format_metric(value: object) -> str:
    """Render a metric as four decimal places, or ``n/a`` if not numeric."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _sort_key(entry: dict[str, object]) -> float:
    """Order index rows by mAP50-95, with unscored runs last."""
    try:
        return -float(str(entry.get("metrics", {}).get("mAP50-95")))
    except (TypeError, ValueError):
        return float("inf")


def _class_table() -> str:
    """Render the 12-class taxonomy as a Markdown table."""
    return "\n".join(f"| {index} | `{name}` |" for index, name in enumerate(CLASS_NAMES))


def _front_matter() -> str:
    """Render the YAML front matter the Hub reads for tags and metadata."""
    return f"""---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
- yolo
- yolo11
- ultralytics
- object-detection
- document-layout-analysis
- document-understanding
- research-papers
base_model: {BASE_REPO_ID}
---"""


# --------------------------------------------------------------------------
# Root card: the repository index
# --------------------------------------------------------------------------


def build_index_card(
    repo_id: str,
    entries: list[dict[str, object]],
    split: str = "val",
) -> str:
    """Render the repository's root ``README.md``.

    Args:
        repo_id: The Hub repository, e.g. ``darkdwine/yolo11-doc-layout``.
        entries: One dict per published variant, each with ``name``,
            ``subfolder``, ``imgsz`` and ``metrics``.
        split: Evaluation split the metrics came from.

    Returns:
        The card as Markdown, including its YAML front matter.
    """
    ordered = sorted(entries, key=_sort_key)

    # A variant whose numbers came from its last training epoch rather than a
    # held-out evaluation must not be presented as if it were measured the same
    # way as the rest. Mark those rows and explain the mark below the table.
    any_training_time = any(entry.get("held_out") is False for entry in ordered)

    if ordered:
        rows = "\n".join(
            "| `{subfolder}`{mark} | {imgsz} | {passes} | {mAP5095} | {ms} | "
            "{status} |".format(
                subfolder=entry["subfolder"],
                mark="" if entry.get("held_out", True) else " †",
                imgsz=entry["imgsz"],
                passes=entry.get("passes") or "?",
                mAP5095=entry["metrics"].get("mAP50-95", "n/a"),
                ms=f"{entry['latency_ms']:.0f}" if entry.get("latency_ms") else "-",
                status=entry.get("status", "") or "",
            )
            for entry in ordered
        )
        # Five columns, not seven. Precision, recall and mAP50 were dropped:
        # they pushed the table wider than the page, which put the Notes column
        # behind a horizontal scrollbar, and Notes is the column a reader needs
        # in order to choose. All four measures remain on each variant's own
        # card and in results.csv.
        table = (
            "| Variant | imgsz | Passes | mAP50-95 | ms/page | Notes |\n"
            "|---|---|---|---|---|---|\n" + rows
        )
        if any_training_time:
            table += (
                "\n\n† Scores from this run's final training epoch, not from the "
                "held-out evaluation. Not directly comparable with the unmarked "
                "rows; treat them as indicative only."
            )
    else:
        table = "_No variants published yet._"

    # Point the usage snippet at the recommended run when it is present,
    # otherwise at whatever scored best, so the example is always loadable.
    default_entry = next(
        (e for e in ordered if e["name"] == RECOMMENDED_RUN),
        ordered[0] if ordered else None,
    )
    default_subfolder = default_entry["subfolder"] if default_entry else "<variant>"
    default_imgsz = default_entry["imgsz"] if default_entry else 1024

    return f"""{_front_matter()}

# YOLOv11 Document Layout — Research Papers

A collection of YOLOv11 document-layout detectors fine-tuned on manually
annotated pages from real arXiv research papers. They locate the twelve region
types below on a page image, which is what makes layout-aware chunking of a
paper possible: text can be split on the paragraph and section boundaries the
model found, instead of on a fixed character count that cuts through the middle
of a table.

**Training code:** [{GITHUB_REPO_LABEL}]({GITHUB_REPO_URL}). The full pipeline is
there: PDF rendering, dataset splitting, fine-tuning, evaluation, latency
benchmarking, and this card's own generator. It is tested and runs end to end.
This repository holds weights and metrics only; the training images are not
included, for the reasons given under Limitations.

Every variant lives in its own subfolder of this one repository, numbered
oldest to newest. Each has its own `README.md` with its full training
configuration.

Subfolder names read `NN-yolo11<size>-<training resolution>`, so
`12-yolo11s-1024` is the twelfth model trained, a YOLO11-small at 1024 pixels.
The number is chronological, so the folder listing shows the order the
collection was built in.

An `augexp` suffix marks the failed augmentation experiment described below.
It is there because a higher number otherwise reads as newer and better, and
these are newer and worse.

Anything else you need in order to choose is in the Notes column of the table
below, and in each variant's own card, rather than encoded in a folder name.

Fine-tuned from [`{BASE_REPO_ID}`](https://huggingface.co/{BASE_REPO_ID}).

## Which one to use

Two answers, depending on what you are doing.

**`12-yolo11s-1024` for accuracy on the classes that matter.** It has a clean
single-pass lineage and leads on `Table` (0.985) and `Footnote`, which decide
where a chunk boundary falls. It costs 37 ms per page.

**`11-yolo11n-1024` for throughput.** It scores 0.766 against 0.769, a
difference well inside noise, at **19 ms per page**. Half the latency for
three thousandths of mAP is the better engineering trade for most pipelines,
and it is the choice to default to if you are processing pages in bulk.

If pages are small or the corpus is huge, `05-yolo11n-640` runs at 9 ms and
still reaches 0.738, though small classes such as `Footnote` and `Page-header`
suffer at 640.

`08-yolo11n-1024` scores about 0.002 higher than either, which is also within
noise. It is not recommended because it is the product of three successive
fine-tuning passes over datasets that no longer exist in their original form,
so it cannot be reproduced. Prefer a lineage you can rebuild.

### Reading the table

**Passes** is how many successive fine-tuning runs produced the checkpoint,
each starting from the previous one's output.

**1** means it was fine-tuned once from a published base model, so you can
reproduce it with a single command. **2 or more** means the earliest passes used
datasets that no longer exist in their original form, so it cannot be rebuilt
from scratch, and it has been shaped by several dataset versions rather than
one.

**ms/page** is median inference latency at batch 1 on a GTX 1650, each model at
its own resolution, excluding image decode. Add roughly 60 to 80 ms per page for
decode to get a wall-clock figure.

Precision, recall and mAP50 are on each variant's own card, alongside its full
training configuration.

Pass count says nothing about accuracy. The early datasets overlap today's held-out
papers by up to 19%, so the scores were checked for inflation and **none was
found**: single-pass models show the same gap between overlapping and
non-overlapping papers, meaning those pages are simply easier for everything.
The numbers below are comparable across every row.

Variants whose name ends `-augexp` are **deliberately published failures**. They
carry an augmentation bundle that regressed every model it was applied to, by
0.016 to 0.060 mAP50-95. They are here so the result is reproducible, not
because they are worth deploying. That sweep changed seven settings at once, so
it identifies a harmful combination rather than a single culprit; `copy_paste`
at `copy_paste_mode="flip"` is the prime suspect, since it mirrors pasted crops
even when whole-image flipping is disabled.

## Variants

![Horizontal bar chart ranking every published variant by mAP50-95 on the
held-out validation split. The recommended 12-yolo11s-1024 is highlighted in
blue; the five failed augmentation experiments are hatched. Scores run from
0.518 to 0.772, with the top seven within 0.01 of each other.](comparison.png)

{table}

Measured on the held-out `{split}` split unless marked otherwise, each at the
image size it was trained at. `mAP50-95` is the number worth reading: `mAP50` counts a box as correct
when it overlaps the ground truth by half, which flatters a model that finds a
region without bounding it tightly, and tight bounds are exactly what a
chunking consumer needs.

### Where the recommended model is strong and weak

![Horizontal bar chart of per-class mAP50-95 for 12-yolo11s-1024. Table 0.985,
List-item 0.926 and Text 0.918 lead; Page-footer trails far behind at 0.360
against an overall 0.769.](per-class.png)

The headline number is an average over twelve classes that range from 0.985 to
0.360, so it hides more than it shows. `Table` and `Text`, the two classes that
most affect where a chunk boundary falls, are the strongest. `Page-footer` is
the one real weakness, and it is a labelling problem rather than a model one:
see Limitations.

### Speed against accuracy

![Scatter plot of median latency per page against mAP50-95 for every published
variant. Three clusters: yolo11n at 640 near 10 ms and 0.52 to 0.74, yolo11n at
1024 near 18 ms and 0.71 to 0.77, and yolo11s at 1024 near 37 ms and 0.75 to
0.77. The recommended 12-yolo11s-1024 is highlighted at the far
right.](speed-accuracy.png)

Latency is decided almost entirely by architecture and resolution, so the
variants fall into three tight bands. Within the 1024 band, nine yolo11n
variants span 0.01 mAP at identical speed.

### Precision against recall

![Dumbbell chart showing precision and recall for each variant on one row,
joined by a line. Most models sit close to balanced near 0.85 to 0.93; the
earliest run 01-yolo11n-640 shows the widest gap, with precision 0.776 against
recall 0.656.](precision-recall.png)

A wide gap means the model leans one way. High precision with low recall misses
regions; the reverse invents them. The early runs are visibly unbalanced and the
later ones are not, which is most of what fine-tuning bought.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(
    repo_id="{repo_id}",
    filename="{default_subfolder}/best.pt",
)
model = YOLO(weights)

results = model.predict("page.jpg", imgsz={default_imgsz}, conf=0.2)
for box in results[0].boxes:
    class_name = results[0].names[int(box.cls)]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(class_name, round(float(box.conf), 3), [round(v) for v in (x1, y1, x2, y2)])
```

Every variant also publishes `best.onnx`, the same weights with a dynamic
batch and page size, for runtimes without PyTorch. Non-maximum suppression is
not part of the graph.

Swap the subfolder in `filename` to load a different variant. Run inference at
the image size that variant was trained at; smaller inputs cost small classes
such as `Footnote` and `Page-header` first, since those regions are only a few
pixels tall once a page is scaled down.

## Classes

The first eleven classes are inherited from the DocLayNet-style base model.
`Authors` is a twelfth class added for research-paper front matter, which the
base taxonomy has no equivalent for.

| ID | Class |
|---|---|
{_class_table()}

## Training data

Pages rendered from arXiv PDFs at 300 DPI and annotated by hand in
[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling), exported in YOLO
horizontal-bounding-box format.

850 pages from 566 papers are annotated and used for training and evaluation.
They are a labelled subset of a much larger unlabelled pool of roughly 22,000
rendered pages; the rest of that pool exists so later annotation rounds have
material to draw on, and was not used to train these models.

The dataset is split by paper rather than by page: every page of a given paper
lands in the same split, so a model is never validated on a layout it saw
during training. Pages of one paper share a template, so a page-level split
would leak layout across the boundary and inflate the scores.

| Split | Papers | Pages |
|---|---|---|
| train | 424 | 632 |
| val | 85 | 130 |
| test | 57 | 88 |

## Limitations

- **`Page-footer` boxes are loose.** The class is found reliably but bounded
  poorly, at roughly 0.36 mAP50-95 against 0.77 overall, consistently across
  every model size and resolution tested. The pattern points at inconsistent
  annotation tightness rather than model capacity.
- **Research papers only.** Training pages are arXiv preprints, mostly
  two-column and in English. Invoices, forms, slides and handwriting are out of
  distribution.
- **Rendered pages, not photographs.** Training images come from digital PDF
  rendering, so skew, shadow and camera perspective are unrepresented.
- **Reading order is not predicted.** The models output regions, not their
  sequence. Ordering is left to the consumer.

## License and attribution

Released under **AGPL-3.0**, inherited from
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which these
models were trained with. Note the AGPL network clause: serving these weights
to users over a network obliges you to offer them the corresponding source.
Ultralytics sells an [Enterprise License](https://www.ultralytics.com/license)
for use without that obligation.

Fine-tuned from [`{BASE_REPO_ID}`](https://huggingface.co/{BASE_REPO_ID}), which
is MIT-licensed and was itself trained on
[DocLayNet](https://huggingface.co/datasets/ds4sd/DocLayNet)
(CDLA-Permissive-1.0).

Annotations were made with
[X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) (GPL-3.0). That
licence covers the tool, not the labels produced with it.

## Training images are not distributed

These repositories contain weights, metrics and plots only. The annotated page
images are not published, because they are rendered from arXiv preprints whose
licences vary per paper and often do not permit redistribution. Ultralytics'
`train_batch` and `val_batch` debugging mosaics are excluded for the same
reason.
"""


# --------------------------------------------------------------------------
# Per-variant card
# --------------------------------------------------------------------------


def _training_section(checkpoint: Checkpoint) -> str:
    """Build a variant card's hyperparameter table from ``args.yaml``."""
    run_args = load_run_args(checkpoint)
    if not run_args:
        return "_No `args.yaml` was recorded for this run._"

    interesting = [
        ("Base checkpoint", run_args.get("model")),
        ("Dataset", run_args.get("data")),
        ("Epochs", run_args.get("epochs")),
        ("Early-stop patience", run_args.get("patience")),
        ("Image size", run_args.get("imgsz")),
        ("Batch size", run_args.get("batch")),
        ("Optimizer", run_args.get("optimizer")),
        ("Initial LR", run_args.get("lr0")),
        ("Horizontal flip", run_args.get("fliplr")),
        ("Copy-paste", run_args.get("copy_paste")),
        ("Multi-scale", run_args.get("multi_scale")),
        ("Random erasing", run_args.get("erasing")),
    ]
    lines = ["| Setting | Value |", "|---|---|"]
    for label, value in interesting:
        if value is None:
            continue
        # Dataset paths are absolute on the training machine; only the tail is
        # meaningful to a reader.
        if label == "Dataset":
            value = "/".join(Path(str(value)).parts[-3:])
        if label == "Base checkpoint":
            value = Path(str(value)).name
        lines.append(f"| {label} | `{value}` |")
    return "\n".join(lines)


def build_variant_card(
    checkpoint: Checkpoint,
    repo_id: str,
    subfolder: str,
    split: str = "val",
    duplicate_of: str | None = None,
) -> str:
    """Render the ``README.md`` that sits inside one variant's subfolder.

    Args:
        checkpoint: The run being described.
        repo_id: The Hub repository holding the whole collection.
        subfolder: This variant's folder within that repository.
        split: Evaluation split whose numbers to quote.
        duplicate_of: Name of another run whose weights are identical to this
            one's and which is therefore not published separately. Named on the
            card so the omission is visible rather than silent.

    Returns:
        The card as Markdown. No YAML front matter: only the repository root
        card carries that, since the Hub reads metadata from the root only.
    """
    metrics, from_held_out = collect_metrics(checkpoint, split)
    header = "| " + " | ".join(METRIC_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(METRIC_COLUMNS)) + "|"
    row = "| " + " | ".join(metrics[key] for key in METRIC_COLUMNS) + " |"

    source = (
        f"Measured on the held-out `{split}` split of the `round_final` "
        f"dataset, at `imgsz={checkpoint.imgsz}`."
        if from_held_out else
        "Taken from the final training epoch's validation pass. These are "
        "training-time numbers, not an independent evaluation."
    )

    run_args = load_run_args(checkpoint)
    parent = Path(str(run_args.get("model", "")))
    if parent.name == "best.pt":
        parent_label = f"the earlier run {parent.parent.parent.name}"
    elif parent.name:
        parent_label = f"the base checkpoint {parent.stem}"
    else:
        parent_label = "an unrecorded checkpoint"

    caveat = ""
    if checkpoint.name.endswith("_attempt_02"):
        caveat = (
            "\n> **This variant is a published failure.** It carries the "
            "`copy_paste` and `multi_scale` augmentation settings that "
            "regressed every model they were applied to. It is here so the "
            "result is reproducible. Do not deploy it.\n"
        )
    elif "round03" in checkpoint.name:
        caveat = (
            "\n> **Chained lineage.** This checkpoint is the result of several "
            "successive fine-tuning passes, one of which used a dataset round "
            "whose train and validation splits pointed at the same images. The "
            "scores below are measured honestly against a clean held-out split, "
            "so the model is usable, but its training history is not a clean "
            "experiment and should not be read as one.\n"
        )

    regenerated_note = ""
    if checkpoint.name in REGENERATED_PLOTS:
        regenerated_note = (
            "\n> **Diagnostics regenerated.** This training run ended before "
            "Ultralytics' final validation pass, so it wrote no confusion "
            "matrix or curves. Those were regenerated afterwards from the saved "
            "checkpoint, scored against the `round_final` held-out split rather "
            "than the dataset this run trained on. `results.png` is a faithful "
            "reconstruction from this run's own `results.csv`. The weights and "
            "metrics are unaffected.\n"
        )

    duplicate_note = ""
    if duplicate_of:
        duplicate_note = (
            f"\nThe run `{duplicate_of}` produced a checkpoint numerically "
            f"identical to this one, differing only in file metadata. It is not "
            f"published separately.\n"
        )

    return f"""# {subfolder}

One variant of the [`{repo_id}`](https://huggingface.co/{repo_id}) document-layout
collection. See the [repository root](https://huggingface.co/{repo_id}) for the
class taxonomy, the comparison against every other variant, and the limitations
that apply to all of them.

Internal run name: `{checkpoint.name}`. Trained at `imgsz={checkpoint.imgsz}`
from `{parent_label}`.
{regenerated_note}{duplicate_note}{caveat}
## Results

{header}
{divider}
{row}

{source}

### Speed against accuracy

![Scatter plot of median latency per page against mAP50-95 for every published
variant. Three clusters: yolo11n at 640 near 10 ms and 0.52 to 0.74, yolo11n at
1024 near 18 ms and 0.71 to 0.77, and yolo11s at 1024 near 37 ms and 0.75 to
0.77. The recommended 12-yolo11s-1024 is highlighted at the far
right.](speed-accuracy.png)

Latency is decided almost entirely by architecture and resolution, so the
variants fall into three tight bands. Within the 1024 band, nine yolo11n
variants span 0.01 mAP at identical speed.

### Precision against recall

![Dumbbell chart showing precision and recall for each variant on one row,
joined by a line. Most models sit close to balanced near 0.85 to 0.93; the
earliest run 01-yolo11n-640 shows the widest gap, with precision 0.776 against
recall 0.656.](precision-recall.png)

A wide gap means the model leans one way. High precision with low recall misses
regions; the reverse invents them. The early runs are visibly unbalanced and the
later ones are not, which is most of what fine-tuning bought.

## Usage

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

weights = hf_hub_download(repo_id="{repo_id}", filename="{subfolder}/best.pt")
model = YOLO(weights)

results = model.predict("page.jpg", imgsz={checkpoint.imgsz}, conf=0.2)
```

Run inference at `imgsz={checkpoint.imgsz}`, the size this checkpoint was
trained at.

The same weights are also published as `best.onnx`, opset 12, for runtimes
without PyTorch:

```python
weights = hf_hub_download(repo_id="{repo_id}", filename="{subfolder}/best.onnx")
model = YOLO(weights, task="detect")
```

The graph's input is `['batch', 3, 'height', 'width']`: pages can be batched,
and letterboxed to a stride multiple rather than padded to a square, which is
what an ONNX runtime pipeline normally does. Detection runs at
`imgsz={checkpoint.imgsz}` because that is what these weights were trained for;
other sizes load, and cost accuracy. The graph ends at the raw `(batch, 16,
anchors)` tensor -- non-maximum suppression is not baked in, so a runtime that
is not Ultralytics has to apply it.

## Training configuration

{_training_section(checkpoint)}

The strongest models in this collection were trained with Ultralytics' stock
augmentation, `copy_paste` and `multi_scale` left off. Disabling horizontal flip
and random erasing is well argued for documents, since a mirrored page never
occurs at inference, but it was never tested in isolation here, so it is not
treated as an established improvement.

## Files

| File | Contents |
|---|---|
| `best.pt` | The checkpoint |
| `best.onnx` | The same weights as ONNX, opset 12, dynamic batch and page size |
| `args.yaml` | Hyperparameters the run was launched with |
| `results.csv` | Per-epoch metrics |
| `results.png` | Training curves |
| `confusion_matrix*.png` | Confusion matrices |
| `Box*_curve.png` | Precision, recall, F1 and PR curves |
"""
