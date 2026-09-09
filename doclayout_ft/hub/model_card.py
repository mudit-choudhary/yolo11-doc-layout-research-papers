"""Generate a Hugging Face model card for one fine-tuned run.

A checkpoint uploaded without a card is close to useless to anyone else: they
cannot tell what it detects, what it was trained on, or whether the number in
the filename means anything. This builds a card from what the run itself
recorded, so the card cannot drift away from the artefact it describes.

Three sources feed it, all optional, and the card degrades gracefully as each
goes missing:

``args.yaml``
    The hyperparameters the run was actually launched with.
``results.csv``
    Per-epoch metrics; the final row gives the training-time scores.
``reports/evaluation_<split>.csv``
    Scores from a deliberate held-out evaluation, which is what the card
    should quote in preference to training-time numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from doclayout_ft.checkpoints import Checkpoint
from doclayout_ft.config import CLASS_NAMES, REPORTS_DIR

#: Hub repo the fine-tuning started from, credited in every card.
BASE_REPO_ID = "Armaggheddon/yolo11-document-layout"

#: Metric keys in the order the card's results table presents them.
METRIC_COLUMNS = ("precision", "recall", "mAP50", "mAP50-95")


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


def _format_metric(value: object) -> str:
    """Render a metric as four decimal places, or ``n/a`` if not numeric."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


def _results_section(checkpoint: Checkpoint, split: str) -> str:
    """Build the card's results table and the sentence explaining its source."""
    eval_row = load_eval_row(checkpoint.name, split)
    if eval_row is not None:
        values = [_format_metric(eval_row.get(key)) for key in METRIC_COLUMNS]
        source = (f"Measured on the held-out `{split}` split of the "
                  f"`round_final` dataset, at `imgsz={checkpoint.imgsz}`.")
    else:
        final = load_final_epoch(checkpoint) or {}
        lookup = {
            "precision": "metrics/precision(B)",
            "recall": "metrics/recall(B)",
            "mAP50": "metrics/mAP50(B)",
            "mAP50-95": "metrics/mAP50-95(B)",
        }
        values = [_format_metric(final.get(lookup[key])) for key in METRIC_COLUMNS]
        source = ("Taken from the final training epoch's validation pass. These "
                  "are training-time numbers, not an independent evaluation.")

    header = "| " + " | ".join(METRIC_COLUMNS) + " |"
    divider = "|" + "|".join(["---"] * len(METRIC_COLUMNS)) + "|"
    row = "| " + " | ".join(values) + " |"
    return f"{header}\n{divider}\n{row}\n\n{source}"


def _training_section(checkpoint: Checkpoint) -> str:
    """Build the card's hyperparameter table from ``args.yaml``."""
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


def build_model_card(
    checkpoint: Checkpoint,
    repo_id: str,
    split: str = "val",
) -> str:
    """Render the full model card for one run.

    Args:
        checkpoint: The run being published.
        repo_id: Destination Hub repository, e.g. ``darkdwine/my-model``.
        split: Evaluation split whose numbers the card should quote.

    Returns:
        The card as Markdown, including its YAML front matter.
    """
    class_table = "\n".join(
        f"| {index} | `{name}` |" for index, name in enumerate(CLASS_NAMES)
    )

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
---

# {checkpoint.name}

A YOLOv11 document-layout detector fine-tuned on manually annotated pages from
real arXiv research papers. It locates the twelve region types below on a page
image, which is what makes layout-aware chunking of a paper possible: text can
be split on paragraph and section boundaries the model found, instead of on a
fixed character count that cuts through the middle of a table.

Fine-tuned from [`{BASE_REPO_ID}`](https://huggingface.co/{BASE_REPO_ID}).

## Classes

The first eleven classes are inherited from the DocLayNet-style base model.
`Authors` is a twelfth class added for research-paper front matter, which the
base taxonomy has no equivalent for.

| ID | Class |
|---|---|
{class_table}

## Results

{_results_section(checkpoint, split)}

`mAP50-95` is the number worth reading. `mAP50` counts a box as correct when it
overlaps the ground truth by half, which flatters a model that finds a region
without bounding it tightly, and tight bounds are exactly what a chunking
consumer needs.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")  # or: hf_hub_download("{repo_id}", "best.pt")

results = model.predict("page.jpg", imgsz={checkpoint.imgsz}, conf=0.2)
for box in results[0].boxes:
    class_name = results[0].names[int(box.cls)]
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(class_name, round(float(box.conf), 3), [round(v) for v in (x1, y1, x2, y2)])
```

Run inference at `imgsz={checkpoint.imgsz}`, the size this checkpoint was trained
at. Smaller inputs cost small classes such as `Footnote` and `Page-header`
first, since those regions are only a few pixels tall once a page is scaled
down.

## Training data

Pages rendered from arXiv PDFs at 300 DPI and annotated by hand in
X-AnyLabeling, exported in YOLO horizontal-bounding-box format. The dataset is
split by paper rather than by page: every page of a given paper lands in the
same split, so a model is never validated on a layout it saw during training.

## Training configuration

{_training_section(checkpoint)}

Two augmentation settings deviate from the Ultralytics defaults, both because
documents are not natural scenes. Horizontal flipping is disabled, since a
mirrored page never occurs at inference and mirrored text destroys the
left-to-right structure the model relies on. Copy-paste augmentation is
disabled, because pasting region crops between unrelated pages produces
structurally impossible layouts; enabling it regressed every model it was
tried on.

## Limitations

- **`Page-footer` boxes are loose.** The class is found reliably but bounded
  poorly, at roughly 0.36 mAP50-95 against 0.75 overall, consistently across
  every model size and resolution tested. The pattern points at inconsistent
  annotation tightness rather than model capacity.
- **Research papers only.** Training pages are arXiv preprints, mostly
  two-column and in English. Invoices, forms, slides and handwriting are out of
  distribution.
- **Rendered pages, not photographs.** Training images come from digital PDF
  rendering, so skew, shadow and camera perspective are unrepresented.
- **Reading order is not predicted.** The model outputs regions, not their
  sequence. Ordering is left to the consumer.

## Provenance

Produced by the [`doclayout_ft`](https://github.com/{repo_id.split('/')[0]}) fine-tuning
pipeline. Run name `{checkpoint.name}`, trained at `imgsz={checkpoint.imgsz}`.
"""
