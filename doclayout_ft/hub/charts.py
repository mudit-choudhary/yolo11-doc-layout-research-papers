"""Build the comparison charts shown on the published model card.

Two charts, each answering one question:

``comparison``
    How do the published variants rank, and what kind of lineage produced each?
    Magnitude across many named things, so horizontal bars sorted by score.
``per_class``
    Where is the recommended model strong and weak? The headline 0.77 is an
    average over twelve classes that range from 0.99 to 0.36, and that spread is
    the useful information.

Both are rendered on a light surface. A model card is a static image on a page
whose theme the reader controls, and a PNG cannot adapt, so an opaque light card
is the option that stays legible either way. Nothing here is transparent.

Only mAP50-95 is plotted. Adding mAP50 beside it would put two measures of
different strictness on one axis and invite reading the flattering one as the
headline; mAP50 counts a box as correct at 50% overlap, which a loose box passes.
Precision and recall stay in the table on the card.

Colour uses the **emphasis** form rather than a categorical palette. One model
is the point of the chart and the other sixteen are context, so the recommended
one carries the accent hue and everything else recedes to gray. Painting all
seventeen in identity colours would have spent the only free channel on
information the ranking already shows, and buried the one row a reader needs.

Failed experiments are separated by hatching rather than a third hue, so the
distinction survives greyscale printing and colour-vision deficiency. The gray
sits below 3:1 against the surface, so every bar carries a direct value label,
which is the documented relief.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display on a training box.
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from doclayout_ft.config import REPORTS_DIR

# --------------------------------------------------------------------------
# Palette: emphasis form. One accent, one de-emphasis gray.
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"

#: Reference categorical slot 1. 4.30:1 against the surface.
ACCENT = "#2a78d6"

#: De-emphasis gray for context bars. 2.09:1, deliberately recessive so the
#: accent dominates; the direct value labels are the relief that allows it.
MUTED = "#b3b1a9"

#: Hatch marking a failed experiment. A texture rather than a third hue, so the
#: distinction survives greyscale and colour-vision deficiency.
FAILED_HATCH = "////"

def _titles(figure, title: str, subtitle: str) -> float:
    """Place the title block and return the top of the plotting rectangle.

    Positions are computed from a fixed offset in inches rather than as a
    fraction of figure height. The two charts differ in height, and a fractional
    offset that clears the title on a tall figure lands on top of it on a short
    one.

    Args:
        figure: Target figure.
        title: Bold headline.
        subtitle: One line of context beneath it.

    Returns:
        The ``rect`` top to pass to ``tight_layout``.
    """
    height = figure.get_size_inches()[1]
    figure.suptitle(title, x=0.012, y=1 - 0.26 / height, ha="left",
                    va="top", fontsize=13.5, color=TEXT_PRIMARY,
                    fontweight="bold")
    figure.text(0.012, 1 - 0.52 / height, subtitle, ha="left", va="top",
                fontsize=9.5, color=TEXT_SECONDARY)
    return 1 - 0.95 / height


def _style(axes) -> None:
    """Apply the recessive grid and axis treatment shared by both charts."""
    axes.set_facecolor(SURFACE)
    axes.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def load_comparison_rows(split: str = "val") -> list[dict]:
    """Read the evaluation table for the published variants.

    Args:
        split: Which ``reports/evaluation_<split>.csv`` to read.

    Returns:
        Rows with ``subfolder``, ``score`` and ``lineage`` keys, sorted best
        first.

    Raises:
        FileNotFoundError: If the evaluation table has not been generated.
    """
    from doclayout_ft.hub.push_to_hub import (
        DUPLICATE_OF, PUBLISH_ORDER, status_for, subfolder_for,
    )

    table = REPORTS_DIR / f"evaluation_{split}.csv"
    if not table.is_file():
        raise FileNotFoundError(
            f"{table} not found. Run doclayout_ft.evaluation.evaluate first.")

    scores = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                scores[row["model"]] = float(row["mAP50-95"])
            except (TypeError, ValueError):
                continue

    rows = []
    for run, _ in PUBLISH_ORDER:
        if run in DUPLICATE_OF or run not in scores:
            continue
        note = status_for(run)
        if "Failed experiment" in note:
            lineage = "failed"
        elif "1 fine-tune from base" in note:
            lineage = "single"
        else:
            lineage = "deep"
        rows.append({
            "subfolder": subfolder_for(run),
            "run": run,
            "score": scores[run],
            "lineage": lineage,
            "recommended": "Recommended" in note,
        })
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def build_comparison_chart(path: Path, split: str = "val") -> Path:
    """Render the variant comparison chart.

    Bars start at zero, as bars must: their length is the encoding. The values
    cluster tightly as a result, and that is the finding rather than a defect.
    Most of these models are within noise of one another, and a chart that
    implied otherwise by cropping the axis would be lying.

    Args:
        path: Destination PNG.
        split: Evaluation split the scores come from.

    Returns:
        The path written.
    """
    rows = load_comparison_rows(split)
    positions = list(range(len(rows)))[::-1]  # best at the top

    figure, axes = plt.subplots(figsize=(9.5, 0.40 * len(rows) + 2.2))
    figure.patch.set_facecolor(SURFACE)
    _style(axes)

    for y, row in zip(positions, rows):
        failed = row["lineage"] == "failed"
        axes.barh(y, row["score"], height=0.62, zorder=2,
                  color=ACCENT if row["recommended"] else MUTED,
                  hatch=FAILED_HATCH if failed else None,
                  edgecolor=SURFACE if failed else "none",
                  linewidth=0)
        # Direct label on every bar: the relief the recessive gray requires,
        # and what lets a reader take a number off the chart without the axis.
        axes.text(row["score"] + 0.008, y, f"{row['score']:.3f}",
                  va="center", ha="left", fontsize=9,
                  color=TEXT_PRIMARY if row["recommended"] else TEXT_SECONDARY,
                  fontweight="bold" if row["recommended"] else "normal")

    axes.set_yticks(positions)
    axes.set_yticklabels([r["subfolder"] for r in rows], fontsize=9,
                         fontfamily="DejaVu Sans Mono")
    for label, row in zip(axes.get_yticklabels(), rows):
        label.set_color(TEXT_PRIMARY if row["recommended"] else TEXT_SECONDARY)
        if row["recommended"]:
            label.set_fontweight("bold")

    axes.set_ylim(-0.7, len(rows) - 0.3)
    axes.set_xlim(0, max(r["score"] for r in rows) * 1.14)
    axes.set_xlabel("mAP50-95 on the held-out validation split",
                    fontsize=9.5, color=TEXT_SECONDARY, labelpad=8)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=ACCENT, linewidth=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, linewidth=0),
        plt.Rectangle((0, 0), 1, 1, facecolor=MUTED, hatch=FAILED_HATCH,
                      edgecolor=SURFACE, linewidth=0),
    ]
    legend = axes.legend(
        handles, ["Recommended", "Other published variants", "Failed experiment"],
        loc="lower right", frameon=False, fontsize=9,
        handlelength=1.3, handleheight=1.2, borderpad=0.9, labelspacing=0.7)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    top = _titles(
        figure, "Document-layout variants, ranked",
        "Most of these sit within noise of each other. "
        "Lineage and reproducibility decide it, not the score.")

    figure.tight_layout(rect=(0, 0, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


def build_per_class_chart(path: Path, csv_path: Path | None = None) -> Path:
    """Render the per-class breakdown for the recommended model.

    A single series, so no legend: the title names what is plotted. The overall
    score is drawn as a reference line, because the point of the chart is how
    far the classes spread around it.

    Args:
        path: Destination PNG.
        csv_path: Per-class CSV. Defaults to the recommended model's.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the per-class CSV has not been generated.
    """
    csv_path = csv_path or (
        REPORTS_DIR / "per_class_yolo11s_doc_layout_imgsz_1024_val.csv")
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} not found. Run doclayout_ft.evaluation.per_class "
            f"--save-csv first.")

    rows, overall = [], None
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            score = float(row["mAP50-95"])
            if row["class"] == "ALL":
                overall = score
            else:
                rows.append((row["class"], score))
    rows.sort(key=lambda r: r[1], reverse=True)
    positions = list(range(len(rows)))[::-1]

    figure, axes = plt.subplots(figsize=(9.5, 0.40 * len(rows) + 2.2))
    figure.patch.set_facecolor(SURFACE)
    _style(axes)

    # One series, so one colour for every bar. Colouring by value would
    # double-encode length as hue and spend the free channel on nothing.
    for y, (_, score) in zip(positions, rows):
        axes.barh(y, score, height=0.62, color=ACCENT, linewidth=0, zorder=2)
        axes.text(score + 0.008, y, f"{score:.3f}", va="center", ha="left",
                  fontsize=9, color=TEXT_SECONDARY)

    if overall is not None:
        axes.axvline(overall, color=TEXT_SECONDARY, linewidth=1.3,
                     linestyle=(0, (4, 3)), zorder=3)
        axes.text(overall - 0.012, -0.55, f"overall {overall:.3f}", fontsize=9,
                  color=TEXT_SECONDARY, ha="right", va="center")

    axes.set_yticks(positions)
    axes.set_yticklabels([c for c, _ in rows], fontsize=9.5, color=TEXT_PRIMARY)
    axes.set_ylim(-1.0, len(rows) - 0.3)
    axes.set_xlim(0, 1.08)
    axes.set_xlabel("mAP50-95 on the held-out validation split",
                    fontsize=9.5, color=TEXT_SECONDARY, labelpad=8)

    top = _titles(
        figure, "Recommended model, class by class",
        "12-yolo11s-1024. Page-footer is found reliably but bounded loosely, "
        "which is an annotation problem.")

    figure.tight_layout(rect=(0, 0, 1, top))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path
