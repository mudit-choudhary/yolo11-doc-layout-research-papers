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
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # No display on a training box.
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

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

#: One typographic scale, used by every chart so they read as one set.
TITLE_SIZE = 14.0
SUBTITLE_SIZE = 9.8
LABEL_SIZE = 9.5
TICK_SIZE = 9.0
VALUE_SIZE = 9.0

#: Left inset for the title block, as a figure fraction.
TITLE_X = 0.012


def _titles(figure, title: str, subtitle: str) -> float:
    """Place the title block and return the top of the plotting rectangle.

    Positions are computed from a fixed offset in inches rather than as a
    fraction of figure height. The charts differ in height, and a fractional
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
    figure.suptitle(title, x=TITLE_X, y=1 - 0.26 / height, ha="left",
                    va="top", fontsize=TITLE_SIZE, color=TEXT_PRIMARY,
                    fontweight="bold")
    figure.text(TITLE_X, 1 - 0.54 / height, subtitle, ha="left", va="top",
                fontsize=SUBTITLE_SIZE, color=TEXT_SECONDARY)
    return 1 - 0.98 / height


def _caption(figure, text: str) -> None:
    """Add a footnote under the plot, for provenance and caveats."""
    figure.text(TITLE_X, 0.012, text, ha="left", va="bottom",
                fontsize=8.4, color=TEXT_SECONDARY)


def _style(axes, axis: str = "x") -> None:
    """Apply the recessive grid and axis treatment shared by every chart.

    Args:
        axes: Target axes.
        axis: Which axis carries gridlines, ``"x"``, ``"y"`` or ``"both"``.
    """
    axes.set_facecolor(SURFACE)
    axes.grid(axis=axis, color=GRID, linewidth=0.7, alpha=0.9)
    axes.set_axisbelow(True)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(GRID)
    axes.spines["bottom"].set_linewidth(0.9)
    axes.tick_params(colors=TEXT_SECONDARY, labelsize=TICK_SIZE, length=0, pad=6)


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
        DUPLICATE_OF, PUBLISH_ORDER, passes_for, status_for, subfolder_for,
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
        # Classify from the data, not from the note text. An earlier version
        # parsed the Notes column, which broke silently the moment that wording
        # was shortened.
        if run.endswith("_attempt_02"):
            lineage = "failed"
        elif (passes_for(run) or 2) == 1:
            lineage = "single"
        else:
            lineage = "deep"
        rows.append({
            "subfolder": subfolder_for(run),
            "run": run,
            "score": scores[run],
            "lineage": lineage,
            "passes": passes_for(run),
            "recommended": "Recommended" in status_for(run),
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

    # Padding above the first bar and below the last was 0.39 data-units on
    # each side (bar height 0.62, ylim margin 0.7), which read as dead space
    # at the top and bottom of a chart that is otherwise tightly packed.
    # Halved to ~0.19 each side.
    axes.set_ylim(-0.5, len(rows) - 0.5)
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
    halo = [path_effects.withStroke(linewidth=3, foreground=SURFACE)]

    for y, (_, score) in zip(positions, rows):
        axes.barh(y, score, height=0.62, color=ACCENT, linewidth=0, zorder=2)
        label = axes.text(score + 0.008, y, f"{score:.3f}", va="center",
                          ha="left", fontsize=VALUE_SIZE, color=TEXT_SECONDARY,
                          zorder=6)
        label.set_path_effects(halo)

    if overall is not None:
        axes.axvline(overall, color=TEXT_SECONDARY, linewidth=1.3,
                     linestyle=(0, (4, 3)), zorder=3)
        note = axes.text(overall - 0.012, -0.55, f"overall {overall:.3f}",
                         fontsize=VALUE_SIZE, color=TEXT_SECONDARY, ha="right",
                         va="center", zorder=6)
        note.set_path_effects(halo)

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
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Speed against accuracy
# --------------------------------------------------------------------------


def load_latency(path: Path | None = None) -> dict[str, dict]:
    """Read the latency table produced by the benchmark.

    Args:
        path: CSV to read. Defaults to ``reports/latency.csv``.

    Returns:
        Mapping of run name to its timing row.

    Raises:
        FileNotFoundError: If the benchmark has not been run.
    """
    path = path or (REPORTS_DIR / "latency.csv")
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Run doclayout_ft.evaluation.benchmark first.")
    rows = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                rows[row["model"]] = {
                    "median_ms": float(row["median_ms"]),
                    "fps": float(row["fps"]),
                    "params_m": float(row["params_m"]),
                }
            except (TypeError, ValueError):
                continue
    return rows


def build_speed_accuracy_chart(path: Path, split: str = "val") -> Path:
    """Render latency against accuracy, the chart that answers "which to run".

    A scatter, because the question is about the relationship between two
    measures rather than the magnitude of one. Faster is left, better is up, so
    the desirable corner is top-left and no legend is needed to say which
    direction is good.

    Both `yolo11n` and `yolo11s` families appear. The point the chart makes is
    that the whole `yolo11n` cluster sits within noise of `yolo11s` on accuracy
    while running meaningfully faster, which the ranked bar chart cannot show.

    Args:
        path: Destination PNG.
        split: Evaluation split the scores come from.

    Returns:
        The path written.
    """
    rows = load_comparison_rows(split)
    timings = load_latency()
    points = [(r, timings[r["run"]]) for r in rows if r["run"] in timings]
    if not points:
        raise FileNotFoundError(
            "No model appears in both the evaluation and latency tables.")

    figure, axes = plt.subplots(figsize=(9.5, 6.4))
    figure.patch.set_facecolor(SURFACE)
    _style(axes, axis="both")

    for row, timing in points:
        failed = row["lineage"] == "failed"
        recommended = row["recommended"]
        axes.scatter(
            timing["median_ms"], row["score"],
            s=210 if recommended else 130,
            facecolor=ACCENT if recommended else (SURFACE if failed else MUTED),
            edgecolor=ACCENT if recommended else MUTED,
            linewidth=1.8 if failed else 1.2,
            zorder=5 if recommended else 4,
        )

    # Latency is essentially decided by architecture and resolution, so the
    # variants pile into three tight clusters. Label one representative of each
    # rather than seventeen overlapping names, and say how many share the spot.
    fastest = min(points, key=lambda p: p[1]["median_ms"])
    best = max(points, key=lambda p: p[0]["score"])
    recommended = next((p for p in points if p[0]["recommended"]), None)

    latencies = [t["median_ms"] for _, t in points]
    span = max(latencies) - min(latencies)
    axes.set_xlim(min(latencies) - span * 0.14, max(latencies) + span * 0.14)

    labelled = []
    for entry, note in ((fastest, "fastest"), (best, "best mAP"),
                        (recommended, "recommended")):
        if entry is None or entry[0]["subfolder"] in {e[0] for e in labelled}:
            continue
        labelled.append((entry[0]["subfolder"], entry, note))

    # Offsets are per role rather than computed, because the three clusters sit
    # in known places and the empty space differs for each: below the fastest,
    # above the other two.
    offsets = {"fastest": (0, -34), "best mAP": (0, 30), "recommended": (0, 30)}
    for subfolder, (row, timing), note in labelled:
        axes.annotate(
            f"{subfolder}\n{note}",
            (timing["median_ms"], row["score"]),
            textcoords="offset points", xytext=offsets[note],
            ha="center", va="center", fontsize=VALUE_SIZE, linespacing=1.6,
            color=TEXT_PRIMARY if row["recommended"] else TEXT_SECONDARY,
            fontweight="bold" if row["recommended"] else "normal",
        )

    # Name the crowd rather than leaving it unexplained, anchored just under it.
    dense = [(r, t) for r, t in points
             if 15 <= t["median_ms"] <= 25 and r["score"] >= 0.70]
    if len(dense) > 3:
        axes.annotate(
            f"{len(dense)} variants sit in this latency band: yolo11n at 1024,\n"
            f"spanning {min(r['score'] for r, _ in dense):.3f} to "
            f"{max(r['score'] for r, _ in dense):.3f} mAP",
            (statistics.fmean(t["median_ms"] for _, t in dense),
             min(r["score"] for r, _ in dense)),
            textcoords="offset points", xytext=(0, -26), ha="center", va="top",
            fontsize=8.6, color=TEXT_SECONDARY, linespacing=1.6)

    axes.set_xlabel("Median latency per page, milliseconds  (lower is better)",
                    fontsize=LABEL_SIZE, color=TEXT_SECONDARY, labelpad=10)
    axes.set_ylabel("mAP50-95  (higher is better)",
                    fontsize=LABEL_SIZE, color=TEXT_SECONDARY, labelpad=10)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=11,
                   markerfacecolor=ACCENT, markeredgecolor=ACCENT),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=MUTED, markeredgecolor=MUTED),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=SURFACE, markeredgecolor=MUTED,
                   markeredgewidth=1.8),
    ]
    legend = axes.legend(handles,
                         ["Recommended", "Other published variants",
                          "Failed experiment"],
                         loc="lower right", frameon=False, fontsize=TICK_SIZE,
                         borderpad=0.9, labelspacing=0.8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    top = _titles(
        figure, "Speed against accuracy",
        "The desirable corner is top-left. The yolo11n cluster is within noise "
        "of yolo11s and meaningfully faster.")
    _caption(figure,
             "Batch 1, GTX 1650, each model at its own training resolution. "
             "Excludes image decode, which is identical for every model.")

    figure.tight_layout(rect=(0, 0.035, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path


# --------------------------------------------------------------------------
# Precision against recall
# --------------------------------------------------------------------------


def build_precision_recall_chart(path: Path, split: str = "val") -> Path:
    """Render each model's precision and recall as a connected pair.

    A dumbbell rather than two bar series. The reader's question is "is this
    model trigger-happy or cautious", which is about the gap between the two
    numbers for one model, not about ranking all the precisions against all the
    recalls. Paired bars would put that gap in two separate rows.

    Args:
        path: Destination PNG.
        split: Evaluation split the scores come from.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the evaluation table is missing.
    """
    from doclayout_ft.hub.push_to_hub import (
        DUPLICATE_OF, PUBLISH_ORDER, subfolder_for,
    )

    table = REPORTS_DIR / f"evaluation_{split}.csv"
    if not table.is_file():
        raise FileNotFoundError(
            f"{table} not found. Run doclayout_ft.evaluation.evaluate first.")

    scores = {}
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                scores[row["model"]] = (float(row["precision"]),
                                        float(row["recall"]),
                                        float(row["mAP50-95"]))
            except (TypeError, ValueError):
                continue

    rows = []
    for run, _ in PUBLISH_ORDER:
        if run in DUPLICATE_OF or run not in scores:
            continue
        precision, recall, score = scores[run]
        rows.append({"subfolder": subfolder_for(run), "precision": precision,
                     "recall": recall, "score": score})
    rows.sort(key=lambda r: r["score"], reverse=True)
    positions = list(range(len(rows)))[::-1]

    figure, axes = plt.subplots(figsize=(9.5, 0.40 * len(rows) + 2.4))
    figure.patch.set_facecolor(SURFACE)
    _style(axes)

    for y, row in zip(positions, rows):
        low, high = sorted((row["precision"], row["recall"]))
        axes.plot([low, high], [y, y], color=MUTED, linewidth=2.4,
                  solid_capstyle="round", zorder=2)
        axes.scatter(row["precision"], y, s=70, facecolor=ACCENT,
                     edgecolor=SURFACE, linewidth=1.4, zorder=4)
        axes.scatter(row["recall"], y, s=70, facecolor=TEXT_SECONDARY,
                     edgecolor=SURFACE, linewidth=1.4, zorder=4)

    axes.set_yticks(positions)
    axes.set_yticklabels([r["subfolder"] for r in rows], fontsize=TICK_SIZE,
                         color=TEXT_PRIMARY, fontfamily="DejaVu Sans Mono")
    axes.set_ylim(-0.8, len(rows) - 0.3)
    lowest = min(min(r["precision"], r["recall"]) for r in rows)
    axes.set_xlim(max(0.0, lowest - 0.025), 1.0)
    axes.set_xlabel("Score  ·  rows ordered by mAP50-95",
                    fontsize=LABEL_SIZE, color=TEXT_SECONDARY, labelpad=10)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=ACCENT, markeredgecolor=ACCENT),
        plt.Line2D([], [], marker="o", linestyle="none", markersize=9,
                   markerfacecolor=TEXT_SECONDARY, markeredgecolor=TEXT_SECONDARY),
    ]
    legend = axes.legend(handles, ["Precision", "Recall"], loc="lower right",
                         frameon=False, fontsize=TICK_SIZE, borderpad=0.9,
                         labelspacing=0.8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)

    top = _titles(
        figure, "Precision against recall",
        "A wide gap means the model leans one way: more misses, or more false "
        "boxes. Most sit close to balanced.")
    _caption(figure,
             "The x-axis starts near the lowest value rather than zero: these "
             "are positions on a scale, not lengths to compare.")

    figure.tight_layout(rect=(0, 0.03, 1, top))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return path
