"""Score every model in a directory on a held-out split and compare them.

Writes a CSV table and a horizontal bar chart to ``reports/``, both sorted by
mAP50-95, and prints the same table to the terminal.

Each model is evaluated at the image size recorded in its own ``args.yaml``,
not at one size for all. A model trained at 1024 and scored at 640 loses
accuracy that has nothing to do with the model, and the resulting table would
rank checkpoints by a resolution mismatch rather than by quality. Baselines
that were never fine-tuned here are scored at 640, the size they were
originally trained at.

Split choice matters when quoting a number:

``val``
    The validation split alone. This is the number to compare against
    training-time metrics, since it is what early stopping watched.
``test``
    The test split alone, untouched by model selection. The honest number for
    an external claim.
``combined``
    val and test merged, via ``data_combined.yaml``. More pages, so a tighter
    estimate, but no longer a clean held-out set because val guided early
    stopping. Reported here because it is what earlier project runs used.

The same checkpoint scores differently across these three, so a headline
figure is only meaningful alongside the split that produced it. See
docs/EVALUATION.md.

Usage::

    python -m doclayout_ft.evaluation.evaluate
    python -m doclayout_ft.evaluation.evaluate --split test
    python -m doclayout_ft.evaluation.evaluate --only yolo11s_doc_layout_imgsz_1024
    python -m doclayout_ft.evaluation.evaluate --models-dir models --out-prefix sweep
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover_many, filter_by_name
from doclayout_ft.config import (
    DEFAULT_ROUND,
    FINETUNED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    data_yaml,
    resolve_device,
)

#: Ultralytics split name to pass for each of our split choices. "combined"
#: is not a real Ultralytics split; it is the ``val:`` key of
#: ``data_combined.yaml``, which points at val+test merged.
ULTRALYTICS_SPLIT = {"combined": "val", "val": "val", "test": "test"}

#: Column order for the output CSV.
CSV_FIELDS = (
    "model", "baseline", "imgsz", "precision", "recall", "mAP50", "mAP50-95", "fitness",
)


def evaluate_one(
    checkpoint: Checkpoint,
    dataset_yaml: Path,
    split: str,
    batch: int,
    device: str | int,
) -> dict[str, object]:
    """Score one checkpoint.

    Args:
        checkpoint: Model to evaluate.
        dataset_yaml: Ultralytics dataset config.
        split: Ultralytics split name, from :data:`ULTRALYTICS_SPLIT`.
        batch: Validation batch size.
        device: CUDA index or ``"cpu"``.

    Returns:
        One row for the comparison table.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint.weights))
    metrics = model.val(
        data=str(dataset_yaml),
        split=split,
        imgsz=checkpoint.imgsz,
        batch=batch,
        device=device,
        plots=False,
        verbose=False,
    )
    results = metrics.results_dict
    return {
        "model": checkpoint.name,
        "baseline": checkpoint.is_baseline,
        "imgsz": checkpoint.imgsz,
        "precision": results.get("metrics/precision(B)"),
        "recall": results.get("metrics/recall(B)"),
        "mAP50": results.get("metrics/mAP50(B)"),
        "mAP50-95": results.get("metrics/mAP50-95(B)"),
        "fitness": results.get("fitness"),
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the comparison table as CSV.

    Args:
        rows: Result rows, already sorted.
        path: Destination file. Parent directories are created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)


def write_chart(rows: list[dict[str, object]], path: Path, title: str) -> None:
    """Write the comparison bar chart.

    Both mAP50 and mAP50-95 are plotted per model. mAP50 alone flatters every
    model, because a loose box that overlaps the truth by half still counts;
    mAP50-95 is what separates a model that finds a region from one that
    bounds it tightly, which is the distinction that matters for chunking.

    Args:
        rows: Result rows, already sorted.
        path: Destination PNG. Parent directories are created.
        title: Chart title.
    """
    import matplotlib
    matplotlib.use("Agg")  # No display on a headless training box.
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    names = [str(row["model"]) for row in rows]
    map50 = [row["mAP50"] or 0.0 for row in rows]
    map5095 = [row["mAP50-95"] or 0.0 for row in rows]
    positions = range(len(rows))

    figure, axes = plt.subplots(figsize=(11, max(4.0, len(rows) * 0.45)))
    axes.barh(positions, map50, height=0.4, align="edge",
              label="mAP50", color="#8ec9e0")
    axes.barh([p - 0.4 for p in positions], map5095, height=0.4, align="edge",
              label="mAP50-95", color="#2f6690")
    axes.set_yticks(list(positions))
    axes.set_yticklabels(names, fontsize=8)
    axes.invert_yaxis()
    axes.set_xlabel("Score")
    axes.set_xlim(0, 1)
    axes.grid(axis="x", alpha=0.25, linestyle=":")
    axes.set_axisbelow(True)
    # pad=20 (default 6.0) leaves real breathing room between the title and
    # the first bar. tight_layout() reads a title's bbox when sizing margins,
    # so raising the pad alone is enough -- no rect/subplots_adjust needed.
    axes.set_title(title, pad=20)
    axes.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def print_table(rows: list[dict[str, object]]) -> None:
    """Print the comparison table to the terminal.

    Args:
        rows: Result rows, already sorted.
    """
    width = max((len(str(row["model"])) for row in rows), default=20)
    header = (f"{'model':<{width}} {'imgsz':>6} {'precision':>10} "
              f"{'recall':>8} {'mAP50':>8} {'mAP50-95':>9}")
    print("\n--- Sorted by mAP50-95 ---")
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(key: str) -> str:
            value = row[key]
            return f"{value:.4f}" if isinstance(value, (int, float)) else "n/a"
        print(f"{str(row['model']):<{width}} {row['imgsz']:>6} {fmt('precision'):>10} "
              f"{fmt('recall'):>8} {fmt('mAP50'):>8} {fmt('mAP50-95'):>9}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, nargs="+",
                        default=[FINETUNED_DIR, MODELS_DIR],
                        help="Directories of models to evaluate, most authoritative "
                             "first (default: FinetunedModels/ then models/). This "
                             "matches what the publisher scores by default, so every "
                             "published variant has a held-out number.")
    parser.add_argument("--round", default=DEFAULT_ROUND,
                        help=f"Dataset round to evaluate against (default: {DEFAULT_ROUND})")
    parser.add_argument("--split", choices=sorted(ULTRALYTICS_SPLIT), default="val",
                        help="Held-out split to score on (default: val). See the "
                             "module docstring on why this choice changes the number.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to these model names (see --list)")
    parser.add_argument("--no-baselines", action="store_true",
                        help="Skip checkpoints that were never fine-tuned here")
    parser.add_argument("--batch", type=int, default=4,
                        help="Validation batch size (default: 4)")
    parser.add_argument("--device", default="0",
                        help="CUDA device index, or 'cpu' (default: 0)")
    parser.add_argument("--out-prefix", default=None,
                        help="Basename for the CSV and PNG written to reports/ "
                             "(default: evaluation_<split>)")
    parser.add_argument("--list", action="store_true",
                        help="Print the discovered models and exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        dataset_yaml = data_yaml(args.round, combined=args.split == "combined")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        found = discover_many(
            args.models_dir,
            include_baselines=not args.no_baselines,
            baseline_suffix="_baseline",
        )
        found = filter_by_name(found, args.only)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not found:
        dirs = ", ".join(str(d) for d in args.models_dir)
        print(f"No models found under {dirs}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(found)} model(s) on split '{args.split}' ({dataset_yaml}):")
    for name, checkpoint in found.items():
        kind = "baseline" if checkpoint.is_baseline else "fine-tuned"
        print(f"  {name:<50} imgsz={checkpoint.imgsz} {kind}")

    if args.list:
        return 0

    rows: list[dict[str, object]] = []
    failures: list[tuple[str, Exception]] = []
    for index, (name, checkpoint) in enumerate(found.items(), start=1):
        print(f"\n=== [{index}/{len(found)}] {name} ===")
        try:
            rows.append(evaluate_one(
                checkpoint, dataset_yaml, ULTRALYTICS_SPLIT[args.split],
                args.batch, resolve_device(args.device),
            ))
        except Exception as exc:  # noqa: BLE001 - score the rest of the sweep
            failures.append((name, exc))
            print(f"{name} FAILED: {exc}", file=sys.stderr)

    if not rows:
        print("\nNo model evaluated successfully.", file=sys.stderr)
        return 1

    # A model that produced no mAP50-95 sorts last rather than crashing the sort.
    rows.sort(key=lambda row: row["mAP50-95"] if isinstance(row["mAP50-95"], (int, float))
              else float("-inf"), reverse=True)

    prefix = args.out_prefix or f"evaluation_{args.split}"
    csv_path = REPORTS_DIR / f"{prefix}.csv"
    png_path = REPORTS_DIR / f"{prefix}.png"

    write_csv(rows, csv_path)
    print(f"\nWrote {csv_path}")
    write_chart(rows, png_path,
                f"Model comparison on {args.round} ({args.split} split)")
    print(f"Wrote {png_path}")

    print_table(rows)

    if failures:
        print(f"\n{len(failures)} model(s) failed to evaluate:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
