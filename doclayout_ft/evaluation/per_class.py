"""Break one model's accuracy down by class.

The headline mAP50-95 is an average across twelve classes, and averages hide
the thing worth knowing. In this project every model scored around 0.75
overall while ``Page-footer`` sat at 0.36 and ``Table`` at 0.98. Only a
per-class view shows that, and it is what tells you whether the next
improvement should come from the model or from the annotations.

Read the two columns together. High precision and recall with low mAP50-95
means the model finds the region reliably but cannot agree with the label on
where its edges are, which usually points at inconsistent annotation tightness
rather than model capacity. Low recall means the class is genuinely being
missed, which more training data can fix.

Usage::

    python -m doclayout_ft.evaluation.per_class
    python -m doclayout_ft.evaluation.per_class --model yolo11n_doc_layout_imgsz_1024
    python -m doclayout_ft.evaluation.per_class --split test --save-csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover
from doclayout_ft.config import (
    DEFAULT_ROUND,
    FINETUNED_DIR,
    REPORTS_DIR,
    data_yaml,
    resolve_device,
)
from doclayout_ft.evaluation.evaluate import ULTRALYTICS_SPLIT

#: The strongest model produced by this project; see docs/EVALUATION.md.
DEFAULT_MODEL = "yolo11s_doc_layout_imgsz_1024"


def collect_per_class(metrics, class_names: dict[int, str]) -> list[dict[str, object]]:
    """Turn an Ultralytics validation result into per-class rows.

    Ultralytics reports per-class arrays indexed by position in
    ``metrics.box.ap_class_index``, not by class id, because classes absent
    from the split are omitted entirely. The index array is what maps a
    position back to a class name.

    Args:
        metrics: Return value of ``model.val()``.
        class_names: Class id to name, from the loaded model.

    Returns:
        One row per class present in the split, sorted by mAP50-95 descending.
    """
    box = metrics.box
    rows: list[dict[str, object]] = []
    for position, class_id in enumerate(box.ap_class_index):
        precision, recall, map50, map5095 = box.class_result(position)
        rows.append({
            "class": class_names.get(int(class_id), f"id_{int(class_id)}"),
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(map50),
            "mAP50-95": float(map5095),
        })
    rows.sort(key=lambda row: row["mAP50-95"], reverse=True)
    return rows


def print_per_class(rows: list[dict[str, object]], overall: dict[str, float]) -> None:
    """Print the per-class table with an overall row appended.

    Args:
        rows: Per-class rows from :func:`collect_per_class`.
        overall: Dataset-wide precision, recall, mAP50 and mAP50-95.
    """
    width = max((len(str(row["class"])) for row in rows), default=14)
    header = (f"{'class':<{width}} {'precision':>10} {'recall':>8} "
              f"{'mAP50':>8} {'mAP50-95':>9}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{str(row['class']):<{width}} {row['precision']:>10.4f} "
              f"{row['recall']:>8.4f} {row['mAP50']:>8.4f} {row['mAP50-95']:>9.4f}")
    print("-" * len(header))
    print(f"{'ALL':<{width}} {overall['precision']:>10.4f} {overall['recall']:>8.4f} "
          f"{overall['mAP50']:>8.4f} {overall['mAP50-95']:>9.4f}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Run name to analyse (default: {DEFAULT_MODEL})")
    parser.add_argument("--models-dir", type=Path, default=FINETUNED_DIR,
                        help="Directory to look the model up in (default: FinetunedModels/)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to a .pt file, bypassing --model lookup entirely")
    parser.add_argument("--round", default=DEFAULT_ROUND,
                        help=f"Dataset round to evaluate against (default: {DEFAULT_ROUND})")
    parser.add_argument("--split", choices=sorted(ULTRALYTICS_SPLIT), default="val",
                        help="Held-out split to score on (default: val)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override the image size (default: the model's own)")
    parser.add_argument("--batch", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--device", default="0",
                        help="CUDA device index, or 'cpu' (default: 0)")
    parser.add_argument("--save-csv", action="store_true",
                        help="Also write the table to reports/per_class_<model>_<split>.csv")
    parser.add_argument("--plots", action="store_true",
                        help="Write confusion matrix and PR curves next to the run")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    from ultralytics import YOLO

    if args.weights is not None:
        if not args.weights.is_file():
            print(f"error: weights not found: {args.weights}", file=sys.stderr)
            return 1
        weights, imgsz, label = args.weights, args.imgsz or 640, args.weights.stem
    else:
        try:
            found = discover(args.models_dir, baseline_suffix="_baseline")
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        checkpoint: Checkpoint | None = found.get(args.model)
        if checkpoint is None:
            print(f"error: no model named '{args.model}' under {args.models_dir}.\n"
                  f"Available: {', '.join(sorted(found)) or '(none)'}", file=sys.stderr)
            return 1
        weights, imgsz, label = checkpoint.weights, args.imgsz or checkpoint.imgsz, args.model

    try:
        dataset_yaml = data_yaml(args.round, combined=args.split == "combined")
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Model:   {label}\nWeights: {weights}\nDataset: {dataset_yaml}")
    print(f"Split:   {args.split}   imgsz: {imgsz}\n")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(dataset_yaml),
        split=ULTRALYTICS_SPLIT[args.split],
        imgsz=imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        plots=args.plots,
        verbose=False,
    )

    rows = collect_per_class(metrics, model.names)
    results = metrics.results_dict
    overall = {
        "precision": results.get("metrics/precision(B)", 0.0),
        "recall": results.get("metrics/recall(B)", 0.0),
        "mAP50": results.get("metrics/mAP50(B)", 0.0),
        "mAP50-95": results.get("metrics/mAP50-95(B)", 0.0),
    }
    print_per_class(rows, overall)

    if args.save_csv:
        out_path = REPORTS_DIR / f"per_class_{label}_{args.split}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["class", "precision", "recall", "mAP50", "mAP50-95"])
            writer.writeheader()
            writer.writerows(rows)
            writer.writerow({"class": "ALL", **overall})
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
