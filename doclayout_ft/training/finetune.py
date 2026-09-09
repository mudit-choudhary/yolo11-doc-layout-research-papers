"""Fine-tune one or more checkpoints on a dataset round.

Every checkpoint found under ``models/`` becomes its own training run, writing
to ``models/<name>_<suffix>/`` so that nothing already on disk is overwritten.
A run whose output directory already exists is skipped, which makes the whole
sweep restartable: if the fifth of twenty runs dies to an out-of-memory error,
re-running picks up from there rather than repeating the first four.

Augmentation defaults depart from Ultralytics' own in two places, both because
documents are not natural scenes:

``fliplr=0.0``
    Ultralytics mirrors half of all training images. A mirrored page never
    occurs at inference, and mirrored text destroys exactly the left-to-right
    structure that distinguishes a caption from a list item.

``copy_paste=0.0``
    Pasting instance crops between unrelated pages produces structurally
    impossible layouts, and Ultralytics' default ``copy_paste_mode="flip"``
    mirrors each pasted crop even when whole-image flipping is off. Enabling
    it regressed every model it was applied to by 0.02 to 0.06 mAP50-95. The
    flag is still exposed, because the finding is worth being able to
    reproduce, but it defaults off. See docs/FINETUNING_STEPS.md.

Mixed precision is left to Ultralytics to decide. On the GTX 1650 this project
was built on, the AMP sanity check fails and training silently falls back to
FP32, which is why memory headroom is tighter than the card's 4 GB suggests.

Usage::

    python -m doclayout_ft.training.finetune --only yolov11s
    python -m doclayout_ft.training.finetune --epochs 210 --batch 2 --imgsz 1024
    python -m doclayout_ft.training.finetune --list
    python -m doclayout_ft.training.finetune --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover, filter_by_name
from doclayout_ft.config import (
    DEFAULT_BATCH,
    DEFAULT_IMGSZ,
    DEFAULT_ROUND,
    MODELS_DIR,
    data_yaml,
    resolve_device,
)

#: Epoch budget. Generous on purpose: ``--patience`` stops the run early once
#: validation metrics stop improving, so a high ceiling costs nothing but
#: gives a slow-converging run room to finish.
DEFAULT_EPOCHS = 210

#: Epochs without validation improvement before stopping early.
DEFAULT_PATIENCE = 60

#: Appended to each run name to namespace one sweep's outputs from the next.
DEFAULT_SUFFIX = "ft"


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--round", default=DEFAULT_ROUND,
                      help=f"Dataset round to train on (default: {DEFAULT_ROUND})")
    data.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                      help="Directory to discover checkpoints in and write runs to "
                           "(default: models/)")

    selection = parser.add_argument_group("checkpoint selection")
    selection.add_argument("--only", nargs="+", default=None,
                           help="Restrict to these checkpoint names (see --list)")
    selection.add_argument("--suffix", default=DEFAULT_SUFFIX,
                           help=f"Appended to each run name (default: {DEFAULT_SUFFIX})")
    selection.add_argument("--list", action="store_true",
                           help="Print the discovered checkpoints and exit")
    selection.add_argument("--dry-run", action="store_true",
                           help="Print what would be trained without training it")

    schedule = parser.add_argument_group("training schedule")
    schedule.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                          help=f"Maximum epochs (default: {DEFAULT_EPOCHS})")
    schedule.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                          help=f"Early-stop patience in epochs (default: {DEFAULT_PATIENCE})")
    schedule.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                          help=f"Training image size (default: {DEFAULT_IMGSZ}). "
                               "Raising this from 640 was the largest single gain "
                               "in this project.")
    schedule.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                          help=f"Batch size (default: {DEFAULT_BATCH}). Sized for 4 GB "
                               "of VRAM at imgsz 1024; raise it on a larger card.")
    schedule.add_argument("--cos-lr", action="store_true",
                          help="Cosine learning-rate schedule instead of linear")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="0",
                         help="CUDA device index, or 'cpu' (default: 0)")
    runtime.add_argument("--cache", default="ram", choices=["ram", "disk", "false"],
                         help="Cache decoded images between epochs (default: ram). "
                              "Drop to 'disk' or 'false' if system RAM is tight.")
    runtime.add_argument("--workers", type=int, default=8,
                         help="Dataloader worker processes (default: 8)")

    aug = parser.add_argument_group("augmentation")
    aug.add_argument("--fliplr", type=float, default=0.0,
                     help="Horizontal flip probability (default: 0.0; Ultralytics "
                          "uses 0.5). Mirrored pages never occur at inference.")
    aug.add_argument("--copy-paste", type=float, default=0.0,
                     help="Copy-paste probability (default: 0.0). Regressed every "
                          "model tested here; see the module docstring.")
    aug.add_argument("--multi-scale", type=float, default=0.0,
                     help="Multi-scale training range as a fraction of imgsz "
                          "(default: 0.0, off)")
    aug.add_argument("--erasing", type=float, default=0.0,
                     help="Random erasing probability (default: 0.0; Ultralytics "
                          "uses 0.4). Erasing can remove the only instance of a "
                          "rare class from a page.")
    return parser


def train_one(
    checkpoint: Checkpoint,
    run_name: str,
    dataset_yaml: Path,
    models_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run a single fine-tuning job.

    Args:
        checkpoint: The starting weights.
        run_name: Output directory name under ``models_dir``.
        dataset_yaml: Ultralytics dataset config to train against.
        models_dir: Ultralytics ``project`` directory.
        args: Parsed command-line arguments supplying the hyperparameters.
    """
    from ultralytics import YOLO

    model = YOLO(str(checkpoint.weights))
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=resolve_device(args.device),
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        cos_lr=args.cos_lr,
        fliplr=args.fliplr,
        copy_paste=args.copy_paste,
        multi_scale=args.multi_scale,
        erasing=args.erasing,
        project=str(models_dir),
        name=run_name,
        exist_ok=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        dataset_yaml = data_yaml(args.round)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        found = discover(args.models_dir, exclude_suffix=args.suffix)
        found = filter_by_name(found, args.only)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not found:
        print(f"No checkpoints found under {args.models_dir}. "
              f"Run doclayout_ft.training.download_base_models first.", file=sys.stderr)
        return 1

    print(f"Discovered {len(found)} checkpoint(s) under {args.models_dir}:")
    for name, checkpoint in found.items():
        kind = "baseline" if checkpoint.is_baseline else f"trained @ {checkpoint.imgsz}"
        print(f"  {name:<50} {kind}")

    if args.list:
        return 0

    print(f"\nDataset: {dataset_yaml}")
    print(f"Schedule: epochs={args.epochs} patience={args.patience} "
          f"imgsz={args.imgsz} batch={args.batch} device={args.device}")
    print(f"Augmentation: fliplr={args.fliplr} copy_paste={args.copy_paste} "
          f"multi_scale={args.multi_scale} erasing={args.erasing}")

    planned: list[tuple[str, Checkpoint]] = []
    for name, checkpoint in found.items():
        run_name = f"{name}_{args.suffix}"
        if (args.models_dir / run_name).exists():
            print(f"\nSkipping {name}: {args.models_dir / run_name} already exists.")
            continue
        planned.append((run_name, checkpoint))

    if not planned:
        print("\nEvery run already exists. Nothing to do.")
        return 0

    if args.dry_run:
        print(f"\nWould train {len(planned)} run(s):")
        for run_name, checkpoint in planned:
            print(f"  {checkpoint.weights} -> {args.models_dir / run_name}")
        return 0

    failures: list[tuple[str, Exception]] = []
    for index, (run_name, checkpoint) in enumerate(planned, start=1):
        print(f"\n=== [{index}/{len(planned)}] {checkpoint.name} -> {run_name} ===")
        try:
            train_one(checkpoint, run_name, dataset_yaml, args.models_dir, args)
        except Exception as exc:  # noqa: BLE001 - a failed run must not kill the sweep
            failures.append((run_name, exc))
            print(f"Run {run_name} FAILED: {exc}", file=sys.stderr)
            print("Continuing with the next checkpoint. If this was a CUDA "
                  "out-of-memory error, lower --batch or --imgsz.", file=sys.stderr)

    print(f"\nFinished {len(planned) - len(failures)}/{len(planned)} run(s).")
    if failures:
        print("Failed runs:", file=sys.stderr)
        for run_name, exc in failures:
            print(f"  {run_name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
