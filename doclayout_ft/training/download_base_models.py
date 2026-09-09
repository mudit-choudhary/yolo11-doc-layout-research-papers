"""Download the pre-trained document-layout checkpoints to fine-tune from.

Fine-tuning here does not start from COCO weights. It starts from checkpoints
already trained for document layout on DocLayNet-style data, published as
``Armaggheddon/yolo11-document-layout`` on the Hugging Face Hub. Starting from
a model that already knows what a caption or a table looks like is what lets a
few hundred annotated pages be enough.

Three sizes are available. On a 4 GB card only the first two are practical:

===========  ==========  ====================================================
Checkpoint   Parameters  Notes
===========  ==========  ====================================================
yolo11n      2.6 M       Fits at batch 4, imgsz 1024. Fastest to train.
yolo11s      9.4 M       Fits at batch 3, imgsz 1024. Best results here.
yolo11m      20 M        Runs out of memory below batch 2 on a GTX 1650.
===========  ==========  ====================================================

Each file is saved into its own directory under ``models/`` so that the
checkpoint discovery in :mod:`doclayout_ft.training.finetune` picks it up as a
fine-tuning starting point.

Usage::

    python -m doclayout_ft.training.download_base_models
    python -m doclayout_ft.training.download_base_models --variants n s
    python -m doclayout_ft.training.download_base_models --dest models --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doclayout_ft.config import MODELS_DIR

#: Hub repository holding the pre-trained document-layout checkpoints.
BASE_REPO_ID = "Armaggheddon/yolo11-document-layout"

#: Size letter -> (filename in the Hub repo, local directory name).
#: The local directory name is what run names are derived from later, so it is
#: kept short and stable.
BASE_MODELS: dict[str, tuple[str, str]] = {
    "n": ("yolo11n_doc_layout.pt", "yolov11n"),
    "s": ("yolo11s_doc_layout.pt", "yolov11s"),
    "m": ("yolo11m_doc_layout.pt", "yolov11m"),
}

#: Sizes downloaded when ``--variants`` is not given. ``m`` is excluded because
#: it cannot be trained on the reference hardware; ask for it explicitly.
DEFAULT_VARIANTS = ("n", "s")


def download_variant(variant: str, dest_root: Path) -> Path:
    """Download one base checkpoint.

    Args:
        variant: Size letter, one of the keys of :data:`BASE_MODELS`.
        dest_root: Directory to create the per-model folder under.

    Returns:
        Path to the downloaded ``.pt`` file.

    Raises:
        KeyError: If ``variant`` is not a known size.
    """
    # Imported lazily so --help works without huggingface_hub installed.
    from huggingface_hub import hf_hub_download

    filename, dir_name = BASE_MODELS[variant]
    target_dir = dest_root / dir_name
    target_dir.mkdir(parents=True, exist_ok=True)

    path = hf_hub_download(
        repo_id=BASE_REPO_ID,
        filename=filename,
        repo_type="model",
        local_dir=str(target_dir),
    )
    return Path(path)


def describe_checkpoint(weights: Path) -> None:
    """Print the class taxonomy a checkpoint was trained with.

    Worth checking once after download: the base checkpoints ship 11 classes,
    and fine-tuning adds a 12th (``Authors``). If these names ever stop
    matching the first 11 entries of
    :data:`doclayout_ft.config.CLASS_NAMES`, every label id in the dataset
    would be pointing at the wrong class.

    Args:
        weights: Path to a ``.pt`` checkpoint.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    print(f"  classes ({len(model.names)}): "
          f"{', '.join(model.names[i] for i in sorted(model.names))}")


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--variants", nargs="+", choices=sorted(BASE_MODELS),
                        default=list(DEFAULT_VARIANTS),
                        help=f"Model sizes to download (default: {' '.join(DEFAULT_VARIANTS)})")
    parser.add_argument("--dest", type=Path, default=MODELS_DIR,
                        help="Directory to download into (default: models/)")
    parser.add_argument("--list", action="store_true",
                        help="Print each checkpoint's class taxonomy after downloading")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(args.variants)} checkpoint(s) from {BASE_REPO_ID} "
          f"into {args.dest}")

    failures: list[tuple[str, Exception]] = []
    for variant in args.variants:
        filename, _ = BASE_MODELS[variant]
        try:
            path = download_variant(variant, args.dest)
            print(f"  {filename} -> {path}")
            if args.list:
                describe_checkpoint(path)
        except Exception as exc:  # noqa: BLE001 - report every failure together
            failures.append((filename, exc))
            print(f"  {filename}: FAILED ({exc})", file=sys.stderr)

    if failures:
        print(f"\n{len(failures)} download(s) failed.", file=sys.stderr)
        return 1

    print("\nDone. These are now discoverable as fine-tuning starting points "
          "by doclayout_ft.training.finetune.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
