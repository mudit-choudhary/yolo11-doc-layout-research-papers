"""Carve a batch of page images out of the image pool for annotation.

Annotation happens in manageable chunks rather than across the whole image
pool at once. This module moves a sampled set of pages into
``images/batch_NN/`` so the annotation tool can be pointed at one folder, and
so it is always obvious which pages have already been handed off.

Sampling deliberately over-represents page 1. Left to a uniform sample, the
``Title``, ``Authors`` and to a lesser extent ``Section-header`` classes would
be badly under-represented, because they occur on the first page of a paper
and essentially nowhere else. A pool that is one part page-1 to roughly three
parts interior pages keeps those classes learnable without drowning out the
body-text layouts that dominate real documents.

The move is destructive by design: a page belongs to exactly one batch, so a
page can never be annotated twice under two different batch folders. Use
``--dry-run`` first, and ``--copy`` if you would rather leave the pool intact.

Usage::

    python -m doclayout_ft.data.make_batches --batch-name batch_06
    python -m doclayout_ft.data.make_batches --batch-name batch_06 --page-one 200 --other 650
    python -m doclayout_ft.data.make_batches --batch-name batch_06 --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

from doclayout_ft.config import IMAGE_DIR, IMAGE_EXTENSIONS

#: Page-1 images per batch. See the module docstring for why these are
#: over-sampled relative to their natural frequency.
DEFAULT_PAGE_ONE_COUNT = 200

#: Interior (non-page-1) images per batch.
DEFAULT_OTHER_COUNT = 650

#: Marker identifying a page-1 image, from the naming scheme in
#: :mod:`doclayout_ft.data.pdf_to_images`.
PAGE_ONE_MARKER = "_page_01."


def pool_images(source_dir: Path) -> list[Path]:
    """List the page images still available for batching.

    Only the top level of ``source_dir`` is scanned. Existing ``batch_NN/``
    subfolders are skipped, so images already handed off for annotation are
    never drawn a second time.

    Args:
        source_dir: The image pool, normally ``images/``.

    Returns:
        Image paths, sorted for reproducibility.
    """
    return sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def partition_by_page_one(images: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split images into (page-1 images, everything else).

    Args:
        images: Candidate page images.

    Returns:
        A ``(page_ones, others)`` pair.
    """
    page_ones = [p for p in images if PAGE_ONE_MARKER in p.name.lower()]
    page_one_set = set(page_ones)
    others = [p for p in images if p not in page_one_set]
    return page_ones, others


def select_batch(
    images: list[Path],
    page_one_count: int,
    other_count: int,
    seed: int | None,
) -> list[Path]:
    """Sample a class-balanced batch from the available pool.

    Both quotas are capped at what the pool can actually supply, so a nearly
    exhausted pool yields a smaller batch rather than an error.

    Args:
        images: Candidate page images.
        page_one_count: Desired number of page-1 images.
        other_count: Desired number of interior pages.
        seed: Seed for reproducible sampling, or None for a fresh shuffle.

    Returns:
        The selected image paths.
    """
    page_ones, others = partition_by_page_one(images)
    rng = random.Random(seed)

    chosen_page_ones = rng.sample(page_ones, min(len(page_ones), page_one_count))
    chosen_others = rng.sample(others, min(len(others), other_count))

    print(f"  page-1 images available: {len(page_ones):>6}  selected: {len(chosen_page_ones)}")
    print(f"  interior images available: {len(others):>4}  selected: {len(chosen_others)}")

    return chosen_page_ones + chosen_others


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--batch-name", required=True,
                        help="Name of the batch folder to create, e.g. batch_06")
    parser.add_argument("--source", type=Path, default=IMAGE_DIR,
                        help="Image pool to draw from (default: images/)")
    parser.add_argument("--page-one", type=int, default=DEFAULT_PAGE_ONE_COUNT,
                        help=f"Page-1 images to include (default: {DEFAULT_PAGE_ONE_COUNT})")
    parser.add_argument("--other", type=int, default=DEFAULT_OTHER_COUNT,
                        help=f"Interior pages to include (default: {DEFAULT_OTHER_COUNT})")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for reproducible sampling (default: unseeded)")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of moving, leaving the pool intact. "
                             "Risks the same page being annotated in two batches.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the selection without creating or moving anything")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if not args.source.is_dir():
        print(f"error: image pool not found: {args.source}", file=sys.stderr)
        return 1

    batch_dir = args.source / args.batch_name
    if batch_dir.exists() and not args.dry_run:
        print(f"error: {batch_dir} already exists. Rename, delete or move it first.",
              file=sys.stderr)
        return 1

    print(f"Scanning {args.source} ...")
    images = pool_images(args.source)
    if not images:
        print(f"error: no unbatched images left in {args.source}", file=sys.stderr)
        return 1

    selection = select_batch(images, args.page_one, args.other, args.seed)
    verb, gerund = ("copy", "Copying") if args.copy else ("move", "Moving")
    print(f"\nWould {verb} {len(selection)} image(s) into {batch_dir}"
          if args.dry_run else
          f"\n{gerund} {len(selection)} image(s) into {batch_dir}")

    if args.dry_run:
        return 0

    batch_dir.mkdir(parents=True)
    transfer = shutil.copy2 if args.copy else shutil.move

    moved = 0
    failures: list[tuple[Path, Exception]] = []
    for image in selection:
        try:
            transfer(str(image), str(batch_dir / image.name))
            moved += 1
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            failures.append((image, exc))

    print(f"Done: {moved} image(s) now in {batch_dir}")
    if failures:
        print(f"{len(failures)} image(s) failed to {verb}:", file=sys.stderr)
        for image, exc in failures:
            print(f"  {image.name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
