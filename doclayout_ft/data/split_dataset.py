"""Split a dataset round into train / val / test lists.

Pages are grouped by paper before splitting, and a whole paper goes to exactly
one split. This matters more than it might look. Pages from the same paper
share a template: the same two-column grid, the same header, the same caption
style, often the same figures. If page 3 of a paper is in train and page 7 is
in val, the model has effectively seen the validation layout during training,
and validation mAP measures memorisation rather than generalisation. Splitting
by paper is what makes the number honest.

This was a mid-project correction. Rounds up to and including ``round_03``
pointed ``train:`` and ``val:`` at the same image folder, so every metric from
that period is meaningless. ``round_final`` is the first round split properly;
see docs/DATASET.md.

The split is non-destructive. Images and labels stay where they are; this only
writes ``train.txt`` / ``val.txt`` / ``test.txt`` and repoints ``data.yaml`` at
them. Re-running with the same ``--seed`` reproduces the same split exactly.

Usage::

    python -m doclayout_ft.data.split_dataset --round round_final
    python -m doclayout_ft.data.split_dataset --round round_final --train 0.8 --val 0.1 --test 0.1
    python -m doclayout_ft.data.split_dataset --round round_final --dry-run
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

from doclayout_ft.config import DEFAULT_ROUND, IMAGE_EXTENSIONS, ROOT, TRAINING_DIR

#: Matches the trailing ``_page_07`` produced by
#: :mod:`doclayout_ft.data.pdf_to_images`. Stripping it turns a page filename
#: back into its paper identity.
PAGE_SUFFIX_RE = re.compile(r"_page_\d+$")

DEFAULT_TRAIN_FRACTION = 0.75
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.10

#: Seed for the paper shuffle. Fixed by default so a split is reproducible
#: and two people running this on the same round get identical files.
DEFAULT_SEED = 42


def paper_key(image_path: Path) -> str:
    """Return the grouping key for one page image.

    Args:
        image_path: Path to a page image, named ``<Paper>_page_NN.jpg``.

    Returns:
        The paper identity: the filename stem with the ``_page_NN`` suffix
        removed. A file that does not carry the suffix becomes its own group,
        which is the safe default -- it can only ever land in one split.
    """
    return PAGE_SUFFIX_RE.sub("", image_path.stem)


def group_by_paper(images: list[Path]) -> dict[str, list[Path]]:
    """Group page images by the paper they came from.

    Args:
        images: Page image paths.

    Returns:
        Mapping of paper key to the pages belonging to it, each page list
        sorted by filename.
    """
    groups: dict[str, list[Path]] = {}
    for image in images:
        groups.setdefault(paper_key(image), []).append(image)
    for pages in groups.values():
        pages.sort()
    return groups


def find_missing_labels(images: list[Path], labels_dir: Path) -> list[Path]:
    """Return the images that have no matching label file.

    An image without a ``.txt`` label is not the same as an image with an
    empty one. Ultralytics reads a missing label as "background, no objects",
    so a page that was simply never annotated would silently train the model
    to predict nothing on pages that look like it. Catching this before the
    split is cheaper than discovering it in the loss curve.

    Args:
        images: Page image paths.
        labels_dir: Directory holding one ``<stem>.txt`` per image.

    Returns:
        The image paths whose label is absent.
    """
    return [p for p in images if not (labels_dir / f"{p.stem}.txt").exists()]


def assign_splits(
    paper_keys: list[str],
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    """Shuffle papers and cut them into train / val / test.

    Test receives the remainder rather than its own rounded count, so the
    three splits always partition the papers exactly with none dropped.

    Args:
        paper_keys: One key per paper.
        train_fraction: Share of papers for training.
        val_fraction: Share of papers for validation.
        seed: Shuffle seed.

    Returns:
        Mapping of split name to the paper keys assigned to it.
    """
    keys = list(paper_keys)
    random.Random(seed).shuffle(keys)

    n_papers = len(keys)
    n_train = round(n_papers * train_fraction)
    n_val = round(n_papers * val_fraction)

    return {
        "train": keys[:n_train],
        "val": keys[n_train:n_train + n_val],
        "test": keys[n_train + n_val:],
    }


def write_split_list(path: Path, images: list[Path]) -> None:
    """Write one split's image list.

    Paths are written absolute and resolved. Ultralytics only prefixes the
    dataset's ``path:`` onto list entries that begin with ``./``; any other
    relative entry is resolved against the working directory at train time,
    not against the dataset directory. Absolute paths sidestep that entirely
    and make the lists correct no matter where training is launched from.

    Args:
        path: The ``.txt`` file to write.
        images: Images belonging to this split.
    """
    lines = sorted(str(p.resolve()) for p in images)
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def update_data_yaml(data_yaml: Path, round_dir: Path, has_test: bool) -> None:
    """Repoint a round's ``data.yaml`` at the split lists just written.

    Also rewrites the ``path:`` key to this round's location **relative to the
    repository root**. It used to be absolute, which went stale whenever the
    repo moved and, worse, was copied into every checkpoint's saved arguments
    and from there into exported ONNX metadata, putting the training machine's
    directory layout into published files.

    A relative root means training must be launched from the repository root.
    Ultralytics keeps a relative ``path:`` as-is only while it resolves against
    the working directory; if it does not, the value is retried against the
    global ``datasets_dir`` instead and the run fails to find its images. Every
    command in ``docs/`` is already run from the repo root.

    The edit is line-based rather than a YAML round-trip so that comments and
    the hand-maintained ``names:`` block survive untouched.

    Args:
        data_yaml: The config file to rewrite in place.
        round_dir: The round directory, whose repo-relative path becomes the
            new ``path:`` value.
        has_test: Whether a ``test.txt`` was written.
    """
    text = data_yaml.read_text()
    try:
        root = round_dir.resolve().relative_to(ROOT)
    except ValueError:
        # A round living outside the repository has no repo-relative spelling,
        # so it keeps an absolute one. Nothing published comes from such a
        # round; this is for scratch datasets kept elsewhere on disk.
        root = round_dir.resolve()
    text = re.sub(r"^path:.*$", f"path: {root}", text, flags=re.MULTILINE)
    text = re.sub(r"^train:.*$", "train: train.txt", text, flags=re.MULTILINE)
    text = re.sub(r"^val:.*$", "val: val.txt", text, flags=re.MULTILINE)

    if has_test:
        if re.search(r"^test:.*$", text, flags=re.MULTILINE):
            text = re.sub(r"^test:.*$", "test: test.txt", text, flags=re.MULTILINE)
        else:
            text = text.replace("val: val.txt", "val: val.txt\ntest: test.txt", 1)

    data_yaml.write_text(text)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--round", default=DEFAULT_ROUND,
                        help=f"Round directory under training_dataset/ (default: {DEFAULT_ROUND})")
    parser.add_argument("--train", type=float, default=DEFAULT_TRAIN_FRACTION,
                        help=f"Training share of papers (default: {DEFAULT_TRAIN_FRACTION})")
    parser.add_argument("--val", type=float, default=DEFAULT_VAL_FRACTION,
                        help=f"Validation share of papers (default: {DEFAULT_VAL_FRACTION})")
    parser.add_argument("--test", type=float, default=DEFAULT_TEST_FRACTION,
                        help=f"Test share of papers (default: {DEFAULT_TEST_FRACTION})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Shuffle seed (default: {DEFAULT_SEED})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the split without writing any file")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        print(f"error: --train + --val + --test must sum to 1.0 "
              f"(got {args.train + args.val + args.test:.4f})", file=sys.stderr)
        return 1

    round_path = TRAINING_DIR / args.round
    images_dir = round_path / "images"
    labels_dir = round_path / "labels"
    data_yaml = round_path / "data.yaml"

    for required in (images_dir, labels_dir, data_yaml):
        if not required.exists():
            print(f"error: expected {required} to exist", file=sys.stderr)
            return 1

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"error: no images found in {images_dir}", file=sys.stderr)
        return 1

    missing = find_missing_labels(images, labels_dir)
    if missing:
        print(f"error: {len(missing)} image(s) have no label file in {labels_dir}.",
              file=sys.stderr)
        for image in missing[:10]:
            print(f"  {image.name}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)
        print("Annotate them, or move them out of the round, then re-run.",
              file=sys.stderr)
        return 1

    groups = group_by_paper(images)
    splits = assign_splits(list(groups), args.train, args.val, args.seed)

    print(f"{len(groups)} paper(s), {len(images)} page(s) in {round_path.name}")
    for name, keys in splits.items():
        pages = sum(len(groups[k]) for k in keys)
        print(f"  {name:<6} {len(keys):>4} paper(s)  {pages:>5} page(s)")

    if args.dry_run:
        print("\nDry run: nothing written.")
        return 0

    for name, keys in splits.items():
        split_images = [img for key in keys for img in groups[key]]
        list_path = round_path / f"{name}.txt"
        write_split_list(list_path, split_images)
        print(f"Wrote {list_path} ({len(split_images)} line(s))")

    update_data_yaml(data_yaml, round_path, has_test=bool(splits["test"]))
    print(f"Updated {data_yaml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
