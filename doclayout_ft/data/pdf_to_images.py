"""Render source PDFs to per-page JPEG images.

This is stage one of the dataset pipeline: every PDF in ``PDFs/`` becomes one
JPEG per page in ``images/``, named ``<PaperName>_page_NN.jpg``. That naming is
not cosmetic -- the ``_page_NN`` suffix is what
:mod:`doclayout_ft.data.split_dataset` strips to group pages by paper, so
pages of one paper never straddle a train/val boundary.

Rendering is done at 300 DPI, the usual floor for text small enough to
annotate reliably. Dropping to 150 makes footnote and page-header text too
soft to place a tight box around; 600 doubles the disk cost for no annotation
benefit at this page size.

Requires the Poppler ``pdftoppm`` binary, which ``pdf2image`` shells out to::

    sudo apt install poppler-utils

Usage::

    python -m doclayout_ft.data.pdf_to_images
    python -m doclayout_ft.data.pdf_to_images --dpi 400 --overwrite
    python -m doclayout_ft.data.pdf_to_images --limit 10 --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

from doclayout_ft.config import IMAGE_DIR, PDF_DIR

#: Render resolution. See the module docstring for why 300 rather than 150/600.
DEFAULT_DPI = 300

#: Pause this many seconds every ``--pause-every`` PDFs. Poppler spawns a
#: subprocess per document and holds every rendered page in RAM before saving;
#: on a long batch that steadily climbs until the machine starts swapping. The
#: pause gives the allocator a chance to hand memory back.
DEFAULT_PAUSE_SECONDS = 7.0
DEFAULT_PAUSE_EVERY = 25


def page_image_name(pdf_stem: str, page_number: int) -> str:
    """Build the image filename for one page.

    Args:
        pdf_stem: PDF filename without its extension.
        page_number: 1-based page number.

    Returns:
        A name of the form ``<pdf_stem>_page_07.jpg``. The page number is
        zero-padded to two digits so filenames sort in reading order, and so
        the suffix regex in the splitter matches predictably.
    """
    return f"{pdf_stem}_page_{page_number:02d}.jpg"


def existing_image_names(dest_dir: Path) -> set[str]:
    """Collect the basenames of every page image already under ``dest_dir``.

    The scan is recursive, because :mod:`doclayout_ft.data.make_batches` moves
    images out of the root into ``batch_NN/`` subfolders as they are handed off
    for annotation. A non-recursive check would see those pages as missing and
    re-render every PDF that has already been annotated.

    Only names are collected, not paths, so this stays cheap even though the
    image tree runs to tens of gigabytes.
    """
    if not dest_dir.is_dir():
        return set()
    return {p.name for p in dest_dir.rglob("*.jpg")}


def already_rendered(pdf_path: Path, existing: set[str]) -> bool:
    """Return True if this PDF appears to have been rendered already.

    Checks only for page 1, which is enough to skip whole documents on a
    re-run without paying to re-open every PDF just to count its pages. A PDF
    whose render was interrupted partway will be re-rendered from scratch when
    ``--overwrite`` is passed.

    Args:
        pdf_path: The PDF being considered.
        existing: Basenames already on disk, from :func:`existing_image_names`.
    """
    return page_image_name(pdf_path.stem, 1) in existing


def convert_pdf(pdf_path: Path, dest_dir: Path, dpi: int) -> int:
    """Render one PDF to JPEGs in ``dest_dir``.

    Args:
        pdf_path: The PDF to render.
        dest_dir: Directory to write page images into.
        dpi: Render resolution.

    Returns:
        Number of pages written.

    Raises:
        Exception: Whatever ``pdf2image`` raises for a malformed or
            password-protected PDF. The caller decides whether one bad file
            should stop the batch.
    """
    # Imported lazily so that --help and the module's tests work without
    # pdf2image and Poppler installed.
    from pdf2image import convert_from_path

    pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="jpeg")
    for index, page in enumerate(pages, start=1):
        page.save(dest_dir / page_image_name(pdf_path.stem, index), "JPEG")
    return len(pages)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, default=PDF_DIR,
                        help="Directory of source PDFs (default: PDFs/)")
    parser.add_argument("--dest", type=Path, default=IMAGE_DIR,
                        help="Directory to write page images to (default: images/)")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI,
                        help=f"Render resolution (default: {DEFAULT_DPI})")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-render PDFs whose page 1 image already exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many PDFs (useful for a trial run)")
    parser.add_argument("--pause-every", type=int, default=DEFAULT_PAUSE_EVERY,
                        help="Pause after this many PDFs to let memory settle")
    parser.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_SECONDS,
                        help="Length of that pause, in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be rendered, write nothing")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if not args.dry_run:
        # Check both halves of the PDF stack up front. Without this, a missing
        # dependency surfaces as one identical "conversion failed" line per
        # PDF, which buries the actual cause under hundreds of lines.
        try:
            import pdf2image  # noqa: F401
        except ImportError:
            print(
                "error: pdf2image is not installed.\n"
                "       pip install pdf2image",
                file=sys.stderr,
            )
            return 1

        if shutil.which("pdftoppm") is None:
            print(
                "error: the 'pdftoppm' binary was not found on PATH.\n"
                "       pdf2image needs Poppler: sudo apt install poppler-utils",
                file=sys.stderr,
            )
            return 1

    if not args.source.is_dir():
        print(f"error: source directory not found: {args.source}", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in args.source.iterdir()
                  if p.is_file() and p.suffix.lower() == ".pdf")
    if not pdfs:
        print(f"No PDFs found in {args.source}")
        return 0

    if not args.overwrite:
        existing = existing_image_names(args.dest)
        pending = [p for p in pdfs if not already_rendered(p, existing)]
        skipped = len(pdfs) - len(pending)
        if skipped:
            print(f"Skipping {skipped} PDF(s) already rendered "
                  f"(pass --overwrite to redo them).")
        pdfs = pending

    if args.limit is not None:
        pdfs = pdfs[: args.limit]

    if not pdfs:
        print("Nothing to do.")
        return 0

    print(f"Rendering {len(pdfs)} PDF(s) at {args.dpi} DPI -> {args.dest}")
    if args.dry_run:
        for pdf in pdfs:
            print(f"  would render {pdf.name}")
        return 0

    total_pages = 0
    failures: list[tuple[Path, Exception]] = []

    for count, pdf in enumerate(pdfs, start=1):
        try:
            pages = convert_pdf(pdf, args.dest, args.dpi)
            total_pages += pages
            print(f"[{count}/{len(pdfs)}] {pdf.name}: {pages} page(s)")
        except Exception as exc:  # noqa: BLE001 - one bad PDF must not stop the batch
            failures.append((pdf, exc))
            print(f"[{count}/{len(pdfs)}] {pdf.name}: FAILED ({exc})", file=sys.stderr)

        if args.pause_every > 0 and count % args.pause_every == 0 and count < len(pdfs):
            time.sleep(args.pause_seconds)

    print(f"\nWrote {total_pages} page image(s) to {args.dest}")
    if failures:
        print(f"{len(failures)} PDF(s) failed:", file=sys.stderr)
        for pdf, exc in failures:
            print(f"  {pdf.name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
