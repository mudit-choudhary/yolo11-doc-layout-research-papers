"""Measure inference latency and model size for every checkpoint.

Accuracy alone cannot answer "which model should I run". A checkpoint that is
0.005 mAP better and twice as slow is the wrong choice for a pipeline chewing
through thousands of pages, and the right one for a careful one-off extraction.
The comparison table cannot say that without timings, so this produces them.

What is measured, and why it is measured this way:

**Warmup is discarded.** The first inferences on a CUDA device pay for kernel
autotuning, memory-pool growth and lazy module initialisation. Including them
would report a number nobody ever experiences after the first page.

**The GPU is synchronised around the timed region.** CUDA calls are
asynchronous, so timing without a sync measures how fast Python can queue work,
not how fast the work finishes.

**Each model runs at its own training resolution.** Timing a 1024-trained model
at 640 would report a latency for a configuration nobody should use.

**Batch size is 1.** Page-at-a-time is how a document pipeline actually calls
this, and it is the latency a user feels. Throughput at larger batches is a
different number and would need its own column.

**Pages are decoded once, up front, and the decode is not timed.** Reading and
decoding a 2550x3301 JPEG costs tens of milliseconds and costs exactly the same
for every model, so leaving it in would add a constant to all seventeen numbers
and compress the differences between them. What is reported is preprocessing,
the forward pass and non-maximum suppression, which is the part that differs.
Add roughly 60 to 80 ms per page for decode if you need the wall-clock figure.

Parameters and GFLOPs are recorded alongside, since those are properties of the
architecture rather than of this particular GPU, and travel to other hardware.

Usage::

    python -m doclayout_ft.evaluation.benchmark
    python -m doclayout_ft.evaluation.benchmark --images 40 --warmup 8
    python -m doclayout_ft.evaluation.benchmark --device cpu --only yolo11s_doc_layout_imgsz_1024
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover_many, filter_by_name
from doclayout_ft.config import (
    DEFAULT_ROUND,
    FINETUNED_DIR,
    IMAGE_EXTENSIONS,
    MODELS_DIR,
    REPORTS_DIR,
    resolve_device,
    round_dir,
)

#: Inferences discarded before timing starts.
DEFAULT_WARMUP = 8

#: Images timed per model. Enough for a stable median without a long sweep.
DEFAULT_IMAGES = 40

CSV_FIELDS = (
    "model", "imgsz", "params_m", "gflops",
    "median_ms", "mean_ms", "p90_ms", "fps", "weights_mb",
)


def sample_images(round_name: str, count: int) -> list[Path]:
    """Pick a stable sample of pages to time against.

    Sorted then truncated rather than randomly sampled, so two runs of this
    command time the same pages and their numbers can be compared.

    Args:
        round_name: Dataset round to draw from.
        count: How many images to take.

    Returns:
        Image paths.

    Raises:
        FileNotFoundError: If the round has no images directory.
    """
    images_dir = round_dir(round_name) / "images"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"No images directory at {images_dir}")
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise FileNotFoundError(f"No images in {images_dir}")
    return images[:count]


def _synchronise(device) -> None:
    """Block until queued CUDA work has finished, if this is a CUDA device."""
    import torch
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_pages(paths: list[Path]) -> list:
    """Decode pages into memory once, so the decode is not timed per model.

    Args:
        paths: Image files to decode.

    Returns:
        Decoded BGR arrays, in the order given.

    Raises:
        FileNotFoundError: If an image cannot be decoded.
    """
    import cv2

    pages = []
    for path in paths:
        page = cv2.imread(str(path))
        if page is None:
            raise FileNotFoundError(f"Could not decode {path}")
        pages.append(page)
    return pages


def benchmark_one(
    checkpoint: Checkpoint,
    pages: list,
    device,
    warmup: int,
) -> dict[str, object]:
    """Time one checkpoint over a fixed set of already-decoded pages.

    Args:
        checkpoint: Model to time.
        pages: Decoded page arrays from :func:`load_pages`.
        device: CUDA index or ``"cpu"``.
        warmup: Inferences to discard before timing.

    Returns:
        One row for the latency table.
    """
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(checkpoint.weights))
    predict = dict(imgsz=checkpoint.imgsz, device=device, verbose=False)

    for page in (pages * ((warmup // len(pages)) + 1))[:warmup]:
        model.predict(source=page, **predict)
    _synchronise(device)

    timings_ms: list[float] = []
    for page in pages:
        start = time.perf_counter()
        model.predict(source=page, **predict)
        _synchronise(device)
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings_ms.sort()
    median = statistics.median(timings_ms)
    return {
        "model": checkpoint.name,
        "imgsz": checkpoint.imgsz,
        "params_m": round(get_num_params(model.model) / 1e6, 2),
        "gflops": round(get_flops(model.model, checkpoint.imgsz), 1),
        "median_ms": round(median, 2),
        "mean_ms": round(statistics.fmean(timings_ms), 2),
        "p90_ms": round(timings_ms[int(len(timings_ms) * 0.9) - 1], 2),
        "fps": round(1000 / median, 2),
        "weights_mb": round(checkpoint.weights.stat().st_size / (1024 * 1024), 1),
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, nargs="+",
                        default=[FINETUNED_DIR, MODELS_DIR],
                        help="Directories of models to time")
    parser.add_argument("--round", default=DEFAULT_ROUND,
                        help=f"Round to draw pages from (default: {DEFAULT_ROUND})")
    parser.add_argument("--images", type=int, default=DEFAULT_IMAGES,
                        help=f"Pages timed per model (default: {DEFAULT_IMAGES})")
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP,
                        help=f"Inferences discarded first (default: {DEFAULT_WARMUP})")
    parser.add_argument("--device", default="0",
                        help="CUDA device index, or 'cpu' (default: 0)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Restrict to these model names")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "latency.csv",
                        help="Destination CSV (default: reports/latency.csv)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        found = discover_many(args.models_dir, include_baselines=False)
        found = filter_by_name(found, args.only)
        images = sample_images(args.round, args.images)
        pages = load_pages(images)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not found:
        print("No models found to benchmark.", file=sys.stderr)
        return 1

    device = resolve_device(args.device)
    print(f"Timing {len(found)} model(s) on {len(images)} pre-decoded page(s), "
          f"batch 1, device={device}, {args.warmup} warmup inference(s).")
    print("Excludes image decode, which is identical for every model.\n")

    rows, failures = [], []
    for index, (name, checkpoint) in enumerate(found.items(), start=1):
        print(f"[{index}/{len(found)}] {name} (imgsz={checkpoint.imgsz}) ... ",
              end="", flush=True)
        try:
            row = benchmark_one(checkpoint, pages, device, args.warmup)
            rows.append(row)
            print(f"{row['median_ms']:.1f} ms/page  ({row['fps']:.1f} FPS)")
        except Exception as exc:  # noqa: BLE001 - time the rest of the sweep
            failures.append((name, exc))
            print(f"FAILED: {exc}")

    if not rows:
        print("\nNothing was timed successfully.", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: r["median_ms"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {args.out}")

    width = max(len(str(r["model"])) for r in rows)
    print(f"\n{'model':<{width}} {'imgsz':>6} {'params':>8} {'GFLOPs':>8} "
          f"{'median':>9} {'p90':>8} {'FPS':>7}")
    for row in rows:
        print(f"{row['model']:<{width}} {row['imgsz']:>6} "
              f"{row['params_m']:>7.2f}M {row['gflops']:>8.1f} "
              f"{row['median_ms']:>7.1f}ms {row['p90_ms']:>6.1f}ms {row['fps']:>7.1f}")

    if failures:
        print(f"\n{len(failures)} model(s) failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
