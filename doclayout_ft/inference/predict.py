"""Run a fine-tuned checkpoint over pages and save detections.

Accepts page images, a directory of them, or a PDF, and writes two things per
page: an annotated preview image, and a JSON file of the raw boxes. The JSON
is the useful output for anything downstream, such as layout-aware chunking of
a document for retrieval; the preview exists so a human can tell at a glance
whether the model is behaving.

Two settings decide what ends up in that JSON:

``--conf``
    Detections below this confidence are dropped. The Ultralytics default of
    0.25 is tuned for natural-scene detection. For layout work the cost of a
    missed paragraph is usually higher than the cost of a spurious box that
    downstream logic can discard, so this defaults lower, at 0.20.

``--imgsz``
    Should match the size the checkpoint was trained at, and does so by
    default. Running a model trained at 1024 over 640-pixel inputs loses small
    classes such as footnotes and page headers first.

Usage::

    python -m doclayout_ft.inference.predict --source page.jpg
    python -m doclayout_ft.inference.predict --source paper.pdf --out-dir /tmp/out
    python -m doclayout_ft.inference.predict --source images/batch_01 --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from doclayout_ft.checkpoints import discover
from doclayout_ft.config import FINETUNED_DIR, IMAGE_EXTENSIONS, ROOT, resolve_device

#: Model used when ``--model`` is not given; the strongest run in the project.
DEFAULT_MODEL = "yolo11s_doc_layout_imgsz_1024"

#: Confidence floor. Lower than the Ultralytics default of 0.25; see above.
DEFAULT_CONF = 0.20

DEFAULT_OUT_DIR = ROOT / "predictions"


def pdf_to_temp_images(pdf_path: Path, dpi: int) -> tuple[Path, list[Path]]:
    """Render a PDF into a temporary directory of page images.

    The same 300 DPI default as the training pipeline is used, so that
    inference sees pages at the scale the model was trained on.

    Args:
        pdf_path: The PDF to render.
        dpi: Render resolution.

    Returns:
        A ``(temp_dir, image_paths)`` pair. The caller owns ``temp_dir`` and
        must delete it.
    """
    from pdf2image import convert_from_path

    temp_dir = Path(tempfile.mkdtemp(prefix="doclayout_pred_"))
    pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="jpeg")
    paths = []
    for index, page in enumerate(pages, start=1):
        path = temp_dir / f"{pdf_path.stem}_page_{index:02d}.jpg"
        page.save(path, "JPEG")
        paths.append(path)
    return temp_dir, paths


def collect_sources(source: Path, dpi: int) -> tuple[list[Path], Path | None]:
    """Resolve a ``--source`` argument into a list of image paths.

    Args:
        source: An image file, a directory of images, or a PDF.
        dpi: Render resolution, used only when ``source`` is a PDF.

    Returns:
        A ``(images, temp_dir)`` pair, where ``temp_dir`` is non-None only when
        a PDF was rendered and needs cleaning up afterwards.

    Raises:
        FileNotFoundError: If the source does not exist or holds no images.
    """
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    if source.is_dir():
        images = sorted(p for p in source.iterdir()
                        if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f"No images in {source}")
        return images, None

    if source.suffix.lower() == ".pdf":
        temp_dir, images = pdf_to_temp_images(source, dpi)
        return images, temp_dir

    if source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source], None

    raise FileNotFoundError(
        f"Unsupported source type: {source.suffix or '(no extension)'}. "
        f"Expected a PDF, an image, or a directory of images."
    )


def detections_to_dict(result, image_path: Path) -> dict[str, object]:
    """Convert one Ultralytics result into a serialisable record.

    Boxes are reported in absolute pixel ``xyxy`` coordinates against the
    original page size, not normalised, because a consumer reconciling these
    against extracted text positions works in page pixels.

    Args:
        result: One element of the list returned by ``model.predict``.
        image_path: The page this result belongs to.

    Returns:
        A dict with the page identity, its pixel dimensions, and its boxes.
    """
    height, width = result.orig_shape
    boxes = []
    for box in result.boxes:
        class_id = int(box.cls.item())
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        boxes.append({
            "class_id": class_id,
            "class_name": result.names.get(class_id, f"id_{class_id}"),
            "confidence": round(float(box.conf.item()), 4),
            "xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        })
    # Reading order is not something the model predicts, but top-to-bottom then
    # left-to-right is a reasonable default for a single-column consumer and
    # costs nothing to provide.
    boxes.sort(key=lambda b: (b["xyxy"][1], b["xyxy"][0]))
    return {
        "image": image_path.name,
        "width": width,
        "height": height,
        "detections": boxes,
    }


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--source", type=Path, required=True,
                        help="An image, a directory of images, or a PDF")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Run name to load (default: {DEFAULT_MODEL})")
    parser.add_argument("--models-dir", type=Path, default=FINETUNED_DIR,
                        help="Directory to look the model up in (default: FinetunedModels/)")
    parser.add_argument("--weights", type=Path, default=None,
                        help="Path to a .pt file, bypassing --model lookup")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Directory for annotated images and JSON (default: predictions/)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"Confidence floor (default: {DEFAULT_CONF})")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold (default: 0.7)")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Inference image size (default: the model's training size)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Render resolution when --source is a PDF (default: 300)")
    parser.add_argument("--device", default="0",
                        help="CUDA device index, or 'cpu' (default: 0)")
    parser.add_argument("--no-images", action="store_true",
                        help="Write only JSON, skipping the annotated previews")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    from ultralytics import YOLO

    if args.weights is not None:
        if not args.weights.is_file():
            print(f"error: weights not found: {args.weights}", file=sys.stderr)
            return 1
        weights, imgsz = args.weights, args.imgsz or 1024
    else:
        try:
            found = discover(args.models_dir, baseline_suffix="_baseline")
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        checkpoint = found.get(args.model)
        if checkpoint is None:
            print(f"error: no model named '{args.model}' under {args.models_dir}.\n"
                  f"Available: {', '.join(sorted(found)) or '(none)'}", file=sys.stderr)
            return 1
        weights, imgsz = checkpoint.weights, args.imgsz or checkpoint.imgsz

    temp_dir: Path | None = None
    try:
        images, temp_dir = collect_sources(args.source, args.dpi)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ImportError:
        print("error: rendering a PDF needs pdf2image and Poppler.\n"
              "       pip install pdf2image && sudo apt install poppler-utils",
              file=sys.stderr)
        return 1

    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Model:  {weights}")
        print(f"Pages:  {len(images)}   imgsz: {imgsz}   conf: {args.conf}")
        print(f"Output: {args.out_dir}\n")

        model = YOLO(str(weights))
        total_detections = 0

        for index, image_path in enumerate(images, start=1):
            results = model.predict(
                source=str(image_path),
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                device=resolve_device(args.device),
                verbose=False,
            )
            record = detections_to_dict(results[0], image_path)
            total_detections += len(record["detections"])

            json_path = args.out_dir / f"{image_path.stem}.json"
            json_path.write_text(json.dumps(record, indent=2))

            if not args.no_images:
                results[0].save(filename=str(args.out_dir / f"{image_path.stem}_annotated.jpg"))

            print(f"[{index}/{len(images)}] {image_path.name}: "
                  f"{len(record['detections'])} detection(s)")

        print(f"\n{total_detections} detection(s) across {len(images)} page(s) "
              f"-> {args.out_dir}")
        return 0
    finally:
        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
