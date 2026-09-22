"""Export every publishable checkpoint's ``best.pt`` to ``best.onnx``.

The ONNX file is written next to the weights it came from, as
``<run>/weights/best.onnx``, which is where
:func:`doclayout_ft.hub.push_to_hub.files_for` looks for it. Export here,
publish with the normal push command, and the Hub subfolder gains a
``best.onnx`` beside its ``best.pt``.

Each run is exported at the image size it was trained at, read from its
``args.yaml``. Exporting at a different size than the model was trained at
costs accuracy, and an ONNX graph baked at the wrong resolution is a silent
version of that mistake, since the file gives no hint of what it expects.

Every export is checked against the checkpoint it came from before it counts as
done: the same random tensor goes through the fused PyTorch model and through
the ONNX graph, and the outputs have to agree. A graph that silently exports
wrong looks exactly like one that exported right, and the point of publishing
these is that someone loads them instead of the weights.

Requires ``ultralytics``, ``onnx`` and ``onnxruntime`` in the environment;
Ultralytics installs the missing export extras itself on first run.

Usage::

    python -m doclayout_ft.hub.export_onnx --list
    python -m doclayout_ft.hub.export_onnx
    python -m doclayout_ft.hub.export_onnx --only yolo11s_doc_layout_imgsz_1024
    python -m doclayout_ft.hub.export_onnx --overwrite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover_many, filter_by_name
from doclayout_ft.config import FINETUNED_DIR, MODELS_DIR, ROOT
from doclayout_ft.hub.push_to_hub import load_ledger, pending_checkpoints, subfolder_for

#: ONNX opset. 12 is old enough for every runtime anyone is likely to load
#: these with (onnxruntime, OpenCV DNN, TensorRT) and new enough for YOLO11's
#: ops. Raise it only if a runtime asks for something newer.
OPSET = 12

#: How far the ONNX graph's output may drift from the PyTorch model's, relative
#: to the largest value in it. Conv/BN fusion and onnxslim reorder float
#: arithmetic, so the two never match bit for bit; observed drift across the
#: seventeen published runs is 1e-6 to 3e-6. Anything near this bound is a
#: broken export, not rounding.
MAX_RELATIVE_DRIFT = 1e-4


def onnx_path(checkpoint: Checkpoint) -> Path:
    """Where this run's ONNX export lives, whether or not it exists yet."""
    return checkpoint.weights.with_suffix(".onnx")


def portable_data_arg(data: str) -> str:
    """Reduce a recorded dataset path to one that means something elsewhere.

    A checkpoint records the dataset it was trained on as an absolute path on
    the machine that trained it, and Ultralytics copies that string into the
    ONNX ``description`` metadata. A published file should not carry somebody's
    home directory, so it is cut down to a repository-relative path, which is
    the part that is actually informative.

    Args:
        data: The ``data`` value recorded in the checkpoint.

    Returns:
        The path relative to the repository root, or its file name if it points
        somewhere outside the repository entirely.
    """
    try:
        return str(Path(data).resolve().relative_to(ROOT))
    except ValueError:
        return Path(data).name


def export_one(checkpoint: Checkpoint, overwrite: bool) -> Path:
    """Export one checkpoint to ONNX at its own training resolution.

    Args:
        checkpoint: The run to export.
        overwrite: Re-export even if ``best.onnx`` is already there.

    Returns:
        Path to the ONNX file.
    """
    out = onnx_path(checkpoint)
    if out.is_file() and not overwrite:
        print(f"  exists, skipped: {out}")
        return out

    from ultralytics import YOLO

    model = YOLO(str(checkpoint.weights))
    data = getattr(model.model, "args", {}).get("data")
    if data:
        model.model.args["data"] = portable_data_arg(data)

    written = model.export(
        format="onnx", imgsz=checkpoint.imgsz, opset=OPSET, simplify=True,
    )
    return Path(written)


def verify(checkpoint: Checkpoint, onnx_file: Path) -> float:
    """Compare the exported graph against the checkpoint it came from.

    Args:
        checkpoint: The run that was exported.
        onnx_file: The graph written for it.

    Returns:
        Largest output difference, relative to the largest output value.

    Raises:
        AssertionError: If the two disagree by more than
            :data:`MAX_RELATIVE_DRIFT`, or if the graph takes an input shape
            other than the one it was exported for.
    """
    import numpy as np
    import onnxruntime as ort
    import torch
    from ultralytics import YOLO

    size = checkpoint.imgsz
    session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    assert shape == [1, 3, size, size], f"{onnx_file} takes {shape}, expected [1, 3, {size}, {size}]"

    sample = torch.rand(1, 3, size, size)
    with torch.no_grad():
        expected = YOLO(str(checkpoint.weights)).model.fuse().eval()(sample)[0].numpy()
    actual = session.run(None, {"images": sample.numpy()})[0]

    drift = float(np.abs(expected - actual).max() / max(np.abs(expected).max(), 1e-9))
    assert drift < MAX_RELATIVE_DRIFT, f"{onnx_file} drifts {drift:.2e} from {checkpoint.weights}"
    return drift


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, nargs="+",
                        default=[FINETUNED_DIR, MODELS_DIR],
                        help="Directories of runs to export, most authoritative "
                             "first (default: FinetunedModels/ then models/)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Export these run names instead of all of them")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-export runs that already have a best.onnx")
    parser.add_argument("--list", action="store_true",
                        help="Show what would be exported, then exit")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    try:
        found = discover_many(args.models_dir, include_baselines=False)
        found = filter_by_name(found, args.only)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The publish queue, not every directory on disk: baselines are never
    # published and duplicate weights are published once, so exporting either
    # would produce a file nothing ever uploads.
    queue = pending_checkpoints(found, load_ledger(), include_published=True)
    if not queue:
        print("No fine-tuned runs found.", file=sys.stderr)
        return 1

    if args.list:
        for checkpoint in queue:
            state = "present" if onnx_path(checkpoint).is_file() else "missing"
            print(f"  {checkpoint.name:<52} imgsz {checkpoint.imgsz:>4}  "
                  f"-> {subfolder_for(checkpoint.name)}/best.onnx  [{state}]")
        return 0

    failures: list[tuple[str, Exception]] = []
    for index, checkpoint in enumerate(queue, start=1):
        print(f"=== [{index}/{len(queue)}] {checkpoint.name} "
              f"(imgsz {checkpoint.imgsz}) ===")
        try:
            out = export_one(checkpoint, args.overwrite)
            print(f"  {out}  {out.stat().st_size / (1024 * 1024):.1f} MB  "
                  f"matches best.pt to {verify(checkpoint, out):.1e}")
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the sweep
            failures.append((checkpoint.name, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)

    print(f"\nExported {len(queue) - len(failures)} of {len(queue)} run(s).")
    if failures:
        print(f"{len(failures)} failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
