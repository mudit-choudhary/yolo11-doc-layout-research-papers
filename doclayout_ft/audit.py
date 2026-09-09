"""Report the health of every training run directory on disk.

A sweep across twenty checkpoints does not fail loudly. Ultralytics writes a
run directory as soon as training starts, so a run that dies in its first epoch
leaves a plausible-looking folder behind containing a couple of plots and no
weights. Nothing downstream complains: checkpoint discovery simply does not
find a ``best.pt`` and moves on, and the run is quietly absent from every table
and from the publish queue.

This makes that visible. Each run directory is classified as:

``complete``
    Has ``weights/best.pt`` and every artefact the publisher looks for.
``incomplete``
    Has weights, but is missing plots. Usually means training was interrupted
    after the weights were saved but before the final validation pass that
    writes the confusion matrix and curves.
``dead``
    No weights at all. The run died early, most often on CUDA out-of-memory.
``base``
    Not a run: a directory of downloaded base checkpoints.

For dead runs it also guesses the cause from ``args.yaml``, since the two
settings that actually killed runs in this project are recoverable from it.

Exit code is 0 unless an *actionable* problem is found, meaning an incomplete
run whose plots can be regenerated. Dead runs are reported but do not fail the
check: they record a training attempt that ran out of memory, so failing on them
would fail forever regardless of what anyone does. ``--strict`` includes them.

Usage::

    python -m doclayout_ft.audit
    python -m doclayout_ft.audit --models-dir models
    python -m doclayout_ft.audit --verbose
    python -m doclayout_ft.audit --strict
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from doclayout_ft.config import FINETUNED_DIR, MODELS_DIR

#: Artefacts a finished Ultralytics run writes and the publisher uploads.
#: Absence of any of these means the run did not reach its final validation.
EXPECTED_ARTIFACTS = (
    "args.yaml",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
)

#: VRAM on the reference GPU, in GB. Used only to phrase the OOM diagnosis.
REFERENCE_VRAM_GB = 3.9


@dataclass
class RunHealth:
    """One run directory's condition.

    Attributes:
        name: Directory name.
        path: Directory path.
        status: One of ``complete``, ``incomplete``, ``dead``, ``base``.
        missing: Expected artefacts that are absent.
        epochs: Rows in ``results.csv``, or None if there is no such file.
        diagnosis: Why a dead run probably died, if it can be inferred.
        detail: Extra context for a base-checkpoint directory.
    """

    name: str
    path: Path
    status: str
    missing: list[str] = field(default_factory=list)
    epochs: int | None = None
    diagnosis: str = ""
    detail: str = ""


def _load_args(run_dir: Path) -> dict:
    """Read a run's ``args.yaml``, tolerating absence and malformed YAML."""
    path = run_dir / "args.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError:
        return {}


def _count_epochs(run_dir: Path) -> int | None:
    """Return how many epochs a run recorded, or None if it recorded none."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def diagnose_dead_run(run_dir: Path) -> str:
    """Infer why a run produced no weights.

    Two settings account for every dead run in this project, and both are
    recorded in ``args.yaml``:

    ``multi_scale``
        Ultralytics computes ``max_imgsz = imgsz * (1 + multi_scale)``, so
        ``multi_scale=0.5`` at ``imgsz=1024`` trains on images up to 1536
        pixels. Activation memory grows with area, making that 2.25 times the
        peak of a plain 1024 run, which does not fit in 4 GB.

    ``batch``
        Straightforwardly too large for the model at that resolution.

    Args:
        run_dir: A run directory with no weights.

    Returns:
        A one-line explanation, or a generic message when nothing stands out.
    """
    run_args = _load_args(run_dir)
    if not run_args:
        return "no args.yaml; cause unrecoverable"

    imgsz = run_args.get("imgsz")
    batch = run_args.get("batch")
    multi_scale = run_args.get("multi_scale") or 0.0

    try:
        multi_scale = float(multi_scale)
        imgsz = int(imgsz)
    except (TypeError, ValueError):
        return "died before the first epoch; args.yaml incomplete"

    if multi_scale >= 0.4:
        peak = int(imgsz * (1 + multi_scale))
        growth = (1 + multi_scale) ** 2
        return (f"likely CUDA OOM: multi_scale={multi_scale} scales images to "
                f"{peak}px, about {growth:.2f}x the activation memory of "
                f"imgsz={imgsz} at batch={batch}")
    if isinstance(batch, int) and batch >= 5:
        return (f"likely CUDA OOM: batch={batch} at imgsz={imgsz} exceeds "
                f"~{REFERENCE_VRAM_GB} GB for this model size")
    return "died before the first epoch; no obvious cause in args.yaml"


def inspect_run(run_dir: Path) -> RunHealth:
    """Classify one directory under a models root.

    Args:
        run_dir: The directory to inspect.

    Returns:
        Its :class:`RunHealth`.
    """
    name = run_dir.name
    has_weights = (run_dir / "weights" / "best.pt").is_file()
    loose = sorted(p.name for p in run_dir.glob("*.pt"))

    if not has_weights and loose:
        return RunHealth(name, run_dir, "base", detail=", ".join(loose))

    if not has_weights:
        return RunHealth(
            name, run_dir, "dead",
            epochs=_count_epochs(run_dir),
            diagnosis=diagnose_dead_run(run_dir),
        )

    missing = [f for f in EXPECTED_ARTIFACTS if not (run_dir / f).is_file()]
    return RunHealth(
        name, run_dir,
        "complete" if not missing else "incomplete",
        missing=missing,
        epochs=_count_epochs(run_dir),
    )


def audit(models_dirs: list[Path]) -> dict[Path, list[RunHealth]]:
    """Inspect every directory under each models root.

    Args:
        models_dirs: Roots to scan.

    Returns:
        Mapping of root to the health of each directory inside it.

    Raises:
        FileNotFoundError: If a root does not exist.
    """
    results: dict[Path, list[RunHealth]] = {}
    for root in models_dirs:
        if not root.is_dir():
            raise FileNotFoundError(f"Models directory not found: {root}")
        results[root] = [inspect_run(d) for d in sorted(root.iterdir()) if d.is_dir()]
    return results


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, nargs="+",
                        default=[FINETUNED_DIR, MODELS_DIR],
                        help="Directories to audit (default: FinetunedModels/ and models/)")
    parser.add_argument("--verbose", action="store_true",
                        help="List every run, not just the ones with problems")
    parser.add_argument("--strict", action="store_true",
                        help="Exit non-zero for dead runs too. Off by default, "
                             "because a run that ran out of memory once is a "
                             "historical fact and would fail the check forever.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 if nothing is wrong, 1 if any run has a problem."""
    args = build_parser().parse_args(argv)

    try:
        results = audit(args.models_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # Two different kinds of finding, and conflating them makes the command
    # useless as a check. An incomplete run is actionable: its plots can be
    # regenerated. A dead run is a historical fact about a training attempt
    # that ran out of memory months ago; nothing can be done about it short of
    # retraining on a bigger card, so failing on it forever is just noise.
    incomplete = 0
    dead = 0
    for root, runs in results.items():
        counts = {s: sum(1 for r in runs if r.status == s)
                  for s in ("complete", "incomplete", "dead", "base")}
        print(f"\n{root}")
        print(f"  {counts['complete']} complete, {counts['incomplete']} incomplete, "
              f"{counts['dead']} dead, {counts['base']} base-checkpoint dir(s)")

        for run in runs:
            if run.status == "complete" and not args.verbose:
                continue
            if run.status == "base":
                if args.verbose:
                    print(f"    base       {run.name}  ({run.detail or 'no .pt files'})")
                continue

            if run.status == "dead":
                dead += 1
                print(f"    dead       {run.name}")
                print(f"               no weights written. {run.diagnosis}")
            elif run.status == "incomplete":
                incomplete += 1
                print(f"    INCOMPLETE {run.name}"
                      + (f"  ({run.epochs} epochs)" if run.epochs else ""))
                print(f"               missing: {', '.join(run.missing)}")
            else:
                print(f"    ok         {run.name}"
                      + (f"  ({run.epochs} epochs)" if run.epochs else ""))

    print()
    if incomplete:
        print(f"{incomplete} run(s) need attention.")
        print("  An incomplete run has usable weights but publishes fewer files than")
        print("  its siblings. Regenerate its plots by validating the checkpoint with")
        print("  plots enabled, then note on its card that they were made after the")
        print("  fact. See docs/RUN_INVENTORY.md.")

    if dead:
        if incomplete:
            print()
        print(f"{dead} dead run(s), for information.")
        print("  These wrote no weights, so they publish nothing and appear in no")
        print("  table. Nothing to fix: they are a record of training attempts that")
        print("  ran out of memory. Retraining them needs a larger card, or a lower")
        print("  --multi-scale and --batch. Pass --strict to fail on these too.")

    if not incomplete and not dead:
        print("Every run is complete.")

    if incomplete or (dead and args.strict):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
