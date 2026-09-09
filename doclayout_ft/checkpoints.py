"""Discovering training runs and checkpoints on disk.

Fine-tuning, evaluation and publishing all need to answer the same question:
what models are in this directory, and which weight file represents each one.
That logic lives here once rather than drifting between three scripts.

Two directory shapes count as a model:

``<dir>/weights/best.pt``
    A finished Ultralytics training run. ``args.yaml`` alongside it records
    the hyperparameters it was trained with, including the image size a later
    evaluation should use.

``<dir>/*.pt``
    Loose checkpoints downloaded straight from the Hub and never fine-tuned.
    These are the baselines an experiment is measured against.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from doclayout_ft.config import BASELINE_IMGSZ


@dataclass(frozen=True)
class Checkpoint:
    """One model on disk, ready to train from, evaluate, or publish.

    Attributes:
        name: Run name, used for output directories and Hub repo names.
        weights: Path to the ``.pt`` file.
        imgsz: Image size this checkpoint was trained at. Evaluating at a
            different size than a model was trained at costs real accuracy, so
            this is carried alongside the weights rather than assumed.
        is_baseline: True for a checkpoint that has never been fine-tuned here.
        run_dir: The directory the checkpoint belongs to.
    """

    name: str
    weights: Path
    imgsz: int
    is_baseline: bool
    run_dir: Path

    @property
    def modified_at(self) -> datetime:
        """When the weights were last written, as an aware UTC datetime.

        Used to order runs oldest-first when publishing a backlog in batches.
        """
        return datetime.fromtimestamp(self.weights.stat().st_mtime, tz=timezone.utc)

    @property
    def args_yaml(self) -> Path:
        """Path to the run's ``args.yaml``, whether or not it exists."""
        return self.run_dir / "args.yaml"

    @property
    def results_csv(self) -> Path:
        """Path to the run's per-epoch ``results.csv``, whether or not it exists."""
        return self.run_dir / "results.csv"


def read_trained_imgsz(run_dir: Path, default: int = BASELINE_IMGSZ) -> int:
    """Read the image size a run was trained at from its ``args.yaml``.

    Args:
        run_dir: A training run directory.
        default: Value to return when ``args.yaml`` is absent or unreadable.

    Returns:
        The recorded ``imgsz``, or ``default``. A malformed ``args.yaml`` is
        treated as absent rather than raised, because one unreadable run
        should not stop a sweep across twenty of them.
    """
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        return default
    try:
        parsed = yaml.safe_load(args_yaml.read_text()) or {}
    except yaml.YAMLError:
        return default
    try:
        return int(parsed.get("imgsz", default))
    except (TypeError, ValueError):
        return default


def discover(
    models_dir: Path,
    exclude_suffix: str | None = None,
    include_baselines: bool = True,
    baseline_suffix: str = "",
) -> dict[str, Checkpoint]:
    """Find every model under ``models_dir``.

    Args:
        models_dir: Directory holding one subdirectory per run.
        exclude_suffix: Skip run directories whose name ends with
            ``_<exclude_suffix>``. Fine-tuning passes its own output suffix
            here: without it, a re-run after a partial failure would treat the
            previous run's outputs as fresh starting points and fine-tune them
            a second time under a doubled name.
        include_baselines: Whether loose, never-fine-tuned ``.pt`` files count.
        baseline_suffix: Appended to baseline names, so a baseline and its
            fine-tuned descendant do not collide in the same table.

    Returns:
        Mapping of run name to :class:`Checkpoint`, sorted by name.

    Raises:
        FileNotFoundError: If ``models_dir`` does not exist.
    """
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    found: dict[str, Checkpoint] = {}

    for run_dir in sorted(models_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if exclude_suffix and run_dir.name.endswith(f"_{exclude_suffix}"):
            continue

        best = run_dir / "weights" / "best.pt"
        if best.is_file():
            found[run_dir.name] = Checkpoint(
                name=run_dir.name,
                weights=best,
                imgsz=read_trained_imgsz(run_dir),
                is_baseline=False,
                run_dir=run_dir,
            )
            # A directory is either a finished run or a bag of base weights,
            # never both. Once best.pt is found, loose .pt files in the same
            # directory are training inputs, not separate models.
            continue

        if not include_baselines:
            continue

        for pt_file in sorted(run_dir.glob("*.pt")):
            found[f"{pt_file.stem}{baseline_suffix}"] = Checkpoint(
                name=f"{pt_file.stem}{baseline_suffix}",
                weights=pt_file,
                imgsz=BASELINE_IMGSZ,
                is_baseline=True,
                run_dir=run_dir,
            )

    return found


def discover_many(
    models_dirs: list[Path],
    exclude_suffix: str | None = None,
    include_baselines: bool = True,
    baseline_suffix: str = "",
) -> dict[str, Checkpoint]:
    """Find every model across several directories.

    The same run name can exist in more than one directory: ``models/`` holds
    working training output and ``FinetunedModels/`` holds a curated copy, and
    seven runs currently appear in both. The first directory listed wins, so
    order the arguments most-authoritative first.

    Args:
        models_dirs: Directories to scan, in priority order.
        exclude_suffix: Passed through to :func:`discover`.
        include_baselines: Passed through to :func:`discover`.
        baseline_suffix: Passed through to :func:`discover`.

    Returns:
        Mapping of run name to :class:`Checkpoint`, sorted by name.

    Raises:
        FileNotFoundError: If any listed directory does not exist.
    """
    merged: dict[str, Checkpoint] = {}
    for models_dir in models_dirs:
        for name, checkpoint in discover(
            models_dir,
            exclude_suffix=exclude_suffix,
            include_baselines=include_baselines,
            baseline_suffix=baseline_suffix,
        ).items():
            merged.setdefault(name, checkpoint)
    return dict(sorted(merged.items()))


def filter_by_name(
    checkpoints: dict[str, Checkpoint],
    only: list[str] | None,
) -> dict[str, Checkpoint]:
    """Restrict a discovery result to an explicit list of names.

    Args:
        checkpoints: Result of :func:`discover`.
        only: Names to keep, or None to keep everything.

    Returns:
        The filtered mapping.

    Raises:
        KeyError: If a requested name was not discovered. Failing loudly beats
            silently evaluating nothing because of a typo'd run name.
    """
    if not only:
        return checkpoints
    unknown = [name for name in only if name not in checkpoints]
    if unknown:
        raise KeyError(
            f"Unknown checkpoint name(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(checkpoints)) or '(none)'}"
        )
    return {name: checkpoints[name] for name in only}
