"""Shared paths, class taxonomy and small helpers used across the pipeline.

Every path here is derived from the repository root, which is located relative
to this file. Nothing in the package hardcodes an absolute path, so the repo
can be moved or cloned to a different machine without editing code. The only
absolute paths that survive are inside the generated dataset artefacts
(``data.yaml`` and the split list files), which are rewritten in place by
:mod:`doclayout_ft.data.split_dataset` whenever it runs.

Large directories (``PDFs/``, ``images/``, ``training_dataset/*/images``,
``models/``, ``FinetunedModels/``) are deliberately untracked by git; see
``docs/DATASET.md`` for how to rebuild them.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------

#: Repository root: two levels up from this file (doclayout_ft/config.py).
ROOT = Path(__file__).resolve().parent.parent

#: Source research papers, one PDF per paper. Untracked.
PDF_DIR = ROOT / "PDFs"

#: Page images rendered from :data:`PDF_DIR`, one JPEG per page, plus the
#: ``batch_NN/`` subfolders carved out of it for annotation. Untracked.
IMAGE_DIR = ROOT / "images"

#: Annotated dataset rounds. Each round is a directory holding ``images/``,
#: ``labels/``, ``classes.txt`` and ``data.yaml``.
TRAINING_DIR = ROOT / "training_dataset"

#: Working directory for training runs. Ultralytics writes one subdirectory
#: per run here.
MODELS_DIR = ROOT / "models"

#: Curated collection of finished runs worth keeping and publishing.
FINETUNED_DIR = ROOT / "FinetunedModels"

#: Evaluation tables and charts.
REPORTS_DIR = ROOT / "reports"

#: The dataset round that all current training and evaluation targets. Earlier
#: rounds are kept for provenance only; see docs/DATASET.md for why round_03
#: and earlier cannot be compared against this one.
DEFAULT_ROUND = "round_final"


# --------------------------------------------------------------------------
# Class taxonomy
# --------------------------------------------------------------------------

#: The 12-class taxonomy, ordered by class id. The first 11 are the
#: DocLayNet-style classes carried over from the base checkpoints; ``Authors``
#: is a custom 12th class added for research-paper front matter.
#:
#: The order is load-bearing: it must match ``classes.txt`` and the ``names:``
#: block of every ``data.yaml``, because YOLO label files reference classes by
#: integer index alone.
CLASS_NAMES: tuple[str, ...] = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
    "Authors",
)

#: Class id -> class name, in the form Ultralytics expects in ``data.yaml``.
CLASS_ID_TO_NAME: dict[int, str] = dict(enumerate(CLASS_NAMES))

#: Image file extensions treated as page images throughout the pipeline.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


# --------------------------------------------------------------------------
# Hardware defaults
# --------------------------------------------------------------------------
#
# These reflect the GTX 1650 (4 GB VRAM) this project was developed on. They
# are argparse defaults, not constraints -- override them on the command line
# for a larger card. docs/FINETUNING_STEPS.md explains how each was chosen.

#: Training resolution. Raising this from the Ultralytics default of 640 was
#: the single largest accuracy gain in the project (+0.03 mAP50-95), because
#: small classes such as Footnote and Page-header are under-resolved at 640.
DEFAULT_IMGSZ = 1024

#: Batch size that fits alongside imgsz=1024 in 4 GB of VRAM for yolo11n/s.
DEFAULT_BATCH = 2

#: Resolution the un-fine-tuned base checkpoints were themselves trained at.
#: Evaluating a baseline at 1024 would understate it, so it is scored at 640.
BASELINE_IMGSZ = 640


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def round_dir(round_name: str = DEFAULT_ROUND) -> Path:
    """Return the directory for one dataset round.

    Args:
        round_name: Directory name under ``training_dataset/``, e.g.
            ``"round_final"``.

    Returns:
        Absolute path to the round directory.

    Raises:
        FileNotFoundError: If the round directory does not exist. The error
            lists the rounds that do, since a typo'd round name is by far the
            most common cause.
    """
    path = TRAINING_DIR / round_name
    if not path.is_dir():
        available = (
            sorted(p.name for p in TRAINING_DIR.iterdir() if p.is_dir())
            if TRAINING_DIR.is_dir()
            else []
        )
        raise FileNotFoundError(
            f"No dataset round at {path}. "
            f"Available rounds: {', '.join(available) if available else '(none)'}"
        )
    return path


def data_yaml(round_name: str = DEFAULT_ROUND, combined: bool = False) -> Path:
    """Return the Ultralytics dataset config for one round.

    Args:
        round_name: Directory name under ``training_dataset/``.
        combined: If True, return ``data_combined.yaml``, whose ``val:`` key
            points at val+test merged. Used to score a model on every held-out
            page at once; see docs/EVALUATION.md for why that number differs
            from the val-only one.

    Returns:
        Absolute path to the YAML file.

    Raises:
        FileNotFoundError: If the round or the YAML file is missing.
    """
    name = "data_combined.yaml" if combined else "data.yaml"
    path = round_dir(round_name) / name
    if not path.is_file():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    return path


def resolve_device(device: str | int) -> str | int:
    """Normalise a ``--device`` argument for Ultralytics.

    Accepts ``"cpu"``, a CUDA index as ``int`` or numeric string, or a
    comma-separated multi-GPU list. Numeric strings are converted to ``int``
    because Ultralytics treats ``"0"`` and ``0`` differently in some code
    paths.

    Args:
        device: Raw value from the command line.

    Returns:
        Either the string ``"cpu"``, a device list string, or an ``int`` index.
    """
    if isinstance(device, int):
        return device
    device = device.strip()
    if device.isdigit():
        return int(device)
    return device


def is_truthy_env(name: str) -> bool:
    """Return True if environment variable ``name`` is set to a truthy value.

    Treats ``1``, ``true``, ``yes`` and ``on`` (case-insensitive) as true.
    """
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
