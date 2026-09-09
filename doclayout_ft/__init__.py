"""Fine-tuning YOLOv11 for document-layout detection on research papers.

The package is organised by pipeline stage, and every module is runnable as a
CLI with ``python -m``:

``doclayout_ft.data``
    Turn source PDFs into annotated, split-ready YOLO datasets.
``doclayout_ft.training``
    Fetch base checkpoints and fine-tune them.
``doclayout_ft.evaluation``
    Score checkpoints against a held-out split and compare them.
``doclayout_ft.inference``
    Run a fine-tuned checkpoint over new pages.
``doclayout_ft.hub``
    Publish checkpoints to the Hugging Face Hub.

Shared paths and the class taxonomy live in :mod:`doclayout_ft.config`.
"""

__version__ = "1.0.0"
