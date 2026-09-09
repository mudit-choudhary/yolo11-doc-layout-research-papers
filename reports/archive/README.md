# Archived evaluation results

Evaluation output from the original fine-tuning sprint, kept for provenance.
Superseded by the tables in `reports/`, which were regenerated after the
pipeline was reorganised.

| File | What it is |
|---|---|
| `2026-09_sprint_evaluation_val.csv` / `.png` | Per-model scores, val split |
| `2026-09_sprint_evaluation_combined.csv` / `.png` | Per-model scores, val + test merged |

The combined-split table was the sprint's headline. It reports lower numbers
than the val-only table for the same checkpoints, because merging val and test
scores a model against more pages including ones it was never selected on. See
[docs/EVALUATION.md](../../docs/EVALUATION.md) on why the split must be named
alongside any figure.

The original filenames were `evaluation_old_models.*` and
`evaluation_combined.*`; the "old_models" name was a hardcoded suffix in the
script rather than a description of its contents.
