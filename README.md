# Document Layout Detection for Research Papers

Fine-tuning YOLOv11 to find the structural regions of a research-paper page:
titles, authors, paragraphs, captions, tables, figures, formulas, footnotes and
page furniture. Twelve classes in all.

The point is not detection for its own sake. It is chunking. Splitting a PDF
for retrieval on a fixed character count cuts through the middle of tables and
strands captions away from their figures. A model that knows where a paragraph
ends lets the split happen on a real boundary instead.

Everything here runs on a single 4 GB GPU.

| | |
|---|---|
| Recommended model | `yolo11s_doc_layout_imgsz_1024` |
| mAP50-95 | 0.769 on held-out validation |
| Base checkpoint | [`Armaggheddon/yolo11-document-layout`](https://huggingface.co/Armaggheddon/yolo11-document-layout) |
| Training data | 850 hand-annotated pages from 566 arXiv papers |
| Unlabelled pool | ~22,000 rendered pages from 1128 PDFs, not used for training |
| Hardware | GTX 1650, 4 GB VRAM |

One checkpoint, `yolo11_doc_layout_v2224_imgsz_1024`, scores about 0.002 higher.
That is within noise. It is recommended against because it is three fine-tuning
passes deep over datasets that no longer exist in their original form, so it
cannot be reproduced, whereas `yolo11s_doc_layout_imgsz_1024` is a single
fine-tune from a published base checkpoint and leads on the classes that decide
chunk boundaries.

Only a labelled subset of the corpus is used. The 1128 source PDFs render to
roughly 22,000 page images; 850 of those pages, across 566 papers, are
annotated and make up the training, validation and test splits. The rest of the
pool is unlabelled and exists so later annotation rounds have material to draw
on.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch first, matched to your CUDA driver. See docs/SETUP.md.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -e .
sudo apt install poppler-utils   # PDF rendering
```

Run a model over a paper:

```bash
python -m doclayout_ft.inference.predict --source paper.pdf --out-dir predictions/
```

That writes one JSON of detected regions and one annotated preview per page.

## What is in here

| Path | Contents |
|---|---|
| `doclayout_ft/` | The pipeline. Every module runs as `python -m`. |
| `docs/` | Setup, dataset, fine-tuning, evaluation and publishing guides. |
| `tests/` | Test suite. No GPU or dataset required. |
| `reports/` | Evaluation tables and charts. |
| `training_dataset/` | Dataset rounds. Configs tracked, images and labels not. |
| `PDFs/`, `images/` | Source papers and the full rendered page pool. Untracked, 24 GB. |
| `models/`, `FinetunedModels/` | Training runs and weights. Untracked. |

## The pipeline

Each stage is one command. Full detail in
[docs/FINETUNING_STEPS.md](docs/FINETUNING_STEPS.md).

```bash
# 1. Render source PDFs to 300 DPI page images
python -m doclayout_ft.data.pdf_to_images

# 2. Carve off a batch to annotate, over-sampling page 1
python -m doclayout_ft.data.make_batches --batch-name batch_06

#    ... annotate in X-AnyLabeling, export YOLO format, drop into a round ...

# 3. Split by paper into train / val / test
python -m doclayout_ft.data.split_dataset --round round_final

# 4. Fetch base checkpoints
python -m doclayout_ft.training.download_base_models --variants n s

# 5. Fine-tune
python -m doclayout_ft.training.finetune --only yolo11s_doc_layout

# 6. Check which runs actually finished (exits 0 unless something is fixable)
python -m doclayout_ft.audit

# 7. Score every run and compare
python -m doclayout_ft.evaluation.evaluate --split val

# 8. Break the best one down by class
python -m doclayout_ft.evaluation.per_class

# 9. Publish to one Hugging Face repo, two variants at a time
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

Every command takes `--help`, and most take `--dry-run` or `--list` so you can
see what would happen before it does.

## Classes

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | Caption | 6 | Picture |
| 1 | Footnote | 7 | Section-header |
| 2 | Formula | 8 | Table |
| 3 | List-item | 9 | Text |
| 4 | Page-footer | 10 | Title |
| 5 | Page-header | 11 | Authors |

The first eleven come from the DocLayNet-style base model. `Authors` is custom,
added because research-paper front matter has no equivalent in that taxonomy
and it matters for citation handling.

## What the experiments found

Full write-up in [docs/EVALUATION.md](docs/EVALUATION.md) and the sprint report
in [docs/SPRINT_REPORT.md](docs/SPRINT_REPORT.md).

- **Resolution was the largest single lever.** Moving from 640 to 1024 pixels
  added roughly 0.03 mAP50-95 across every checkpoint, and helped small classes
  such as `Footnote` and `Page-header` most. Those regions are only a few
  pixels tall once a page is scaled to 640.
- **Splitting by paper rather than by page was a correction, not a refinement.**
  Pages of one paper share a template, so a page-level split validates the
  model on layouts it trained on. Every metric from before that fix is void.
- **Older lineages overlap the held-out set but gain nothing from it.** Fifteen
  of nineteen runs descend from a round that saw up to 19% of today's validation
  papers. Tested directly: clean-lineage models show the same gap on those
  papers, so they are simply easier pages, not memorised ones.
- **`yolo11s` beat `yolo11n` slightly**, on the classes that matter most for
  chunking, at roughly five times the training time.
- **`yolo11m` could not be trained on this hardware.** It runs out of memory
  below batch size 2, and batch size 1 makes batch-norm statistics unreliable.
- **`multi_scale` is a memory multiplier, not a tweak.** At 0.5 it trains on
  images 1.5x wider, which is 2.25x the activation memory. That killed four runs
  outright on a 4 GB card, silently, before any weights were written.
- **The `attempt_02` augmentation sweep regressed every model it touched**, by
  0.016 to 0.060 mAP50-95. It changed seven settings at once, so it condemns the
  bundle rather than any one setting. `copy_paste` is the prime suspect on
  mechanistic grounds: Ultralytics mirrors each pasted crop even when
  whole-image flipping is off, and compositing regions between unrelated pages
  produces layouts that cannot occur.
- **The best models used Ultralytics' stock flipping and erasing.** Turning
  those off is well argued for documents, but it was only ever tried inside the
  failed bundle, so it remains untested here. The fine-tuning defaults match the
  best run rather than the better argument.
- **`Page-footer` is a labelling problem, not a model problem.** It is found
  reliably and bounded loosely, at 0.36 mAP50-95 against 0.75 overall,
  identically across every size and resolution tried.

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests -q
```

The tests build synthetic datasets and runs in temporary directories, so they
need neither a GPU nor the real data.

## License

Copyright (C) 2026 Mudit Choudhary.

Licensed under **AGPL-3.0**, inherited from
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which these
models were trained with. The full text is in [LICENSE](LICENSE).

Note the AGPL network clause: serving these models to users over a network
obliges you to offer them the corresponding source. Ultralytics sells an
[Enterprise License](https://www.ultralytics.com/license) for use without that
obligation.

See [docs/LICENSING.md](docs/LICENSING.md) for the full upstream chain and the
caveats, including that the base checkpoint declares MIT.
