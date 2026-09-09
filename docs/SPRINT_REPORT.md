# YOLOv11 Document-Layout Fine-Tuning — Weekend Sprint Report

> **Historical document.** This is the write-up as it stood at the end of the
> fine-tuning sprint, kept for provenance. It records what was tried, in what
> order, and what went wrong along the way, which the reference docs
> deliberately leave out.
>
> For current numbers see [EVALUATION.md](EVALUATION.md), which re-runs the
> evaluation on a clean val-only split and supersedes the tables below. For
> how to run the pipeline see [FINETUNING_STEPS.md](FINETUNING_STEPS.md).
> Script names mentioned here predate the reorganisation into the
> `doclayout_ft` package.

**Project:** RAG_Setup — Document Layout Detection for PDF Chunking
**Window:** Friday ~8:00 PM – Sunday ~7:00 PM (~47 hours wall-clock)
**Hardware:** NVIDIA GTX 1650 (3.9 GB VRAM), Ubuntu 24.04 LTS
**Dataset:** Manually annotated real arXiv research papers, 12-class taxonomy, X-AnyLabeling

---

## 1. Objective

Fine-tune a YOLOv11 document-layout detection model to identify paragraph, caption, table, figure, and header/footer regions on scanned research-paper pages, in order to drive paragraph-aware PDF chunking for a RAG pipeline. Success is ultimately measured by chunk-boundary quality (word coverage, leakage, contamination), not just detection mAP — but this sprint focused on establishing a strong detection baseline first.

## 2. Data Pipeline

### 2.1 Source and annotation
- Source PDFs pulled from arXiv, converted to page images.
- Annotated manually in X-AnyLabeling (Dockerized), exported in YOLO HBB format.
- 12-class taxonomy: `Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture, Section-header, Table, Text, Title, Authors` — the base DocLayNet-style 11 classes plus a custom `Authors` class.

### 2.2 PDF → image conversion
- `pdf2image` at **300 DPI**, JPEG output — standard for OCR/annotation-grade text legibility.
- Rate-limited (7s pause every 25 PDFs) to avoid overloading the conversion process on batch runs.

### 2.3 Batching for annotation
- Images pulled into per-batch folders (e.g. `batch_05`) via random sampling, deliberately over-representing page 1 (up to 200 page-1 images per batch vs. ~650 random interior pages) — likely to ensure adequate coverage of `Title`/`Authors`/`Section-header`, which concentrate on first pages.

### 2.4 Train/val/test split
- Split **by paper, not by page** (`split_dataset.py`), keyed on filename with the trailing `_page_NN` suffix stripped. This prevents sibling pages of the same paper — which share near-identical layout style — from leaking across splits and inflating validation metrics.
- Default ratios: 75% train / 15% val / 10% test.
- This was a mid-sprint correction: earlier rounds (`round_03`) used a flat, unsplit image pool where `train:` and `val:` pointed at the same folder in `data.yaml`, invalidating all validation numbers from that period. `round_final` (632 train / 130 val) is the first dataset with a methodologically sound split.

### 2.5 Dataset growth across rounds

| Round | Approx. pages | Split integrity |
|---|---|---|
| round_01 | ~150 | Unknown/undocumented (Jan finetune) |
| round_02 | ~450 | Unknown/undocumented (Jan finetune) |
| round_03 | 850 | **Invalid** — train/val pointed to same folder |
| round_final | 632 train / 130 val (762 total) | **Valid** — paper-level split |

## 3. Training Infrastructure

- Ultralytics YOLOv11, PyTorch `cu118` (driver pinned to 535.x / CUDA 12.2 due to external-monitor compatibility constraints — later builds compiled against CUDA 13.0 failed to initialize against this driver and had to be downgraded).
- **AMP (mixed precision) is disabled for every run** — the GTX 1650 (Turing, non-RTX) fails Ultralytics' AMP sanity check and silently falls back to FP32 to avoid NaN losses, which is why VRAM headroom is tighter than the card's spec would suggest.
- Fine-tuning driven by `finetune_all_models.py`: auto-discovers all checkpoints under `models/`, launches one training run per checkpoint, skips runs whose output directory already exists (suffix-based namespacing), and points every run at a single shared `data.yaml`.

### Operational issues encountered mid-sprint
- **CUDA/PyTorch driver mismatch**: an environment update pulled in `torch+cu130`, incompatible with the 12.2 driver; resolved by reinstalling `cu118`.
- **exFAT filesystem corruption on the external SSD** (`DarkDwine`): an unclean prior unmount left the volume with an unflushed dirty bit; a subsequent burst of directory writes (moving/deleting annotation files during a cleanup pass) hit the corrupted region and the kernel force-remounted the drive read-only mid-annotation-session. Diagnosed via `dmesg`, repaired with `fsck.exfat -y` (`exfatprogs`) after a precautionary full backup. All 27 flagged entries were either already-empty quarantine folders or regenerable `venv`/`.git` artifacts — no annotation data was lost.
- **CUDA OOM tuning**: batch size and resolution were iteratively tuned against the 3.9 GB VRAM ceiling; `yolo11m` proved impractical on this hardware even at batch=1 (see §5.4).

## 4. Experiments Conducted

### Phase 1 — Baseline (640px, default augmentation)
Initial fine-tune of `yolo11n` and several prior checkpoints (`v2`, `v22`, `v222`, `v2224`, `v2224_round03`, `v222_round03`) at the Ultralytics default `imgsz=640`.

### Phase 2 — Resolution increase (1024px)
Re-ran the same checkpoint lineage at `imgsz=1024, batch=4` (GTX 1650 memory ceiling), hypothesizing that small classes (`Footnote`, `Page-header`, `Page-footer`) were under-resolved at 640px. Also re-ran on the corrected `round_final` split.

### Phase 3 — Model size sweep
- `yolo11n` (2.6M params) — fits comfortably at batch=4, 1024px.
- `yolo11s` (9.4M params) — fits at batch=3, 1024px (~3.5–3.6 GB).
- `yolo11m` (20M params) — **could not be trained reliably**: OOM'd at batch=5 and batch=2, only completed a forward pass at batch=1, which risks unstable BatchNorm statistics (a single-sample batch produces unreliable mean/variance estimates) independent of gradient-accumulation settings. Abandoned as infeasible on this GPU.

### Phase 4 — Augmentation tuning (`attempt_02`)
Applied `fliplr=0.0` (disable horizontal flip — a mirrored document/reading order never occurs at inference), `copy_paste=0.2–0.3`, `multi_scale=0.23–0.5`, `erasing=0.0`, extended `epochs=210 / patience=60`.

## 5. Results

### 5.1 Full comparison (validation, `round_final`)

| Model | imgsz | Precision | Recall | mAP50 | **mAP50-95** |
|---|---|---|---|---|---|
| yolo11_doc_layout_v2224_imgsz_1024 | 1024 | 0.878 | 0.888 | 0.919 | **0.7535** |
| **yolo11s_doc_layout_imgsz_1024** | 1024 | 0.923 | 0.912 | 0.933 | **0.7530** |
| yolo11_doc_layout_v222_round03_imgsz_1024 | 1024 | 0.889 | 0.904 | 0.925 | 0.7480 |
| yolo11n_doc_layout_imgsz_1024 | 1024 | 0.879 | 0.879 | 0.925 | 0.7490 |
| yolo11_doc_layout_v2_imgsz_1024 | 1024 | 0.908 | 0.886 | 0.922 | 0.7488 |
| yolo11_doc_layout_v222_imgsz_1024 | 1024 | 0.894 | 0.863 | 0.915 | 0.7471 |
| yolo11_doc_layout_v2224_round03 | 640 | 0.863 | 0.873 | 0.899 | 0.7230 |
| yolo11_doc_layout_v222_round03 | 640 | 0.868 | 0.845 | 0.898 | 0.7208 |
| yolo11s_doc_layout_attempt_02 | 1024 | 0.874 | 0.876 | 0.915 | 0.7274 ↓ |
| yolo11s_doc_layout_imgsz_1024_attempt_02 | 1024 | 0.882 | 0.885 | 0.906 | 0.7253 ↓ |
| yolo11_doc_layout_v222_imgsz_1024_attempt_02 | 1024 | 0.860 | 0.851 | 0.888 | 0.7065 ↓ |
| yolo11_doc_layout_v222_round03_attempt_02 | 1024 | 0.844 | 0.829 | 0.874 | 0.7007 ↓ |
| yolo11_doc_layout_v222_round03_imgsz_1024_attempt_02 | 1024 | 0.858 | 0.826 | 0.862 | 0.6912 ↓ |
| yolo11_doc_layout_v2224 | 640 | 0.824 | 0.808 | 0.851 | 0.6521 |
| yolo11_doc_layout_v22 / v222 | 640 | 0.712 | 0.712 | 0.732 | 0.5331 |
| yolo11m_doc_layout_baseline (unfinetuned) | 640 | 0.718 | 0.565 | — | 0.4546 |
| yolo11n_doc_layout_baseline (unfinetuned) | 640 | 0.683 | 0.552 | — | 0.4227 |
| yolo11s_doc_layout_baseline (unfinetuned) | 640 | 0.637 | 0.577 | — | 0.4200 |

### 5.2 Best model — per-class breakdown (`yolo11s_doc_layout_imgsz_1024`, val-only)

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Table | 1.000 | 0.928 | 0.995 | **0.985** |
| List-item | 0.946 | 0.987 | 0.991 | 0.926 |
| Text | 0.962 | 0.948 | 0.984 | 0.919 |
| Title | 0.987 | 0.966 | 0.989 | 0.887 |
| Authors | 0.974 | 0.906 | 0.959 | 0.809 |
| Picture | 0.880 | 0.871 | 0.896 | 0.797 |
| Formula | 0.892 | 0.960 | 0.978 | 0.772 |
| Caption | 0.947 | 0.872 | 0.929 | 0.762 |
| Footnote | 0.885 | 0.797 | 0.891 | 0.720 |
| Page-header | 0.928 | 0.954 | 0.960 | 0.695 |
| Section-header | 0.898 | 0.975 | 0.944 | 0.642 |
| **Page-footer** | 0.929 | 0.908 | 0.861 | **0.361** |
| **Overall** | **0.936** | **0.923** | **0.948** | **0.773** |

*Note: this val-only figure (0.773) differs from the "combined split" evaluation script's number for the same checkpoint (0.753, §5.1) because the combined-split script evaluates against val+test together rather than val alone — a methodology difference worth reconciling before quoting a single headline number externally.*

## 6. Key Findings

1. **Resolution (640→1024px) was the single largest lever**, adding roughly +0.03 mAP50-95 across the board and disproportionately helping small-object classes like `Footnote` and `Authors`, consistent with the hypothesis that small text regions are under-resolved at 640px.
2. **The paper-level train/val split correction was necessary, not optional.** Numbers from `round_03` (flat split) cannot be meaningfully compared to `round_final` (paper-level split) — the former likely overstated true generalization.
3. **`yolo11s` gave a small, real edge over `yolo11n`** on the classes that matter most for chunking (`Footnote`: 0.720 vs 0.611; `Table`: 0.985 vs 0.956), at acceptable cost (batch=3 vs batch=4, ~3.6h vs ~40min training time). `yolo11s_doc_layout_imgsz_1024` is the strongest model produced this sprint.
4. **`yolo11m` is not practically trainable on the GTX 1650** at 1024px — OOMs even at batch=2, and batch=1 risks unreliable BatchNorm statistics. Would need a larger-VRAM GPU (cloud/Colab) to evaluate fairly.
5. **`Page-footer` is a persistent, resolution- and architecture-independent weak point** (mAP50-95 stuck at 0.36–0.39 across every model/config tested) despite strong precision/recall — this pattern (good at *finding* the region, poor at *bounding* it tightly) points to inconsistent annotation tightness in the `Page-footer` labels rather than a model capacity problem. Deprioritized per current project scope, but worth a manual annotation audit if revisited.
6. **The `attempt_02` augmentation combination (`copy_paste`, `multi_scale`) regressed every model it was applied to**, by 0.02–0.06 mAP50-95, consistently. The most likely cause: Ultralytics' default `copy_paste_mode=flip` mirrors individual pasted instance crops even when whole-image `fliplr` is disabled, silently reintroducing the mirrored-text problem at the instance level. Additionally, `copy_paste` composites instances from unrelated images into structurally implausible positions — a reasonable augmentation for natural-scene object detection (its original COCO use case), but a poor fit for document layouts, where spatial/structural coherence between regions carries real signal. **Recommendation: do not use `copy_paste` for this task**; rare-class exposure should instead come from more annotated real examples.

## 7. Recommended Production Model

**`yolo11s_doc_layout_imgsz_1024`** — mAP50-95 = 0.753 (combined split) / 0.773 (val-only), trained at `imgsz=1024, batch=3`, default augmentation apart from disabled AMP (hardware-forced). Strongest result on the two classes most likely to affect chunk-boundary quality (`Footnote`, `Table`), and the only `s`-class run not compromised by the augmentation regression.

## 8. Next Steps

1. Reconcile the val-only vs. combined-split evaluation discrepancy before treating either number as the canonical headline metric.
2. Move from box-level metrics (mAP) to **task-level chunking metrics** — word coverage, word leakage, orphan runs, boundary offset, contamination rate — computed directly against PyMuPDF ground-truth word positions, since this is what actually determines chunking quality and box metrics have plateaued around 0.75.
3. Optional: manual annotation audit of `Page-footer` boxes for tightness consistency, if this class is reprioritized later.
4. Optional: evaluate `yolo11m` on a higher-VRAM GPU (cloud/Colab) as a genuine capacity test, since it was never fairly assessed on local hardware.
5. Runtime end-to-end verification of the full RAG pipeline (YOLO inference + CUDA embedding path) with `YOLO_MODEL_PATH` pointed at `yolo11s_doc_layout_imgsz_1024/weights/best.pt`.
