# Experiment log: the attempt_02 sweep

Recovered from each run's own `args.yaml`, which is authoritative: it records
what Ultralytics was actually launched with, not what anyone remembers
intending.

Kept because the conclusion is easy to overstate. The sweep regressed
everything, so it is tempting to blame the setting that looks most suspicious.
The configs show seven settings changed together, which does not support that.

## What changed

Against the checkpoint each run was fine-tuned from, for the three pairs where
image size did **not** also change:

| Setting | Before | After | Expected effect |
|---|---|---|---|
| `copy_paste` | 0.0 | 0.2 – 0.3 | Suspect |
| `multi_scale` | 0.0 | 0.23 – 0.5 | Suspect |
| `fliplr` | 0.5 | 0.0 | Should **help** a document task |
| `erasing` | 0.4 | 0.0 | Should **help** rare classes |
| `epochs` | 100 – 150 | 210 | Neutral; early stopping governs |
| `patience` | 45 | 60 | Neutral |
| `batch` | 3 – 5 | 2 – 3 | Weak confound; see below |

Unchanged throughout: `lr0: 0.01`, `lrf: 0.01`, `warmup_epochs: 3.0`,
`optimizer: auto`, `nbs: 64`, `mosaic: 1.0`, `mixup: 0.0`, `scale: 0.5`,
`translate: 0.1`, `degrees: 0.0`, `flipud: 0.0`, and the HSV jitter.

Four older runs (`v22`, `v222`, `v2224`, and their 640-pixel siblings) also
moved from `imgsz: 640` to `1024`, which confounds them further. Their diffs
additionally show keys such as `cls_pw`, `cls_remap`, `dgrad` and a changed
`tracker` default. Those are Ultralytics version drift between when the parent
and the child were trained, not deliberate choices.

## Results

| Starting checkpoint | Base | With attempt_02 | Delta |
|---|---|---|---|
| yolo11s_doc_layout_imgsz_1024 | 0.7694 | 0.7533 | -0.0161 |
| yolo11_doc_layout_v222_round03 | 0.7442 | 0.7138 | -0.0304 |
| yolo11_doc_layout_v222_imgsz_1024 | 0.7623 | 0.7108 | -0.0515 |
| yolo11_doc_layout_v222_round03_imgsz_1024 | 0.7666 | 0.7070 | -0.0596 |

mAP50-95 on the held-out `val` split of `round_final`.

## Ruling things out

**Batch size is a weak confound.** Ultralytics accumulates gradients to a
nominal batch of 64 (`nbs: 64`) whatever the micro-batch, so the effective
optimisation batch was unchanged and the fixed `lr0: 0.01` stayed appropriate.
Batch-norm statistics do come from the micro-batch, so 2 is noisier than 5, but
that is second-order next to an augmentation change of this size.

**Training length is not the explanation.** Two of the three cleanest
`attempt_02` runs early-stopped *sooner* than their parents despite a higher cap
and higher patience:

| Run | Parent epochs | attempt_02 epochs |
|---|---|---|
| yolo11_doc_layout_v222_imgsz_1024 | 93 | 73 |
| yolo11_doc_layout_v222_round03_imgsz_1024 | 101 | 61 |
| yolo11s_doc_layout_imgsz_1024 | 100 | 189 |

They plateaued earlier and lower. The third trained nearly twice as long and
still finished worse.

**Two changes should have helped.** `fliplr` 0.5 to 0.0 and `erasing` 0.4 to
0.0 are the settings usually recommended for document layout. If that reasoning
is right, the damage done by `copy_paste` and `multi_scale` is *larger* than the
measured delta, since it had to overcome two improvements to land where it did.
If the reasoning is wrong, those two changes are themselves part of the cause.
The experiment cannot distinguish these.

## What can be concluded

**`copy_paste` and `multi_scale`, applied together at these strengths, cost
0.016 to 0.060 mAP50-95 on this task.** Which of the two matters, or whether it
takes both, was not isolated.

The mechanistic case against `copy_paste` stands independently of this
experiment. The runs recorded `copy_paste_mode: flip`, which mirrors each pasted
instance crop *even when whole-image `fliplr` is 0*. These runs reintroduced
mirrored text at the instance level while believing mirroring was off.
Copy-paste also composites regions from unrelated pages into positions that
cannot occur, and in a document, unlike a natural scene, the spatial
relationships between regions carry real signal.

That is a reason for suspicion, not a measurement.

## What is still open

Four runs, changing one thing each against `yolo11s_doc_layout_imgsz_1024`,
would settle it:

1. `--copy-paste 0.3` alone.
2. `--multi-scale 0.25` alone.
3. `--fliplr 0.0` alone. Currently believed helpful and never verified.
4. `--erasing 0.0` alone. Same.

Runs 3 and 4 matter beyond this question, because the argument for them is
strong enough that people will assume it. Until they are run, the fine-tuning
defaults match the best measured run (`fliplr=0.5`, `erasing=0.4`) rather than
the better argument.

## Reproducing this analysis

```bash
python -m doclayout_ft.evaluation.evaluate --models-dir FinetunedModels models --split val
```

Then compare any run against its parent by reading the `model:` key of its
`args.yaml`, which names the checkpoint it started from.
