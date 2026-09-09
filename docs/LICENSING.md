# Licensing

Not legal advice. This lays out what each upstream component requires so you
can make an informed decision, or take one to someone qualified.

## Where the LICENSE file came from

**I added it, unprompted.** During the reorganisation I downloaded the
AGPL-3.0 text from `gnu.org` and committed it as `LICENSE`, because Ultralytics
is AGPL-3.0 and that propagates. You did not ask for it and did not choose it.

If you disagree with the reasoning below, delete it. Nothing in the code depends
on that file existing.

## What AGPL-3.0 requires, in short

The GNU Affero General Public License v3 is a strong copyleft licence. Its
practical effect:

- **You may** use, modify, and distribute the work, including commercially.
- **If you distribute it**, you must ship the complete corresponding source
  under AGPL-3.0, and keep the copyright and licence notices.
- **Anyone you give it to gets the same rights**, which you cannot restrict.
- **No warranty.**

The clause that makes AGPL different from plain GPL is **section 13, the
network clause**. Under GPL, running modified software on a server for users to
interact with over a network is not "distribution", so no source needs to be
released. AGPL closes that: if users interact with your modified version over a
network, you must offer them the corresponding source.

For this project, that means a hosted API that runs these weights would trigger
the source-offer obligation, where the same service built on an MIT-licensed
model would not.

## Do you actually need it?

**Ultralytics YOLO is AGPL-3.0.** Confirmed on the installed package:

```
Name: ultralytics
Version: 8.4.138
License: AGPL-3.0
```

Every source file carries `# Ultralytics 🚀 AGPL-3.0 License`.

Ultralytics' stated position is that models trained with their code are
derivative works of it, and are therefore covered. They sell an **Enterprise
License** at <https://www.ultralytics.com/license> specifically for people who
want to avoid the AGPL obligations. The existence of that product is the
clearest statement of how they read it.

Two honest caveats:

1. **Whether trained weights are legally a "derivative work" of the training
   framework is not settled.** Weights are numbers produced by running code, not
   copies of that code. Reasonable people disagree, and it has not been tested
   in court. Ultralytics' interpretation is the strict one.
2. **The base checkpoint's author disagrees in practice.**
   [`Armaggheddon/yolo11-document-layout`](https://huggingface.co/Armaggheddon/yolo11-document-layout),
   which every model here descends from, declares **MIT** despite also being
   trained with Ultralytics YOLO11.

So there is a real inconsistency upstream. Your options:

| Choice | What it means |
|---|---|
| **Keep AGPL-3.0** (current) | Safest reading. Costs nothing unless you later want to run this in a closed hosted service. |
| **Match the base model's MIT** | Consistent with your upstream, but takes a position against the framework author's stated terms. |
| **Buy an Ultralytics Enterprise License** | The clean route if you want to use these commercially without copyleft. |

If in doubt, AGPL is the conservative default and is what the cards currently
declare. Changing it later is easy while you are the sole copyright holder;
changing it after other people contribute is not.

## X-AnyLabeling: no obligation on your annotations

X-AnyLabeling is **GPL-3.0**. That licence covers **the software**, not what you
produce with it.

GPL has never claimed ownership of a program's output. Using GIMP does not make
your image GPL; using a GPL compiler does not make your binary GPL. The same
applies here: your label files, your dataset and your trained weights are your
own work.

You would only take on GPL obligations if you **distributed X-AnyLabeling
itself**, modified or not. You have not, and nothing in this repository does.

The one courtesy worth extending is attribution, which the docs already give.

## The training images and source PDFs

The 850 annotated pages come from arXiv preprints, and arXiv papers carry
**varied licences**. Many are under the default arXiv non-exclusive
distribution licence, which does **not** grant you redistribution rights.
Others are CC-BY, CC-BY-SA or CC-BY-NC.

This matters for one reason: **redistributing the page images would require
checking each paper's licence individually.** The publishing pipeline does not
distribute them, deliberately. See "What gets published" below.

Training a model on them and publishing the weights is a different act from
redistributing the corpus, and is widely treated as acceptable. That question is
also not fully settled, but it is not one this project takes an unusual position
on.

## Upstream chain

| Component | Licence | Bears on you |
|---|---|---|
| Ultralytics YOLO11 | AGPL-3.0 | The main question. See above. |
| `Armaggheddon/yolo11-document-layout` | MIT | Permissive. Attribution only. |
| DocLayNet, the base model's training data | CDLA-Permissive-1.0 | Permissive. Upstream of the base model, not of you directly. |
| X-AnyLabeling | GPL-3.0 | Software only. No claim on your annotations. |
| Your annotations and weights | Your choice | Currently AGPL-3.0. |

## What gets published to the Hub

**No page images are uploaded.** Verified against a real run directory. Each
variant folder receives:

| File | Contains page content? |
|---|---|
| `best.pt` | No. Model weights. |
| `args.yaml` | No. Hyperparameters. |
| `results.csv` | No. Per-epoch metrics. |
| `results.png` | No. Training curves. |
| `confusion_matrix*.png` | No. Class-by-class counts. |
| `Box*_curve.png` | No. Precision and recall curves. |
| `labels.jpg` | No. Class counts and box position and size histograms. |

Explicitly **not** uploaded, and this is the important one:

- `train_batch*.jpg` and `val_batch*_labels.jpg` / `val_batch*_pred.jpg`.
  Ultralytics writes these as debugging mosaics, and **they are composites of
  real annotated pages**. They would redistribute paper content. They are
  excluded from `RUN_ARTIFACTS` in `doclayout_ft/hub/push_to_hub.py`.
- `last.pt`, redundant with `best.pt`.
- The dataset itself: no images, no label files, no split lists.

If you ever add a filename to `RUN_ARTIFACTS`, check what it depicts first.
