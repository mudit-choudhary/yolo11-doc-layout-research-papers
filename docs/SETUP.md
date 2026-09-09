# Setup

## The short version

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch first, matched to your driver. See below before running this.
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

pip install -e .
sudo apt install poppler-utils
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m doclayout_ft.training.finetune --list
```

## Install PyTorch before anything else

This is the step that goes wrong, and it goes wrong quietly.

`ultralytics` will happily install whatever PyTorch build pip finds first. If
that build was compiled against a newer CUDA than your driver supports, nothing
fails at install time. It fails later, at the first CUDA call, with an error
that reads like a hardware fault rather than a version mismatch.

Pick your build from the [PyTorch install
selector](https://pytorch.org/get-started/locally/) using your actual driver
version:

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
```

The driver version caps the CUDA runtime you can use. A newer runtime against
an older driver does not degrade, it refuses to initialise.

### The reference machine

| | |
|---|---|
| GPU | NVIDIA GTX 1650, 4 GB |
| Driver | 535.x, CUDA 12.2 |
| PyTorch | 2.7.1+cu118 |
| OS | Ubuntu 24.04 |

The driver is pinned at 535.x for reasons unrelated to this project, involving
external-monitor support. Mid-project an unrelated environment update pulled in
a `cu130` build of PyTorch, which could not initialise against that driver.
Reinstalling the `cu118` build fixed it. If CUDA suddenly stops working after
an upgrade, check the build suffix first:

```bash
python -c "import torch; print(torch.__version__)"   # want a +cu118 suffix here
```

## Poppler

PDF rendering shells out to Poppler's `pdftoppm`, which pip cannot install:

```bash
sudo apt install poppler-utils     # Debian / Ubuntu
brew install poppler               # macOS
```

Only the two PDF-reading entry points need it. Training, evaluation and
image-based inference do not.

## Mixed precision on older GPUs

On the GTX 1650, Ultralytics' AMP sanity check fails and training falls back to
full FP32. This is correct behaviour, not a bug: Turing cards without tensor
cores produce NaN losses under Ultralytics' AMP path.

The consequence is worth knowing when tuning batch size. FP32 activations cost
roughly twice the memory of FP16, so the card behaves like a smaller one than
its 4 GB suggests. On an RTX card AMP engages automatically and the batch sizes
in `docs/FINETUNING_STEPS.md` can be raised considerably.

## Hugging Face authentication

Only needed to publish models. Downloading base checkpoints works anonymously.

```bash
hf auth login          # newer huggingface_hub
huggingface-cli login  # older
```

The publish script never handles a token itself; it relies on this login. Do
not put a token in a file inside the repository.

## Directory expectations

The package resolves every path from the repository root, so the checkout can
live anywhere. The large directories are recreated rather than cloned:

| Directory | Size | How to get it |
|---|---|---|
| `PDFs/` | 4.4 GB | Supply your own; see docs/DATASET.md |
| `images/` | 19 GB | `python -m doclayout_ft.data.pdf_to_images` |
| `training_dataset/*/images`, `labels` | 1.4 GB | Annotate; see docs/DATASET.md |
| `models/`, `FinetunedModels/` | 780 MB | Train, or pull from the Hub |

One absolute path does persist, inside each round's `data.yaml`. It is rewritten
automatically every time `doclayout_ft.data.split_dataset` runs, so a moved or
freshly cloned checkout self-heals on the next split.
