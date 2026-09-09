# Pre-publish checklist

Two sections. The first is what has been **verified by running it**, so you do
not need to re-check it. The second is what still needs you.

---

## Verified — no action needed

Each of these was checked by executing something and reading the result, not by
assumption. The command that verified it is given so you can repeat it.

- [x] **No page images are published.** Checked the upload list against a real
      run directory. Of the files uploaded, only `labels.jpg` is an image, and
      it holds class counts and box histograms, not page content. Opened and
      inspected it directly.
- [x] **The `train_batch` / `val_batch` mosaics are excluded.** These are
      composites of real annotated arXiv pages and are the actual redistribution
      risk. Three tests enforce it:
      `test_training_page_mosaics_are_never_uploaded`,
      `test_upload_list_contains_no_page_imagery_by_name`,
      `test_dataset_files_are_never_uploaded`.
- [x] **No dataset files are published.** No images, labels, split lists,
      `classes.txt` or `data.yaml`.
- [x] **Everything goes into one repository.** Confirmed by dry run: 17 variants
      as subfolders of `darkdwine/yolo11-doc-layout-research-papers`, not 17
      repositories.
- [x] **Subfolder names are unique, sortable, Hub-safe, and cover every run on
      disk.** 19 run directories map to 17 published names plus 2 declared
      duplicates, with nothing unaccounted for. Names carry only order,
      architecture and resolution; internal lineage tokens were removed after
      confirming the sequence number alone makes every name unique.
- [x] **Older lineages do not gain from overlapping the held-out set.** Fifteen
      of nineteen runs descend from a round that saw up to 19% of today's
      validation papers. Measured directly by scoring on the seen and unseen
      halves separately: clean-lineage models show the same gap, so the
      published scores are comparable. See [EVALUATION.md](EVALUATION.md).
- [x] **Two runs are genuine duplicates and are skipped.**
      `yolo11_doc_layout_v222` and `yolo11_doc_layout_v222_imgsz_1024` are
      numerically identical to their `v22` counterparts, verified by comparing
      state dicts tensor by tensor. Not published twice.
- [x] **All 22 models have real held-out scores.** Evaluated on the `val` split
      of `round_final`, so no card falls back to training-time numbers.
- [x] **Upstream licences confirmed.** `ultralytics 8.4.138` is AGPL-3.0 (read
      off the installed package); the base model declares MIT; DocLayNet is
      CDLA-Permissive-1.0; X-AnyLabeling is GPL-3.0 and imposes nothing on your
      annotations. See [LICENSING.md](LICENSING.md).
- [x] **Baselines are excluded.** The un-fine-tuned checkpoints belong to their
      original author and never enter the queue.
- [x] **Every command runs.** All nine entry points execute from an arbitrary
      working directory; 76 tests pass.
- [x] **Nothing has been uploaded.** The ledger is empty and every run so far has
      been a dry run.

Re-confirm the whole set at any time:

```bash
python -m pytest tests -q
python -m doclayout_ft.hub.push_to_hub --models-dir FinetunedModels models --limit 0
```

---

## Decided

Settled on 2026-09-09. Recorded so the reasoning is not re-litigated later.

- [x] **Licence: AGPL-3.0.** Kept. See [LICENSING.md](LICENSING.md) for why it
      propagates from Ultralytics and what the network clause means.
- [x] **Publish all 17, including the 5 failed `augexp` runs.** A documented
      negative result is more useful than an undocumented one. Their cards and
      the root table both say plainly not to deploy them. **This is now the
      default**, so no flag is needed and none can be forgotten partway through
      the backlog.
- [x] **Repository: `darkdwine/yolo11-doc-layout-research-papers`.**
- [x] **Private first**, made public later from the browser. `--private` must
      be on the **first** publish, which is the one that creates the repository.

---

## The LICENSE file itself

Leave the text alone. The AGPL-3.0 text is meant to be copied verbatim and says
so in its own header: "changing it is not allowed". You do not put your name or
project into it.

The copyright line goes elsewhere, and the repository has one now in
`README.md`. That is the conventional place, alongside source-file headers.

---

## The commands, in order

Copy-paste, top to bottom.

```bash
source /home/mudit/Desktop/fenv/bin/activate
cd /media/mudit/DarkDwine1/ResearchPapersYOLO_FT
```

### 1. Confirm the tree is sound

```bash
python -m pytest tests -q
```

### 2. Authenticate

First create a token at <https://huggingface.co/settings/tokens>:

- **New token** → token type **Write** (a Read token logs in fine and then
  fails when the repository is created)
- Name it anything, `doclayout-publish` for instance
- Copy it; the Hub shows it once

Then:

```bash
hf auth login
```

It prompts for the token. Paste it and press Enter. The paste is invisible,
which is normal. Answer **n** to "Add token as git credential?" unless you also
plan to `git push` to the Hub.

Non-interactive alternative, if you prefer not to paste at a prompt:

```bash
hf auth login --token "$(cat ~/path/to/token.txt)"
```

Do not put the token in a shell command you type directly, or it lands in
`~/.bash_history`. Never commit it.

Confirm:

```bash
hf auth whoami
```

It should print `darkdwine`. If it prints something else, that is your real
username, and `DEFAULT_REPO_ID` in `doclayout_ft/hub/push_to_hub.py` needs to
match it.

The token is stored at `~/.cache/huggingface/token`. `hf auth logout` removes
it.

### 3. Refresh the metrics the cards quote

Skip only if nothing has been retrained since the last evaluation.

```bash
python -m doclayout_ft.evaluation.evaluate --split val
```

Scores all 22 models by default, which is what the 17 published variants need.
Takes a few minutes on the GTX 1650.

### 4. Dry run, and read the preflight lines

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2
```

Expect:

```
  preflight: logged in as 'darkdwine', token role 'write'
  preflight: 'darkdwine/yolo11-doc-layout-research-papers' will be created automatically on publish.
```

If it warns about a read-only token or a namespace mismatch, fix that before
going further. Nothing uploads without `--yes`.

### 5. First publish — this creates the repository

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --private --yes
```

`--private` matters **only on this run**. It is ignored once the repository
exists.

### 6. Check it

Open <https://huggingface.co/darkdwine/yolo11-doc-layout-research-papers>.
Confirm the root card renders, the table looks right, and `01-yolo11n-640/`
holds `best.pt` plus its own `README.md`.

### 7. Every following week

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

No `--private` needed; the repository keeps its visibility. The ledger tracks
what has gone, so this continues rather than repeats. Nine passes clears all 17.

Watch the last line for progress:

```
Published 2 variant(s) into darkdwine/... 13 still pending; run this again to continue.
```

### 8. When you are ready to go public

In the browser: repository → **Settings** → **Change visibility** → Public.
Nothing to run locally.

---

## If something needs fixing after publishing

```bash
# rebuild only the root comparison table
python -m doclayout_ft.hub.push_to_hub --refresh-index --yes

# re-upload one variant, card and all
python -m doclayout_ft.hub.push_to_hub \
    --only yolo11s_doc_layout_imgsz_1024 --include-published --yes
```

## To publish the best model first instead of oldest-first

```bash
python -m doclayout_ft.hub.push_to_hub \
    --only yolo11s_doc_layout_imgsz_1024 --private --yes
```

It lands in `12-yolo11s-1024/` regardless, so the numbering is unaffected and
the remaining backlog continues from the oldest.
