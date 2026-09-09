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

## Needs you — decisions

- [ ] **Keep AGPL-3.0?** Currently yes. The reasoning and the caveats are in
      [LICENSING.md](LICENSING.md). Nothing to do if you keep it.
- [ ] **Publish the 5 failed `augexp` runs?** Default excludes them. Including
      them documents a negative result; their cards say plainly not to deploy
      them. Controlled by whether you pass `--models-dir FinetunedModels models`.
- [ ] **Repository name.** `darkdwine/yolo11-doc-layout-research-papers`. Change
      `DEFAULT_REPO_ID` in `doclayout_ft/hub/push_to_hub.py` if you want another.
- [ ] **Public or private first?** Publishing private and flipping to public
      after inspection is the low-risk order. **Decide this before the first
      publish:** `--private` applies only when the repository is created, so
      passing it later does nothing. You do not create the repository by hand;
      the first publish makes it.

---

## The LICENSE file itself

Leave the text alone. The AGPL-3.0 text is meant to be copied verbatim and says
so in its own header: "changing it is not allowed". You do not put your name or
project into it.

The copyright line goes elsewhere, and the repository has one now in
`README.md`. That is the conventional place, alongside source-file headers.

---

## The commands, in order

Copy-paste, top to bottom. Run from the repository root with the venv active.

```bash
source /home/mudit/Desktop/fenv/bin/activate
cd /media/mudit/DarkDwine1/ResearchPapersYOLO_FT
```

### 1. Confirm the tree is sound

```bash
python -m pytest tests -q
```

Expect all tests to pass. This includes the guards that keep page imagery off
the Hub, so it is worth running rather than skipping.

### 2. Log in to Hugging Face

```bash
hf auth login
hf auth whoami
```

Paste a token with **write** access, created at
<https://huggingface.co/settings/tokens>. A read-only token authenticates fine
and then fails when the repository is created. No token goes in this
repository.

The username `hf auth whoami` prints must match the namespace in
`DEFAULT_REPO_ID`, currently `darkdwine`. The dry run in step 5 checks this for
you and says so.

You do **not** need to create the repository in the browser. The first publish
creates it.

### 3. Refresh the metrics the cards quote

Skip only if you have not retrained anything since the last evaluation.

```bash
python -m doclayout_ft.evaluation.evaluate \
    --models-dir FinetunedModels models --split val
```

Takes a few minutes on the GTX 1650. Writes `reports/evaluation_val.csv`.

### 4. Read the queue

```bash
python -m doclayout_ft.hub.push_to_hub --list
```

Add `--models-dir FinetunedModels models` here and in every command below if
you decided to include the failed `augexp` runs.

### 5. Dry run, and actually read the output

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2
```

Read two things. First the `preflight:` lines, which confirm your login has
write scope and that the namespace is yours. Then the file list under each
variant: you want nothing ending `.jpg` except `labels.jpg`.

Nothing is uploaded without `--yes`.

### 6. Publish the first two

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

The repository is created here, automatically. Add `--private` if you want to
inspect before anyone else can see it, and add it **on this first run**: it is
ignored once the repository exists. Making it public later is one setting in
the Hub UI.

### 7. Check the result

Open <https://huggingface.co/darkdwine/yolo11-doc-layout-research-papers>.
Confirm the root card renders, the comparison table looks right, and one
variant subfolder contains `best.pt` and its own `README.md`.

### 8. Next week, and the week after

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

Same command each time. The ledger tracks what has gone, so it continues rather
than repeats. Seven or nine passes clears the backlog.

### If a card needs fixing after publishing

```bash
# rebuild only the root comparison table
python -m doclayout_ft.hub.push_to_hub --refresh-index --yes

# re-upload one variant
python -m doclayout_ft.hub.push_to_hub \
    --only yolo11s_doc_layout_imgsz_1024 --include-published --yes
```

### To jump the queue and publish the best model first

```bash
python -m doclayout_ft.hub.push_to_hub \
    --only yolo11s_doc_layout_imgsz_1024 --yes
```

It lands in `12-yolo11s-1024/` regardless of publish order, so doing this does
not disturb the numbering.
