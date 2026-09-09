# Pre-publish checklist

Run through this once before the first push. Steps 1 to 3 are decisions; 4
onward are commands.

## 1. Decide the licence

The repository is currently AGPL-3.0 because Ultralytics is, and I chose that
default without asking. Read [LICENSING.md](LICENSING.md) and confirm you want
it. The base checkpoint you fine-tuned from declares MIT despite also being
trained with Ultralytics, so there is a real inconsistency upstream worth a
moment's thought.

Changing it now is trivial. Changing it after other people build on the
published models is not.

If you keep AGPL, nothing to do. If you change it, edit `LICENSE`, and the
`license:` line in `_front_matter()` in `doclayout_ft/hub/model_card.py`.

## 2. Decide whether the failed runs are published

The default publishes the 14 curated runs in `FinetunedModels/`. Adding
`--models-dir FinetunedModels models` also publishes the 5 `attempt_02` runs,
which carry the augmentation bundle that regressed every model.

Publishing them documents a negative result, and their cards say plainly not to
deploy them. Leaving them out keeps the collection to things worth using.
Either is defensible.

## 3. Confirm the repository name

`darkdwine/yolo11-doc-layout-research-papers`, set as `DEFAULT_REPO_ID` in
`doclayout_ft/hub/push_to_hub.py`. Everything goes into this one repository,
with each variant in a subfolder. Override per-run with `--repo-id`.

## 4. Authenticate

```bash
hf auth login
```

No token goes in this repository, ever. The publisher reads your CLI login.

## 5. Generate the metrics the cards quote

Without this, cards fall back to final-training-epoch numbers and mark
themselves as such.

```bash
python -m doclayout_ft.evaluation.evaluate --models-dir FinetunedModels models --split val
```

Takes a few minutes for 22 models. Writes `reports/evaluation_val.csv`.

## 6. Confirm nothing sensitive is staged

```bash
python -m pytest tests -q          # includes the guards below
```

Three tests exist specifically to keep page imagery off the Hub:

- `test_training_page_mosaics_are_never_uploaded`
- `test_upload_list_contains_no_page_imagery_by_name`
- `test_dataset_files_are_never_uploaded`

The concern is real: Ultralytics writes `train_batch*.jpg` and `val_batch*.jpg`
into every run directory, and those are composites of actual annotated pages
from arXiv papers. They are excluded, and the tests keep it that way.

## 7. Dry run, and actually read it

```bash
python -m doclayout_ft.hub.push_to_hub --list      # the queue
python -m doclayout_ft.hub.push_to_hub --limit 2   # what would be written
```

Check the file list under each variant. You are looking for anything ending
`.jpg` that is not `labels.jpg`, which is a statistics plot and safe.

## 8. Publish

```bash
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

Add `--private` if you want to inspect the result before anyone sees it. You
can make it public later from the Hub settings, and that is the low-risk order.

## 9. Check the published page

Open the repository. Confirm the root card renders, the comparison table is
right, and a variant subfolder looks correct. Then:

```bash
# next week, and the week after
python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
```

The ledger at `.hf_publish_ledger.json` tracks what has gone, so each run picks
up where the last stopped. Back it up if you might publish from another machine.
