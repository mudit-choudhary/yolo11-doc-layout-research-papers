"""Publish fine-tuned checkpoints into a single Hugging Face repository.

Everything goes into **one** Hub repo, with each checkpoint in its own
subfolder::

    darkdwine/yolo11-doc-layout-research-papers/
    ├── README.md                   <- comparison across all variants
    ├── 01-yolo11n-640-v2/
    ├── ...
    ├── 12-yolo11s-1024/            <- recommended
    │   ├── README.md               <- this variant's own card
    │   ├── best.pt
    │   ├── args.yaml
    │   └── ...curves and matrices...
    └── 17-yolo11s-1024-augexp2/
        └── ...

One repository rather than seventeen, because these are variants of a single
model family, not seventeen unrelated models. Someone comparing them should not
have to open fourteen pages, and the root card puts the comparison table in
front of them on arrival.

Uploads are staged rather than done in bulk: runs go up oldest first, a couple
at a time, and a ledger records what has already gone so the same command next
week continues rather than repeats. The root card is regenerated on every
publish from the full ledger, so it always describes the whole collection and
not just the batch that was uploaded.

Authentication comes from the Hugging Face CLI, and is not handled here::

    hf auth login          # or: huggingface-cli login

Nothing is uploaded without ``--yes``. The default is a dry run that prints
exactly what would be written and where, which is worth reading once before the
first real push, since a Hub repository is public by default and its history is
not quietly rewritable.

Usage::

    python -m doclayout_ft.hub.push_to_hub --list
    python -m doclayout_ft.hub.push_to_hub --limit 2
    python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
    python -m doclayout_ft.hub.push_to_hub --models-dir FinetunedModels models --list
    python -m doclayout_ft.hub.push_to_hub --only yolo11s_doc_layout_imgsz_1024 --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover_many, filter_by_name
from doclayout_ft.config import FINETUNED_DIR, ROOT
from doclayout_ft.hub.model_card import (
    build_index_card,
    build_variant_card,
    collect_metrics,
)

#: The single repository every checkpoint is published into.
DEFAULT_REPO_ID = "darkdwine/yolo11-doc-layout-research-papers"

#: How many runs one invocation publishes by default. Sized for a weekly pass
#: through the backlog rather than a single bulk upload.
DEFAULT_LIMIT = 2

#: Records which runs are already in the repository, so repeated runs advance
#: the backlog. It also caches each variant's metrics, so the root card can be
#: rebuilt from the full collection without re-reading every run directory.
#: Machine-local bookkeeping, and git-ignored.
LEDGER_PATH = ROOT / ".hf_publish_ledger.json"

#: Files copied from a run directory when they exist. Weights and the variant
#: card are handled separately, since they are always required.
RUN_ARTIFACTS = (
    "args.yaml",
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
)


#: Publish order and public subfolder name for each run, oldest first.
#:
#: Names carry three things and no more: **when** it was trained, **what
#: architecture**, and **at what resolution**. That is what someone choosing a
#: checkpoint actually decides on.
#:
#: Earlier drafts also encoded the internal lineage (``v2``, ``v22``,
#: ``v2224``). That was dropped. The sequence number already makes every name
#: unique, so the lineage token did no disambiguating work, and ``v2224`` means
#: nothing to anyone outside this repository. Lineage is real information, but
#: it belongs on the variant card, which states the parent checkpoint and the
#: full training history in words.
#:
#: The one status kept in the name is ``augexp``, and it earns its place for a
#: specific reason: a higher sequence number reads as newer and therefore
#: better. Runs 13 to 17 are newer than run 12 and worse. Without the suffix the
#: numbering would actively mislead.
#:
#: Ordering is by hand rather than by file timestamp because most runs were
#: copied into FinetunedModels/ at once and share a meaningless mtime. This
#: table is the chronology, reconstructed from surviving timestamps in models/
#: and from each run's parent checkpoint.
PUBLISH_ORDER: tuple[tuple[str, str], ...] = (
    ("yolo11_doc_layout_v2",                                 "01-yolo11n-640"),
    ("yolo11_doc_layout_v22",                                "02-yolo11n-640"),
    ("yolo11_doc_layout_v2224",                              "03-yolo11n-640"),
    ("yolo11_doc_layout_v222_round03",                       "04-yolo11n-640"),
    ("yolo11_doc_layout_v2224_round03",                      "05-yolo11n-640"),
    ("yolo11_doc_layout_v2_imgsz_1024",                      "06-yolo11n-1024"),
    ("yolo11_doc_layout_v22_imgsz_1024",                     "07-yolo11n-1024"),
    ("yolo11_doc_layout_v2224_imgsz_1024",                   "08-yolo11n-1024"),
    ("yolo11_doc_layout_v222_round03_imgsz_1024",            "09-yolo11n-1024"),
    ("yolo11_doc_layout_v2224_round03_imgsz_1024",           "10-yolo11n-1024"),
    ("yolo11n_doc_layout_imgsz_1024",                        "11-yolo11n-1024"),
    ("yolo11s_doc_layout_imgsz_1024",                        "12-yolo11s-1024"),
    ("yolo11_doc_layout_v222_imgsz_1024_attempt_02",         "13-yolo11n-1024-augexp"),
    ("yolo11_doc_layout_v222_round03_attempt_02",            "14-yolo11n-1024-augexp"),
    ("yolo11_doc_layout_v222_round03_imgsz_1024_attempt_02", "15-yolo11n-1024-augexp"),
    ("yolo11s_doc_layout_imgsz_1024_attempt_02",             "16-yolo11s-1024-augexp"),
    ("yolo11s_doc_layout_attempt_02",                        "17-yolo11s-1024-augexp"),
)

#: One-line status shown against each variant in the root comparison table.
#: This is where "which should I use" gets answered, rather than in a folder
#: name a reader has to decode.
#:
#: "Chained lineage" means the checkpoint is the product of several successive
#: fine-tuning passes, the earliest of which used a dataset round whose train
#: and validation splits pointed at the same images, and which overlaps the
#: current held-out set by up to 19% of its papers. That was tested for score
#: inflation and none was found: clean-lineage models show the same gap between
#: overlapping and non-overlapping papers, so the gap is a property of those
#: pages being easier, not of the models having memorised them. The note is
#: therefore about reproducibility, not about the numbers being wrong. See
#: docs/EVALUATION.md.
RUN_STATUS: dict[str, str] = {
    "yolo11s_doc_layout_imgsz_1024": "**Recommended.** Single fine-tune from base",
    "yolo11n_doc_layout_imgsz_1024": "Fastest. Single fine-tune from base",
    "yolo11_doc_layout_v2": "Early run, superseded",
    "yolo11_doc_layout_v22": "Early run, superseded",
    "yolo11_doc_layout_v2224": "Early run, superseded",
    "yolo11_doc_layout_v2_imgsz_1024": "Chained lineage",
    "yolo11_doc_layout_v22_imgsz_1024": "Chained lineage",
    "yolo11_doc_layout_v2224_imgsz_1024": "Chained lineage. Tops the table by ~0.002, within noise",
    "yolo11_doc_layout_v222_round03": "Chained lineage",
    "yolo11_doc_layout_v2224_round03": "Chained lineage",
    "yolo11_doc_layout_v222_round03_imgsz_1024": "Chained lineage",
    "yolo11_doc_layout_v2224_round03_imgsz_1024": "Chained lineage",
    "yolo11_doc_layout_v222_imgsz_1024_attempt_02": "Failed experiment, do not deploy",
    "yolo11_doc_layout_v222_round03_attempt_02": "Failed experiment, do not deploy",
    "yolo11_doc_layout_v222_round03_imgsz_1024_attempt_02": "Failed experiment, do not deploy",
    "yolo11s_doc_layout_imgsz_1024_attempt_02": "Failed experiment, do not deploy",
    "yolo11s_doc_layout_attempt_02": "Failed experiment, do not deploy",
}

#: Runs whose weights are numerically identical to another run's, verified by
#: comparing state dicts tensor by tensor. The files differ only in metadata.
#: Publishing both would put the same model in the collection twice under two
#: names, so the alias is skipped and named in the card of the run it duplicates.
DUPLICATE_OF: dict[str, str] = {
    "yolo11_doc_layout_v222": "yolo11_doc_layout_v22",
    "yolo11_doc_layout_v222_imgsz_1024": "yolo11_doc_layout_v22_imgsz_1024",
}

_SUBFOLDER_BY_RUN = dict(PUBLISH_ORDER)
_ORDER_BY_RUN = {run: index for index, (run, _) in enumerate(PUBLISH_ORDER)}


def subfolder_for(run_name: str) -> str:
    """Return the public subfolder name for a run.

    Known runs use their entry in :data:`PUBLISH_ORDER`. Anything else, such as
    a run trained after this table was written, falls back to the run name with
    underscores turned into hyphens, which is valid but carries none of the
    ordering or architecture information the curated names do. Add new runs to
    the table rather than relying on the fallback.

    Args:
        run_name: A run directory name, e.g. ``yolo11s_doc_layout_imgsz_1024``.

    Returns:
        A subfolder name, e.g. ``12-yolo11s-1024``.
    """
    known = _SUBFOLDER_BY_RUN.get(run_name)
    return known if known is not None else run_name.replace("_", "-").lower()


def status_for(run_name: str) -> str:
    """Return the one-line status shown against a variant, or an empty string."""
    return RUN_STATUS.get(run_name, "")


def publish_rank(run_name: str) -> int:
    """Return a run's position in the publish order.

    Runs missing from :data:`PUBLISH_ORDER` sort after every known one, keeping
    a newly trained model at the end of the queue rather than silently ahead of
    the curated sequence.
    """
    return _ORDER_BY_RUN.get(run_name, len(PUBLISH_ORDER))


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, dict]:
    """Read the record of already-published runs.

    Args:
        path: Ledger file.

    Returns:
        Mapping of run name to its publish record. A missing or corrupt ledger
        reads as empty: the worst case is re-uploading a run, which the Hub
        handles as a no-op commit, whereas refusing to run would block the
        backlog entirely.
    """
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"warning: could not read ledger at {path}; treating it as empty.",
              file=sys.stderr)
        return {}
    return data if isinstance(data, dict) else {}


def record_published(
    run_name: str,
    entry: dict,
    path: Path = LEDGER_PATH,
) -> dict[str, dict]:
    """Add one run to the ledger and return the updated ledger.

    Written after each upload rather than once at the end, so that an
    interrupted batch does not lose track of what it already pushed.

    Args:
        run_name: The run that was published.
        entry: Its record: subfolder, imgsz, metrics and repo.
        path: Ledger file.

    Returns:
        The ledger including the new entry.
    """
    ledger = load_ledger(path)
    ledger[run_name] = {
        **entry,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    return ledger


def pending_checkpoints(
    found: dict[str, Checkpoint],
    ledger: dict[str, dict],
    include_published: bool,
) -> list[Checkpoint]:
    """Order the publish queue, oldest run first.

    Args:
        found: Discovered checkpoints.
        ledger: Already-published runs.
        include_published: If True, keep runs already in the ledger, for
            re-uploading a card or a corrected artefact.

    Returns:
        Checkpoints in publish order.
    """
    queue = [
        c for c in found.values()
        if (include_published or c.name not in ledger) and c.name not in DUPLICATE_OF
    ]
    # Curated order, not mtime: most runs were copied into FinetunedModels/ in
    # one go and share a meaningless timestamp. Unknown runs fall to the end,
    # ordered among themselves by age.
    queue.sort(key=lambda c: (publish_rank(c.name), c.modified_at))
    return queue


def files_for(checkpoint: Checkpoint) -> list[tuple[Path, str]]:
    """List the files one run contributes, as (local path, name in its folder).

    Args:
        checkpoint: The run being published.

    Returns:
        Pairs of source path and destination filename, relative to the
        variant's subfolder. Weights are uploaded as ``best.pt`` regardless of
        run name, so every variant has the same entry point.
    """
    files: list[tuple[Path, str]] = [(checkpoint.weights, "best.pt")]
    for name in RUN_ARTIFACTS:
        path = checkpoint.run_dir / name
        if path.is_file():
            files.append((path, name))
    return files


def index_entries(ledger: dict[str, dict]) -> list[dict[str, object]]:
    """Turn the ledger into the rows the root card's table needs.

    Args:
        ledger: The publish ledger.

    Returns:
        One entry per published variant.
    """
    return [
        {
            "name": name,
            "rank": publish_rank(name),
            "subfolder": record.get("subfolder", subfolder_for(name)),
            "run_name": record.get("run_name", name),
            "status": record.get("status", status_for(name)),
            "imgsz": record.get("imgsz", "?"),
            "metrics": record.get("metrics", {}),
            "held_out": record.get("metrics_from_held_out_eval", True),
        }
        for name, record in ledger.items()
    ]


def publish_one(
    api,
    checkpoint: Checkpoint,
    repo_id: str,
    split: str,
    dry_run: bool,
) -> dict:
    """Upload one variant into its subfolder of the collection repository.

    Args:
        api: An ``HfApi`` instance, or None on a dry run.
        checkpoint: The run to publish.
        repo_id: The collection repository.
        split: Evaluation split whose metrics the cards should quote.
        dry_run: If True, print the plan and upload nothing.

    Returns:
        The ledger entry describing what was published.
    """
    subfolder = subfolder_for(checkpoint.name)
    metrics, from_held_out = collect_metrics(checkpoint, split)
    alias = next((dup for dup, canon in DUPLICATE_OF.items()
                  if canon == checkpoint.name), None)
    card = build_variant_card(checkpoint, repo_id, subfolder,
                              split=split, duplicate_of=alias)
    files = files_for(checkpoint)

    source = "held-out eval" if from_held_out else "final training epoch"
    print(f"  folder:   {repo_id}/{subfolder}/")
    print(f"  run age:  {checkpoint.modified_at.date()}   "
          f"imgsz: {checkpoint.imgsz}   mAP50-95: "
          f"{metrics.get('mAP50-95', 'n/a')} ({source})")
    print(f"  writes:   {subfolder}/README.md ({len(card.splitlines())} lines)")
    for path, name in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"            {subfolder}/{name:<34} {size_mb:>7.1f} MB")

    entry = {
        "repo_id": repo_id,
        "run_name": checkpoint.name,
        "subfolder": subfolder,
        "status": status_for(checkpoint.name),
        "imgsz": checkpoint.imgsz,
        "metrics": metrics,
        "metrics_from_held_out_eval": from_held_out,
        "source_dir": str(checkpoint.run_dir.parent),
    }

    if dry_run:
        return entry

    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo=f"{subfolder}/README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add {subfolder} model card",
    )
    for path, name in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"{subfolder}/{name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {subfolder}/{name}",
        )
    return entry


def update_index(api, repo_id: str, ledger: dict[str, dict], split: str,
                 dry_run: bool) -> None:
    """Regenerate and upload the repository's root card.

    Called after every batch, from the full ledger rather than from the batch,
    so the comparison table always covers the whole collection.

    Args:
        api: An ``HfApi`` instance, or None on a dry run.
        repo_id: The collection repository.
        ledger: The publish ledger, already updated.
        split: Evaluation split the metrics came from.
        dry_run: If True, print what would be written and upload nothing.
    """
    card = build_index_card(repo_id, index_entries(ledger), split=split)
    print(f"Root card: {repo_id}/README.md "
          f"({len(card.splitlines())} lines, {len(ledger)} variant(s) listed)")
    if dry_run:
        return
    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Update index for {len(ledger)} variant(s)",
    )


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, nargs="+", default=[FINETUNED_DIR],
                        help="Directories of runs to publish, most authoritative "
                             "first (default: FinetunedModels/). Pass both "
                             "FinetunedModels and models to include the "
                             "attempt_02 runs.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                        help=f"The single collection repository every variant is "
                             f"published into (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Publish these run names instead of the oldest pending ones")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"How many runs to publish this pass (default: "
                             f"{DEFAULT_LIMIT}); 0 for all pending")
    parser.add_argument("--split", default="val",
                        help="Evaluation split the cards should quote (default: val)")
    parser.add_argument("--private", action="store_true",
                        help="Create the repository private (first publish only)")
    parser.add_argument("--include-published", action="store_true",
                        help="Do not skip runs already in the ledger")
    parser.add_argument("--refresh-index", action="store_true",
                        help="Regenerate and upload only the root card, publishing "
                             "no new variants")
    parser.add_argument("--list", action="store_true",
                        help="Show the publish queue and the ledger, then exit")
    parser.add_argument("--yes", action="store_true",
                        help="Actually upload. Without it, this is a dry run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    ledger = load_ledger()

    if args.refresh_index:
        if not ledger:
            print("Nothing published yet, so there is no index to refresh.",
                  file=sys.stderr)
            return 1
        api = None
        if args.yes:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=args.repo_id, repo_type="model",
                            private=args.private, exist_ok=True)
        update_index(api, args.repo_id, ledger, args.split, not args.yes)
        return 0

    try:
        found = discover_many(args.models_dir, include_baselines=False)
        found = filter_by_name(found, args.only)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not found:
        dirs = ", ".join(str(d) for d in args.models_dir)
        print(f"No fine-tuned runs found under {dirs}", file=sys.stderr)
        return 1

    queue = pending_checkpoints(found, ledger, args.include_published or bool(args.only))

    if args.list:
        print(f"Repository: {args.repo_id}\n")
        print(f"Discovered {len(found)} run(s) across "
              f"{', '.join(str(d.name) for d in args.models_dir)}\n")
        print(f"Already published ({len(ledger)}):")
        for name, record in sorted(ledger.items()):
            print(f"  {name:<52} -> {record.get('subfolder', '?')}/")
        print(f"\nPending, oldest first ({len(queue)}):")
        for checkpoint in queue:
            print(f"  {checkpoint.modified_at.date()}  {checkpoint.name:<52} "
                  f"-> {subfolder_for(checkpoint.name)}/")
        return 0

    if not queue:
        print("Nothing pending. Every discovered run is already in the ledger.")
        return 0

    batch = queue[: args.limit] if args.limit > 0 else queue

    mode = "PUBLISHING" if args.yes else "DRY RUN (pass --yes to upload)"
    print(f"{mode}\nRepository: {args.repo_id}")
    print(f"Batch: {len(batch)} of {len(queue)} pending run(s), oldest first.\n")

    api = None
    if args.yes:
        from huggingface_hub import HfApi
        api = HfApi()
        url = api.create_repo(repo_id=args.repo_id, repo_type="model",
                              private=args.private, exist_ok=True)
        print(f"Repository ready: {url}\n")

    failures: list[tuple[str, Exception]] = []
    for index, checkpoint in enumerate(batch, start=1):
        print(f"=== [{index}/{len(batch)}] {checkpoint.name} ===")
        try:
            entry = publish_one(api, checkpoint, args.repo_id, args.split, not args.yes)
            if args.yes:
                ledger = record_published(checkpoint.name, entry)
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
            failures.append((checkpoint.name, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)
        print()

    # Rebuild the root card from the whole ledger, including entries from
    # earlier weeks, so the comparison table is never partial. On a dry run
    # this previews the index as it would look after the batch.
    preview = dict(ledger)
    if not args.yes:
        for checkpoint in batch:
            metrics, from_held_out = collect_metrics(checkpoint, args.split)
            preview.setdefault(checkpoint.name, {
                "subfolder": subfolder_for(checkpoint.name),
                "run_name": checkpoint.name,
                "status": status_for(checkpoint.name),
                "imgsz": checkpoint.imgsz,
                "metrics": metrics,
                "metrics_from_held_out_eval": from_held_out,
            })
    if preview:
        update_index(api, args.repo_id, preview, args.split, not args.yes)

    remaining = len(queue) - len(batch)
    print()
    if args.yes:
        print(f"Published {len(batch) - len(failures)} variant(s) into "
              f"{args.repo_id}. {remaining} still pending; run this again to "
              f"continue.")
    else:
        print(f"Dry run complete. {remaining} run(s) would remain after this batch.")

    if failures:
        print(f"\n{len(failures)} run(s) failed:", file=sys.stderr)
        for name, exc in failures:
            print(f"  {name}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
