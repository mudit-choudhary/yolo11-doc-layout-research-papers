"""Publish fine-tuned checkpoints to the Hugging Face Hub, oldest first.

This is built for a backlog cleared in instalments rather than one bulk upload.
Runs are ordered by when their weights were written and published a couple at a
time; a ledger records what has already gone up, so running the same command
next week continues where the last one stopped instead of repeating itself.

Each published repository gets the weights, the run's own ``args.yaml`` and
``results.csv``, whichever training curves and confusion matrices the run
produced, and a generated model card. That is enough for someone else to see
what the checkpoint is, how it was trained and how well it scored, without
this repository.

Authentication comes from the Hugging Face CLI, and is not handled here::

    hf auth login          # or: huggingface-cli login

Nothing is uploaded without ``--yes``. The default is a dry run that prints
exactly what each repository would receive, which is worth reading once before
the first real push, since a Hub repository is public by default and its
history is not quietly rewritable.

Usage::

    python -m doclayout_ft.hub.push_to_hub --list
    python -m doclayout_ft.hub.push_to_hub --limit 2
    python -m doclayout_ft.hub.push_to_hub --limit 2 --yes
    python -m doclayout_ft.hub.push_to_hub --only yolo11s_doc_layout_imgsz_1024 --yes
    python -m doclayout_ft.hub.push_to_hub --only yolo11s_doc_layout_imgsz_1024 --private --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from doclayout_ft.checkpoints import Checkpoint, discover, filter_by_name
from doclayout_ft.config import FINETUNED_DIR, ROOT
from doclayout_ft.hub.model_card import build_model_card

#: Hub account or organisation the repositories are created under.
DEFAULT_NAMESPACE = "darkdwine"

#: How many runs one invocation publishes by default. Sized for a weekly pass
#: through the backlog rather than a single bulk upload.
DEFAULT_LIMIT = 2

#: Records which runs have been published, so repeated runs advance the
#: backlog. Machine-local bookkeeping, and git-ignored: it describes what this
#: machine has uploaded, which is not a fact about the source tree.
LEDGER_PATH = ROOT / ".hf_publish_ledger.json"

#: Files copied from a run directory when they exist. Weights and the model
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


def repo_name_for(run_name: str) -> str:
    """Derive a Hub repository name from a run name.

    Underscores become hyphens, which reads better in a URL and matches Hub
    convention. The transformation is one-to-one, so a repository name can
    always be traced back to the run that produced it.

    Args:
        run_name: A run directory name, e.g. ``yolo11s_doc_layout_imgsz_1024``.

    Returns:
        A Hub repository name, e.g. ``yolo11s-doc-layout-imgsz-1024``.
    """
    return run_name.replace("_", "-").lower()


def load_ledger(path: Path = LEDGER_PATH) -> dict[str, dict[str, str]]:
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
    repo_id: str,
    url: str,
    path: Path = LEDGER_PATH,
) -> None:
    """Add one run to the ledger.

    Written after each upload rather than once at the end, so that an
    interrupted batch does not lose track of the repositories it already
    created.

    Args:
        run_name: The run that was published.
        repo_id: Destination repository.
        url: The repository's URL.
        path: Ledger file.
    """
    ledger = load_ledger(path)
    ledger[run_name] = {
        "repo_id": repo_id,
        "url": url,
        "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")


def pending_checkpoints(
    found: dict[str, Checkpoint],
    ledger: dict[str, dict[str, str]],
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
    queue = [c for c in found.values() if include_published or c.name not in ledger]
    queue.sort(key=lambda c: c.modified_at)
    return queue


def files_for(checkpoint: Checkpoint) -> list[tuple[Path, str]]:
    """List the files one run contributes, as (local path, name in the repo).

    Args:
        checkpoint: The run being published.

    Returns:
        Pairs of source path and destination filename. Weights are uploaded as
        ``best.pt`` regardless of run name, so every published repository has
        the same entry point.
    """
    files: list[tuple[Path, str]] = [(checkpoint.weights, "best.pt")]
    for name in RUN_ARTIFACTS:
        path = checkpoint.run_dir / name
        if path.is_file():
            files.append((path, name))
    return files


def publish_one(
    checkpoint: Checkpoint,
    repo_id: str,
    split: str,
    private: bool,
    dry_run: bool,
) -> str | None:
    """Create a Hub repository and upload one run into it.

    Args:
        checkpoint: The run to publish.
        repo_id: Destination repository, ``namespace/name``.
        split: Evaluation split whose metrics the model card should quote.
        private: Whether to create the repository private.
        dry_run: If True, print the plan and upload nothing.

    Returns:
        The repository URL, or None on a dry run.
    """
    card = build_model_card(checkpoint, repo_id, split=split)
    files = files_for(checkpoint)

    print(f"  repo:       {repo_id} ({'private' if private else 'public'})")
    print(f"  run age:    {checkpoint.modified_at.date()}")
    print(f"  model card: {len(card.splitlines())} line(s)")
    for path, name in files:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  upload:     {name:<34} {size_mb:>7.1f} MB")

    if dry_run:
        return None

    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(repo_id=repo_id, repo_type="model",
                          private=private, exist_ok=True)

    api.upload_file(
        path_or_fileobj=card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    for path, name in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {name}",
        )
    return str(url)


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--models-dir", type=Path, default=FINETUNED_DIR,
                        help="Directory of runs to publish (default: FinetunedModels/)")
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE,
                        help=f"Hub account or organisation (default: {DEFAULT_NAMESPACE})")
    parser.add_argument("--repo-id", default=None,
                        help="Full repo id for a single upload, overriding --namespace. "
                             "Only valid with exactly one --only.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Publish these run names instead of the oldest pending ones")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"How many runs to publish this pass (default: {DEFAULT_LIMIT})")
    parser.add_argument("--split", default="val",
                        help="Evaluation split the model cards should quote (default: val)")
    parser.add_argument("--private", action="store_true",
                        help="Create the repositories private")
    parser.add_argument("--include-published", action="store_true",
                        help="Do not skip runs already in the ledger")
    parser.add_argument("--no-baselines", action="store_true", default=True,
                        help="Skip never-fine-tuned checkpoints (default: on; these "
                             "belong to their original authors, not this project)")
    parser.add_argument("--list", action="store_true",
                        help="Show the publish queue and the ledger, then exit")
    parser.add_argument("--yes", action="store_true",
                        help="Actually upload. Without it, this is a dry run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)

    if args.repo_id and (not args.only or len(args.only) != 1):
        print("error: --repo-id names a single repository, so it requires "
              "exactly one --only run name.", file=sys.stderr)
        return 1

    try:
        found = discover(args.models_dir, include_baselines=not args.no_baselines)
        found = filter_by_name(found, args.only)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not found:
        print(f"No runs found under {args.models_dir}", file=sys.stderr)
        return 1

    ledger = load_ledger()
    queue = pending_checkpoints(found, ledger, args.include_published or bool(args.only))

    if args.list:
        print(f"Published already ({len(ledger)}):")
        for name, record in sorted(ledger.items()):
            print(f"  {name:<50} {record.get('repo_id', '?')}")
        print(f"\nPending, oldest first ({len(queue)}):")
        for checkpoint in queue:
            print(f"  {checkpoint.modified_at.date()}  {checkpoint.name:<50} "
                  f"-> {args.namespace}/{repo_name_for(checkpoint.name)}")
        return 0

    if not queue:
        print("Nothing pending. Every run under "
              f"{args.models_dir} is already in the ledger.")
        return 0

    batch = queue[: args.limit] if args.limit > 0 else queue

    mode = "PUBLISHING" if args.yes else "DRY RUN (pass --yes to upload)"
    print(f"{mode}: {len(batch)} of {len(queue)} pending run(s), oldest first.\n")

    failures: list[tuple[str, Exception]] = []
    for index, checkpoint in enumerate(batch, start=1):
        repo_id = args.repo_id or f"{args.namespace}/{repo_name_for(checkpoint.name)}"
        print(f"=== [{index}/{len(batch)}] {checkpoint.name} ===")
        try:
            url = publish_one(checkpoint, repo_id, args.split, args.private, not args.yes)
            if url is not None:
                record_published(checkpoint.name, repo_id, url)
                print(f"  published:  {url}")
        except Exception as exc:  # noqa: BLE001 - one failure must not sink the batch
            failures.append((checkpoint.name, exc))
            print(f"  FAILED: {exc}", file=sys.stderr)
        print()

    remaining = len(queue) - len(batch)
    if args.yes:
        print(f"Published {len(batch) - len(failures)} run(s). "
              f"{remaining} still pending; run this again to continue.")
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
