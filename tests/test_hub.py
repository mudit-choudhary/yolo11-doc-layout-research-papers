"""Tests for Hub repo naming, the publish ledger and model-card generation.

Nothing here touches the network. The publish queue and the card are pure
functions of what is on disk, which is exactly the part worth testing before
anything is uploaded to a public repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doclayout_ft.checkpoints import discover
from doclayout_ft.hub import model_card, push_to_hub
from tests.test_checkpoints import make_run


@pytest.mark.parametrize(
    ("run_name", "expected"),
    [
        ("yolo11s_doc_layout_imgsz_1024", "yolo11s-doc-layout-imgsz-1024"),
        ("yolo11_doc_layout_v2", "yolo11-doc-layout-v2"),
        ("Already-Hyphenated", "already-hyphenated"),
    ],
)
def test_subfolder_is_a_reversible_transformation(run_name, expected):
    assert push_to_hub.subfolder_for(run_name) == expected


def test_everything_publishes_into_one_repository():
    """The whole collection is one repo with subfolders, not one repo each."""
    assert push_to_hub.DEFAULT_REPO_ID.count("/") == 1
    names = ["yolo11s_doc_layout_imgsz_1024", "yolo11n_doc_layout_imgsz_1024"]
    folders = {push_to_hub.subfolder_for(n) for n in names}
    assert len(folders) == len(names), "each variant needs its own subfolder"
    assert not any("/" in f for f in folders), "a subfolder is one path segment"


def test_ledger_round_trips(tmp_path):
    ledger = tmp_path / "ledger.json"
    push_to_hub.record_published(
        "run_a", {"repo_id": "ns/coll", "subfolder": "run-a", "imgsz": 1024}, ledger)
    push_to_hub.record_published(
        "run_b", {"repo_id": "ns/coll", "subfolder": "run-b", "imgsz": 640}, ledger)

    loaded = push_to_hub.load_ledger(ledger)

    assert set(loaded) == {"run_a", "run_b"}
    assert loaded["run_a"]["subfolder"] == "run-a"
    assert loaded["run_a"]["repo_id"] == loaded["run_b"]["repo_id"], \
        "every variant belongs to the same repository"
    assert "published_at" in loaded["run_a"]


def test_missing_ledger_reads_as_empty(tmp_path):
    assert push_to_hub.load_ledger(tmp_path / "absent.json") == {}


def test_corrupt_ledger_reads_as_empty_rather_than_blocking_the_backlog(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text("{not json")
    assert push_to_hub.load_ledger(ledger) == {}


def test_ledger_holding_a_json_list_is_ignored(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(["run_a"]))
    assert push_to_hub.load_ledger(ledger) == {}


def test_queue_is_oldest_first_and_skips_published_runs(tmp_path):
    import os

    make_run(tmp_path, "older", imgsz=640)
    make_run(tmp_path, "newer", imgsz=1024)
    make_run(tmp_path, "done", imgsz=1024)
    # Force a known ordering rather than relying on filesystem timestamp
    # resolution, which on some filesystems is coarser than the test runtime.
    for name, mtime in (("older", 1_000), ("done", 2_000), ("newer", 3_000)):
        os.utime(tmp_path / name / "weights" / "best.pt", (mtime, mtime))

    found = discover(tmp_path)
    ledger = {"done": {"repo_id": "ns/done"}}

    queue = push_to_hub.pending_checkpoints(found, ledger, include_published=False)
    assert [c.name for c in queue] == ["older", "newer"]

    everything = push_to_hub.pending_checkpoints(found, ledger, include_published=True)
    assert [c.name for c in everything] == ["older", "done", "newer"]


def test_files_for_always_normalises_the_weights_name(tmp_path):
    """Every published repo should expose the same entry point."""
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    (run_dir / "results.csv").write_text("epoch\n1\n")
    (run_dir / "confusion_matrix.png").write_bytes(b"png")

    checkpoint = discover(tmp_path)["run_a"]
    names = [name for _, name in push_to_hub.files_for(checkpoint)]

    assert names[0] == "best.pt"
    assert "args.yaml" in names
    assert "results.csv" in names
    assert "confusion_matrix.png" in names
    assert "results.png" not in names, "absent artefacts must not be listed"


def test_load_final_epoch_tolerates_padded_column_names(tmp_path):
    """Some Ultralytics versions pad results.csv headers with spaces."""
    run_dir = make_run(tmp_path, "run_a", imgsz=640)
    (run_dir / "results.csv").write_text(
        "epoch,   metrics/mAP50-95(B)\n1,0.42\n"
    )
    checkpoint = discover(tmp_path)["run_a"]

    final = model_card.load_final_epoch(checkpoint)

    assert final is not None
    assert final["metrics/mAP50-95(B)"] == "0.42"


def test_index_entries_default_held_out_for_older_ledger_records():
    """Records written before the flag existed must not gain a bogus dagger."""
    entries = push_to_hub.index_entries(
        {"run_a": {"subfolder": "run-a", "imgsz": 1024, "metrics": {}}})
    assert entries[0]["held_out"] is True


def test_refresh_index_without_a_ledger_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(push_to_hub, "LEDGER_PATH", tmp_path / "absent.json")
    assert push_to_hub.main(["--refresh-index"]) == 1


def test_dry_run_uploads_nothing_and_leaves_no_ledger(tmp_path, monkeypatch):
    make_run(tmp_path, "run_a", imgsz=1024)
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(push_to_hub, "LEDGER_PATH", ledger)

    exit_code = push_to_hub.main(["--models-dir", str(tmp_path), "--limit", "1"])

    assert exit_code == 0
    assert not ledger.exists(), "a dry run must not record a publish"


def test_baselines_are_never_published(tmp_path, monkeypatch):
    """Base checkpoints belong to their original author, not to this project."""
    from tests.test_checkpoints import make_baseline

    make_run(tmp_path, "run_a", imgsz=1024)
    make_baseline(tmp_path, "yolov11s", "yolo11s_doc_layout.pt")
    monkeypatch.setattr(push_to_hub, "LEDGER_PATH", tmp_path / "ledger.json")

    from doclayout_ft.checkpoints import discover_many
    found = discover_many([tmp_path], include_baselines=False)

    assert set(found) == {"run_a"}


# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------


def test_only_the_root_card_carries_front_matter(tmp_path):
    """The Hub reads repository metadata from the root README only."""
    make_run(tmp_path, "run_a", imgsz=1024)
    checkpoint = discover(tmp_path)["run_a"]

    index = model_card.build_index_card("ns/coll", [
        {"name": "run_a", "subfolder": "run-a", "imgsz": 1024, "metrics": {}},
    ])
    variant = model_card.build_variant_card(checkpoint, "ns/coll", "run-a")

    assert index.startswith("---\n"), "Hub needs YAML front matter first"
    assert "pipeline_tag: object-detection" in index
    assert "library_name: ultralytics" in index
    assert not variant.startswith("---\n"), "a subfolder card needs no front matter"


def test_index_card_lists_every_variant_and_the_taxonomy():
    index = model_card.build_index_card("ns/coll", [
        {"name": "a", "subfolder": "a", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.7000"}},
        {"name": "b", "subfolder": "b", "imgsz": 640,
         "metrics": {"mAP50-95": "0.8000"}},
    ])

    assert "`a`" in index and "`b`" in index
    assert "`Authors`" in index
    assert index.count("| `") >= 14, "twelve classes plus two variants"
    assert "Limitations" in index


def test_index_card_orders_variants_by_score():
    index = model_card.build_index_card("ns/coll", [
        {"name": "worse", "subfolder": "worse", "imgsz": 640,
         "metrics": {"mAP50-95": "0.5000"}},
        {"name": "better", "subfolder": "better", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.9000"}},
    ])
    assert index.index("`better`") < index.index("`worse`")


def test_index_card_sorts_unscored_variants_last():
    index = model_card.build_index_card("ns/coll", [
        {"name": "unscored", "subfolder": "unscored", "imgsz": 640,
         "metrics": {"mAP50-95": "n/a"}},
        {"name": "scored", "subfolder": "scored", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.5000"}},
    ])
    assert index.index("`scored`") < index.index("`unscored`")


def test_index_card_marks_training_time_numbers_as_not_comparable():
    """A training-time score must not sit unmarked beside held-out ones."""
    index = model_card.build_index_card("ns/coll", [
        {"name": "clean", "subfolder": "clean", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.9000"}, "held_out": True},
        {"name": "guess", "subfolder": "guess", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.8000"}, "held_out": False},
    ])

    assert "`guess` †" in index
    assert "`clean` †" not in index
    assert "final training epoch" in index


def test_index_card_omits_the_footnote_when_every_score_is_held_out():
    index = model_card.build_index_card("ns/coll", [
        {"name": "clean", "subfolder": "clean", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.9000"}, "held_out": True},
    ])
    assert "†" not in index


def test_index_card_handles_an_empty_collection():
    assert "No variants published yet" in model_card.build_index_card("ns/coll", [])


def test_index_card_usage_snippet_points_at_a_real_variant():
    """The example must be copy-pasteable, so it needs a folder that exists."""
    index = model_card.build_index_card("ns/coll", [
        {"name": "only_one", "subfolder": "only-one", "imgsz": 640,
         "metrics": {"mAP50-95": "0.5000"}},
    ])
    assert 'filename="only-one/best.pt"' in index
    assert "<variant>" not in index


def test_variant_card_points_at_its_own_subfolder(tmp_path):
    make_run(tmp_path, "run_a", imgsz=1024)
    checkpoint = discover(tmp_path)["run_a"]

    card = model_card.build_variant_card(checkpoint, "ns/coll", "run-a")

    assert 'filename="run-a/best.pt"' in card
    assert 'repo_id="ns/coll"' in card
    assert "imgsz=1024" in card


def test_variant_card_warns_on_a_published_failure(tmp_path):
    make_run(tmp_path, "run_x_attempt_02", imgsz=1024)
    checkpoint = discover(tmp_path)["run_x_attempt_02"]

    card = model_card.build_variant_card(checkpoint, "ns/coll", "run-x-attempt-02")

    assert "published failure" in card
    assert "Do not deploy it" in card


def test_variant_card_warns_on_compromised_lineage(tmp_path):
    make_run(tmp_path, "run_round03_thing", imgsz=1024)
    checkpoint = discover(tmp_path)["run_round03_thing"]

    assert "Compromised lineage" in model_card.build_variant_card(
        checkpoint, "ns/coll", "x")


def test_metrics_fall_back_to_training_numbers_without_an_eval_table(tmp_path):
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.5,0.5,0.5\n"
        "2,0.9123,0.8765,0.9456,0.7531\n"
    )
    checkpoint = discover(tmp_path)["run_a"]

    metrics, from_held_out = model_card.collect_metrics(checkpoint, "val")

    assert metrics["mAP50-95"] == "0.7531", "the final epoch's score"
    assert from_held_out is False
    assert "training-time numbers" in model_card.build_variant_card(
        checkpoint, "ns/coll", "run-a")


def test_variant_card_survives_a_run_with_no_recorded_metrics(tmp_path):
    # imgsz=None means no args.yaml is written, so the card has nothing to
    # describe the training configuration with.
    make_run(tmp_path, "run_a", imgsz=None)
    checkpoint = discover(tmp_path)["run_a"]

    card = model_card.build_variant_card(checkpoint, "ns/coll", "run-a")

    assert "n/a" in card
    assert "No `args.yaml` was recorded" in card


# --------------------------------------------------------------------------
# Guard: never distribute page imagery
# --------------------------------------------------------------------------


def test_training_page_mosaics_are_never_uploaded(tmp_path):
    """train_batch/val_batch images are composites of real annotated pages.

    The corpus is rendered from arXiv preprints whose licences vary per paper
    and often forbid redistribution, so these must not reach the Hub. This is
    the one upload rule that is a licensing matter rather than a tidiness one.
    """
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    for name in ("train_batch0.jpg", "train_batch12.jpg",
                 "val_batch0_labels.jpg", "val_batch0_pred.jpg",
                 "val_batch2_pred.jpg"):
        (run_dir / name).write_bytes(b"page imagery")

    uploaded = {name for _, name in push_to_hub.files_for(discover(tmp_path)["run_a"])}

    assert not any("train_batch" in n or "val_batch" in n for n in uploaded), \
        f"page mosaics must never be uploaded, found: {sorted(uploaded)}"
    assert "best.pt" in uploaded, "the weights themselves still go up"


def test_upload_list_contains_no_page_imagery_by_name():
    """A second line of defence over the artefact list itself."""
    forbidden = ("train_batch", "val_batch", "mosaic")
    for name in push_to_hub.RUN_ARTIFACTS:
        assert not any(f in name for f in forbidden), \
            f"{name} looks like it depicts training pages"


def test_dataset_files_are_never_uploaded(tmp_path):
    """Label files and split lists are dataset, not model artefacts."""
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    for name in ("train.txt", "val.txt", "classes.txt", "data.yaml", "labels.cache"):
        (run_dir / name).write_bytes(b"dataset")

    uploaded = {name for _, name in push_to_hub.files_for(discover(tmp_path)["run_a"])}

    assert uploaded & {"train.txt", "val.txt", "classes.txt",
                       "data.yaml", "labels.cache"} == set()
