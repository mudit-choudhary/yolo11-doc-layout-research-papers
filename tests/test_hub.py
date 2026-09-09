"""Tests for Hub repo naming, the publish ledger and model-card generation.

Nothing here touches the network. The publish queue and the card are pure
functions of what is on disk, which is exactly the part worth testing before
anything is uploaded to a public repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doclayout_ft.checkpoints import Checkpoint, discover
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
def test_repo_name_is_a_reversible_transformation(run_name, expected):
    assert push_to_hub.repo_name_for(run_name) == expected


def test_ledger_round_trips(tmp_path):
    ledger = tmp_path / "ledger.json"
    push_to_hub.record_published("run_a", "ns/run-a", "https://example/run-a", ledger)
    push_to_hub.record_published("run_b", "ns/run-b", "https://example/run-b", ledger)

    loaded = push_to_hub.load_ledger(ledger)

    assert set(loaded) == {"run_a", "run_b"}
    assert loaded["run_a"]["repo_id"] == "ns/run-a"
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


def test_model_card_has_front_matter_and_the_full_taxonomy(tmp_path):
    make_run(tmp_path, "run_a", imgsz=1024)
    checkpoint = discover(tmp_path)["run_a"]

    card = model_card.build_model_card(checkpoint, "darkdwine/run-a")

    assert card.startswith("---\n"), "Hub needs YAML front matter first"
    assert "pipeline_tag: object-detection" in card
    assert "library_name: ultralytics" in card
    assert card.count("| `") >= 12, "all twelve classes should be listed"
    assert "`Authors`" in card
    assert "imgsz=1024" in card
    assert "Limitations" in card


def test_model_card_falls_back_to_training_metrics_without_an_eval_table(tmp_path):
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    (run_dir / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),"
        "metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.5,0.5,0.5,0.5\n"
        "2,0.9123,0.8765,0.9456,0.7531\n"
    )
    checkpoint = discover(tmp_path)["run_a"]

    card = model_card.build_model_card(checkpoint, "darkdwine/run-a")

    assert "0.7531" in card, "the final epoch's score should be quoted"
    assert "training-time numbers" in card


def test_model_card_survives_a_run_with_no_recorded_metrics(tmp_path):
    make_run(tmp_path, "run_a", imgsz=None)
    checkpoint = discover(tmp_path)["run_a"]

    card = model_card.build_model_card(checkpoint, "darkdwine/run-a")

    assert "n/a" in card
    assert "No `args.yaml` was recorded" in card


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


def test_repo_id_without_a_single_only_is_rejected(tmp_path):
    """--repo-id names one repository, so it cannot cover a batch."""
    make_run(tmp_path, "run_a", imgsz=1024)
    exit_code = push_to_hub.main(
        ["--models-dir", str(tmp_path), "--repo-id", "ns/thing"]
    )
    assert exit_code == 1


def test_dry_run_uploads_nothing_and_leaves_no_ledger(tmp_path, monkeypatch):
    make_run(tmp_path, "run_a", imgsz=1024)
    ledger = tmp_path / "ledger.json"
    monkeypatch.setattr(push_to_hub, "LEDGER_PATH", ledger)

    exit_code = push_to_hub.main(["--models-dir", str(tmp_path), "--limit", "1"])

    assert exit_code == 0
    assert not ledger.exists(), "a dry run must not record a publish"
