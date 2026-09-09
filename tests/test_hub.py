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
        ("yolo11s_doc_layout_imgsz_1024", "12-yolo11s-1024"),
        ("yolo11_doc_layout_v2", "01-yolo11n-640"),
        ("yolo11s_doc_layout_attempt_02", "17-yolo11s-1024-augexp"),
    ],
)
def test_curated_runs_get_their_published_name(run_name, expected):
    assert push_to_hub.subfolder_for(run_name) == expected


def test_unknown_run_falls_back_to_a_valid_name():
    """A model trained after the table was written must still publish."""
    assert push_to_hub.subfolder_for("some_new_run") == "some-new-run"


def test_published_names_are_unique():
    """Two runs sharing a subfolder would overwrite each other on the Hub."""
    folders = [folder for _, folder in push_to_hub.PUBLISH_ORDER]
    assert len(folders) == len(set(folders))


def test_published_names_sort_into_publish_order():
    """The numeric prefix is what makes the Hub file browser show chronology."""
    folders = [folder for _, folder in push_to_hub.PUBLISH_ORDER]
    assert folders == sorted(folders)


def test_published_names_are_hub_safe():
    folders = [folder for _, folder in push_to_hub.PUBLISH_ORDER]
    for folder in folders:
        assert folder == folder.lower()
        assert "/" not in folder and " " not in folder
        assert all(c.isalnum() or c == "-" for c in folder), folder


def test_published_names_state_architecture_and_resolution():
    """Architecture and training resolution are what a consumer picks on."""
    for _, folder in push_to_hub.PUBLISH_ORDER:
        assert "yolo11n" in folder or "yolo11s" in folder, folder
        assert "-640" in folder or "-1024" in folder, folder


def test_published_names_carry_no_internal_lineage_tokens():
    """v2/v22/v2224 mean nothing outside this repo and disambiguate nothing."""
    for _, folder in push_to_hub.PUBLISH_ORDER:
        assert "-v2" not in folder, f"{folder} leaks internal lineage"
        assert "round03" not in folder, f"{folder} leaks internal round naming"


def test_names_stay_unique_without_lineage_tokens():
    """The sequence number is what carries uniqueness now."""
    folders = [f for _, f in push_to_hub.PUBLISH_ORDER]
    assert len(folders) == len(set(folders))


def test_failed_runs_keep_a_marker_in_the_name():
    """A higher number reads as newer and better; these are newer and worse."""
    for run, folder in push_to_hub.PUBLISH_ORDER:
        if run.endswith("_attempt_02"):
            assert folder.endswith("-augexp"), folder
        else:
            assert not folder.endswith("-augexp"), folder


def test_every_failed_run_is_labelled_as_one():
    """A reader must not mistake a published failure for an option."""
    for run, _ in push_to_hub.PUBLISH_ORDER:
        if run.endswith("_attempt_02"):
            assert "Failed experiment" in push_to_hub.status_for(run), run
    assert "Recommended" in push_to_hub.status_for("yolo11s_doc_layout_imgsz_1024")


def test_every_run_has_a_pass_count():
    """The Passes column replaced prose, so it has to be complete."""
    for run, _ in push_to_hub.PUBLISH_ORDER:
        count = push_to_hub.passes_for(run)
        assert isinstance(count, int) and count >= 1, f"{run}: {count!r}"


def test_single_pass_runs_are_the_reproducible_ones():
    assert push_to_hub.passes_for("yolo11s_doc_layout_imgsz_1024") == 1
    assert push_to_hub.passes_for("yolo11n_doc_layout_imgsz_1024") == 1
    assert push_to_hub.passes_for("yolo11_doc_layout_v2224_round03_imgsz_1024") == 4


def test_status_map_only_names_runs_that_are_published():
    published = {run for run, _ in push_to_hub.PUBLISH_ORDER}
    assert set(push_to_hub.RUN_STATUS) <= published


def test_index_table_shows_the_notes_column():
    index = model_card.build_index_card("ns/coll", [
        {"name": "a", "subfolder": "12-yolo11s-1024", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.7694"}, "status": "**Recommended**"},
    ])
    assert "| Notes |" in index
    assert "**Recommended**" in index


def test_publish_rank_orders_known_runs_and_sinks_unknown_ones():
    ranks = [push_to_hub.publish_rank(run) for run, _ in push_to_hub.PUBLISH_ORDER]
    assert ranks == sorted(ranks)
    assert push_to_hub.publish_rank("not_in_table") >= len(push_to_hub.PUBLISH_ORDER)


def test_duplicate_runs_point_at_a_run_that_is_published():
    """An alias must defer to a run that actually reaches the Hub."""
    published = {run for run, _ in push_to_hub.PUBLISH_ORDER}
    for alias, canonical in push_to_hub.DUPLICATE_OF.items():
        assert canonical in published, f"{alias} defers to unpublished {canonical}"
        assert alias not in published, f"{alias} is both an alias and published"


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


def test_duplicate_weights_are_not_published_twice(tmp_path):
    """v222 is byte-different from v22 but numerically identical to it."""
    alias, canonical = next(iter(push_to_hub.DUPLICATE_OF.items()))
    make_run(tmp_path, alias, imgsz=1024)
    make_run(tmp_path, canonical, imgsz=1024)

    queue = push_to_hub.pending_checkpoints(
        discover(tmp_path), ledger={}, include_published=False)

    names = [c.name for c in queue]
    assert canonical in names
    assert alias not in names, "the duplicate must not get its own subfolder"


def test_queue_follows_the_curated_order_not_file_mtime(tmp_path):
    """Most runs were copied at once, so mtimes carry no chronology."""
    import os
    early, late = push_to_hub.PUBLISH_ORDER[0][0], push_to_hub.PUBLISH_ORDER[5][0]
    make_run(tmp_path, early, imgsz=640)
    make_run(tmp_path, late, imgsz=1024)
    # Give the chronologically earlier run the NEWER mtime, so mtime ordering
    # would put it second.
    os.utime(tmp_path / early / "weights" / "best.pt", (9_000, 9_000))
    os.utime(tmp_path / late / "weights" / "best.pt", (1_000, 1_000))

    queue = push_to_hub.pending_checkpoints(
        discover(tmp_path), ledger={}, include_published=False)

    assert [c.name for c in queue] == [early, late]


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

    assert "Chained lineage" in model_card.build_variant_card(
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


# --------------------------------------------------------------------------
# Preflight: catch a bad login or namespace before an upload starts
# --------------------------------------------------------------------------


class _FakeApi:
    """Stands in for HfApi.whoami() without touching the network."""

    def __init__(self, payload):
        self._payload = payload

    def whoami(self):
        if self._payload is None:
            raise RuntimeError("not logged in")
        return self._payload


def _patch_api(monkeypatch, payload):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: _FakeApi(payload))


def test_preflight_reports_a_missing_login(monkeypatch, capsys):
    _patch_api(monkeypatch, None)
    assert push_to_hub.preflight("darkdwine/coll") is False
    assert "not logged in" in capsys.readouterr().out


def test_preflight_accepts_your_own_namespace(monkeypatch, capsys):
    _patch_api(monkeypatch, {"name": "darkdwine",
                             "auth": {"accessToken": {"role": "write"}}})
    assert push_to_hub.preflight("darkdwine/coll") is True
    assert "created automatically" in capsys.readouterr().out


def test_preflight_rejects_a_namespace_you_do_not_own(monkeypatch, capsys):
    """Publishing to someone else's namespace is refused by the Hub."""
    _patch_api(monkeypatch, {"name": "someone_else",
                             "auth": {"accessToken": {"role": "write"}}})
    assert push_to_hub.preflight("darkdwine/coll") is False
    assert "neither your username nor an org" in capsys.readouterr().out


def test_preflight_accepts_an_org_you_belong_to(monkeypatch):
    _patch_api(monkeypatch, {"name": "mudit", "orgs": [{"name": "darkdwine"}],
                             "auth": {"accessToken": {"role": "write"}}})
    assert push_to_hub.preflight("darkdwine/coll") is True


def test_preflight_flags_a_read_only_token(monkeypatch, capsys):
    """A read token authenticates fine and then fails at create_repo."""
    _patch_api(monkeypatch, {"name": "darkdwine",
                             "auth": {"accessToken": {"role": "read"}}})
    assert push_to_hub.preflight("darkdwine/coll") is False
    assert "read-only" in capsys.readouterr().out


def test_preflight_survives_a_payload_without_auth_details(monkeypatch):
    """Older hub versions omit the token role; that is not an error."""
    _patch_api(monkeypatch, {"name": "darkdwine"})
    assert push_to_hub.preflight("darkdwine/coll") is True


def test_preflight_never_blocks_publishing(monkeypatch):
    """It is advisory: a False result must not stop a deliberate --yes."""
    import inspect
    source = inspect.getsource(push_to_hub.main)
    assert "preflight(args.repo_id)" in source
    assert "if not args.yes:" in source


def test_card_flags_regenerated_diagnostics(tmp_path):
    """A confusion matrix scored on a different dataset must say so."""
    run = next(iter(model_card.REGENERATED_PLOTS))
    make_run(tmp_path, run, imgsz=640)
    checkpoint = discover(tmp_path)[run]

    card = model_card.build_variant_card(checkpoint, "ns/coll", "02-yolo11n-640")

    assert "Diagnostics regenerated" in card
    assert "held-out split" in card


def test_card_stays_quiet_for_runs_with_original_plots(tmp_path):
    make_run(tmp_path, "some_normal_run", imgsz=1024)
    checkpoint = discover(tmp_path)["some_normal_run"]
    assert "Diagnostics regenerated" not in model_card.build_variant_card(
        checkpoint, "ns/coll", "x")


def test_regenerated_plots_names_a_real_published_run():
    published = {run for run, _ in push_to_hub.PUBLISH_ORDER}
    assert model_card.REGENERATED_PLOTS <= published


def test_ledger_path_override_actually_takes_effect(tmp_path, monkeypatch):
    """A default argument would bind LEDGER_PATH at import time, so overriding
    the module attribute would silently do nothing and tests would read the
    real ledger."""
    fake = tmp_path / "ledger.json"
    monkeypatch.setattr(push_to_hub, "LEDGER_PATH", fake)

    assert push_to_hub.load_ledger() == {}
    push_to_hub.record_published("run_a", {"subfolder": "01-x", "imgsz": 640})

    assert fake.is_file(), "the override must be where the write landed"
    assert set(push_to_hub.load_ledger()) == {"run_a"}


def test_notes_come_from_the_current_table_not_the_ledger(tmp_path, monkeypatch):
    """Editing a note must reach a variant published weeks ago."""
    run = "yolo11s_doc_layout_imgsz_1024"
    stale = {run: {"subfolder": "12-yolo11s-1024", "imgsz": 1024,
                   "metrics": {}, "status": "something written long ago"}}

    entry = push_to_hub.index_entries(stale)[0]

    assert entry["status"] == push_to_hub.status_for(run)
    assert entry["status"] != "something written long ago"


def test_a_run_dropped_from_the_table_keeps_its_stored_note():
    stale = {"run_no_longer_listed": {"subfolder": "99-x", "imgsz": 640,
                                      "metrics": {}, "status": "kept"}}
    assert push_to_hub.index_entries(stale)[0]["status"] == "kept"


def test_notes_stay_short_enough_not_to_widen_the_table():
    """Sentences here pushed the table past the page and hid the column behind
    a horizontal scrollbar, which defeated its purpose."""
    for run, _ in push_to_hub.PUBLISH_ORDER:
        note = push_to_hub.status_for(run)
        assert len(note) <= 32, f"{run}: note too long for the column: {note!r}"


def test_a_removed_note_is_not_resurrected_from_the_ledger():
    """Most rows are deliberately blank; truthiness fallback would undo that."""
    run = "yolo11_doc_layout_v222_round03_imgsz_1024"
    assert push_to_hub.status_for(run) == "", "this row is intentionally blank"

    stale = {run: {"subfolder": "09-yolo11n-1024", "imgsz": 1024,
                   "metrics": {}, "status": "Chained lineage"}}

    assert push_to_hub.index_entries(stale)[0]["status"] == ""
