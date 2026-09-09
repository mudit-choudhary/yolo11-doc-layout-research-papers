"""Tests for run and checkpoint discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from doclayout_ft import checkpoints
from doclayout_ft.config import BASELINE_IMGSZ


def make_run(models_dir: Path, name: str, imgsz: int | None = None) -> Path:
    """Create a finished training run: weights/best.pt plus optional args.yaml."""
    run_dir = models_dir / name
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "best.pt").write_bytes(b"weights")
    if imgsz is not None:
        (run_dir / "args.yaml").write_text(f"imgsz: {imgsz}\nepochs: 100\n")
    return run_dir


def make_baseline(models_dir: Path, dir_name: str, file_name: str) -> Path:
    """Create a directory holding a loose, never-fine-tuned checkpoint."""
    run_dir = models_dir / dir_name
    run_dir.mkdir(parents=True)
    (run_dir / file_name).write_bytes(b"weights")
    return run_dir


def test_discovers_finished_runs_and_reads_their_imgsz(tmp_path):
    make_run(tmp_path, "run_a", imgsz=1024)
    make_run(tmp_path, "run_b", imgsz=640)

    found = checkpoints.discover(tmp_path)

    assert set(found) == {"run_a", "run_b"}
    assert found["run_a"].imgsz == 1024
    assert found["run_b"].imgsz == 640
    assert not found["run_a"].is_baseline


def test_run_without_args_yaml_falls_back_to_the_baseline_imgsz(tmp_path):
    make_run(tmp_path, "run_no_args", imgsz=None)
    assert checkpoints.discover(tmp_path)["run_no_args"].imgsz == BASELINE_IMGSZ


def test_malformed_args_yaml_does_not_stop_discovery(tmp_path):
    """One unreadable run should not sink a sweep across twenty of them."""
    run_dir = make_run(tmp_path, "run_bad", imgsz=None)
    (run_dir / "args.yaml").write_text("imgsz: [this: is not: valid\n")
    assert checkpoints.discover(tmp_path)["run_bad"].imgsz == BASELINE_IMGSZ


def test_non_numeric_imgsz_falls_back(tmp_path):
    run_dir = make_run(tmp_path, "run_weird", imgsz=None)
    (run_dir / "args.yaml").write_text("imgsz: null\n")
    assert checkpoints.discover(tmp_path)["run_weird"].imgsz == BASELINE_IMGSZ


def test_baselines_are_discovered_and_suffixed(tmp_path):
    make_baseline(tmp_path, "yolov11s", "yolo11s_doc_layout.pt")

    found = checkpoints.discover(tmp_path, baseline_suffix="_baseline")

    assert "yolo11s_doc_layout_baseline" in found
    assert found["yolo11s_doc_layout_baseline"].is_baseline
    assert found["yolo11s_doc_layout_baseline"].imgsz == BASELINE_IMGSZ


def test_baselines_can_be_excluded(tmp_path):
    make_run(tmp_path, "run_a", imgsz=1024)
    make_baseline(tmp_path, "yolov11s", "yolo11s_doc_layout.pt")
    found = checkpoints.discover(tmp_path, include_baselines=False)
    assert set(found) == {"run_a"}


def test_exclude_suffix_keeps_a_sweeps_own_output_out_of_its_input(tmp_path):
    """Without this, re-running a sweep would fine-tune its own outputs."""
    make_run(tmp_path, "run_a", imgsz=1024)
    make_run(tmp_path, "run_a_ft", imgsz=1024)

    found = checkpoints.discover(tmp_path, exclude_suffix="ft")

    assert set(found) == {"run_a"}


def test_loose_weights_beside_a_finished_run_are_training_inputs_not_models(tmp_path):
    """A directory is either a finished run or a bag of base weights."""
    run_dir = make_run(tmp_path, "run_a", imgsz=1024)
    (run_dir / "yolo11n.pt").write_bytes(b"input weights")

    found = checkpoints.discover(tmp_path)

    assert set(found) == {"run_a"}


def test_missing_models_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoints.discover(tmp_path / "nope")


def test_filter_by_name_rejects_a_typo_loudly(tmp_path):
    """Silently evaluating nothing because of a typo wastes a whole sweep."""
    make_run(tmp_path, "run_a", imgsz=1024)
    found = checkpoints.discover(tmp_path)

    assert set(checkpoints.filter_by_name(found, ["run_a"])) == {"run_a"}
    assert set(checkpoints.filter_by_name(found, None)) == {"run_a"}
    with pytest.raises(KeyError, match="run_typo"):
        checkpoints.filter_by_name(found, ["run_typo"])


def test_modified_at_is_timezone_aware(tmp_path):
    make_run(tmp_path, "run_a", imgsz=640)
    stamp = checkpoints.discover(tmp_path)["run_a"].modified_at
    assert stamp.tzinfo is not None


def test_discover_many_merges_directories_with_the_first_winning(tmp_path):
    """models/ and FinetunedModels/ share seven run names; one must win."""
    curated = tmp_path / "curated"
    working = tmp_path / "working"
    make_run(curated, "shared", imgsz=1024)
    make_run(working, "shared", imgsz=640)
    make_run(working, "working_only", imgsz=640)

    found = checkpoints.discover_many([curated, working])

    assert set(found) == {"shared", "working_only"}
    assert found["shared"].imgsz == 1024, "the first directory listed wins"
    assert found["shared"].run_dir.parent.name == "curated"


def test_discover_many_returns_names_sorted(tmp_path):
    models = tmp_path / "m"
    for name in ("zeta", "alpha", "mid"):
        make_run(models, name, imgsz=640)
    assert list(checkpoints.discover_many([models])) == ["alpha", "mid", "zeta"]


def test_discover_many_raises_on_a_missing_directory(tmp_path):
    make_run(tmp_path / "real", "run_a", imgsz=640)
    with pytest.raises(FileNotFoundError):
        checkpoints.discover_many([tmp_path / "real", tmp_path / "absent"])
