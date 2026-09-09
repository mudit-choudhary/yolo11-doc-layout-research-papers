"""Tests for run-directory health auditing.

A sweep that loses runs loses them silently: Ultralytics creates the directory
at startup, so a run that dies in epoch 1 leaves a plausible-looking folder
with no weights, and checkpoint discovery just skips it. These tests cover the
classification that makes that visible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doclayout_ft import audit


def make_complete(root: Path, name: str, epochs: int = 100) -> Path:
    run = root / name
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"w")
    for artefact in audit.EXPECTED_ARTIFACTS:
        (run / artefact).write_bytes(b"x")
    (run / "results.csv").write_text(
        "epoch\n" + "".join(f"{i}\n" for i in range(1, epochs + 1)))
    return run


def make_dead(root: Path, name: str, **args) -> Path:
    """A run that died before writing weights: plots only, no results.csv."""
    run = root / name
    run.mkdir(parents=True)
    (run / "labels.jpg").write_bytes(b"x")
    (run / "train_batch0.jpg").write_bytes(b"x")
    if args:
        lines = "\n".join(f"{k}: {v}" for k, v in args.items())
        (run / "args.yaml").write_text(lines + "\n")
    return run


def test_complete_run_is_reported_complete(tmp_path):
    make_complete(tmp_path, "good")
    health = audit.inspect_run(tmp_path / "good")
    assert health.status == "complete"
    assert health.missing == []
    assert health.epochs == 100


def test_missing_plots_make_a_run_incomplete(tmp_path):
    run = make_complete(tmp_path, "partial")
    (run / "confusion_matrix.png").unlink()
    (run / "results.png").unlink()

    health = audit.inspect_run(run)

    assert health.status == "incomplete"
    assert set(health.missing) == {"confusion_matrix.png", "results.png"}
    assert health.epochs == 100, "weights and metrics are still intact"


def test_run_without_weights_is_dead(tmp_path):
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.5)
    health = audit.inspect_run(tmp_path / "died")
    assert health.status == "dead"
    assert health.epochs is None


def test_directory_of_loose_checkpoints_is_a_base_dir(tmp_path):
    run = tmp_path / "yolov11s"
    run.mkdir()
    (run / "yolo11s_doc_layout.pt").write_bytes(b"w")
    health = audit.inspect_run(run)
    assert health.status == "base"
    assert "yolo11s_doc_layout.pt" in health.detail


def test_oom_from_multi_scale_is_diagnosed(tmp_path):
    """max_imgsz = imgsz * (1 + multi_scale); 0.5 at 1024 means 1536px."""
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.5)
    diagnosis = audit.diagnose_dead_run(tmp_path / "died")
    assert "multi_scale" in diagnosis
    assert "1536" in diagnosis
    assert "2.25" in diagnosis


def test_oom_from_batch_size_is_diagnosed(tmp_path):
    make_dead(tmp_path, "died", imgsz=1024, batch=5, multi_scale=0.0)
    diagnosis = audit.diagnose_dead_run(tmp_path / "died")
    assert "batch=5" in diagnosis


def test_a_modest_multi_scale_is_not_blamed(tmp_path):
    """0.25 survived in practice, so it must not be reported as the cause."""
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.25)
    diagnosis = audit.diagnose_dead_run(tmp_path / "died")
    assert "multi_scale" not in diagnosis
    assert "no obvious cause" in diagnosis


def test_missing_args_yaml_is_admitted_not_guessed(tmp_path):
    make_dead(tmp_path, "died")
    assert "unrecoverable" in audit.diagnose_dead_run(tmp_path / "died")


def test_malformed_args_yaml_does_not_raise(tmp_path):
    run = make_dead(tmp_path, "died")
    (run / "args.yaml").write_text("imgsz: [broken: yaml\n")
    assert audit.diagnose_dead_run(run)


def test_audit_walks_several_roots(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    make_complete(a, "one")
    make_dead(b, "two", imgsz=1024, batch=8, multi_scale=0.5)

    results = audit.audit([a, b])

    assert [r.status for r in results[a]] == ["complete"]
    assert [r.status for r in results[b]] == ["dead"]


def test_audit_raises_on_a_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        audit.audit([tmp_path / "absent"])


def test_incomplete_runs_fail_the_check(tmp_path):
    """Incomplete is actionable: the plots can be regenerated."""
    run = make_complete(tmp_path, "partial")
    (run / "confusion_matrix.png").unlink()
    assert audit.main(["--models-dir", str(tmp_path)]) == 1


def test_dead_runs_do_not_fail_the_check_by_default(tmp_path):
    """A run that OOM'd once is history; failing on it forever is noise."""
    make_complete(tmp_path, "good")
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.5)
    assert audit.main(["--models-dir", str(tmp_path)]) == 0


def test_strict_fails_on_dead_runs(tmp_path):
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.5)
    assert audit.main(["--models-dir", str(tmp_path), "--strict"]) == 1


def test_a_healthy_tree_passes(tmp_path):
    make_complete(tmp_path, "good")
    assert audit.main(["--models-dir", str(tmp_path)]) == 0
    assert audit.main(["--models-dir", str(tmp_path), "--strict"]) == 0


def test_advice_matches_what_was_actually_found(tmp_path, capsys):
    """Do not explain incomplete runs when there are none."""
    make_complete(tmp_path, "good")
    make_dead(tmp_path, "died", imgsz=1024, batch=3, multi_scale=0.5)

    audit.main(["--models-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert "dead run(s), for information" in out
    assert "need attention" not in out, "no incomplete runs, so no such advice"
    assert "Regenerate its plots" not in out


def test_incomplete_advice_appears_only_for_incomplete_runs(tmp_path, capsys):
    run = make_complete(tmp_path, "partial")
    (run / "results.png").unlink()

    audit.main(["--models-dir", str(tmp_path)])
    out = capsys.readouterr().out

    assert "need attention" in out
    assert "Regenerate its plots" in out
    assert "dead run(s)" not in out


def test_healthy_tree_says_so_and_offers_no_advice(tmp_path, capsys):
    make_complete(tmp_path, "good")
    audit.main(["--models-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "Every run is complete." in out
    assert "need attention" not in out
    assert "dead run(s)" not in out


def test_base_checkpoint_dirs_do_not_count_as_problems(tmp_path):
    make_complete(tmp_path, "good")
    base = tmp_path / "yolov11s"
    base.mkdir()
    (base / "yolo11s_doc_layout.pt").write_bytes(b"w")
    assert audit.main(["--models-dir", str(tmp_path)]) == 0
