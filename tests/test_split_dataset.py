"""Tests for the paper-level train/val/test split.

The property that matters is leakage: no paper may appear in two splits. These
tests build a small synthetic round on disk rather than mocking, because the
splitter's job is largely about what it writes to disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doclayout_ft.data import split_dataset


def make_round(tmp_path: Path, papers: int, pages_per_paper: int) -> Path:
    """Create a synthetic dataset round with images, labels and a data.yaml."""
    round_dir = tmp_path / "round_test"
    images = round_dir / "images"
    labels = round_dir / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    for paper in range(papers):
        for page in range(1, pages_per_paper + 1):
            stem = f"Paper_{paper:03d}_page_{page:02d}"
            (images / f"{stem}.jpg").write_bytes(b"")
            (labels / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")

    (round_dir / "data.yaml").write_text(
        "path: /stale/path/that/should/be/rewritten\n"
        "train: images\n"
        "val: images\n"
        "\nnames:\n  0: Caption\n"
    )
    return round_dir


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Attention_Is_All_You_Need_page_07.jpg", "Attention_Is_All_You_Need"),
        ("Paper_page_01.jpg", "Paper"),
        ("Paper_page_123.jpg", "Paper"),
        ("no_page_suffix.jpg", "no_page_suffix"),
        ("Talk_about_page_layout.jpg", "Talk_about_page_layout"),
    ],
)
def test_paper_key_strips_only_a_real_page_suffix(filename, expected):
    assert split_dataset.paper_key(Path(filename)) == expected


def test_no_paper_appears_in_two_splits(tmp_path):
    """The whole point of splitting by paper rather than by page."""
    round_dir = make_round(tmp_path, papers=40, pages_per_paper=4)
    images = sorted((round_dir / "images").iterdir())
    groups = split_dataset.group_by_paper(images)
    splits = split_dataset.assign_splits(list(groups), 0.75, 0.15, seed=42)

    seen: set[str] = set()
    for keys in splits.values():
        overlap = seen & set(keys)
        assert not overlap, f"papers in more than one split: {overlap}"
        seen |= set(keys)
    assert seen == set(groups)


def test_every_paper_lands_somewhere(tmp_path):
    """Test takes the remainder, so no paper may be dropped by rounding."""
    round_dir = make_round(tmp_path, papers=37, pages_per_paper=3)
    groups = split_dataset.group_by_paper(sorted((round_dir / "images").iterdir()))
    splits = split_dataset.assign_splits(list(groups), 0.75, 0.15, seed=1)
    assert sum(len(keys) for keys in splits.values()) == len(groups)


def test_split_is_reproducible_for_a_fixed_seed(tmp_path):
    round_dir = make_round(tmp_path, papers=25, pages_per_paper=2)
    groups = list(split_dataset.group_by_paper(sorted((round_dir / "images").iterdir())))
    first = split_dataset.assign_splits(groups, 0.75, 0.15, seed=7)
    second = split_dataset.assign_splits(groups, 0.75, 0.15, seed=7)
    assert first == second
    different = split_dataset.assign_splits(groups, 0.75, 0.15, seed=8)
    assert different != first


def test_missing_labels_are_detected(tmp_path):
    round_dir = make_round(tmp_path, papers=5, pages_per_paper=2)
    images = sorted((round_dir / "images").iterdir())
    (round_dir / "labels" / f"{images[0].stem}.txt").unlink()
    missing = split_dataset.find_missing_labels(images, round_dir / "labels")
    assert [p.name for p in missing] == [images[0].name]


def test_main_writes_absolute_paths_and_repoints_data_yaml(tmp_path, monkeypatch):
    """Absolute paths, because Ultralytics resolves relative list entries
    against the working directory rather than the dataset directory."""
    round_dir = make_round(tmp_path, papers=20, pages_per_paper=3)
    monkeypatch.setattr(split_dataset, "TRAINING_DIR", tmp_path)

    assert split_dataset.main(["--round", "round_test", "--seed", "42"]) == 0

    train_lines = (round_dir / "train.txt").read_text().split()
    assert train_lines, "train.txt should not be empty"
    assert all(Path(line).is_absolute() for line in train_lines)
    assert all(Path(line).exists() for line in train_lines)

    yaml_text = (round_dir / "data.yaml").read_text()
    assert "train: train.txt" in yaml_text
    assert "val: val.txt" in yaml_text
    assert "test: test.txt" in yaml_text
    assert f"path: {round_dir.resolve()}" in yaml_text
    assert "/stale/path" not in yaml_text
    assert "names:" in yaml_text, "the hand-maintained names block must survive"


def test_main_rejects_fractions_that_do_not_sum_to_one(tmp_path, monkeypatch):
    make_round(tmp_path, papers=5, pages_per_paper=1)
    monkeypatch.setattr(split_dataset, "TRAINING_DIR", tmp_path)
    assert split_dataset.main(
        ["--round", "round_test", "--train", "0.9", "--val", "0.5", "--test", "0.1"]
    ) == 1


def test_dry_run_writes_nothing(tmp_path, monkeypatch):
    round_dir = make_round(tmp_path, papers=10, pages_per_paper=2)
    monkeypatch.setattr(split_dataset, "TRAINING_DIR", tmp_path)
    original = (round_dir / "data.yaml").read_text()

    assert split_dataset.main(["--round", "round_test", "--dry-run"]) == 0

    assert not (round_dir / "train.txt").exists()
    assert (round_dir / "data.yaml").read_text() == original
