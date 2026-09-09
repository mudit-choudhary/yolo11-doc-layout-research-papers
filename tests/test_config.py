"""Tests for shared configuration and path resolution."""

from __future__ import annotations

import pytest

from doclayout_ft import config


def test_class_names_match_the_yolo_label_contract():
    """Class order is load-bearing: YOLO labels reference classes by index."""
    assert len(config.CLASS_NAMES) == 12
    assert config.CLASS_NAMES[0] == "Caption"
    assert config.CLASS_NAMES[11] == "Authors"
    assert config.CLASS_ID_TO_NAME[8] == "Table"
    assert len(set(config.CLASS_NAMES)) == len(config.CLASS_NAMES)


def test_class_names_agree_with_the_shipped_classes_txt():
    """classes.txt and CLASS_NAMES must not drift apart.

    They are two copies of the same taxonomy: one read by the annotation tool,
    one by this package. If they disagree, annotations get written against one
    ordering and trained against another.
    """
    classes_txt = config.TRAINING_DIR / config.DEFAULT_ROUND / "classes.txt"
    if not classes_txt.is_file():
        pytest.skip(f"{classes_txt} is not present in this checkout")
    on_disk = tuple(classes_txt.read_text().split())
    assert on_disk == config.CLASS_NAMES


@pytest.mark.parametrize(
    ("given", "expected"),
    [("0", 0), ("1", 1), (0, 0), ("cpu", "cpu"), (" cpu ", "cpu"), ("0,1", "0,1")],
)
def test_resolve_device_normalises_cli_input(given, expected):
    assert config.resolve_device(given) == expected


def test_round_dir_names_the_alternatives_when_the_round_is_missing():
    """A typo'd round name is the common failure; the message should help."""
    with pytest.raises(FileNotFoundError, match="Available rounds"):
        config.round_dir("round_that_does_not_exist")


def test_is_truthy_env(monkeypatch):
    monkeypatch.setenv("DOCLAYOUT_TEST_FLAG", "yes")
    assert config.is_truthy_env("DOCLAYOUT_TEST_FLAG")
    monkeypatch.setenv("DOCLAYOUT_TEST_FLAG", "0")
    assert not config.is_truthy_env("DOCLAYOUT_TEST_FLAG")
    monkeypatch.delenv("DOCLAYOUT_TEST_FLAG")
    assert not config.is_truthy_env("DOCLAYOUT_TEST_FLAG")
