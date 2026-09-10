"""Tests for the latency benchmark.

The measurement itself needs a GPU and real weights, so what is tested here is
the surrounding logic: sampling stability, page loading, and that the reported
columns are the ones the charts read.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doclayout_ft.evaluation import benchmark


def make_round(tmp_path: Path, pages: int) -> Path:
    round_dir = tmp_path / "round_test" / "images"
    round_dir.mkdir(parents=True)
    for index in range(pages):
        (round_dir / f"Paper_{index:03d}_page_01.jpg").write_bytes(b"")
    return round_dir


def test_sampling_is_stable_across_runs(tmp_path, monkeypatch):
    """Two benchmark runs must time the same pages to be comparable."""
    make_round(tmp_path, 20)
    monkeypatch.setattr(benchmark, "round_dir", lambda name: tmp_path / name)

    first = benchmark.sample_images("round_test", 5)
    second = benchmark.sample_images("round_test", 5)

    assert first == second
    assert len(first) == 5


def test_sampling_caps_at_what_exists(tmp_path, monkeypatch):
    make_round(tmp_path, 3)
    monkeypatch.setattr(benchmark, "round_dir", lambda name: tmp_path / name)
    assert len(benchmark.sample_images("round_test", 40)) == 3


def test_sampling_reports_an_empty_round(tmp_path, monkeypatch):
    (tmp_path / "round_test" / "images").mkdir(parents=True)
    monkeypatch.setattr(benchmark, "round_dir", lambda name: tmp_path / name)
    with pytest.raises(FileNotFoundError, match="No images"):
        benchmark.sample_images("round_test", 5)


def test_undecodable_page_is_reported(tmp_path):
    bad = tmp_path / "not-an-image.jpg"
    bad.write_bytes(b"nonsense")
    with pytest.raises(FileNotFoundError, match="decode"):
        benchmark.load_pages([bad])


def test_csv_columns_match_what_the_charts_read():
    """The charts index into these by name; a rename would break them."""
    for column in ("model", "median_ms", "fps", "params_m"):
        assert column in benchmark.CSV_FIELDS


def test_warmup_default_is_not_zero():
    """The first CUDA inferences pay for autotuning nobody ever waits for
    again, so timing them would report a latency that never recurs."""
    assert benchmark.DEFAULT_WARMUP >= 1
