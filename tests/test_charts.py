"""Tests for the model-card charts.

These check the things that make a chart wrong rather than ugly: reading the
right column, covering the right rows, and failing usefully when their inputs
are missing. Appearance is checked by looking at the rendered PNG.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from doclayout_ft.hub import charts


@pytest.fixture
def reports(tmp_path, monkeypatch):
    """Point the chart builders at a synthetic reports directory."""
    monkeypatch.setattr(charts, "REPORTS_DIR", tmp_path)
    return tmp_path


def write_eval(path: Path, rows: list[tuple[str, float]]) -> None:
    with (path / "evaluation_val.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "baseline", "imgsz", "precision", "recall",
                         "mAP50", "mAP50-95", "fitness"])
        for name, score in rows:
            writer.writerow([name, "False", 1024, 0.9, 0.9, 0.9, score, score])


def test_comparison_rows_are_sorted_best_first(reports):
    from doclayout_ft.hub.push_to_hub import PUBLISH_ORDER
    runs = [r for r, _ in PUBLISH_ORDER][:4]
    write_eval(reports, list(zip(runs, [0.5, 0.9, 0.7, 0.6])))

    rows = charts.load_comparison_rows()

    assert [r["score"] for r in rows] == [0.9, 0.7, 0.6, 0.5]


def test_comparison_rows_exclude_duplicates(reports):
    from doclayout_ft.hub.push_to_hub import DUPLICATE_OF, PUBLISH_ORDER
    alias = next(iter(DUPLICATE_OF))
    runs = [r for r, _ in PUBLISH_ORDER][:3] + [alias]
    write_eval(reports, [(r, 0.7) for r in runs])

    names = {r["run"] for r in charts.load_comparison_rows()}

    assert alias not in names, "a duplicate must not appear twice in the chart"


def test_lineage_classes_are_assigned_from_the_notes(reports):
    write_eval(reports, [
        ("yolo11s_doc_layout_imgsz_1024", 0.77),        # recommended, single
        ("yolo11_doc_layout_v2224_imgsz_1024", 0.78),   # 3 fine-tunes deep
        ("yolo11s_doc_layout_attempt_02", 0.75),        # failed experiment
    ])
    by_run = {r["run"]: r for r in charts.load_comparison_rows()}

    assert by_run["yolo11s_doc_layout_imgsz_1024"]["lineage"] == "single"
    assert by_run["yolo11s_doc_layout_imgsz_1024"]["recommended"]
    assert by_run["yolo11_doc_layout_v2224_imgsz_1024"]["lineage"] == "deep"
    assert not by_run["yolo11_doc_layout_v2224_imgsz_1024"]["recommended"]
    assert by_run["yolo11s_doc_layout_attempt_02"]["lineage"] == "failed"


def test_unscored_runs_are_skipped_not_plotted_as_zero(reports):
    """A model with no score must be absent, not shown as a zero-length bar."""
    write_eval(reports, [("yolo11s_doc_layout_imgsz_1024", 0.77)])
    rows = charts.load_comparison_rows()
    assert [r["run"] for r in rows] == ["yolo11s_doc_layout_imgsz_1024"]


def test_a_malformed_score_is_skipped(reports):
    with (reports / "evaluation_val.csv").open("w", newline="") as handle:
        handle.write("model,mAP50-95\nyolo11s_doc_layout_imgsz_1024,\n")
    assert charts.load_comparison_rows() == []


def test_missing_evaluation_table_names_the_fix(reports):
    with pytest.raises(FileNotFoundError, match="evaluate"):
        charts.load_comparison_rows()


def test_comparison_chart_renders_a_png(reports, tmp_path):
    write_eval(reports, [("yolo11s_doc_layout_imgsz_1024", 0.77),
                         ("yolo11n_doc_layout_imgsz_1024", 0.76)])
    out = charts.build_comparison_chart(tmp_path / "c.png")
    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_per_class_chart_renders_a_png(tmp_path):
    csv_path = tmp_path / "per_class.csv"
    csv_path.write_text(
        "class,precision,recall,mAP50,mAP50-95\n"
        "Table,1.0,0.93,0.99,0.985\n"
        "Page-footer,0.93,0.91,0.86,0.360\n"
        "ALL,0.93,0.91,0.94,0.769\n")

    out = charts.build_per_class_chart(tmp_path / "p.png", csv_path=csv_path)

    assert out.is_file()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_per_class_chart_needs_its_input(tmp_path):
    with pytest.raises(FileNotFoundError, match="per_class"):
        charts.build_per_class_chart(tmp_path / "p.png",
                                     csv_path=tmp_path / "absent.csv")


def test_charts_are_opaque(reports, tmp_path):
    """A model card renders on a theme the reader controls, and a PNG cannot
    adapt, so a transparent background would be illegible in one of them."""
    from PIL import Image
    write_eval(reports, [("yolo11s_doc_layout_imgsz_1024", 0.77)])
    out = charts.build_comparison_chart(tmp_path / "c.png")

    image = Image.open(out)
    assert image.mode in ("RGB", "P") or (
        image.mode == "RGBA"
        and image.convert("RGBA").getchannel("A").getextrema()[0] == 255)


def test_the_card_references_both_charts():
    from doclayout_ft.hub.model_card import build_index_card
    card = build_index_card("ns/coll", [
        {"name": "a", "subfolder": "12-yolo11s-1024", "imgsz": 1024,
         "metrics": {"mAP50-95": "0.7694"}},
    ])
    assert "](comparison.png)" in card
    assert "](per-class.png)" in card
    assert card.count("![") >= 2, "both images need alt text"
