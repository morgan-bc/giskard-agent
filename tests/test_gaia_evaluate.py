"""Tests for the GAIA evaluation script."""

import json

import evaluate_gaia


def _row(task_id: str, prediction: str, ground_truth: str, level: int, error: str | None = None):
    return {
        "task_id": task_id,
        "prediction": prediction,
        "ground_truth": ground_truth,
        "level": level,
        "error": error,
    }


def test_score_results_scores_each_row():
    rows = [
        _row("a", "41", "41", 1),
        _row("b", "crow", "seagull", 2),
        _row("c", "", "41", 1, error="timeout"),
    ]
    scored = evaluate_gaia.score_results(rows)
    assert scored[0]["score"] is True
    assert scored[0]["score_detail"] == "number"
    assert scored[1]["score"] is False
    assert scored[2]["score"] is False  # empty prediction never scores


def test_write_scored_roundtrip(tmp_path):
    rows = [_row("a", "41", "41", 1)]
    out = evaluate_gaia.write_scored(evaluate_gaia.score_results(rows), tmp_path)
    assert out == tmp_path / "scored.jsonl"
    loaded = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert loaded[0]["score"] is True


def test_format_summary_includes_breakdowns():
    rows = [
        _row("a", "41", "41", 1),
        _row("b", "x", "seagull", 1, error="timeout"),
        _row("c", "17", "17", 2),
    ]
    summary = evaluate_gaia.format_summary(evaluate_gaia.score_results(rows))
    assert "Level 1" in summary
    assert "Level 2" in summary
    assert "Errors" in summary
    assert "Wrong answers" in summary


def test_load_results_dedups_and_skips_malformed(tmp_path, capsys):
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        json.dumps(_row("a", "wrong", "41", 1))
        + "\n{broken\n"
        + json.dumps(_row("a", "41", "41", 1))
        + "\n",
        encoding="utf-8",
    )
    rows = evaluate_gaia.load_results(results_dir)
    assert len(rows) == 1
    assert rows[0]["prediction"] == "41"  # last row wins
    out = capsys.readouterr().out
    assert "malformed" in out
    assert "duplicate" in out


def test_missing_dataset_rows_warning(tmp_path, capsys):
    dataset = tmp_path / "dev.json"
    dataset.write_text(json.dumps([{"task_id": "a"}, {"task_id": "zzz"}]), encoding="utf-8")
    results_dir = tmp_path / "run"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        json.dumps(_row("a", "41", "41", 1)) + "\n", encoding="utf-8"
    )
    evaluate_gaia.main_with(
        results_dir=results_dir, dataset=dataset
    )
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
