"""Score a GAIA run directory with the official question scorer.

Reads ``results.jsonl`` from the run dir, scores every row, writes
``scored.jsonl`` (idempotent — reruns overwrite it), and prints a summary:
overall accuracy, per-level accuracy, error stats, and the wrong-answer list.

Usage:
    PYTHONPATH=src python benchmarks/gaia/evaluate_gaia.py --results-dir benchmarks/gaia/runs/<ts>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from gaia_scorer import question_scorer


def load_results(results_dir: Path) -> list[dict]:
    """Load result rows from ``results.jsonl``."""
    results_path = results_dir / "results.jsonl"
    rows = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def score_results(rows: list[dict]) -> list[dict]:
    """Add ``score`` and ``score_detail`` to every row."""
    scored = []
    for row in rows:
        score, detail = question_scorer(row.get("prediction") or "", row["ground_truth"])
        scored.append({**row, "score": score, "score_detail": detail})
    return scored


def write_scored(scored: list[dict], results_dir: Path) -> Path:
    """Write scored rows to ``scored.jsonl`` (overwrites)."""
    out = results_dir / "scored.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out


def format_summary(scored: list[dict]) -> str:
    """Render the human-readable summary report."""
    total = len(scored)
    correct = sum(1 for r in scored if r["score"])
    if not total:
        return "No results."
    lines = [f"Accuracy: {correct}/{total} = {correct / total:.1%}"]

    by_level: dict[int, list[dict]] = defaultdict(list)
    for row in scored:
        by_level[row["level"]].append(row)
    for level in sorted(by_level):
        rows = by_level[level]
        level_correct = sum(1 for r in rows if r["score"])
        lines.append(f"  Level {level}: {level_correct}/{len(rows)} = {level_correct / len(rows):.1%}")

    errors = Counter(row["error"] for row in scored if row.get("error"))
    if errors:
        lines.append(f"Errors: {dict(errors)}")

    wrong = [row for row in scored if not row["score"]]
    if wrong:
        lines.append("Wrong answers:")
        for row in wrong:
            lines.append(f"  - [{row['level']}] {row['task_id']} error={row.get('error')!r}")
            lines.append(f"      pred={row.get('prediction')!r}")
            lines.append(f"      gt  ={row['ground_truth']!r}")
    return "\n".join(lines)


def main_with(*, results_dir: Path, dataset: Path | None = None) -> None:
    """Score the run directory and print the summary (testable entry point)."""
    rows = load_results(results_dir)
    if dataset is not None:
        dataset_ids = {t["task_id"] for t in json.loads(dataset.read_text(encoding="utf-8"))}
        missing = dataset_ids - {row["task_id"] for row in rows}
        if missing:
            print(f"WARNING: {len(missing)} dataset tasks have no result row, e.g. {sorted(missing)[:3]}")
    scored = score_results(rows)
    out = write_scored(scored, results_dir)
    print(format_summary(scored))
    print(f"\nScored results written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a GAIA run directory.")
    parser.add_argument("--results-dir", type=Path, required=True, help="Run directory containing results.jsonl")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset JSON for missing-task cross-check")
    args = parser.parse_args()
    main_with(results_dir=args.results_dir, dataset=args.dataset)


if __name__ == "__main__":
    main()
