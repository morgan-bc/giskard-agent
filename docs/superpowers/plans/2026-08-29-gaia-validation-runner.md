# GAIA Validation Runner & Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the GAIA validation set (103 tasks, `benchmarks/gaia/validation/dev.json`) through the current giskard harness task-by-task, and score the outputs with the official GAIA `question_scorer`.

**Architecture:** Three standalone scripts under `benchmarks/gaia/`: a pure-function scorer module (official GAIA scorer port, zero third-party deps), a sequential runner (one shared agent instance, per-task fresh session, JSONL incremental persistence + resume, per-task timeout + assistant-turn cap), and an evaluator (reads results.jsonl, writes scored.jsonl, prints summary). Config comes from `.env` (`LLM_API_KEY` / `LLM_BASE_URL` / `BASE_MODEL`) via python-dotenv, matching `examples/yolo_agent.py`.

**Tech Stack:** Python 3.10+, giskard harness (`create_harness_agent`, `ParallelSearchClient`, `OpenAIChatCompletionClient`), pytest (config in `pyproject.toml`, `asyncio_mode=auto`).

**Spec:** `docs/superpowers/specs/2026-08-29-gaia-validation-runner-design.md`

**Important constraints:**
- Never `git add benchmarks/` wholesale — the dataset `benchmarks/gaia/validation/dev.json` is untracked on purpose; only add the specific files named in each commit step.
- `tests/` has no `conftest.py` yet; Task 1 creates it so tests can import the benchmark scripts.
- Run everything from the repo root `d:\Projects\giskard`.

---

### Task 0: Commit spec amendment + this plan

**Files:**
- Modify: `docs/superpowers/specs/2026-08-29-gaia-validation-runner-design.md` (already amended: official scorer list-branch requires equal length + all elements match; string normalization removes all whitespace + punctuation + lowercases, NO article removal)

- [ ] **Step 1: Commit docs**

```bash
git add docs/superpowers/specs/2026-08-29-gaia-validation-runner-design.md docs/superpowers/plans/2026-08-29-gaia-validation-runner.md
git commit -m "docs(benchmarks): amend gaia spec to official scorer semantics, add implementation plan"
```

---

### Task 1: `gaia_scorer.py` — official scorer port (TDD)

**Files:**
- Create: `benchmarks/gaia/gaia_scorer.py`
- Create: `tests/conftest.py`
- Create: `tests/test_gaia_scorer.py`

- [ ] **Step 1: Create `tests/conftest.py`** — makes benchmark scripts importable from tests:

```python
"""Shared test setup."""

import sys
from pathlib import Path

# Make benchmark scripts importable from tests (gaia_scorer, evaluate_gaia).
_BENCH_GAIA = str(Path(__file__).resolve().parents[1] / "benchmarks" / "gaia")
if _BENCH_GAIA not in sys.path:
    sys.path.insert(0, _BENCH_GAIA)
```

- [ ] **Step 2: Write the failing test `tests/test_gaia_scorer.py`**

```python
"""Tests for the GAIA official question scorer port."""

import math

import pytest

from gaia_scorer import _words_to_number, normalize_number_str, normalize_str, question_scorer


class TestWordsToNumber:
    def test_hyphenated_tens(self):
        assert _words_to_number("forty-one") == 41

    def test_hundreds(self):
        assert _words_to_number("one hundred twenty three") == 123

    def test_with_and(self):
        assert _words_to_number("one hundred and five") == 105

    def test_bare_hundred(self):
        assert _words_to_number("hundred") == 100

    def test_unknown_word_raises(self):
        with pytest.raises(ValueError):
            _words_to_number("banana")


class TestNormalizeNumberStr:
    def test_plain(self):
        assert normalize_number_str("41") == 41.0

    def test_commas(self):
        assert normalize_number_str("1,000") == 1000.0

    def test_dollar(self):
        assert normalize_number_str("$100") == 100.0

    def test_percent(self):
        assert normalize_number_str("50%") == 50.0

    def test_english_words(self):
        assert normalize_number_str("forty-one") == 41.0

    def test_garbage_is_inf(self):
        assert math.isinf(normalize_number_str("banana"))


class TestNormalizeStr:
    def test_lowercase(self):
        assert normalize_str("Seagull") == "seagull"

    def test_all_whitespace_removed(self):
        assert normalize_str("sea gull") == "seagull"

    def test_punct_removed(self):
        assert normalize_str("seagull.") == "seagull"

    def test_keep_punct(self):
        assert normalize_str("a.b", remove_punct=False) == "a.b"


class TestQuestionScorer:
    # number branch
    def test_number_exact(self):
        assert question_scorer("41", "41") == (True, "number")

    def test_number_dollar_comma(self):
        assert question_scorer("$1,000", "1000")[0] is True

    def test_number_wrong(self):
        assert question_scorer("40", "41") == (False, "number")

    def test_number_unrounded_differs(self):
        assert question_scorer("40.4", "41")[0] is False

    def test_number_garbage(self):
        assert question_scorer("forty two", "41")[0] is False

    # string branch
    def test_string_case_punct(self):
        assert question_scorer("seagull.", "seagull") == (True, "string")

    def test_string_whitespace_insensitive(self):
        assert question_scorer("sea gull", "seagull")[0] is True

    def test_string_mismatch(self):
        assert question_scorer("crow", "seagull")[0] is False

    def test_empty_prediction_string_branch(self):
        assert question_scorer("", "seagull")[0] is False

    # list branch
    def test_list_exact(self):
        assert question_scorer("34689, 33063", "34689,33063") == (True, "list")

    def test_list_length_mismatch(self):
        score, detail = question_scorer("34689", "34689,33063")
        assert score is False
        assert detail.startswith("list: length mismatch")

    def test_list_element_mismatch(self):
        assert question_scorer("34689, 33064", "34689,33063")[0] is False

    def test_list_mixed_numeric_string_elements(self):
        assert question_scorer("2, Barack Obama", "2,barack obama")[0] is True

    def test_semicolon_gt_comma_prediction(self):
        # gt uses ';', prediction uses ',' — both split on [;,]
        assert question_scorer("34689,33063", "34689;33063")[0] is True

    # None prediction must not crash
    def test_none_prediction(self):
        assert question_scorer(None, "seagull")[0] is False
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_gaia_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gaia_scorer'`

- [ ] **Step 4: Implement `benchmarks/gaia/gaia_scorer.py`**

```python
"""GAIA official question scorer port.

Ported from the official GAIA ``evaluation.py`` (``question_scorer``), with one
addition: when a numeric ground truth cannot be matched because the model answer
is an English number word (e.g. "forty-one"), a lightweight built-in 0-999
word-to-number converter is tried before giving up (returning ``inf``, which
always scores False). This keeps the module dependency-free.
"""

from __future__ import annotations

import re
import string

_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _words_to_number(text: str) -> int:
    """Convert an English number word (0-999, e.g. "forty-one") to an int.

    Raises:
        ValueError: If the text is not a supported English number word.
    """
    words = [w for w in re.split(r"[\s-]+", text.strip().lower()) if w and w != "and"]
    if not words:
        raise ValueError(f"not a number word: {text!r}")
    total = 0
    pending = 0  # accumulates ones/tens before a "hundred"
    for word in words:
        if word in _ONES:
            pending += _ONES[word]
        elif word in _TENS:
            pending += _TENS[word]
        elif word == "hundred":
            total += max(pending, 1) * 100
            pending = 0
        else:
            raise ValueError(f"unknown number word {word!r} in {text!r}")
    return total + pending


def normalize_number_str(number_str: str) -> float:
    """Normalize a number string ($, %, commas removed) to a float.

    Falls back to the built-in English word converter (0-999) when the string
    is not numeric; returns ``inf`` when conversion is impossible (an ``inf``
    comparison is always False, matching official behavior).
    """
    for char in ("$", "%", ","):
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        pass
    try:
        return float(_words_to_number(number_str))
    except ValueError:
        return float("inf")


def split_string(s: str, char_list: list[str] | None = None) -> list[str]:
    """Split on any of the given separator characters (default ',' and ';')."""
    if char_list is None:
        char_list = [",", ";"]
    pattern = f"[{''.join(char_list)}]"
    return re.split(pattern, s)


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalize a string: remove all whitespace, optionally punctuation, lowercase."""
    no_spaces = re.sub(r"\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _is_float(element: str | None) -> bool:
    try:
        float(element)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def question_scorer(model_answer: str | None, ground_truth: str) -> tuple[bool, str]:
    """Score a model answer against the ground truth (official GAIA rules).

    Three branches, mirroring the official scorer:

    1. Numeric ground truth: prediction is stripped of ``$``/``%``/``,``, and
       compared as a float (English number words supported via the fallback).
    2. List ground truth (contains ``,`` or ``;``): both sides are split on
       ``[;,]``; element counts must match and every element must match
       (numeric elements compare as floats, string elements keep punctuation).
    3. String ground truth: both sides are normalized (all whitespace removed,
       punctuation removed, lowercased) and compared for equality.

    Returns:
        ``(score, detail)`` where ``detail`` names the branch taken.
    """
    model_answer = model_answer or ""

    if _is_float(ground_truth):
        normalized_answer = normalize_number_str(model_answer)
        return normalized_answer == float(ground_truth), "number"

    if any(char in ground_truth for char in (",", ";")):
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)
        if len(gt_elems) != len(ma_elems):
            return False, f"list: length mismatch ({len(ma_elems)} vs {len(gt_elems)})"
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _is_float(gt_elem):
                if normalize_number_str(ma_elem) != float(gt_elem):
                    return False, "list: numeric element mismatch"
            elif normalize_str(ma_elem, remove_punct=False) != normalize_str(
                gt_elem, remove_punct=False
            ):
                return False, "list: string element mismatch"
        return True, "list"

    score = normalize_str(model_answer) == normalize_str(ground_truth)
    return score, "string"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_gaia_scorer.py -v`
Expected: all PASS

- [ ] **Step 6: Lint**

Run: `ruff check benchmarks/gaia tests/conftest.py tests/test_gaia_scorer.py`
Expected: no errors

- [ ] **Step 7: Commit**

```bash
git add benchmarks/gaia/gaia_scorer.py tests/conftest.py tests/test_gaia_scorer.py
git commit -m "feat(benchmarks): port official gaia question scorer"
```

---

### Task 2: `run_gaia.py` — sequential runner with resume

**Files:**
- Create: `benchmarks/gaia/run_gaia.py`

No unit tests (requires live API); verified by `compileall` + ruff here and by the smoke run in Task 4.

- [ ] **Step 1: Implement `benchmarks/gaia/run_gaia.py`**

```python
"""Run the GAIA validation set through the giskard harness, task by task.

Sequential execution with JSONL incremental persistence and resume: completed
task_ids in ``results.jsonl`` are skipped on rerun (override with ``--force``).

Usage:
    PYTHONPATH=src python benchmarks/gaia/run_gaia.py --limit 3
    PYTHONPATH=src python benchmarks/gaia/run_gaia.py --level 1
    PYTHONPATH=src python benchmarks/gaia/run_gaia.py --output-dir benchmarks/gaia/runs/my_run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from giskard import create_harness_agent
from giskard.providers import OpenAIChatCompletionClient
from giskard.tools.web_search import ParallelSearchClient

GAIA_TASK_TEMPLATE = """\
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use units unless specified, don't use commas to separate digits, use digits. If asked for a string, don't use articles, don't use abbreviations (e.g. for cities), and write the numbers in digits. If asked for a comma separated list, apply the above rules depending on each element.

Question: {question}
"""

FINAL_ANSWER_MARKER = "FINAL ANSWER:"


def extract_final_answer(text: str) -> str | None:
    """Return the text after the LAST ``FINAL ANSWER:`` marker (same line)."""
    idx = text.upper().rfind(FINAL_ANSWER_MARKER)
    if idx == -1:
        return None
    rest = text[idx + len(FINAL_ANSWER_MARKER):].strip()
    if not rest:
        return None
    return rest.splitlines()[0].strip()


def load_tasks(
    path: Path,
    *,
    level: int | None,
    task_ids: set[str] | None,
    limit: int | None,
) -> list[dict]:
    """Load and filter dataset tasks by level, task_ids, then limit."""
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if level is not None:
        tasks = [t for t in tasks if t["Level"] == level]
    if task_ids:
        tasks = [t for t in tasks if t["task_id"] in task_ids]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


async def run_task(
    agent,  # noqa: ANN001 — Agent type import would create a cycle risk; duck-typed
    question: str,
    *,
    timeout_s: float,
    max_turns: int,
) -> dict:
    """Run one GAIA task; return prediction/error/turns/duration/updates."""
    session = agent.create_session()
    started = time.monotonic()
    all_text: list[str] = []
    updates: list[dict] = []
    message_ids: set[str] = set()
    turns_over_limit = False
    error: str | None = None

    async def consume() -> None:
        nonlocal turns_over_limit
        stream = agent.run(
            GAIA_TASK_TEMPLATE.format(question=question), stream=True, session=session
        )
        async for update in stream:
            updates.append(update.to_dict())
            if update.message_id and update.role == "assistant":
                message_ids.add(update.message_id)
                if len(message_ids) > max_turns:
                    turns_over_limit = True
                    break
            if update.text:
                all_text.append(update.text)

    try:
        await asyncio.wait_for(consume(), timeout=timeout_s)
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as exc:  # noqa: BLE001 — one task must never kill the run
        error = f"api_error:{type(exc).__name__}: {exc}"

    if turns_over_limit and not error:
        error = "max_turns"

    prediction = extract_final_answer("\n".join(all_text)) or ""
    if prediction == "" and error is None:
        error = "missing_final_answer"

    return {
        "prediction": prediction,
        "error": error,
        "turns": len(message_ids),
        "duration_s": round(time.monotonic() - started, 1),
        "updates": updates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GAIA validation tasks through the giskard harness."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "validation" / "dev.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to benchmarks/gaia/runs/<YYYYmmdd-HHMMSS>/",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N selected tasks")
    parser.add_argument("--level", type=int, choices=(1, 2, 3), default=None)
    parser.add_argument(
        "--task-ids", type=str, default=None, help="Comma-separated task_ids to run"
    )
    parser.add_argument(
        "--timeout", type=float, default=1800.0, help="Per-task wall-clock timeout in seconds"
    )
    parser.add_argument(
        "--max-turns", type=int, default=60, help="Per-task max assistant turns"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run tasks already present in results.jsonl"
    )
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        sys.exit("LLM_API_KEY not set (check .env)")
    model = os.getenv("BASE_MODEL") or "unknown"

    run_dir = args.output_dir or (
        Path(__file__).resolve().parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir / "workdir"
    workdir.mkdir(exist_ok=True)
    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    results_path = run_dir / "results.jsonl"

    task_id_filter = (
        {t.strip() for t in args.task_ids.split(",") if t.strip()} if args.task_ids else None
    )
    tasks = load_tasks(
        args.dataset, level=args.level, task_ids=task_id_filter, limit=args.limit
    )

    done_ids: set[str] = set()
    if results_path.exists() and not args.force:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["task_id"])
        tasks = [t for t in tasks if t["task_id"] not in done_ids]

    total = len(tasks)
    print(f"Run dir: {run_dir}")
    print(f"Tasks to run: {total} (skipped {len(done_ids)} already done)")

    client = OpenAIChatCompletionClient(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("BASE_MODEL"),
    )
    parallel = ParallelSearchClient()
    agent = create_harness_agent(
        name="gaia_agent",
        client=client,
        workdir=workdir,
        tool_approval_rule="yolo",
        web_search_client=parallel,
    )

    try:
        for index, task in enumerate(tasks, start=1):
            record = {
                "task_id": task["task_id"],
                "id": task["id"],
                "level": task["Level"],
                "question": task["Question"],
                "prediction": "",
                "ground_truth": task["answer"],
                "duration_s": 0.0,
                "turns": 0,
                "error": None,
                "model": model,
            }
            try:
                result = await run_task(
                    agent, task["Question"], timeout_s=args.timeout, max_turns=args.max_turns
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # noqa: BLE001 — belt and braces around run_task
                result = {
                    "prediction": "",
                    "error": f"api_error:{type(exc).__name__}: {exc}",
                    "turns": 0,
                    "duration_s": 0.0,
                    "updates": [],
                }
            record.update(result)

            # Write the transcript first, then strip it from the JSONL record.
            updates = record.pop("updates", [])
            (transcripts_dir / f"{task['task_id']}.json").write_text(
                json.dumps(updates, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            with results_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(
                f"[{index}/{total}] {task['task_id']} level={task['Level']} "
                f"turns={record['turns']} {record['duration_s']}s error={record['error']} "
                f"pred={record['prediction']!r}"
            )
    finally:
        await parallel.close()


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nInterrupted — completed tasks are already in results.jsonl; rerun to resume.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax + import check (no API call)**

Run: `python -m compileall benchmarks/gaia -q`
Expected: no output (success)

Run: `$env:PYTHONPATH='src'; python -c "import sys; sys.argv=['run_gaia.py','--help']; import runpy; runpy.run_path('benchmarks/gaia/run_gaia.py', run_name='__main__')"`
Expected: prints usage/help text and exits (does not call the API)

- [ ] **Step 3: Lint**

Run: `ruff check benchmarks/gaia/run_gaia.py`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add benchmarks/gaia/run_gaia.py
git commit -m "feat(benchmarks): add gaia validation runner with resume"
```

---

### Task 3: `evaluate_gaia.py` — evaluator (TDD)

**Files:**
- Create: `benchmarks/gaia/evaluate_gaia.py`
- Create: `tests/test_gaia_evaluate.py`

- [ ] **Step 1: Write the failing test `tests/test_gaia_evaluate.py`**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gaia_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate_gaia'`

- [ ] **Step 3: Implement `benchmarks/gaia/evaluate_gaia.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gaia_evaluate.py tests/test_gaia_scorer.py -v`
Expected: all PASS

- [ ] **Step 5: Lint**

Run: `ruff check benchmarks/gaia tests`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add benchmarks/gaia/evaluate_gaia.py tests/test_gaia_evaluate.py
git commit -m "feat(benchmarks): add gaia evaluation script with summary report"
```

---

### Task 4: Artifacts hygiene + smoke run

**Files:**
- Create: `benchmarks/gaia/.gitignore`

- [ ] **Step 1: Create `benchmarks/gaia/.gitignore`** — keeps run artifacts out of git without touching the user's root `.gitignore`:

```
runs/
```

- [ ] **Step 2: Commit**

```bash
git add benchmarks/gaia/.gitignore
git commit -m "chore(benchmarks): ignore gaia run artifacts"
```

- [ ] **Step 3: Smoke run (requires live API + network)**

Run: `$env:PYTHONPATH='src'; python benchmarks/gaia/run_gaia.py --limit 1 --output-dir benchmarks/gaia/runs/smoke --timeout 600`
Expected: prints `Run dir: ...`, `Tasks to run: 1`, then one `[1/1] ...` line with a non-empty prediction or a documented error; `benchmarks/gaia/runs/smoke/` contains `results.jsonl`, `transcripts/<task_id>.json`, `workdir/`.

- [ ] **Step 4: Smoke evaluate**

Run: `$env:PYTHONPATH='src'; python benchmarks/gaia/evaluate_gaia.py --results-dir benchmarks/gaia/runs/smoke`
Expected: prints `Accuracy: ...` and `Scored results written to ...` with `scored.jsonl` created.

- [ ] **Step 5: Full test suite**

Run: `pytest tests -v`
Expected: all PASS (scorer + evaluator + existing harness tests)

---

## Self-Review Checklist (completed)

- **Spec coverage:** scorer port (Task 1), runner with all CLI params + resume + timeout/max-turns + transcripts + FINAL ANSWER extraction + ParallelSearchClient lifecycle (Task 2), evaluator with overall/per-level/error stats + wrong list + idempotent scored.jsonl (Task 3), runs/ ignored + smoke verification (Task 4). Error taxonomy (`timeout` / `max_turns` / `missing_final_answer` / `api_error:*`) implemented in Task 2.
- **Placeholder scan:** none — every step has complete code or an exact command.
- **Type consistency:** `question_scorer -> tuple[bool, str]` matches Task 3 usage; `run_task` dict keys (`prediction`, `error`, `turns`, `duration_s`, `updates`) match the record merge in `run()`; `main_with(results_dir=..., dataset=...)` matches the test call.
