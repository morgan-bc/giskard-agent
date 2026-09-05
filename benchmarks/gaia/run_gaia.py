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

from giskard import Agent, create_harness_agent
from giskard.core.types import AgentResponse
from giskard.providers import OpenAIChatCompletionClient
from giskard.tools.web_search import ParallelSearchClient
from giskard.tools.python import LocalPythonExecutor

GAIA_TASK_TEMPLATE = """\
You are a helpful AI assistant that uses tools to complete tasks.

## General guidelines

- Think through the task before acting. Break complex work into clear steps.
- Use the tools available to you to gather information, perform actions, and verify results.
- Explain your reasoning and thought process as you work through tasks.
- Explain what you learned and what you are going to do next between tool calls, \
so the user can follow along with your thought process.
- Avoid making more than 4 tool calls in a row without explaining what you are doing.
- If a tool call fails or returns unexpected results, adapt your approach rather than \
repeating the same call.
- When you have completed the task, present a clear and concise summary of what you did \
and what you found.

## Answer format
- Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER].
- YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use units unless specified, don't use commas to separate digits, use digits. If asked for a string, don't use articles, don't use abbreviations (e.g. for cities), and write the numbers in digits. If asked for a comma separated list, apply the above rules depending on each element.

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
    agent: Agent,
    question: str,
    *,
    timeout_s: float,
    max_turns: int,
) -> dict:
    """Run one GAIA task; return prediction/error/turns/duration/trajectory."""
    session = agent.create_session()
    started = time.monotonic()
    error: str | None = None
    final_response: AgentResponse | None = None

    agent.client.function_invocation_configuration["max_iterations"] = max_turns

    async def consume() -> AgentResponse:
        stream = agent.run(question, stream=True, session=session)
        return await stream.get_final_response()

    try:
        final_response = await asyncio.wait_for(consume(), timeout=timeout_s)
    except asyncio.TimeoutError:
        error = "timeout"
    except Exception as exc:  # noqa: BLE001 — one task must never kill the run
        error = f"api_error:{type(exc).__name__}: {exc}"

    full_text = final_response.text if final_response is not None else ""
    prediction = extract_final_answer(full_text) or ""
    if prediction == "" and error is None:
        error = "missing_final_answer"
        print(f"Answer not found in response: {full_text}")

    in_memory = session.state.get("in_memory", {})
    messages = [msg.to_dict() for msg in in_memory.get("messages", [])]
    last_invocation = session.last_invocation or {}
    trajectory = {
        "system_prompt": last_invocation.get("system_prompt"),
        "tools": last_invocation.get("tools", []),
        "messages": messages,
    }
    # get_final_response() consumes the whole stream, so the turn cap is a
    # post-hoc label rather than a mid-run abort.
    turns = sum(1 for m in messages if m.get("role") == "assistant")

    return {
        "prediction": prediction,
        "error": error,
        "turns": turns,
        "duration_s": round(time.monotonic() - started, 1),
        "trajectory": trajectory,
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
        "--timeout", type=float, default=180, help="Per-task wall-clock timeout in seconds"
    )
    parser.add_argument(
        "--max-turns", type=int, default=20, help="Per-task max assistant turns"
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
        Path(__file__).resolve().parent / "runs" / datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    workdir = run_dir / "workdir"
    workdir.mkdir(exist_ok=True)
    transcripts_dir = run_dir / "transcripts"
    transcripts_dir.mkdir(exist_ok=True)
    results_path = run_dir / "results.jsonl"

    task_id_filter = None
    if args.task_ids:
        task_id_filter = {t.strip() for t in args.task_ids.split(",") if t.strip()}
        if not task_id_filter:
            sys.exit("--task-ids given but no valid ids parsed")
    tasks = load_tasks(
        args.dataset, level=args.level, task_ids=task_id_filter, limit=args.limit
    )
    if task_id_filter and not tasks:
        sys.exit("No dataset tasks matched --task-ids")

    done_ids: set[str] = set()
    if results_path.exists() and not args.force:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                done_ids.add(json.loads(line)["task_id"])
            except json.JSONDecodeError:
                print(f"WARNING: skipping malformed line in results.jsonl: {line[:80]!r}")
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
    python_executor = LocalPythonExecutor(workdir=workdir)
    tools = python_executor.get_tools()
    agent = create_harness_agent(
        name="gaia_agent",
        client=client,
        harness_instructions=GAIA_TASK_TEMPLATE,
        workdir=workdir,
        tool_approval_rule="yolo",
        web_search_client=parallel,
        disable_file_memory=True,
        disable_todo=True,
        disable_file_access=True,
        disable_mode=True,
        tools=tools,
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
                print(f"[{index}/{total}] {task['task_id']} level={task['Level']}")
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
                    "trajectory": None,
                }
            record.update(result)

            # Write the trajectory first, then strip it from the JSONL record.
            trajectory = record.pop("trajectory", None)
            (transcripts_dir / f"{task['task_id']}.json").write_text(
                json.dumps(trajectory, ensure_ascii=False, indent=2), encoding="utf-8"
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
