# Harness Agent Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework `create_harness_agent` — default web search via `ParallelSearchClient`, default shell via `LocalShellTool.as_function()`, a unified `workdir` root, a `"yolo"` tool-approval preset, and removal of experimental warnings.

**Architecture:** The factory stops calling client protocol methods (`get_web_search_tool`/`get_shell_tool`) and wires tools directly from the tools package (lazy imports, matching the existing `LocalShellTool` pattern). One resolved `workdir` propagates to the file-access store, file-memory store, shell tool, and the YOLO rule. The YOLO rule is a `ToolApprovalRuleCallback` appended to `ToolApprovalMiddleware.auto_approval_rules`; it approves by tool-name classification and escalates deletions (including destructive shell commands) to humans.

**Tech Stack:** Python 3.10+, pytest, giskard core (`Content`, `ToolApprovalMiddleware`, `ContextProvider`), `giskard.tools.shell.LocalShellTool`, `giskard.tools.web_search.parrallel.ParallelSearchClient`.

**Spec:** `docs/superpowers/specs/2026-08-28-harness-agent-rework-design.md`

**Key facts for all tasks:**

- `ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]` — receives the **function-call** `Content` (with `.name`, `.parse_arguments()`); return `True` = auto-approve, `False` = escalate to standing rules / human prompt.
- Tool names (verified in source):
  - FileAccessProvider: `write_file`, `read_file`, `delete_file`, `ls`, `glob`, `grep`, `edit_file`, `edit_file_lines`
  - FileMemoryProvider: `file_memory_write`, `file_memory_read`, `file_memory_delete`, `file_memory_ls`, `file_memory_replace`, `file_memory_replace_lines`, `file_memory_grep`
  - ParallelSearchClient: `web_search`, `web_fetch`
  - LocalShellTool default function name: `run_shell` (single `command: str` argument)
- `FileSystemAgentFileStore` exposes `root_path` property; providers expose `.store`; `Agent.default_options["tools"]` holds the tool list; `Agent.context_providers` holds providers.
- Test command prefix: run from repo root `d:\Projects\giskard` with `python -m pytest`.

---

### Task 1: YOLO approval rule in `tool_approval.py`

**Files:**
- Modify: `src/giskard/core/harness/tool_approval.py`
- Create: `tests/test_harness_yolo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_harness_yolo.py`:

```python
"""Tests for the YOLO tool-approval rule and the harness factory rework."""

from __future__ import annotations

from pathlib import Path

import pytest

from giskard.core.harness.tool_approval import create_yolo_approval_rule
from giskard.core.types import Content


def _function_call(name: str, arguments: dict | None = None) -> Content:
    return Content.from_function_call(call_id="c1", name=name, arguments=arguments or {})


@pytest.fixture()
def yolo() -> object:
    return create_yolo_approval_rule(Path.cwd().resolve())


class TestDestructiveShellDetection:
    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf build",
            "rmdir tmp",
            "del file.txt",
            "erase file.txt",
            "rd /s /q build",
            "Remove-Item -Recurse -Force .",
            "remove-item foo",
            "cd foo && rm bar.txt",
            "Set-Location ..; Remove-Item x",
            "Get-ChildItem . | Remove-Item -Force",
        ],
    )
    def test_destructive_commands_escalate(self, yolo, command):
        assert yolo(_function_call("run_shell", {"command": command})) is False

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "python script.py",
            "cd build && python -m pytest -q",
            "Get-ChildItem -Recurse -Filter *.py",
            "echo hello",
        ],
    )
    def test_safe_commands_are_approved(self, yolo, command):
        assert yolo(_function_call("run_shell", {"command": command})) is True

    def test_shell_without_command_argument_escalates(self, yolo):
        assert yolo(_function_call("run_shell")) is False


class TestYoloApprovalMatrix:
    @pytest.mark.parametrize(
        "name",
        ["read_file", "ls", "glob", "grep", "web_search", "web_fetch",
         "write_file", "edit_file", "edit_file_lines",
         "file_memory_write", "file_memory_read", "file_memory_ls",
         "file_memory_grep", "file_memory_replace", "file_memory_replace_lines"],
    )
    def test_approved_tools(self, yolo, name):
        assert yolo(_function_call(name)) is True

    @pytest.mark.parametrize(
        "name",
        ["delete_file", "file_memory_delete", "some_unknown_mcp_tool"],
    )
    def test_escalated_tools(self, yolo, name):
        assert yolo(_function_call(name)) is False

    def test_none_name_escalates(self, yolo):
        # Runtime edge: a function_call content without a name escalates.
        # Constructed directly since from_function_call types name as str.
        assert yolo(Content("function_call", call_id="c1", name=None)) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: FAIL — `ImportError: cannot import name 'create_yolo_approval_rule'`

- [ ] **Step 3: Implement the YOLO rule**

In `src/giskard/core/harness/tool_approval.py`, add imports `re` and `Path` at the top (alphabetical in the stdlib/stdlib blocks):

```python
import re
from pathlib import Path
```

Add the following after `ToolApprovalRuleCallback = Callable[[Content], bool | Awaitable[bool]]` (around line 38):

```python
# Tools auto-approved by the YOLO rule: read-only tools, workdir-confined write
# tools, read-only web tools, and the shell tool (subject to the destructive
# command check below). Deletions and any unknown tool escalate to the human.
_YOLO_APPROVED_TOOLS: frozenset[str] = frozenset({
    "read_file",
    "ls",
    "glob",
    "grep",
    "write_file",
    "edit_file",
    "edit_file_lines",
    "file_memory_write",
    "file_memory_read",
    "file_memory_ls",
    "file_memory_grep",
    "file_memory_replace",
    "file_memory_replace_lines",
    "web_search",
    "web_fetch",
})

# Dangerous deletions always require human confirmation.
_YOLO_ESCALATED_TOOLS: frozenset[str] = frozenset({"delete_file", "file_memory_delete"})

# Matches (word-boundary, case-insensitive) a destructive command name inside a
# shell command segment's leading token. Covers POSIX rm/rmdir, Windows
# del/erase/rd, and PowerShell Remove-Item plus its aliases (ri, rm, del, rd,
# erase). A path-prefixed binary such as /bin/rm also matches.
_YOLO_DESTRUCTIVE_TOKEN_RE = re.compile(r"\b(rm|rmdir|del|erase|rd|ri|remove-item)\b", re.IGNORECASE)

# Segment separators: command chaining and pipelines.
_YOLO_COMMAND_SEGMENT_RE = re.compile(r"&&|;|\||\r?\n")


def _is_destructive_shell_command(command: str) -> bool:
    """Return whether any segment of ``command`` starts with a delete-style command.

    Only the leading token of each segment (split on ``&&``, ``;``, ``|``, and
    newlines) is inspected. Known limitations (by design — approval is a UX
    boundary, not a hard security boundary): ``sudo rm``, ``xargs rm``, shell
    aliases, and scripts that delete internally are not caught.
    """
    for segment in _YOLO_COMMAND_SEGMENT_RE.split(command):
        leading = re.match(r"\s*(\S+)", segment)
        if leading and _YOLO_DESTRUCTIVE_TOKEN_RE.search(leading.group(1)):
            return True
    return False


def create_yolo_approval_rule(workdir: Path) -> ToolApprovalRuleCallback:
    """Create a YOLO auto-approval rule bounded by ``workdir``.

    Classification:

    - workdir reads/writes/executes (read tools, write/edit tools, memory
      tools, web search, non-destructive shell commands) -> auto-approve.
    - deletions (``delete_file``, ``file_memory_delete``, destructive shell
      commands) -> escalate to human confirmation.
    - anything else (unknown or MCP tools) -> escalate.

    Args:
        workdir: The resolved working directory that bounds the agent's file
            tools. The file/memory stores already confine paths to this root,
            so no per-call path check is performed here; the argument pins the
            boundary for future argument-aware checks.

    Returns:
        A callback suitable for :class:`ToolApprovalMiddleware`'s
        ``auto_approval_rules``.
    """
    del workdir  # Reserved: the store confines paths; see docstring.

    def _yolo_rule(function_call: Content) -> bool:
        name = function_call.name
        if name is None:
            return False
        if name in _YOLO_ESCALATED_TOOLS:
            return False
        if name == "run_shell":
            arguments = function_call.parse_arguments()
            command = arguments.get("command") if isinstance(arguments, Mapping) else None
            if not isinstance(command, str) or _is_destructive_shell_command(command):
                return False
            return True
        return name in _YOLO_APPROVED_TOOLS

    return _yolo_rule
```

Add to the module's `__all__` list: `"create_yolo_approval_rule"` (keep alphabetical order).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: all tests PASS (30+ parameterized cases)

- [ ] **Step 5: Commit**

```bash
git add src/giskard/core/harness/tool_approval.py tests/test_harness_yolo.py
git commit -m "feat(harness): add yolo tool approval rule with destructive shell command detection"
```

---

### Task 2: `workdir` parameter threading in the factory

**Files:**
- Modify: `src/giskard/core/harness/agents.py`
- Modify: `tests/test_harness_yolo.py`

- [ ] **Step 1: Write failing tests for validation and workdir propagation**

Append to `tests/test_harness_yolo.py`:

```python
from unittest.mock import MagicMock

from giskard import create_harness_agent
from giskard.core.harness.file_access import (
    FileAccessProvider,
    FileMemoryProvider,
    FileSystemAgentFileStore,
)


class TestWorkdirThreading:
    def test_workdir_roots_file_access_store(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        provider = next(p for p in agent.context_providers if isinstance(p, FileAccessProvider))
        assert isinstance(provider.store, FileSystemAgentFileStore)
        assert provider.store.root_path == tmp_path.resolve()

    def test_workdir_roots_file_memory_store(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        provider = next(p for p in agent.context_providers if isinstance(p, FileMemoryProvider))
        assert isinstance(provider.store, FileSystemAgentFileStore)
        assert provider.store.root_path == (tmp_path.resolve() / "agent-file-memory")

    def test_workdir_reaches_default_shell_tool(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        class _SpyShellTool(LocalShellTool):
            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)  # type: ignore[arg-type]
                captured.update(kwargs)

        monkeypatch.setattr("giskard.tools.shell.LocalShellTool", _SpyShellTool)
        create_harness_agent(MagicMock(), workdir=tmp_path)
        assert captured["workdir"] == str(tmp_path.resolve())


class TestToolApprovalRuleValidation:
    def test_yolo_with_disabled_auto_approval_raises(self) -> None:
        with pytest.raises(ValueError, match="disable_tool_auto_approval"):
            create_harness_agent(MagicMock(), tool_approval_rule="yolo", disable_tool_auto_approval=True)

    def test_unknown_rule_value_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_approval_rule"):
            create_harness_agent(MagicMock(), tool_approval_rule="bogus")
```

Note: `LocalShellTool` is already imported in this test file's Task 1 version — add `from giskard.tools.shell import LocalShellTool` to the imports at the top.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness_yolo.py::TestWorkdirThreading tests/test_harness_yolo.py::TestToolApprovalRuleValidation -v`
Expected: FAIL — `TypeError: create_harness_agent() got an unexpected keyword argument 'workdir'`

- [ ] **Step 3: Implement the parameter and threading**

In `src/giskard/core/harness/agents.py`:

1. Extend the typing import (line 17):

```python
from typing import TYPE_CHECKING, Any, Literal, TypedDict
```

2. In the `TYPE_CHECKING` block (around line 43), add:

```python
    from giskard.tools.web_search import ParallelSearchClient
```

3. In the `create_harness_agent` signature, insert three keyword parameters — `workdir` immediately before `disable_file_access` (line 323), `web_search_client` immediately before `disable_web_search` (line 336), `tool_approval_rule` immediately before `disable_tool_auto_approval` (line 337):

```python
    disable_file_memory: bool = False,
    file_memory_store: AgentFileStore | None = None,
    workdir: str | Path | None = None,
    disable_file_access: bool = False,
    file_access_store: AgentFileStore | None = None,
    file_access_disable_write_tools: bool = False,
    file_access_enable_extra_tools: bool = False,
    file_access_disable_readonly_tool_approval: bool = False,
    file_access_disable_write_tool_approval: bool = False,
    skills_provider: SkillsProvider | None = None,
    skills_paths: str | Path | Sequence[str | Path] | None = None,
    background_agents: Sequence[SupportsAgentRun] | None = None,
    background_agents_instructions: str | None = None,
    disable_shell: bool = False,
    shell_executor: ShellExecutor | None = None,
    shell_environment_provider_options: ShellEnvironmentProviderOptions | None = None,
    web_search_client: ParallelSearchClient | None = None,
    disable_web_search: bool = False,
    tool_approval_rule: Literal["yolo"] | None = None,
    disable_tool_auto_approval: bool = False,
```

4. Add validation right after the existing token-param validations (after line 555):

```python
    if tool_approval_rule not in (None, "yolo"):
        raise ValueError(f"tool_approval_rule must be None or 'yolo', got {tool_approval_rule!r}.")
    if tool_approval_rule == "yolo" and disable_tool_auto_approval:
        raise ValueError(
            "tool_approval_rule='yolo' requires the tool auto-approval middleware; "
            "set disable_tool_auto_approval=False."
        )
```

5. Replace the file-access default-root block (lines 570-573) with the workdir resolution plus store defaults:

```python
    # One working directory roots all file I/O: the shared file-access store,
    # the session file-memory store, the default shell tool, and the YOLO
    # approval boundary. Existing explicit stores still win.
    resolved_workdir = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    if file_access_store is None and not disable_file_access:
        file_access_store = FileSystemAgentFileStore(resolved_workdir)
```

6. In `_assemble_context_providers`'s file-memory default (lines 186-190), change the store root — the function receives the resolved workdir via a new parameter. Change the signature to add `workdir: Path` (after `file_memory_store`) and the body:

```python
    if not disable_file_memory:
        memory_store = file_memory_store or FileSystemAgentFileStore(
            (workdir / "agent-file-memory").resolve()
        )
        providers.append(FileMemoryProvider(memory_store))
```

At the call site (line 608), pass `workdir=resolved_workdir` after `file_memory_store=file_memory_store,`.

7. Change the default shell construction (lines 591-598) to pass workdir:

```python
    if shell_executor is None and not disable_shell:
        from giskard.tools.shell import LocalShellTool

        shell_executor = LocalShellTool(workdir=resolved_workdir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: all PASS (Task 1 tests still green, new tests green)

- [ ] **Step 5: Commit**

```bash
git add src/giskard/core/harness/agents.py tests/test_harness_yolo.py
git commit -m "feat(harness): add workdir param rooting file access, file memory, and default shell"
```

---

### Task 3: Tool assembly decoupled from client protocol methods

**Files:**
- Modify: `src/giskard/core/harness/agents.py`
- Modify: `tests/test_harness_yolo.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_harness_yolo.py`:

```python
from giskard.tools.web_search import ParallelSearchClient


class TestDefaultToolAssembly:
    def _tool_names(self, agent) -> list[str]:
        return [getattr(t, "name", "") for t in agent.default_options["tools"] or []]

    def test_default_web_search_tools_injected_without_client_protocol(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        names = self._tool_names(agent)
        assert "web_search" in names
        assert "web_fetch" in names

    def test_disable_web_search_skips_web_tools(self, tmp_path: Path) -> None:
        agent = create_harness_agent(MagicMock(), workdir=tmp_path, disable_web_search=True)
        names = self._tool_names(agent)
        assert "web_search" not in names
        assert "web_fetch" not in names

    def test_supplied_web_search_client_is_used(self, tmp_path: Path) -> None:
        client = ParallelSearchClient()
        agent = create_harness_agent(MagicMock(), workdir=tmp_path, web_search_client=client)
        assert self._tool_names(agent).count("web_search") == 1

    def test_shell_tool_wired_without_supports_shell_tool(self, tmp_path: Path) -> None:
        # MagicMock does NOT implement SupportsShellTool; previously this path
        # logged a warning and skipped the shell tool entirely.
        agent = create_harness_agent(MagicMock(), workdir=tmp_path)
        assert "run_shell" in self._tool_names(agent)
```

Note: add `from giskard.tools.web_search import ParallelSearchClient` to imports.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness_yolo.py::TestDefaultToolAssembly -v`
Expected: FAIL — `test_shell_tool_wired_without_supports_shell_tool` fails (shell tool skipped for non-SupportsShellTool client); `test_supplied_web_search_client_is_used` fails (unexpected kwarg `web_search_client`)

- [ ] **Step 3: Implement the decoupling**

In `src/giskard/core/harness/agents.py`:

1. Remove `SupportsShellTool` and `SupportsWebSearchTool` from the module import (line 20):

```python
from ..clients import SupportsChatGetResponse
```

2. Replace `_assemble_shell` (lines 225-266) — drop the `client` parameter, the `SupportsShellTool` check, and the warning; call `as_function()` directly:

```python
def _assemble_shell(
    shell_executor: ShellExecutor | None,
    shell_environment_provider_options: ShellEnvironmentProviderOptions | None,
) -> tuple[ToolTypes | None, ContextProvider | None]:
    """Build the shell tool and environment provider when a shell executor is supplied.

    The shell tool is wrapped from the executor's own ``as_function()`` — no
    client support is required. The environment provider is still opt-in: it is
    only built when a shell executor is present.

    Returns a ``(tool, provider)`` tuple. Both are ``None`` when no shell
    executor is provided.

    Raises:
        TypeError: If ``shell_executor`` does not expose a callable ``as_function()`` method.
    """
    if shell_executor is None:
        return None, None

    # ShellExecutor is a protocol without ``as_function()``, so the
    # contract is validated at runtime: a shell tool such as
    # LocalShellTool/DockerShellTool exposes it.
    as_function = getattr(shell_executor, "as_function", None)
    if not callable(as_function):
        raise TypeError(
            f"shell_executor must expose a callable 'as_function()' method "
            f"(e.g. a LocalShellTool or DockerShellTool), "
            f"but got {type(shell_executor).__name__}."
        )

    # Imported lazily: the shell types live in giskard.tools.shell,
    # which depends on core, so core cannot import them at module load time.
    from giskard.tools.shell import ShellEnvironmentProvider

    shell_tool = as_function()
    shell_provider = ShellEnvironmentProvider(shell_executor, shell_environment_provider_options)
    return shell_tool, shell_provider
```

Update the call site (line 601) to the new signature:

```python
    shell_tool, shell_provider = _assemble_shell(
        shell_executor,
        shell_environment_provider_options,
    )
```

3. Replace the web-search assembly block (lines 633-643):

```python
    # Assemble tools, auto-adding web search via ParallelSearchClient.
    assembled_tools: list[ToolTypes | Callable[..., Any]] = []
    if not disable_web_search:
        # Imported lazily: giskard.tools depends on core, so core cannot import
        # it at module load time. The default client connects lazily on first
        # invocation; a caller-supplied instance is owned (and closed) by the
        # caller.
        from giskard.tools.web_search.parrallel import ParallelSearchClient

        search_client = web_search_client or ParallelSearchClient()
        assembled_tools.extend(search_client.get_tools())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/giskard/core/harness/agents.py tests/test_harness_yolo.py
git commit -m "feat(harness): wire web search and shell tools directly, dropping client protocol dependency"
```

---

### Task 4: YOLO wiring and experimental warning removal

**Files:**
- Modify: `src/giskard/core/harness/agents.py`
- Modify: `tests/test_harness_yolo.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_harness_yolo.py`:

```python
import warnings

from giskard.core._feature_stage import ExperimentalWarning


class TestYoloWiring:
    def test_yolo_rule_reaches_middleware(self, tmp_path: Path) -> None:
        agent = create_harness_agent(
            MagicMock(),
            workdir=tmp_path,
            tool_approval_rule="yolo",
        )
        from giskard.core.harness.tool_approval import ToolApprovalMiddleware

        middleware = next(m for m in agent.middleware if isinstance(m, ToolApprovalMiddleware))
        rule_names = {r.__qualname__ for r in middleware.auto_approval_rules}
        assert "_yolo_rule" in rule_names

    def test_experimental_warnings_removed(self, tmp_path: Path) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ExperimentalWarning)
            # loop_should_continue previously triggered the harness experimental warning.
            create_harness_agent(MagicMock(), workdir=tmp_path, loop_should_continue=lambda response: False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness_yolo.py::TestYoloWiring -v`
Expected: FAIL — `test_yolo_rule_reaches_middleware` fails (no yolo rule in middleware); `test_experimental_warnings_removed` fails (ExperimentalWarning raised as error)

- [ ] **Step 3: Implement**

In `src/giskard/core/harness/agents.py`:

1. Change the tool_approval import (line 33) to include the rule factory:

```python
from .tool_approval import ToolApprovalMiddleware, create_yolo_approval_rule
```

2. Delete the `_warn_experimental_harness_params` function (lines 278-298) and its call site (lines 557-568, the `experimental_params` block and `_warn_experimental_harness_params(...)` call).

3. Remove the now-unused import (line 22):

```python
from .._feature_stage import ExperimentalFeature, warn_experimental_feature
```

4. Build the resolved auto-approval rules — inside the middleware assembly section (lines 665-667), replace:

```python
    # Resolve auto-approval rules: caller-supplied heuristics plus the YOLO
    # preset when requested. Evaluated after standing rules, before prompting.
    resolved_auto_approval_rules = list(auto_approval_rules or [])
    if tool_approval_rule == "yolo":
        resolved_auto_approval_rules.append(create_yolo_approval_rule(resolved_workdir))

    assembled_middleware: list[MiddlewareTypes] = []
    if not disable_tool_auto_approval:
        assembled_middleware.append(ToolApprovalMiddleware(auto_approval_rules=resolved_auto_approval_rules or None))
```

(Leave the rest of the middleware assembly — loop insertion, MessageInjectionMiddleware — unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/giskard/core/harness/agents.py tests/test_harness_yolo.py
git commit -m "feat(harness): wire yolo approval preset and remove experimental warnings"
```

---

### Task 5: Docstrings, cleanup, and full verification

**Files:**
- Modify: `src/giskard/core/harness/agents.py`
- Modify: `tests/test_harness_yolo.py`

- [ ] **Step 1: Update the factory docstring**

In `create_harness_agent`'s docstring:

1. Delete the `.. note:: Experimental features` block (lines 372-377).
2. In the feature bullet list, replace the **Shell tool** and web-search mentions:

```markdown
    - **Web search** — ``web_search`` and ``web_fetch`` tools via
      ``ParallelSearchClient`` (on by default; disable via
      ``disable_web_search`` or supply your own via ``web_search_client``)
    - **Shell tool** — local shell command execution (on by default via
      ``LocalShellTool`` anchored at ``workdir``; disable via ``disable_shell``
      or customize via ``shell_executor``)
```

3. Add keyword-arg documentation entries (keep alphabetical-ish placement next to related args):

```markdown
        workdir: The working directory that roots all file I/O — the shared
            file-access store, the session file-memory store
            (``{workdir}/agent-file-memory``), the default shell tool's
            execution directory, and the YOLO approval boundary. When None
            (default), the current working directory is used. Explicitly
            supplied ``file_access_store`` / ``file_memory_store`` /
            ``shell_executor`` are not overridden by this value.
        web_search_client: Optional ``ParallelSearchClient`` supplying the
            ``web_search`` and ``web_fetch`` tools. When None (default) and
            ``disable_web_search`` is False, a new client is created (it
            connects lazily on first invocation; the caller owns any supplied
            instance and its lifecycle).
        tool_approval_rule: Optional approval preset. ``"yolo"`` auto-approves
            workdir reads, writes, and executions (including web search and
            non-destructive shell commands); deletion operations
            (``delete_file``, ``file_memory_delete``, destructive shell
            commands such as ``rm``/``Remove-Item``) and any unknown tool
            still require human approval. Requires
            ``disable_tool_auto_approval=False``. Known limitation: the shell
            check inspects each command segment's leading token only, so
            ``sudo rm`` / ``xargs rm`` / aliases are not caught — approval is a
            UX boundary, not a sandbox.
```

4. Update the `disable_shell` / `shell_executor` / `disable_web_search` entries to remove the "client does not implement SupportsWebSearchTool/SupportsShellTool" warning wording (the warnings no longer exist):

- `disable_shell`: "When True, skip the shell tool and ShellEnvironmentProvider. When False (default), a ``LocalShellTool`` anchored at ``workdir`` is created automatically (platform default shell, persistent mode, approval required per command)."
- `shell_executor`: "Optional shell executor overriding the default ``LocalShellTool``. When provided, the shell tool and a ``ShellEnvironmentProvider`` are wired from it. The object must expose ``as_function()`` and satisfy the ``ShellExecutor`` protocol -- e.g. a ``LocalShellTool`` or ``DockerShellTool``. The caller owns the executor's lifecycle and its workdir configuration (the harness does not inject ``workdir``)."
- `disable_web_search`: "When True, skip the web search tools. When False (default), ``web_search`` and ``web_fetch`` are added via ``ParallelSearchClient``."

- [ ] **Step 2: Lint-level self-check**

Run: `python -m py_compile src/giskard/core/harness/agents.py src/giskard/core/harness/tool_approval.py`
Expected: no output (compiles clean). Also confirm no lingering references:

Run: search `SupportsWebSearchTool|SupportsShellTool|_warn_experimental_harness_params` in `src/giskard/core/harness/agents.py` — expected: no matches. (`SupportsChatGetResponse` in the TYPE_CHECKING block stays.)

- [ ] **Step 3: Run the full new test suite**

Run: `python -m pytest tests/test_harness_yolo.py -v`
Expected: all PASS

- [ ] **Step 4: Run the pre-existing harness test file to check for regressions**

Run: `python -m pytest tests/test_harness.py -v`
Expected: existing tests behave as before (note: `test_harness.py` contains a manual/LLM-gated section that skips without env vars — that is pre-existing behavior, not a regression).

- [ ] **Step 5: Commit**

```bash
git add src/giskard/core/harness/agents.py tests/test_harness_yolo.py
git commit -m "docs(harness): document workdir, web_search_client, and yolo approval preset"
```
