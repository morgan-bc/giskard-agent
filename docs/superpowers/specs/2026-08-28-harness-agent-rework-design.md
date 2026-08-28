# Design: `create_harness_agent` Rework — workdir, YOLO Approval, Default Tools

Date: 2026-08-28
Status: Approved
Scope: `src/giskard/core/harness/agents.py`, `src/giskard/core/harness/tool_approval.py`, tests

## Goals

1. Decouple default tool assembly from client protocol methods (`get_web_search_tool`,
   `get_shell_tool`). Web search defaults to `ParallelSearchClient`; shell defaults to
   `LocalShellTool.as_function()`.
2. Introduce a single `workdir` parameter that roots all file I/O (file access tools,
   file memory) and shell execution.
3. Add a `tool_approval_rule` parameter with a `"yolo"` preset: within `workdir`,
   read + write + execute are auto-approved; outside `workdir`, only reads are
   auto-approved; deletion operations (`rm`, `delete_file`) always require human
   confirmation.
4. Remove the experimental-feature warnings from the harness factory.
5. Analyze and confirm the necessity of the current context providers and middleware.

## Non-Goals

- No changes to `SupportsWebSearchTool` / `SupportsShellTool` protocols in
  `clients.py` — providers keep their methods; the harness simply stops using them.
- No changes to `_feature_stage.py` or other `@experimental` decoration sites.
- No removal of existing providers or middleware (see Analysis below).

## 1. API Changes

Three new keyword-only parameters on `create_harness_agent`, placed next to their
semantically related parameters:

```python
workdir: str | Path | None = None,                       # before disable_file_access
web_search_client: ParallelSearchClient | None = None,   # next to disable_web_search
tool_approval_rule: Literal["yolo"] | None = None,       # next to disable_tool_auto_approval
```

### workdir

- Resolved once: `Path(workdir).resolve()` when provided, otherwise
  `Path.cwd().resolve()`.
- Roots (existing explicit stores still win when supplied):
  - `FileSystemAgentFileStore(resolved_workdir)` — default file-access store
    (replaces the current `Path.cwd()` default).
  - `FileSystemAgentFileStore(resolved_workdir / "agent-file-memory")` — default
    file-memory store (replaces the current `Path.cwd() / "agent-file-memory"`).
  - `LocalShellTool(workdir=resolved_workdir)` — default shell tool.
  - YOLO approval rule boundary.
- No existence check or auto-creation at the harness layer; behavior is delegated to
  the store / shell tool (unchanged from status quo).

### web_search_client

- `disable_web_search=True` → no web tools wired, no warning.
- Otherwise: the harness uses the supplied `web_search_client`, or constructs
  `ParallelSearchClient()` when None. Its `get_tools()` result (`web_search`,
  `web_fetch`) is appended to the assembled tools.
- Imported lazily inside the factory (same pattern as `LocalShellTool`:
  `giskard.tools` depends on core, so core cannot import it at module load time).
- Lifecycle: **caller owns** the supplied instance (mirrors the `shell_executor`
  convention); the harness-created default instance connects lazily on first
  invocation.

### tool_approval_rule

- `None` (default) → current behavior: tools follow their own `approval_mode`.
- `"yolo"` → the factory appends `create_yolo_approval_rule(resolved_workdir)` to
  `auto_approval_rules`.
- `tool_approval_rule="yolo"` combined with `disable_tool_auto_approval=True` raises
  `ValueError` (contradictory configuration).

## 2. Tool Assembly Decoupled from the Client

### Web search (replaces the current `SupportsWebSearchTool` branch)

```python
if not disable_web_search:
    from giskard.tools.web_search.parrallel import ParallelSearchClient
    search = web_search_client or ParallelSearchClient()
    assembled_tools.extend(search.get_tools())
```

- The `isinstance(client, SupportsWebSearchTool)` check and the
  "Web search tool not available" warning are deleted.

### Shell (replaces `_assemble_shell`'s client branch)

- `_assemble_shell` no longer checks `SupportsShellTool` and no longer calls
  `client.get_shell_tool(func=...)`. It wires `shell_executor.as_function()` directly.
- The "Shell tool not available" warning is deleted.
- Default construction becomes `LocalShellTool(workdir=resolved_workdir)`. When the
  caller supplies `shell_executor`, the harness does **not** inject `workdir`
  (caller owns the executor and its configuration).
- `ShellEnvironmentProvider` wiring is unchanged (still opt-in via
  `shell_environment_provider_options` / shell_executor presence).

## 3. YOLO Approval Rule

New public builder in `tool_approval.py`:

```python
def create_yolo_approval_rule(workdir: Path) -> ToolApprovalRuleCallback: ...
```

The callback receives a function-call `Content` and returns `True` (auto-approve) or
`False` (fall through to standing rules / human prompt). Auto-approval rules can only
approve — non-matches escalate, which matches the YOLO semantics.

### Classification matrix

| Tool(s) | Decision | Rationale |
|---|---|---|
| `read_file`, `ls`, `glob`, `grep` | approve | read-only (reads allowed outside workdir per "non-workdir: read_only"; stores confine paths lexically regardless) |
| `web_search`, `web_fetch` | approve | read-only, no filesystem access |
| `write_file`, `edit_file`, `edit_file_lines` | approve only if `file_name` resolves inside workdir; otherwise escalate | path-gated (defense-in-depth, symlink-following) |
| `file_memory_read` / `ls` / `grep` | approve | read-only |
| `file_memory_write` / `replace` / `replace_lines` | approve only if `file_name` resolves inside workdir; otherwise escalate | path-gated |
| `delete_file`, `file_memory_delete` | escalate | dangerous deletion |
| `run_shell` | approve unless destructive pattern matches | see below |
| anything else (unknown tools, MCP tools) | escalate | "others need approval" |

Shell commands are **assumed** workdir-executions and are NOT path-verified: the
rule cannot reliably parse arbitrary shell syntax for path arguments. The default
`LocalShellTool(workdir=...)` re-anchors each persistent command (`confine_workdir`)
but is not hard confinement; the destructive-pattern escalation is the only
shell-level gate.

## 3a. Amendment: write-tool path containment

After review, the YOLO rule additionally validates the `file_name` argument of the
six write-capable file tools (`write_file`, `edit_file`, `edit_file_lines`,
`file_memory_write`, `file_memory_replace`, `file_memory_replace_lines`):
relative arguments are joined onto `workdir`, absolute arguments must already be
inside it, resolution follows symlinks, and anything resolving outside the
boundary — or missing/unparseable arguments — escalates to a human. Read tools
stay blanket-approved (the spec allows non-workdir reads, and the stores reject
absolute paths and `..` traversal lexically). This closes the gap where a
caller-supplied store with weaker confinement would have been trusted implicitly.

### Shell destructive-command detection

- Split the command on `&&`, `;`, `|`, and newlines into segments.
- For each segment, word-boundary match (case-insensitive) the leading token against:
  `rm`, `rmdir`, `del`, `erase`, `Remove-Item`, `rd`.
- Any match → escalate (return `False`); otherwise approve.

Known limitations (documented in the docstring): `sudo rm`, `xargs rm`, shell aliases,
and scripts that delete internally are not caught. Approval is a UX boundary, not a
hard security boundary — consistent with `LocalShellTool`'s documented stance that
`confine_workdir` is a re-anchor, not confinement.

## 4. Experimental Warning Removal

- Delete `_warn_experimental_harness_params` and its call site.
- Delete the "Experimental features" note block from the factory docstring.
- Remove now-unused imports (`ExperimentalFeature`, `warn_experimental_feature`).
- Scope is limited to the harness factory; `_feature_stage.py` and other decorated
  APIs are untouched.

## 5. Provider / Middleware Necessity Analysis (requirement 5)

**Conclusion: keep everything.** None of the components are dead weight.

| Component | Verdict | Reason |
|---|---|---|
| `InMemoryHistoryProvider` | required | `require_per_service_call_history_persistence=True` depends on it |
| `CompactionProvider` + before-strategy | keep | core token-budget management; dormant unless token params given |
| `TodoProvider` | keep | independent task-tracking role; opt-out via `disable_todo` |
| `AgentModeProvider` | keep (most debatable) | injects strong plan/execute workflow instructions and is heavy for simple agents, but is opt-out via `disable_mode`; removal is a breaking change with no substitute |
| `FileMemoryProvider` | keep | session-scoped memory differs from the shared FileAccessProvider store; now rooted under workdir |
| `FileAccessProvider` | keep (core) | primary workdir anchor for file tools |
| `SkillsProvider` / `BackgroundAgentsProvider` | keep | opt-in, zero default cost |
| `ShellEnvironmentProvider` | keep | bound to the shell tool, opt-in |
| `ToolApprovalMiddleware` | required | mount point for the YOLO rule |
| `AgentLoopMiddleware` | keep | opt-in via `loop_should_continue` |
| `MessageInjectionMiddleware` | keep | always-on but a no-op with an empty queue |
| Per-service-call history persistence | required | core mechanism |

## 6. Error Handling

- New validation: `tool_approval_rule="yolo"` with `disable_tool_auto_approval=True`
  → `ValueError`.
- Existing validation (token params) unchanged.

## 7. Testing Plan

New `tests/test_harness_yolo.py` (plus coverage of the factory changes):

- YOLO rule unit tests:
  - approval matrix: read tools, web tools, write tools → approved;
    `delete_file`, unknown tools → escalated;
  - shell detection: `rm -rf x`, `Remove-Item -Recurse`, multi-segment
    (`cd x && rm y`), PowerShell casing → escalated; `ls`, `python script.py` → approved.
- Factory integration tests:
  - default wiring injects `web_search` + `web_fetch` without requiring
    `SupportsWebSearchTool` on the client;
  - shell tool wires via `as_function()` without `SupportsShellTool`;
  - `workdir` propagates to the file-access store, file-memory store, and shell tool;
  - `tool_approval_rule="yolo"` + `disable_tool_auto_approval=True` raises `ValueError`;
  - no `ExperimentalWarning` is emitted when enabling `background_agents` /
    `loop_should_continue` (inverse assertion).
