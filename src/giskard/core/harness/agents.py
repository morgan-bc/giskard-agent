# Copyright (c) Microsoft. All rights reserved.

"""Harness agent factory: a pre-configured bundled agent with batteries included.

This module provides :func:`create_harness_agent`, a factory function that assembles
the full agent pipeline from a chat client, wiring up function invocation,
per-service-call history persistence, compaction, and a rich set of default
context providers (todo, mode, memory, skills).
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from ..agents import Agent, SupportsAgentRun
from ..compaction import CompactionProvider, ContextWindowCompactionStrategy
from ..sessions import ContextProvider, HistoryProvider, InMemoryHistoryProvider, MessageInjectionMiddleware
from ..skills import SkillsProvider
from ..telemetry import FeatureIndex, mark_feature_used
from ..types import ChatOptions
from .background_agents import BackgroundAgentsProvider
from .file_access import AgentFileStore, FileAccessProvider, FileSystemAgentFileStore
from .file_memory import FileMemoryProvider
from .loop import DEFAULT_MAX_ITERATIONS, AgentLoopMiddleware
from .mode import AgentModeProvider
from .todo import TodoProvider
from .tool_approval import ToolApprovalMiddleware, create_yolo_approval_rule

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    from collections.abc import Mapping

    from giskard.tools.shell import ShellEnvironmentProviderOptions, ShellExecutor
    from giskard.tools.web_search import ParallelSearchClient

    from ..clients import SupportsChatGetResponse
    from ..compaction import CompactionStrategy, TokenizerProtocol
    from ..middleware import MiddlewareTypes
    from ..tools import ToolTypes
    from .loop import NextMessageCallable, ShouldContinueCallable
    from .tool_approval import ToolApprovalRuleCallback

DEFAULT_HARNESS_INSTRUCTIONS = """\
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
"""


def _assemble_instructions(
    harness_instructions: str | None,
    agent_instructions: str | None,
) -> str | None:
    """Assemble final instructions from harness + agent instructions."""
    harness = harness_instructions if harness_instructions is not None else DEFAULT_HARNESS_INSTRUCTIONS

    return f"{harness}\n\n{agent_instructions or ''}".strip() or None


def _assemble_compaction(
    *,
    disable_compaction: bool,
    max_context_window_tokens: int | None,
    max_output_tokens: int | None,
    history_source_id: str,
    before_compaction_strategy: CompactionStrategy | None,
    after_compaction_strategy: CompactionStrategy | None,
    tokenizer: TokenizerProtocol | None,
) -> tuple[CompactionStrategy | None, CompactionProvider | None]:
    """Resolve the harness compaction strategies into their execution sites.

    The token-budget default (``ContextWindowCompactionStrategy``) is used for **both** the before
    and after phases, and is only applied when the token params are provided. A single shared
    instance is reused across both phases (it is stateless). Caller-supplied strategies always win
    per phase.

    Because the harness enables per-service-call history persistence, running the before-phase
    compaction as a ``CompactionProvider.before_run`` hook is a no-op: the agent skips
    ``HistoryProvider.before_run``, so the provider only ever sees an empty context.
    Instead the before-strategy is returned separately so the caller can wire it as the agent's
    ``compaction_strategy`` chat option. That runs it inside ``BaseChatClient.get_response`` —
    per model call, inner of ``PerServiceCallHistoryPersistingMiddleware`` (which has already
    loaded the full history into the outgoing messages) and outer of the leaf client.

    The after-strategy stays on a ``CompactionProvider`` (its ``after_run`` compacts the persisted
    session history in place and works correctly).

    Returns a ``(before_strategy, after_provider)`` tuple. Both elements are ``None`` when
    compaction is disabled or when the corresponding phase has no strategy (no custom strategy
    and no token budget to build the default).
    """
    if disable_compaction:
        return None, None

    # Token-budget default, shared across both phases when token params are available.
    default_strategy: CompactionStrategy | None = None
    if max_context_window_tokens is not None and max_output_tokens is not None:
        default_strategy = ContextWindowCompactionStrategy(
            max_context_window_tokens=max_context_window_tokens,
            max_output_tokens=max_output_tokens,
            tokenizer=tokenizer,
        )

    # Resolve each phase: caller-supplied strategy wins; otherwise fall back to the shared default.
    before_strategy = before_compaction_strategy if before_compaction_strategy is not None else default_strategy
    after_strategy = after_compaction_strategy if after_compaction_strategy is not None else default_strategy

    # The after-strategy runs post-turn against the persisted history via a CompactionProvider;
    # the before-strategy runs per model call via the agent's compaction_strategy option.
    after_provider = (
        CompactionProvider(
            before_strategy=None,
            after_strategy=after_strategy,
            tokenizer=tokenizer,
            history_source_id=history_source_id,
        )
        if after_strategy is not None
        else None
    )
    return before_strategy, after_provider


def _assemble_context_providers(
    *,
    history_provider: HistoryProvider,
    compaction_provider: CompactionProvider | None,
    disable_todo: bool,
    todo_provider: TodoProvider | None,
    disable_mode: bool,
    mode_provider: AgentModeProvider | None,
    disable_file_memory: bool,
    file_memory_store: AgentFileStore | None,
    workdir: Path,
    file_access_store: AgentFileStore | None,
    file_access_disable_write_tools: bool,
    file_access_enable_extra_tools: bool,
    file_access_disable_readonly_tool_approval: bool,
    file_access_disable_write_tool_approval: bool,
    skills_provider: SkillsProvider | None,
    skills_paths: str | Path | Sequence[str | Path] | None,
    background_agents: Sequence[SupportsAgentRun] | None,
    background_agents_instructions: str | None,
    shell_context_provider: ContextProvider | None,
    extra_context_providers: Sequence[ContextProvider] | None,
) -> list[ContextProvider]:
    """Assemble the ordered list of context providers."""
    providers: list[ContextProvider] = []

    # History first so other providers can access loaded messages.
    providers.append(history_provider)

    # Compaction runs after history loads messages.
    if compaction_provider is not None:
        providers.append(compaction_provider)

    if not disable_todo:
        providers.append(todo_provider or TodoProvider())

    if not disable_mode:
        providers.append(mode_provider or AgentModeProvider())

    # File-based session memory (on by default). Default store is rooted at
    # ``{workdir}/agent-file-memory``; the provider isolates memories per session
    # via its default ``scope=session_id``.
    if not disable_file_memory:
        memory_store = file_memory_store or FileSystemAgentFileStore(
            (workdir / "agent-file-memory").resolve()
        )
        providers.append(FileMemoryProvider(memory_store))

    # Shared file access (opt-in). Only added when a store is supplied.
    if file_access_store is not None:
        providers.append(
            FileAccessProvider(
                file_access_store,
                disable_write_tools=file_access_disable_write_tools,
                enable_extra_tools=file_access_enable_extra_tools,
                disable_readonly_tool_approval=file_access_disable_readonly_tool_approval,
                disable_write_tool_approval=file_access_disable_write_tool_approval,
            )
        )

    # Skills are opt-in: only added when skills_provider or skills_paths is provided.
    if skills_provider:
        providers.append(skills_provider)
    if skills_paths:
        providers.append(SkillsProvider.from_paths(skills_paths))

    # Background agents are opt-in: only added when agents are provided.
    if background_agents:
        providers.append(BackgroundAgentsProvider(background_agents, instructions=background_agents_instructions))

    # Shell environment provider is opt-in: only added when a shell tool was wired.
    if shell_context_provider is not None:
        providers.append(shell_context_provider)

    # Append any user-supplied additional providers.
    if extra_context_providers:
        providers.extend(extra_context_providers)

    return providers


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


HARNESS_AGENT_PROVIDER_NAME = "giskard.harness"

OptionsCoT = TypeVar(
    "OptionsCoT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="ChatOptions[None]",
)


def create_harness_agent(
    client: SupportsChatGetResponse[OptionsCoT],
    *,
    id: str | None = None,
    name: str | None = None,
    description: str | None = None,
    harness_instructions: str | None = None,
    agent_instructions: str | None = None,
    tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
    max_context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    history_provider: HistoryProvider | None = None,
    disable_compaction: bool = False,
    before_compaction_strategy: CompactionStrategy | None = None,
    after_compaction_strategy: CompactionStrategy | None = None,
    tokenizer: TokenizerProtocol | None = None,
    disable_todo: bool = False,
    todo_provider: TodoProvider | None = None,
    disable_mode: bool = False,
    mode_provider: AgentModeProvider | None = None,
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
    auto_approval_rules: Sequence[ToolApprovalRuleCallback] | None = None,
    loop_should_continue: ShouldContinueCallable | None = None,
    loop_next_message: NextMessageCallable | None = None,
    loop_max_iterations: int | None = DEFAULT_MAX_ITERATIONS,
    otel_provider_name: str | None = None,
    context_providers: Sequence[ContextProvider] | None = None,
    middleware: MiddlewareTypes | Sequence[MiddlewareTypes] | None = None,
    default_options: Mapping[str, Any] | None = None,
) -> Agent[OptionsCoT]:
    """Create a pre-configured agent with batteries included.

    Assembles an :class:`~giskard.Agent` from a chat client, automatically wiring:

    - **Function invocation** — automatic tool calling loop
    - **Per-service-call history persistence** — persists history after every model call
    - **Compaction** — context-window compaction before/after each run
    - **TodoProvider** — todo list management
    - **AgentModeProvider** — plan/execute mode tracking
    - **FileMemoryProvider** — file-based session memory (on by default)
    - **FileAccessProvider** — shared file read/write tools (on by default,
      backed by a ``FileSystemAgentFileStore`` rooted at the current directory;
      disable via ``disable_file_access`` or customize via ``file_access_store``)
    - **Web search** — ``web_search`` and ``web_fetch`` tools via
      ``ParallelSearchClient`` (on by default; disable via
      ``disable_web_search`` or supply your own via ``web_search_client``)
    - **Shell tool** — local shell command execution (on by default via
      ``LocalShellTool`` anchored at ``workdir``; disable via ``disable_shell``
      or customize via ``shell_executor``)
    - **SkillsProvider** — skill discovery and progressive loading
    - **BackgroundAgentsProvider** — delegate work to background sub-agents
    - **Tool approval** — "don't ask again" standing approval rules plus heuristic
      auto-approval callbacks
    - **Looping** — re-run the agent until a ``should_continue`` predicate is satisfied
    - **OpenTelemetry** — observability via ``AgentTelemetryLayer``

    Each feature can be disabled or customized via keyword arguments.

    Examples:
        Basic usage:

        .. code-block:: python

            from giskard import create_harness_agent
            from giskard.providers import OpenAIChatClient

            agent = create_harness_agent(
                OpenAIChatClient(model="gpt-4o"),
            )
            session = agent.create_session()
            response = await agent.run("Plan a weekend trip to Seattle", session=session)

        With customization:

        .. code-block:: python

            agent = create_harness_agent(
                client=client,
                max_context_window_tokens=200_000,
                max_output_tokens=32_000,
                name="research-agent",
                agent_instructions="Focus on academic sources.",
                disable_todo=True,
                skills_paths=["./skills", "./custom-skills"],
            )

    Args:
        client: The chat client providing access to the underlying AI model.

    Keyword Args:
        id: Optional agent ID (auto-generated UUID if omitted).
        name: Optional agent name.
        description: Optional agent description.
        harness_instructions: Override the default harness-level system instructions that
            govern agent behavior (how to use tools, report progress, structure responses).
            These provide general "operating guidelines" independent of any specific task.
            When None, ``DEFAULT_HARNESS_INSTRUCTIONS`` is used. Set to empty string ``""``
            to omit harness instructions entirely.
        agent_instructions: Domain or task-specific instructions appended after harness
            instructions. Use this for the agent's purpose, persona, or specialization
            (e.g., "You are a research assistant focused on academic sources.").
        tools: Additional tools to include in the agent's toolset.
        max_context_window_tokens: Maximum tokens the model's context window supports.
            Used to construct the default token-budget-aware compaction strategies. When None
            (default) and no custom ``before_compaction_strategy`` / ``after_compaction_strategy``
            is provided, compaction is automatically disabled.
        max_output_tokens: Maximum output tokens per response.
            Used to construct the default compaction strategies and sets a default max_tokens
            chat option. When None (default), no default max_tokens option is set, and unless a
            custom compaction strategy is provided, compaction is automatically disabled.
        history_provider: Custom history provider. When None, an InMemoryHistoryProvider is used.
        disable_compaction: When True, skip compaction provider setup.
        before_compaction_strategy: Custom before-run compaction strategy. When provided,
            compaction runs even if token params are omitted. Defaults to
            ContextWindowCompactionStrategy (token-budget aware) when token params are provided.
        after_compaction_strategy: Custom after-run compaction strategy. When provided,
            compaction runs even if token params are omitted. Defaults to the same
            ContextWindowCompactionStrategy used for the before phase when token params are
            provided.
        tokenizer: Custom tokenizer for compaction strategies.
        disable_todo: When True, skip the TodoProvider.
        todo_provider: Custom TodoProvider instance. Ignored when disable_todo is True.
        disable_mode: When True, skip the AgentModeProvider.
        mode_provider: Custom AgentModeProvider instance. Ignored when disable_mode is True.
        disable_file_memory: When True, skip the FileMemoryProvider. When False (default),
            a FileMemoryProvider is added, giving the agent session-scoped, file-based memory.
        file_memory_store: Custom AgentFileStore backing the FileMemoryProvider. When None
            (and disable_file_memory is False), a FileSystemAgentFileStore rooted at
            ``{workdir}/agent-file-memory`` is created. Ignored when disable_file_memory is True.
        workdir: The working directory that roots all file I/O — the shared
            file-access store, the session file-memory store
            (``{workdir}/agent-file-memory``), the default shell tool's
            execution directory, and the YOLO approval boundary. When None
            (default), the current working directory is used. Explicitly
            supplied ``file_access_store`` / ``file_memory_store`` /
            ``shell_executor`` are not overridden by this value. When ``workdir``
            is None, the default ``LocalShellTool`` is anchored at ``Path.cwd()``
            and re-anchors each persistent-shell command there (``confine_workdir``),
            whereas previously it was constructed without a workdir.
        disable_file_access: When True, skip the FileAccessProvider. When False (default),
            file access tools are enabled: when ``file_access_store`` is None, a
            ``FileSystemAgentFileStore`` rooted at ``workdir`` is created.
        file_access_store: AgentFileStore backing the FileAccessProvider. When None and
            ``disable_file_access`` is False (default), a store rooted at
            ``workdir`` is created. When set, the supplied store backs the agent's
            shared read/write file tools.
        file_access_disable_write_tools: When True, the FileAccessProvider advertises only its
            read-only tools (read_file, glob, grep, and the optional ls); the write tools
            (write_file, delete_file, edit_file, edit_file_lines) are hidden. When False
            (default), all enabled tools are advertised. Only used when file_access_store is set.
        file_access_enable_extra_tools: When True, the FileAccessProvider's optional tools (ls,
            delete_file, edit_file_lines) are advertised in addition to the default tools
            (read_file, write_file, edit_file, glob, grep). When False (default), only the default
            tools are advertised. Only used when file_access_store is set.
        file_access_disable_readonly_tool_approval: When True, the FileAccessProvider's read-only
            tools (read_file, ls, glob, grep) are registered with ``approval_mode="never_require"`` so they
            run without host approval. When False (default), they require approval. Only used when
            file_access_store is set.
        file_access_disable_write_tool_approval: When True, the FileAccessProvider's write tools
            (write_file, delete_file, edit_file, edit_file_lines) are registered with
            ``approval_mode="never_require"`` so they run without host approval. When False
            (default), they require approval. Only used when file_access_store is set.
        skills_provider: Custom SkillsProvider instance for code-defined skills.
            Can be combined with ``skills_paths`` to aggregate file and code-based skills.
            **Security:** if the provider is configured with an external skill source (e.g.
            :class:`~giskard.MCPSkillsSource`), the skill content it loads is untrusted input
            — only enable sources you trust; see :class:`~giskard.SkillsSource`.
        skills_paths: Paths for file-based skill discovery (looks for SKILL.md files).
            Accepts a single ``str`` or :class:`~pathlib.Path`, or a sequence of
            ``str | Path``. Can be combined with ``skills_provider``. When neither
            ``skills_provider`` nor ``skills_paths`` is provided, no SkillsProvider
            is added.
        background_agents: Collection of agents available for background task delegation.
            When provided, a ``BackgroundAgentsProvider`` is automatically included,
            enabling the agent to start, monitor, and retrieve results from background tasks.
            Each agent must have a non-empty, unique name (case-insensitive).
            **Security:** supplied agents receive text input from this agent and their output is fed
            back into its context, so only supply agents you have vetted and trust — see
            :class:`~giskard.BackgroundAgentsProvider` for the exfiltration and
            prompt-injection risks of untrusted agents.
        background_agents_instructions: Optional instruction override for the
            ``BackgroundAgentsProvider``. May include ``{background_agents}`` placeholder
            which will be replaced with the agent listing.
        disable_shell: When True, skip the shell tool and ShellEnvironmentProvider. When
            False (default), a ``LocalShellTool`` anchored at ``workdir`` is created
            automatically (platform default shell, persistent mode, approval required per
            command).
        shell_executor: Optional shell executor overriding the default ``LocalShellTool``. When
            provided, the shell tool and a ``ShellEnvironmentProvider`` are wired from it. The
            object must expose ``as_function()`` and satisfy the ``ShellExecutor`` protocol --
            e.g. a ``LocalShellTool`` or ``DockerShellTool``. The caller owns the executor's
            lifecycle and its workdir configuration (the harness does not inject ``workdir``).
        shell_environment_provider_options: Optional ``ShellEnvironmentProviderOptions``
            (from ``agent-framework-tools``) used to customize the ``ShellEnvironmentProvider``
            environment probing and instructions. Only used when ``shell_executor`` is provided.
        web_search_client: Optional ``ParallelSearchClient`` supplying the
            ``web_search`` and ``web_fetch`` tools. When None (default) and
            ``disable_web_search`` is False, a new client is created (it
            connects lazily on first invocation; the caller owns any supplied
            instance and its lifecycle).
        disable_web_search: When True, skip the web search tools. When False (default),
            ``web_search`` and ``web_fetch`` are added via ``ParallelSearchClient``.
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
        disable_tool_auto_approval: When True, do not wire the tool auto-approval middleware.
            When False (default), a :class:`~giskard.ToolApprovalMiddleware` is added
            (outermost) to coordinate "don't ask again" standing approval rules and queued
            approval prompts; callers must pass an :class:`~giskard.AgentSession` to
            :meth:`~giskard.Agent.run` when enabled.
        auto_approval_rules: Optional heuristic callbacks that can auto-approve a function call
            that would otherwise require approval. Each callback receives the ``function_call``
            content and returns ``True`` to approve it. Rules are evaluated after standing rules
            (derived from prior user approvals) but before prompting the user. Only used when
            ``disable_tool_auto_approval`` is False.
        loop_should_continue: Optional predicate that enables the looping middleware. When provided, the
            agent is re-run in a loop (via :class:`~giskard.AgentLoopMiddleware`, wired as
            the outermost middleware so each iteration is a full agent run including tool approval)
            for as long as the predicate returns ``True``, up to ``loop_max_iterations``. If an
            iteration returns a pending tool-approval request, the loop stops and returns it so the
            caller can approve before continuing. When None (default), no loop is added.
        loop_next_message: Optional callable controlling the input for the next loop iteration.
            Only takes effect when ``loop_should_continue`` is set (otherwise no loop is added and
            this is ignored).
        loop_max_iterations: Safety cap on the number of loop iterations. ``None`` means unbounded;
            a positive integer caps the loop (defaults to the loop middleware's default cap). Only
            takes effect when ``loop_should_continue`` is set (otherwise no loop is added and this
            is ignored).
        otel_provider_name: Custom OpenTelemetry provider/source name for telemetry.
        context_providers: Additional context providers to include after the built-in ones.
        middleware: Additional middleware to include.
        default_options: Provider-specific chat options (temperature, max_tokens, etc.).

    Returns:
        A fully configured :class:`~giskard.Agent` instance.

    Raises:
        ValueError: If max_context_window_tokens is provided and <= 0, or
            max_output_tokens is provided and <= 0, or max_output_tokens >=
            max_context_window_tokens when both are provided, or
            tool_approval_rule is not None or "yolo", or tool_approval_rule
            is "yolo" and disable_tool_auto_approval is True.
    """
    if max_context_window_tokens is not None and max_context_window_tokens <= 0:
        raise ValueError("max_context_window_tokens must be positive.")
    if max_output_tokens is not None and max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive.")
    if (
        max_context_window_tokens is not None
        and max_output_tokens is not None
        and max_output_tokens >= max_context_window_tokens
    ):
        raise ValueError("max_output_tokens must be less than max_context_window_tokens.")
    if tool_approval_rule not in (None, "yolo"):
        raise ValueError(f"tool_approval_rule must be None or 'yolo', got {tool_approval_rule!r}.")
    if tool_approval_rule == "yolo" and disable_tool_auto_approval:
        raise ValueError(
            "tool_approval_rule='yolo' requires the tool auto-approval middleware; "
            "set disable_tool_auto_approval=False."
        )

    # One working directory roots all file I/O: the shared file-access store,
    # the session file-memory store, the default shell tool, and the YOLO
    # approval boundary. Existing explicit stores still win.
    resolved_workdir = Path(workdir).resolve() if workdir is not None else Path.cwd().resolve()
    if file_access_store is None and not disable_file_access:
        file_access_store = FileSystemAgentFileStore(resolved_workdir)

    # Build history provider.
    resolved_history = history_provider or InMemoryHistoryProvider()

    # Build compaction. The before-strategy is wired as the agent's compaction_strategy option
    # (runs per model call, inner of per-service-call persistence); the after-strategy stays on a
    # CompactionProvider that compacts the persisted history post-turn. See issue #7011.
    before_compaction, compaction_provider = _assemble_compaction(
        disable_compaction=disable_compaction,
        max_context_window_tokens=max_context_window_tokens,
        max_output_tokens=max_output_tokens,
        history_source_id=resolved_history.source_id,
        before_compaction_strategy=before_compaction_strategy,
        after_compaction_strategy=after_compaction_strategy,
        tokenizer=tokenizer,
    )

    # Shell tooling is on by default: create a LocalShellTool unless the caller
    # supplies an executor or opts out. Imported lazily: the shell types live in
    # giskard.tools.shell, which depends on core, so core cannot import them at
    # module load time.
    if shell_executor is None and not disable_shell:
        from giskard.tools.shell import LocalShellTool

        shell_executor = LocalShellTool(workdir=resolved_workdir)

    # Build the shell tool and environment provider (opt-in via shell_executor).
    shell_tool, shell_provider = _assemble_shell(
        shell_executor,
        shell_environment_provider_options,
    )

    # Build context providers.
    assembled_providers = _assemble_context_providers(
        history_provider=resolved_history,
        compaction_provider=compaction_provider,
        disable_todo=disable_todo,
        todo_provider=todo_provider,
        disable_mode=disable_mode,
        mode_provider=mode_provider,
        disable_file_memory=disable_file_memory,
        file_memory_store=file_memory_store,
        workdir=resolved_workdir,
        file_access_store=file_access_store,
        file_access_disable_write_tools=file_access_disable_write_tools,
        file_access_enable_extra_tools=file_access_enable_extra_tools,
        file_access_disable_readonly_tool_approval=file_access_disable_readonly_tool_approval,
        file_access_disable_write_tool_approval=file_access_disable_write_tool_approval,
        skills_provider=skills_provider,
        skills_paths=skills_paths,
        background_agents=background_agents,
        background_agents_instructions=background_agents_instructions,
        shell_context_provider=shell_provider,
        extra_context_providers=context_providers,
    )

    # Build instructions.
    instructions = _assemble_instructions(harness_instructions, agent_instructions)

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
    if shell_tool is not None:
        assembled_tools.append(shell_tool)
    if tools is not None:
        if isinstance(tools, Sequence):
            assembled_tools.extend(tools)  # pyright: ignore[reportUnknownArgumentType]
        else:
            assembled_tools.append(tools)
    final_tools: list[ToolTypes | Callable[..., Any]] | None = assembled_tools or None

    # Build default options dict.
    default_opts: dict[str, Any] = dict(default_options) if default_options else {}
    if max_output_tokens is not None:
        default_opts.setdefault("max_tokens", max_output_tokens)

    # Assemble middleware. Tool approval is enabled by default (like the .NET harness) and is
    # placed first so it sits outermost: it intercepts inbound "always approve" responses and
    # outbound approval requests at the caller boundary, and its re-invocation loop re-runs any
    # user-supplied middleware. ToolApprovalMiddleware requires an AgentSession at run time.
    # When should_continue is supplied, the loop is prepended ahead of tool approval so it sits
    # outermost of all: each loop iteration is a full agent run (including tool approval), and the
    # loop's approval escape hatch returns any pending approval request to the caller.
    # Resolve auto-approval rules: caller-supplied heuristics plus the YOLO
    # preset when requested. Evaluated after standing rules, before prompting.
    resolved_auto_approval_rules = list(auto_approval_rules or [])
    if tool_approval_rule == "yolo":
        resolved_auto_approval_rules.append(create_yolo_approval_rule(resolved_workdir))

    assembled_middleware: list[MiddlewareTypes] = []
    if not disable_tool_auto_approval:
        assembled_middleware.append(ToolApprovalMiddleware(auto_approval_rules=resolved_auto_approval_rules or None))
    if loop_should_continue is not None:
        assembled_middleware.insert(
            0,
            AgentLoopMiddleware(
                loop_should_continue,
                max_iterations=loop_max_iterations,
                next_message=loop_next_message,
            ),
        )
    # Message injection is always on. It is a no-op when no messages are queued for the session,
    # so there is no opt-out.
    assembled_middleware.append(MessageInjectionMiddleware())
    # Bare-source normalization (a single middleware object or a MiddlewareBundle is
    # one element) is owned by _as_middleware_list.
    from ..middleware import _as_middleware_list  # pyright: ignore[reportPrivateUsage]

    assembled_middleware.extend(_as_middleware_list(middleware))

    agent = Agent(
        client,
        instructions,
        id=id,
        name=name,
        description=description,
        tools=final_tools,
        default_options=default_opts,  # type: ignore[arg-type]
        context_providers=assembled_providers,
        middleware=assembled_middleware or None,
        compaction_strategy=before_compaction,
        tokenizer=tokenizer,
        require_per_service_call_history_persistence=True,
    )

    # Set the telemetry provider name after construction.
    agent.otel_provider_name = otel_provider_name or HARNESS_AGENT_PROVIDER_NAME
    mark_feature_used(FeatureIndex.CORE_HARNESS_AGENT)

    return agent
