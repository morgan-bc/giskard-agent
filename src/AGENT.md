# AGENT.md — giskard

Python agent framework. Package name `giskard-agent`, import root `giskard`.
Python 3.10+, src layout, flit-core build.

## Commands

```bash
# Run code (no install needed)
$env:PYTHONPATH='src'; python <script.py>          # PowerShell
PYTHONPATH=src python <script.py>                   # bash

# Build
flit build                                          # outputs to dist/

# Compile check (fast syntax validation)
python -m compileall src/giskard -q

# Lint / typecheck
ruff check src tests
pyright                                             # config in pyproject.toml
```

No test suite yet (`tests/` is empty). Verify changes with `compileall` + a smoke script under `PYTHONPATH=src`.

## Layout

```
src/giskard/
├── __init__.py          # __version__ via importlib.metadata("giskard-agent"); exports Agent, create_harness_agent
├── core/                # Framework core
│   ├── agents.py        # Agent class, SupportsAgentRun protocol
│   ├── clients.py       # BaseChatClient, chat client protocols
│   ├── tools.py         # FunctionTool, @tool decorator, tool normalization
│   ├── types.py         # Content, Message, ChatOptions, ChatResponse
│   ├── mcp.py           # MCPTool + MCPStdioTool/MCPStreamableHTTPTool/MCPWebsocketTool
│   ├── middleware.py    # Middleware pipeline, MiddlewareBundle
│   ├── sessions.py      # AgentSession, history persistence
│   ├── skills.py        # SkillResource ABC, FileSkillResource, SkillsProvider (~3.7k lines)
│   ├── compaction.py    # ContextWindowCompactionStrategy
│   ├── harness/         # agents.py: create_harness_agent factory; todo/mode/memory/file_access providers
│   ├── workflows/       # Graph-based multi-agent workflows
│   └── _feature_stage.py# ExperimentalFeature enum + @experimental decorator
├── providers/
│   └── openai/          # OpenAIChatClient etc. Azure support REMOVED by design
└── tools/
    ├── shell/           # LocalShellTool (cross-OS shell execution)
    └── web_search/      # ParallelSearchClient wrapping https://search.parallel.ai/mcp
```

## Conventions

- **Imports**: always `giskard.core.X` / `giskard.providers.X` / `giskard.tools.X`. No legacy package names anywhere.
- **Version**: single source of truth is `pyproject.toml [project].version`. `__init__.py` reads it via `importlib.metadata.version("giskard-agent")`. Never hardcode versions elsewhere.
- **Experimental features**: gate with `@experimental(feature_id=ExperimentalFeature.X)` from `_feature_stage.py`. Unused enum members may be pruned.
- **Lazy exports**: `core/__init__.py` uses `__getattr__` lazy module resolution — keep module paths in `_LAZY_MODULE_EXPORTS` in sync with actual filenames (files have no underscore prefix; e.g. `.middleware` not `._middleware`).
- **Circular imports**: `giskard/__init__.py` must define `__version__` BEFORE importing `.core.agents` (observability reads it at import time). Use local imports inside functions to break cycles.
- **Style**: ruff + pyright strict-ish; type hints everywhere; Google-style docstrings.
