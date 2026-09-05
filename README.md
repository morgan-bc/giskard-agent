# Giskard

Giskard 是一个独立的 Python Agent 框架，改编自 Microsoft Agent Framework（agent-framework-core），保留了其核心抽象（Agent、会话、中间件、工具调用循环），并在此基础上提供功能齐备、开箱即用的 **Harness Agent** 与 GAIA 基准评测流水线。

项目完全独立，不依赖 `agent_framework` 包；所有原语（`Agent`、`AgentSession`、中间件、工具装饰器等）均由本仓库自行实现。

## 特性

- **Agent**：支持流式/非流式输出、多轮工具调用循环、结构化输出、会话状态持久化与恢复
- **Harness Agent**（`create_harness_agent`）：开箱即用的组合配置
  - Per-service-call 历史持久化（每次模型调用后立即落盘，支持中途恢复）
  - 上下文工程：文件记忆、todo、计划/执行模式、技能（Skills）、后台子代理
  - 工具审批（YOLO 规则）：只读与工作目录内写操作自动放行，删除类操作需人工确认
  - 压缩（Compaction）、OpenTelemetry 可观测性
- **Provider**：OpenAI Chat Completions 与 Responses API 客户端，支持自定义响应解析（如 reasoning content）
- **工具**：本地 Shell、Python 执行器、并行网络搜索（`ParallelSearchClient` / Tavily）、MCP 工具接入
- **中间件**：Chat / Function / Agent 三层中间件管道，支持审批、消息注入、历史持久化等横切逻辑
- **评测**：GAIA 验证集跑分与评分脚本（FINAL ANSWER 协议 + 官方 scorer 移植）

## 安装

要求 Python >= 3.10。

```bash
# 开发模式安装（推荐 uv）
uv sync
# 或 pip
pip install -e .
```

## 快速开始

### 1. 配置环境变量

在项目根目录创建 `.env`：

```ini
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-endpoint/v1   # OpenAI 兼容 endpoint
BASE_MODEL=your-model-name
TAVILY_API_KEY=your-tavily-key          # 可选，web_search 工具所需
```

### 2. 运行示例

```bash
$env:PYTHONPATH='src'; python examples/yolo_agent.py
```

示例会创建一个 YOLO 审批规则的 harness agent，执行联网调研任务，并将完整轨迹（system prompt、工具列表、消息历史）保存到 `workdir/`。

### 3. 最小用法

```python
import asyncio
from giskard import Agent, create_harness_agent
from giskard.providers import OpenAIChatCompletionClient

async def main():
    client = OpenAIChatCompletionClient(
        api_key="...",
        base_url="https://your-endpoint/v1",
        model="your-model",
    )
    agent = create_harness_agent(
        name="assistant",
        client=client,
        workdir="./workdir",           # 文件操作与 shell 的根目录
        tool_approval_rule="yolo",     # 工具自动审批
    )
    session = agent.create_session()
    response = await agent.run("调研 AI Agent 最新进展，并把结果保存下来", session=session)
    print(response.text)

asyncio.run(main())
```

也可以直接使用裸 `Agent` 自行组装 provider / 中间件 / 上下文提供者。

## Harness Agent 常用配置

| 参数 | 说明 | 默认 |
| --- | --- | --- |
| `workdir` | shell 与文件工具的根目录 | 当前目录 |
| `tool_approval_rule` | `"yolo"` 启用自动审批规则 | `None` |
| `harness_instructions` | 追加到系统提示的 harness 指令 | `None` |
| `web_search_client` / `disable_web_search` | ParallelSearchClient搜索客户端 / 关闭联网搜索 | 自动创建 / 开启 |
| `shell_executor` / `disable_shell` | 自定义 shell 执行器 / 关闭 shell | `LocalShellTool` / 开启 |
| `disable_file_memory` / `disable_todo` / `disable_mode` / `disable_file_access` | 关闭对应内置提供者 | 全部开启 |
| `require_per_service_call_history_persistence` | 每次模型调用后持久化历史 | `True` |
| `middleware` / `context_providers` | 追加自定义中间件 / 提供者 | `None` |

## GAIA 基准测试

基于 `benchmarks/gaia/validation/dev.json`（含 ground truth）：

```bash
# 顺序执行全部任务（结果与轨迹增量写入 runs/<时间戳>/）
$env:PYTHONPATH='src'; python benchmarks/gaia/run_gaia.py

# 可选参数：--limit N / --level 1|2|3 / --task-ids id1,id2 / --timeout 1800 / --max-turns 30 / --force

# 评分（官方 GAIA scorer 规则，输出准确率与错误分布）
$env:PYTHONPATH='src'; python benchmarks/gaia/evaluate_gaia.py --results-dir benchmarks/gaia/runs/<时间戳>
```

产物说明：
- `results.jsonl` — 每任务一行的增量结果（预测、错误、轮数、耗时）
- `transcripts/<task_id>.json` — 完整轨迹（system_prompt / tools / messages）
- 支持断点续跑：重跑时自动跳过 `results.jsonl` 中已有的任务（`--force` 强制重跑）

## 测试

```bash
$env:PYTHONPATH='src'; python -m pytest tests/
```

## 项目结构

```
src/giskard/
├── core/
│   ├── agents.py            # Agent、运行上下文、options 合并
│   ├── harness/             # create_harness_agent 及内置提供者/中间件
│   ├── clients.py           # 聊天客户端基类
│   ├── sessions.py          # AgentSession、HistoryProvider、消息注入
│   ├── middleware.py        # 三层中间件管道
│   ├── tools.py             # 工具装饰器与函数调用循环
│   ├── security.py          # 安全策略与审批
│   ├── compaction.py        # 上下文压缩
│   ├── mcp.py               # MCP 工具接入
│   └── types.py             # Message/Content/ChatResponse 等核心类型
├── providers/openai/        # OpenAI Chat Completions / Responses 客户端
└── tools/
    ├── shell/               # LocalShellTool（本地命令执行）
    ├── python/              # LocalPythonExecutor（Python 代码执行）
    └── web_search/          # ParallelSearchClient / Tavily
examples/yolo_agent.py       # harness agent 完整示例
benchmarks/gaia/             # GAIA 跑分与评测脚本
tests/                       # 单元与回归测试
```

## License

MIT
