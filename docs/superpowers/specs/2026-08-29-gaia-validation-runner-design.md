# Design: GAIA 验证集运行与评估脚本

Date: 2026-08-29
Status: Approved
Scope: `benchmarks/gaia/gaia_scorer.py`, `benchmarks/gaia/run_gaia.py`, `benchmarks/gaia/evaluate_gaia.py`, `tests/test_gaia_scorer.py`

## Goals

1. 用当前 harness（`create_harness_agent` + YOLO 审批 + `ParallelSearchClient` + `LocalShellTool`）
   在 GAIA 验证集（`benchmarks/gaia/validation/dev.json`，103 题，含 ground_truth，无附件）上
   逐题运行 agent 并落盘结果。
2. 用 GAIA 官方 `question_scorer` 的移植版对结果打分，输出总体/分组准确率与错误分类。

## Non-Goals

- 不做并发执行（顺序跑通优先，`--workers` 留待将来）。
- 不支持数据集附件（本题集 `file_name` 全空）。
- 不修改 harness 本体。

## 1. 文件结构与数据流

```
benchmarks/gaia/validation/dev.json
        │
        ▼
run_gaia.py ──逐题──▶ runs/<timestamp>/
   │                    ├── results.jsonl        # 每题一条，实时追加
   │                    ├── transcripts/<task_id>.json
   │                    └── workdir/              # agent 工作目录（YOLO 边界）
        │
        ▼
evaluate_gaia.py ──▶ runs/<timestamp>/scored.jsonl + 控制台汇总
```

| 文件 | 职责 |
|---|---|
| `benchmarks/gaia/gaia_scorer.py` | 官方 `question_scorer` 移植，纯函数、零第三方依赖 |
| `benchmarks/gaia/run_gaia.py` | 运行器：加载题目 → 逐题跑 harness → 落盘 |
| `benchmarks/gaia/evaluate_gaia.py` | 评估器：读 results.jsonl → 打分 → 汇总报告 |

## 2. gaia_scorer.py（官方 scorer 移植）

- 移植 GAIA 官方 `question_scorer` 三分支逻辑：
  1. **数字题**：ground_truth 可 float → 预测归一化（去 `,` `$` `%`）后数值比较
  2. **列表题**：ground_truth 含 `,` 或 `;` → 按分隔符拆分逐项比对，匹配率 > 50% 计分
  3. **字符串题**：小写、去标点、去冠词（a/an/the）后精确比较
- 官方在数字解析失败时用 `word2number` 转换英文数字词；为不加依赖，内置一个覆盖
  0–999 组合的轻量 `w2n` fallback，解析失败返回 NaN → 判 False
- 返回 `(score: bool, detail: str)`，detail 记录走了哪个分支

## 3. run_gaia.py（运行器）

**CLI 参数**：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | `benchmarks/gaia/validation/dev.json` | 题目文件 |
| `--output-dir` | `benchmarks/gaia/runs/<YYYYmmdd-HHMMSS>/` | 本次运行产物目录 |
| `--limit N` | 无 | 只跑前 N 题（配合 level/task-ids 筛选后） |
| `--level 1\|2\|3` | 无 | 只跑指定 Level |
| `--task-ids` | 无 | 只跑指定 task_id（逗号分隔） |
| `--timeout` | 1800 | 单题墙钟超时（秒） |
| `--max-turns` | 60 | 单题最大 assistant 回合数 |
| `--force` | 关 | 忽略断点续跑，重跑所有选中题 |

**单题执行流程**：

1. 断点续跑：启动时读 results.jsonl 中已有 task_id，跳过（`--force` 时重跑）
2. `create_harness_agent(workdir=runs/<ts>/workdir, tool_approval_rule="yolo",
   web_search_client=共享的 ParallelSearchClient)`；一个 agent 实例复用，每题新建 session
3. Prompt = GAIA 官方 FINAL ANSWER 指令 + 题目 Question
4. `asyncio.wait_for` 包裹整题（超时 → error=`timeout`，prediction 置空）
5. 流式迭代 updates：
   - 按 `message_id` 分组计数 assistant 回合，超过 `--max-turns` 主动 break 中断流
     （error=`max_turns`）
   - 同时把全部 update 序列收集为 transcript
6. 从 `response.text` 提取最后一个 `FINAL ANSWER:` 标记后的内容作为 prediction；
   无标记 → prediction 空，error=`missing_final_answer`
7. 追加一条到 results.jsonl：
   `{task_id, id, level, question, prediction, ground_truth, duration_s, turns, error, model}`
8. 写 transcripts/<task_id>.json（全部 updates 的 to_dict）

**生命周期**：整个运行共享一个 `ParallelSearchClient`，结束后统一 `await close()`
（沿用 `examples/yolo_agent.py` 模式，规避 MCP 连接残留问题）。
配置从 `.env` 读 `LLM_API_KEY` / `LLM_BASE_URL` / `BASE_MODEL`。

**运行方式**：`PYTHONPATH=src python benchmarks/gaia/run_gaia.py ...`（遵循仓库约定）。

## 4. evaluate_gaia.py（评估器）

**CLI**：`--results-dir`（必填，指向 run 目录）、`--dataset`（默认 dev.json）

- 逐题调用 scorer 打分 → 写 `scored.jsonl`（results 原字段 + `score` + `score_detail`）
- 控制台输出：
  - 总体准确率 + 各 Level 分组准确率
  - error 分类统计（timeout / max_turns / missing_final_answer / api_error）
  - 错题清单（task_id + prediction vs ground_truth，便于人工复核）
- 幂等，重跑覆盖 scored.jsonl

## 5. 错误处理

- 每题独立 try/except：API 异常记 `error=api_error:<类名>`、prediction 空串，继续下一题
- `KeyboardInterrupt` 安全：已完成的题已在 jsonl 中，重启自动续跑
- 轮数超限不视为 crash：中断流后若已有可提取的文本仍尝试提取 prediction

## 6. 测试

- `tests/test_gaia_scorer.py`：scorer 纯函数单测（数字归一化、`$`/`%`/逗号、冠词剔除、
  列表拆分与 50% 阈值、英文数字词 fallback）
- runner 链路冒烟：`PYTHONPATH=src python benchmarks/gaia/run_gaia.py --limit 1` 手动验证
  （需要真实 API）
