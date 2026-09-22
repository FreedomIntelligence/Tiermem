# 使用示例

先完成 [API 配置](../README.md#2-配置-api)。以下命令均在 Tiermem 仓库根目录执行；启动脚本会自动创建环境并安装所需依赖。

## Demo 1：写入记忆，再重新打开使用

```bash
bash jev_tiermem/run_demo.sh coding
```

脚本完整执行以下流程，无需手工准备 summary 或原文：

1. 执行仓库中的真实 router 单元测试，生成 `tests.log`。
2. 用 `observe` 原样保存测试输出。
3. 用普通文本模型生成简短 summary，绑定原文 `raw_ids`。
4. 关闭客户端，再用同一个 session 打开记忆。
5. 追问包含 `network_failure` 的完整测试名及其结果，由 Jev 路由并返回答案与证据。

### 原文保存成了什么记忆

下面是一次实际运行保存的内容。测试日志由程序执行命令产生，summary 由文本模型当场生成。

**① 原文：写入 `raw.sqlite3` 的测试日志。** `observe` 原样保存，返回原文 ID `r-a488cdf77476b87040ec8a45`：

```text
Command: python -m unittest jev_tiermem.tests.test_memory.RouterTests -v
Exit code: 0
test_all_write_conditions_required (jev_tiermem.tests.test_memory.RouterTests.test_all_write_conditions_required) ... ok
test_invalid_or_missing_probability_fails_closed (jev_tiermem.tests.test_memory.RouterTests.test_invalid_or_missing_probability_fails_closed) ... ok
test_network_failure_fails_closed (jev_tiermem.tests.test_memory.RouterTests.test_network_failure_fails_closed) ... ok
test_threshold_and_usage (jev_tiermem.tests.test_memory.RouterTests.test_threshold_and_usage) ... ok

----------------------------------------------------------------------
Ran 4 tests in 0.000s

OK
```

**② 记忆：实际落盘的 `MEMORY.md`。** `compact` 调用普通文本模型概括日志，再由 Jev TierMem 保存摘要与来源索引：

```markdown
# Memory

<!-- jev_tiermem {"id": "s-f62b689845bc513b1d3bcf2b", "raw_ids": ["r-a488cdf77476b87040ec8a45"], "kind": "summary"} -->
- Tool: `python -m unittest jev_tiermem.tests.test_memory.RouterTests -v` passed.
- 4 tests OK: write conditions, invalid/missing probability fails closed, network failure fails closed, threshold/usage.
<!-- /jev_tiermem -->
```

其中 `s-...` 是摘要 ID，`raw_ids` 指向上面的原始日志。摘要记住了“4 项测试通过”，原文保留完整测试名等细节。索引由 Jev TierMem 根据实际保存的记录建立。

**③ 后续提问：新客户端只接收问题，通过记忆找回细节。** 问的是“之前日志中包含 `network_failure` 的完整测试名是什么？结果如何？”demo 实际传入的英文问题是：

> What is the fully qualified name of the test containing 'network_failure' in tests.log, and what was its result?

**④ 模型最终回答：**

> jev_tiermem.tests.test_memory.RouterTests.test_network_failure_fails_closed — ok

引用证据为 `r-a488cdf77476b87040ec8a45`。摘要保存了测试通过的概况；完整函数路径从该原始日志恢复。此次路由与检查结果：

```text
summary: 不足
raw history: 足够
route: R
结果：写入和回查均通过
```

运行 demo 时会直接打印从存储读回的原文和 `MEMORY.md`，相同内容也保存在本次 `result.json` 的 `stored_raw`、`notes`、`memory_markdown` 中。

每次模型生成的笔记和路由可能不同。脚本保留真实的充分性判断；写入、来源索引或回查检查未通过时返回非零退出码。

## Demo 2：Agent 按需调用 MCP 记忆工具

```bash
bash jev_tiermem/run_demo.sh mcp
```

这个 demo 使用真实文本模型作为宿主 agent，并连接本地 Jev TierMem MCP 服务。模型自行选择工具、生成摘要和最终回答；脚本只安排三个任务并检查实际行为。

| 顺序 | 任务 | 检查内容 |
| --- | --- | --- |
| 1 | 运行 router 测试，记住结果 | 真实测试通过，完整输出写入 raw，摘要指向正确来源 |
| 2 | 回答 `2 + 2` | 普通问题没有调用记忆工具 |
| 3 | 新 agent 回查之前的完整测试名及结果 | 新实例没有旧对话，使用 `retrieve` 找回证据并引用，不重新跑测试 |

一次实跑的工具调用和输出节选（证据 ID 已简写）：

```text
[1/3] 运行测试并记住结果
  tool → run_router_tests
  tool → jev_memory_observe
  tool → jev_memory_add_summary
  summary: Router unit tests passed: 4 tests ran successfully (OK).
  来源: r-…

[2/3] 2 + 2
4

[3/3] 新 agent 回查
  tool → jev_memory_retrieve
  summary: 不足
  raw: 足够
jev_tiermem.tests.test_memory.RouterTests.test_network_failure_fails_closed — ok.
Evidence: r-…
普通问题记忆调用次数：0
结果：全部通过
```

这次 MCP demo 的原文是同一组真实测试输出（不含 Demo 1 额外添加的 `Command` / `Exit code` 两行），通过 `jev_memory_observe` 保存为 `r-4fea38ce678fa91240814aca`。宿主 agent 生成一句摘要，调用 `jev_memory_add_summary` 后，实际文件内容为：

```markdown
# Memory

<!-- jev_tiermem {"id": "s-bd8c20a88d3930f7170d6832", "raw_ids": ["r-4fea38ce678fa91240814aca"], "kind": "summary"} -->
Router unit tests passed: 4 tests ran successfully (OK).
<!-- /jev_tiermem -->
```

**记录由 Jev TierMem 完成。** 两个 demo 都将原文存入我们的 SQLite，将带来源的摘要写入我们的 `MEMORY.md`；区别在于谁生成摘要：

| 步骤 | Demo 1：直接使用记忆 | Demo 2：宿主通过 MCP 接入 |
| --- | --- | --- |
| 保存原文 | `observe` 写入 SQLite | agent 调用 `jev_memory_observe`，服务端写入 SQLite |
| 生成摘要 | `compact` 调用配置的普通文本模型 | 宿主 agent 自己生成摘要 |
| 保存摘要及索引 | Jev TierMem 写入 `MEMORY.md`，绑定原文 IDs | `jev_memory_add_summary` 写入 `MEMORY.md`，验证并保存宿主传入的原文 IDs |
| 判断证据是否足够 | Jev | Jev |

MCP demo 也会读回并显示实际原文和记忆文件；这部分展示不会注入新 agent 的对话。

新 agent 被问到同样的完整测试名和结果时，实际最终回答为：

> jev_tiermem.tests.test_memory.RouterTests.test_network_failure_fails_closed — ok. Evidence: r-4fea38ce678fa91240814aca

这条回答来自 `jev_memory_retrieve` 返回的历史证据：summary 只写了“4 tests ran successfully”，Jev 判断不足后回查 raw，找到完整名称和 `ok`。新 agent 没有拿到上一轮对话，也没有重新执行测试。

写入工具 `jev_memory_observe` 和 `jev_memory_add_summary` 不调用模型：摘要由宿主 agent 提供。检索工具 `jev_memory_retrieve` 调用 Jev 判断证据是否足够，返回证据供宿主作答；必要时 deepsearch 使用服务端文本模型改写查询。

保存工具输出时，模型选择结果引用，由 demo 的宿主代码将其解析成完整原文，再发送标准 MCP 写入请求，避免模型重新抄写日志。

示例为宿主暴露三个核心记忆工具，服务端仍保留完整的八个 MCP 工具。连接 Codex/OpenClaw 时使用同一套[标准 MCP 接口和配置](../integrations/README.md)。

## 检索流程与 deepsearch

**summary 判断决定是否升级到原文，raw 判断决定是否继续搜索。** 默认过程如下：

1. 取最多 5 条相关 summary，将原问题与摘要交给 Jev。充分性分数达到 `0.8` 就返回 S；低于阈值则进入 R。判断调用失败时也保守进入 R，并在 trace 中记录错误。
2. 首轮 raw 检索沿摘要的 `raw_ids` 取最多 4 页原文，再用原问题做 SQLite 全文检索，取最多 4 页；合并去重。
3. 将原问题与累计找到的原文交给 Jev。充分性分数达到 `0.8` 就停止；否则在剩余预算内补搜。
4. 普通文本模型根据问题、摘要、已找到的原文和用过的查询，提出最多 3 个新搜索词串。程序执行本地全文检索，每个查询最多取 4 页，排除已读原文；再交给 Jev 判断。

默认最多 **3 轮 raw 检索，包含首轮**，因此最多进行两次查询改写。每轮判断使用累计证据；找到足够证据、没有新证据或预算用完时提前结束。相关上限可通过 Python API 的 [Config](../config.py) 设置：

| 配置 | 默认值 | 含义 |
| --- | --- | --- |
| `sufficient_threshold` | `0.8` | summary 和 raw 的充分性判断阈值 |
| `max_search_rounds` | `3` | raw 检索总轮数，包含首轮 |
| `raw_top_k` | `4` | 每个查询最多召回的原文页数；也限制首轮来源索引取页数 |
| `max_raw_pages` | `12` | 一次检索最多累计的原文页数 |
| `evidence_max_chars` | `24000` | 累计原文正文的字符上限，不是 token 数 |

`deepsearch=False` 表示先判断 summary，仍可按需进入上述多轮检索；`deepsearch=True` 表示跳过 summary 判断，直接从 raw 开始。

### 问题和原文怎样传给模型

Jev 接收原问题与证据文本，返回充分性分数。下面是兼容接口的请求结构示意，证据内容、元数据和判断指令已简化；summary 阶段将 `evidence` 换成摘要记录：

```json
{
  "model": "jev-1.13.0",
  "state": {
    "query": "包含 network_failure 的完整测试名及结果是什么？",
    "evidence": [
      {
        "id": "r-...",
        "text": "test_network_failure_fails_closed (...) ... ok",
        "speaker": "tool"
      }
    ]
  },
  "questions": {
    "sufficient": {
      "type": "noul",
      "instructions": "这些证据是否明确支持问题所需的完整答案？"
    }
  }
}
```

需要补搜时，普通文本模型收到 `task: "search"`，以及 `query`、`summaries`、`raw`、`previous_queries` 四个字段，分别是原问题、相关摘要、累计原文和已经用过的查询。它返回类似 `{"queries": ["network_failure", "RouterTests"]}` 的搜索建议，再由程序查询 SQLite。Jev 负责判断，普通文本模型负责提出查询，SQLite 负责实际检索。

### 怎样看运行结果

| 输出 | 含义 |
| --- | --- |
| `summary: 不足` | Jev 成功返回分数，但分数低于阈值，需要回查原文 |
| `raw history: 足够` | 累计原文已达到充分性阈值，停止检索 |
| `raw history: 不足` | 累计原文尚未达到阈值，继续补搜或按停止条件结束 |
| `判断失败：TypeSafeAuthenticationError` | Jev 鉴权失败，没有得到分数；检查接口地址、模型与 key 是否匹配 |
| `route: R` | 已进入原文检索；是否通过还要看充分性判断和 demo 检查结果 |

`result.json` 中的 `trace.stop_reason` 记录停止原因：`summary_sufficient` / `raw_sufficient` 表示证据足够；`no_new_evidence` 表示没有新增原文；`round_budget` / `raw_budget` 表示达到轮数或原文预算。文本模型可能根据已找到的日志给出答案，但 Jev 判断失败或未达到充分性阈值时，demo 仍保留未通过状态。

使用兼容 Jev 服务时，需设置完整 `JEV_API_URL`，否则默认使用 TypeSafe 官方 SDK。也可以显式指定，例如：

```bash
bash jev_tiermem/run_demo.sh coding \
  --jev-api-url https://jevtypesafeai.com/api/v1/decide \
  --jev-model jev-1.13.0
```

## 配置和输出

两个 demo 都读取 README 中配置的 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`、`TYPESAFE_API_KEY`、`TYPESAFE_DEFAULT_MODEL`、`JEV_API_URL`。也可以显式传入 `--model`、`--jev-model`、`--jev-api-url`。

`run_demo.sh` 首次运行时按照 `pyproject.toml` 创建本机 `.venv` 并安装依赖；后续复用该环境。启动脚本和依赖声明随仓库发布，`.venv` 与运行数据留在本机。

安装完成后也可直接用 `python jev_tiermem/examples/mcp_agent.py`，缺依赖时会切换到项目环境，保留参数和 API 环境变量。显式指定解释器也可以：

```bash
jev_tiermem/.venv/bin/python jev_tiermem/examples/mcp_agent.py
```

首次使用时，若项目环境尚未安装，先运行：

```bash
python -m venv jev_tiermem/.venv
jev_tiermem/.venv/bin/python -m pip install -e './jev_tiermem[live,mcp]'
```

如果环境变量已写在 bashrc，加载后直接运行：

```bash
source ~/.bashrc
bash jev_tiermem/run_demo.sh mcp
```

运行结果默认位于 `jev_tiermem/.runs/coding-demo/` 或 `jev_tiermem/.runs/mcp-agent/`。每次创建新目录和 session，终端打印 `result.json` 的位置；其中保存实际笔记、证据、路由和检查结果。`MEMORY.md` 与 `raw.sqlite3` 保留在对应 memory 子目录，可用 `--store` 指定其他输出位置。

## Skill：回查使用约定

这个例子读取仓库中已有的 [记忆排障 skill](skills/jev-memory-triage/SKILL.md)，记录后再追问命令细节：

```bash
python -m jev_tiermem.agent_loop \
  --workspace jev_tiermem/examples/skills \
  --store jev_tiermem/.runs/memory --session skill-demo \
  --task '读取 jev-memory-triage/SKILL.md，概述这个 skill 的用途。'

python -m jev_tiermem.agent_loop \
  --workspace jev_tiermem/examples/skills \
  --store jev_tiermem/.runs/memory --session skill-demo \
  --task '不要重新读文件，通过 recall 找回 skill 中查看笔记和搜索原文的两个命令，需要设置哪些变量？引用原始证据。'
```

原文件包含 `show`、`search` 命令，以及 `MEMORY_STORE`、`MEMORY_SESSION` 和 `SEARCH_TERMS` 三个变量。可以将该文件换成自己的 skill 或生成产物，保持两次启动的 store/session 一致。

## 接入自己的 agent

需要保存一个工具结果时，用 `observe` 写入原文，再用 `add_summary` 保存宿主生成的阶段摘要。下面先生成一份真实测试输出，再回查它：

```python
import os
import subprocess
import sys
from jev_tiermem import Config, JevTierMem

config = Config(
    model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    jev_model=os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"),
    jev_api_url=os.environ.get("JEV_API_URL") or None,
)

with JevTierMem("jev_tiermem/.runs/memory", "python-demo", config) as memory:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "jev_tiermem.tests.test_memory.RouterTests", "-v"],
        capture_output=True, text=True, check=True,
    )
    output = completed.stdout + completed.stderr
    raw_ids = memory.observe(output, speaker="tool")
    memory.add_summary("已执行 RouterTests，完整结果见测试日志。", raw_ids)

    result = memory.retrieve("网络异常测试的完整函数名及结果是什么？")
    print(result["route"], result["sufficient"], result["evidence"])
```

`retrieve` 返回证据，宿主据此回答。`route` 为 `S` 时使用 summary，为 `R` 时已进入原文检索；回答前检查 `sufficient` 并保留证据 ID。需要直接搜索原文时使用 `retrieve(query, deepsearch=True)`。

| 接口 | 用法 |
| --- | --- |
| `observe(text, speaker="tool")` | 原样保存事件，返回 raw IDs |
| `add_summary(text, raw_ids)` | 保存宿主生成的摘要及来源索引 |
| `compact()` | 调用普通文本模型，为尚未摘要的原文生成笔记 |
| `retrieve(query)` | 检索 summary，按需回查 raw，返回证据和充分性判断 |
| `answer(query)` | 检索并调用文本模型生成回答，返回 `Answer` 对象 |
| `promote(text, quotes)` | 提交有原文依据的新事实，交由 Jev 判断是否写回 |

`promote` 的 `quotes` 格式为 `[{'raw_id': '原文 ID', 'quote': '原文中的逐字引文'}]`。引文必须匹配；通过依据、长期价值和新颖性判断后才写入笔记。
