# 使用示例

**昨天修好的 bug，今天换一个 agent，还能解释为什么这样改吗？**

这两个 demo 围绕同一个 coding 任务：联系人 CSV 导入支持普通 UTF-8，却在读取带 BOM 的 Excel 导出文件时报错。修复需要保留客户编号的前导零和姓名中的逗号。项目、CSV 和回归测试都[随仓库提供](csv_project/README.md)，是可复现的示例；日志来自现场执行。

先完成 [API 配置](../README.md#2-配置-api)，然后在 Tiermem 仓库根目录运行：

```bash
bash jev_tiermem/run_demo.sh coding  # 最短流程：修复 → 写记忆 → 换会话回查
bash jev_tiermem/run_demo.sh mcp     # 完整 agent：模型自己修复，按需调用 MCP 记忆工具
```

首次执行会自动安装依赖。两个入口都在 `.runs/` 中创建项目副本，结束后保留修复代码、完整日志与记忆。

## Demo 1：写入记忆，再重新打开使用

`coding` 入口自动完成以下过程：

1. 读取示例代码和 CSV 字节，运行测试，复现 `KeyError: 'customer_id'`。
2. 脚本将 `contacts.py` 的读取编码从 `utf-8` 改成 `utf-8-sig`，重新运行三项回归测试。
3. 用 `observe` 原样保存代码检查、失败日志、实际 diff 和成功日志；用 `compact` 生成简短 summary，绑定来源 `raw_ids`。
4. 关闭客户端，重新打开相同 session。新客户端只有问题，通过记忆恢复先前工作。

这个入口的代码修复由脚本执行；想看模型自己选择修复和记忆工具，运行下面的 MCP demo。

### 原文保存成了什么记忆

以下内容来自一次成功实跑，summary 和最终答案均由模型当场生成。

**① 原文：真实文件、失败日志、diff、回归结果。** `coding.log` 完整保存执行过程，下面是节选：

```text
Fixture: fixtures/excel_export.csv
First three bytes (hex): ef bb bf
Decoded with utf-8 (repr): '\ufeffcustomer_id,name,email\n00073,"Chen, Mei",mei@example.com\n'

KeyError: 'customer_id'
FAILED (errors=1)
```

实际修复只有一行：

```diff
-    with open(path, encoding="utf-8", newline="") as stream:
+    with open(path, encoding="utf-8-sig", newline="") as stream:
```

修复后，实际测试输出包含：

```text
Regression values: customer_id='00073', name='Chen, Mei'
test_excel_utf8_bom (check_contacts.ContactImportTests.test_excel_utf8_bom) ... ok
test_plain_utf8 (check_contacts.ContactImportTests.test_plain_utf8) ... ok
test_preserves_leading_zero_and_quoted_comma (check_contacts.ContactImportTests.test_preserves_leading_zero_and_quoted_comma) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.000s

OK
```

**② 记忆：实际保存的 `MEMORY.md`。** Jev TierMem 将完整原文分页写入 SQLite，普通文本模型概括成笔记，程序绑定来源索引：

```markdown
# Memory

<!-- jev_tiermem {"id": "s-b0e622ee3c8e918c265a48b4", "raw_ids": ["r-f5a753bd21284cf1e27ab7c3", "r-25079f93c71b04e46485bad4"], "kind": "summary"} -->
- Tool task: repair CSV import **without changing customer IDs/names**.
- Obs: BOM CSV caused `KeyError 'customer_id'`; plain CSV passed.
- Fix applied in `contacts.py`: `utf-8`→`utf-8-sig`.
- Result: all 3 tests passed.
<!-- /jev_tiermem -->
```

这条 summary 记住了修复原因、编码选择与测试结果，没有保存文件的十六进制字节和客户数据。`raw_ids` 指向完整原文的两页，细节仍能找回。**原文与笔记均由 Jev TierMem 存储，不依赖宿主保留旧对话。**

**③ 新会话提问：**

> 上次 CSV 导入为什么不能直接用 utf-8？请从之前的记录找出：修复后用的编码、触发失败的 CSV 文件的前三个字节（十六进制），以及回归测试保留的客户编号和姓名。请引用记忆证据。

**④ Jev 路由：**

```text
summary: 不足
raw history: 足够
route: R
```

Jev 先检查问题与摘要，发现具体值不足，再沿来源索引找到 raw。此次第一轮原文检索就已足够；没有启动额外补搜，也没有重新读文件或运行测试。

**⑤ 模型实际回答：**

> 上次不能直接用 utf-8，是因为触发失败的 CSV 是带 BOM 的 UTF-8 文件；用 utf-8 解码后内容开头变成 '\ufeffcustomer_id'，导致读取时表头不是 'customer_id'，从而在导入时触发 KeyError: 'customer_id'。修复后改用的编码是 utf-8-sig。触发失败的 CSV 文件前三个字节（十六进制）是 ef bb bf。回归测试保留的客户编号和姓名是 customer_id='00073'，name='Chen, Mei'。

引用证据：`r-f5a753bd21284cf1e27ab7c3`, `r-25079f93c71b04e46485bad4`。

终端会展示从存储读回的原文、`MEMORY.md`、路由与答案。`result.json` 保存相同内容并检查：bug 确实复现、修复后测试通过、原文无损、summary 来源有效、Jev 判断证据足够、回答恢复了具体值并引用证据。

模型每次的笔记、措辞和路由可能不同。若摘要已经足够，可以走 S；demo 不强制走 R，也不把失败的鉴权或充分性判断算作通过。

## Demo 2：Agent 按需调用 MCP 记忆工具

```bash
bash jev_tiermem/run_demo.sh mcp
```

同一个 CSV 项目，这次由真实文本模型自己选择读代码、运行测试、修改 `contacts.py`，然后调用标准 MCP 工具保存记忆。脚本只提供小项目、工具和任务，并检查实际结果。

一次成功实跑的调用顺序：

```text
[1/3] Agent 复现并修复 CSV 导入 bug，保存记忆
  tool → inspect_project
  tool → run_import_tests
  tool → write_importer
  tool → run_import_tests
  tool → coding_history
  tool → jev_memory_observe
  tool → jev_memory_add_summary

[2/3] 独立问题：Python 列表如何去重并保持顺序？
list(dict.fromkeys(items))

[3/3] 新 agent：没有旧对话、没有文件访问工具
  tool → jev_memory_retrieve
  summary: 不足
  raw: 足够
Jev route=R, sufficient=True
普通问题记忆调用次数：0
结果：全部通过
```

第一次测试确实失败，模型提交的实际 diff 将 `utf-8` 改为 `utf-8-sig`，再次测试后通过。随后 agent 请求完整 `coding_history`，用 `jev_memory_observe` 保存这份原文，再自行撰写摘要交给 `jev_memory_add_summary`。实际保存的文件：

```markdown
# Memory

<!-- jev_tiermem {"id": "s-cf886fafb85af60ab4db8707", "raw_ids": ["r-37b9fa7dfa8cfe33a4b35478", "r-31ec7493cbfea02107594a43"], "kind": "summary"} -->
将 contacts.py 改为使用 utf-8-sig 读取，修复 Excel UTF-8 BOM 联系人 CSV 导入失败，同时保持客户编号前导零、姓名内容和普通 UTF-8 CSV 读取正常。
<!-- /jev_tiermem -->
```

新 agent 被问到与 Demo 1 相同的问题后，最终回答如下：

> 不能直接用 `utf-8` 的原因是失败的那个 CSV 实际带有 UTF-8 BOM；记录里显示 `excel_export.csv` 的前三个字节是 `ef bb bf`，用纯 `utf-8` 解码后表头会变成 `\ufeffcustomer_id`，所以读取 `row["customer_id"]` 时触发 `KeyError`。[r-37b9fa7dfa8cfe33a4b35478]
>
> 之前修复后改用的编码是 `utf-8-sig`；补丁把 `open(path, encoding="utf-8")` 改成了 `open(path, encoding="utf-8-sig")`，随后测试全部通过。[r-31ec7493cbfea02107594a43]
>
> 回归测试保留的客户编号和姓名是：
> - 客户编号：`00073`
> - 姓名：`Chen, Mei`
>
> 这也在测试输出的回归值里直接记录了：`customer_id='00073', name='Chen, Mei'`。[r-37b9fa7dfa8cfe33a4b35478][r-31ec7493cbfea02107594a43]

新 agent 仅通过 `jev_memory_retrieve` 获取旧信息。终端中展示原文和记忆文件的步骤不会把这些内容注入它的对话。

### 谁负责“记”和“用”

| 步骤 | Demo 1：Python API | Demo 2：宿主通过 MCP 接入 |
| --- | --- | --- |
| 执行修复 | 脚本应用一行修改，实际运行测试 | 模型选择工具并生成修改代码，实际运行测试 |
| 保存原文 | `observe` 写入 SQLite | agent 选择 `jev_memory_observe`，服务端写入 SQLite |
| 生成 summary | `compact` 调用配置的普通文本模型 | 宿主 agent 生成一句摘要 |
| 保存笔记及索引 | Jev TierMem 写入 `MEMORY.md`，绑定 raw IDs | `jev_memory_add_summary` 验证来源 IDs 后保存 |
| 判断是否回查原文 | Jev | Jev |
| 最终回答 | `answer` 调用普通文本模型 | MCP 返回证据，由宿主 agent 回答 |

此 demo 在修复完成这个节点记一次；回忆时才检索。你也可以让自己的 agent 在关键测试、阶段结束等节点主动写入。写入工具 `observe` / `add_summary` 本身不调用模型；摘要由宿主提供时无需再调用一次摘要模型。

为了无损保存日志，模型选择 `coding_history` 返回的引用，宿主代码把引用解析为完整原文后发起标准 MCP 写入请求；不会让模型手工重抄日志。这只是 demo 的本地工具辅助，MCP 服务仍接收普通文本。

示例向宿主暴露三个核心记忆工具；服务端保留完整的八个工具，Codex/OpenClaw 可通过[相同标准接口接入](../integrations/README.md)。

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
    "query": "上次 CSV 导入失败的文件前三个字节是什么？",
    "evidence": [
      {
        "id": "r-...",
        "text": "Fixture: fixtures/excel_export.csv\nFirst three bytes (hex): ef bb bf",
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

需要补搜时，普通文本模型收到 `task: "search"`，以及 `query`、`summaries`、`raw`、`previous_queries` 四个字段，分别是原问题、相关摘要、累计原文和已经用过的查询。它返回类似 `{"queries": ["excel_export", "First three bytes"]}` 的搜索建议，再由程序查询 SQLite。Jev 负责判断，普通文本模型负责提出查询，SQLite 负责实际检索。

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

需要保存一个工具结果时，用 `observe` 写入原文，再用 `add_summary` 保存宿主生成的阶段摘要。下面读取示例 CSV，保存记录，再通过记忆检索细节：

```python
import os
from pathlib import Path
from jev_tiermem import Config, JevTierMem

config = Config(
    model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
    jev_model=os.environ.get("TYPESAFE_DEFAULT_MODEL", "jev-latest"),
    jev_api_url=os.environ.get("JEV_API_URL") or None,
)

path = Path("jev_tiermem/examples/csv_project/fixtures/excel_export.csv")
data = path.read_bytes()
output = f"File: {path.name}\nFirst three bytes: {data[:3].hex(' ')}\n{data.decode('utf-8')!r}"

with JevTierMem("jev_tiermem/.runs/memory", "python-demo", config) as memory:
    raw_ids = memory.observe(output, speaker="tool")
    memory.add_summary("已检查联系人 CSV 导出文件，原始字节与内容见来源记录。", raw_ids)

with JevTierMem("jev_tiermem/.runs/memory", "python-demo", config) as memory:
    result = memory.retrieve("之前 CSV 文件的前三个字节是什么？客户编号和姓名是什么？")
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
