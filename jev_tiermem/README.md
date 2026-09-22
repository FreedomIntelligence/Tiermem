# Jev TierMem

**Fast recall from notes. Precise recovery from raw history.**  
**快速回忆重点，准确找回细节。**

Jev TierMem 为长任务 agent 提供轻量记忆：用普通 Markdown 笔记保留进展，用可搜索的原始历史保存细节，再由 **Jev 判断何时需要回查原文**。适合持续调试、跨会话编码，以及需要回看工具输出的工作流。

- **按需读写：** 现有 agent 需要记住进展时写入，需要回忆时检索；也提供边运行边记忆的普通 agent。
- **摘要可追溯：** 每条 summary 都保存 `raw_ids`，指向对应原文。
- **按需找回细节：** 笔记不足时搜索 raw data，必要时多轮 deepsearch；有依据且值得保留的新事实可按条件写回。
- **轻量接入：** 提供普通 agent、Python API 和 MCP 工具；无需训练 router、mem0 或向量数据库。

当前为实验版本。需要一个普通文本模型 API 和一个 Jev API；记忆保存在本地 Markdown 与 SQLite 中。

[运行 demo](#两个可运行-demo) · [快速开始](#快速开始) · [使用示例](examples/README.md) · [Python API](examples/README.md#接入自己的-agent) · [OpenClaw / Codex 接入](integrations/README.md)

## 两个可运行 demo

配置下方的 [API 环境变量](#2-配置-api)后，从 Tiermem 仓库根目录运行。首次执行会自动创建本机环境并安装依赖：

```bash
# 1. 直接体验：复现并修复 CSV 导入 bug → 写入记忆 → 新会话回查。
bash jev_tiermem/run_demo.sh coding

# 2. Agent 接入：由模型决定何时调用标准 MCP 记忆工具。
bash jev_tiermem/run_demo.sh mcp
```

启动脚本根据仓库中的 `pyproject.toml` 安装依赖，之后复用本机的 `jev_tiermem/.venv`，无需退出 Conda `(base)`。`.venv` 由各使用者在本机生成，不需要提交到 Git。

**场景：昨天修好了 Excel CSV 导入，今天换一个 agent 继续 review。** 新 agent 被问到：「为什么改成 `utf-8-sig`？出错文件的前三个字节是什么？客户编号的前导零还在吗？」它需要从记忆中找回修复原因与测试细节。

仓库附带一个可复现的小型 [CSV 导入项目](examples/csv_project/README.md)。第一个 demo 自动执行一行修复，展示失败日志、修复 diff、回归测试，以及实际保存的 `MEMORY.md` 和来源索引；然后关闭客户端重新打开，仅通过记忆回答。可先看[原文 → 记忆 → 回查的实跑记录](examples/README.md#原文保存成了什么记忆)。

第二个 demo 连续运行三个场景：

| 任务 | Agent 如何使用记忆 |
| --- | --- |
| 自己检查 CSV 导入代码、复现并修复 bug | 模型选择读文件、修改代码和测试工具，再用 `observe` / `add_summary` 记住完整过程 |
| 回答「Python 列表如何去重并保持顺序」 | 正常回答，不需要调用记忆工具 |
| 新 agent 追问修复原因、文件字节和客户数据 | 调用 `retrieve`，由 Jev 判断 summary / raw，再回答并引用证据 |

终端展示实际工具调用和检查结果。代码修复发生在 `jev_tiermem/.runs/` 下的项目副本中，每次使用新的 session。示例数据随仓库提供，失败和通过的日志均由现场运行产生。两个 demo 都调用真实模型；MCP demo 使用标准接口，Codex/OpenClaw 可通过[相同接口接入](integrations/README.md)。详细步骤见 [demo 说明](examples/README.md)。

## 快速开始

### 1. 安装

需要 Python 3.10+ 和支持 FTS5 的 SQLite。以下命令均在 **Tiermem 仓库根目录**执行：

```bash
git clone https://github.com/FreedomIntelligence/Tiermem.git
cd Tiermem

python -m venv jev_tiermem/.venv
jev_tiermem/.venv/bin/python -m pip install -e './jev_tiermem[live,mcp]'
source jev_tiermem/.venv/bin/activate
```

只运行普通 agent 时，可将安装项改为 `'./jev_tiermem[live]'`。

### 2. 配置 API

配置普通文本模型；使用兼容服务时，替换地址和模型名：

```bash
export OPENAI_API_KEY='your-text-model-key'
export OPENAI_BASE_URL='https://api.openai.com/v1'
export OPENAI_MODEL='gpt-4.1-mini'
```

两种 Jev 接口都从 **`TYPESAFE_API_KEY`** 读取 key；变量名不决定请求发往哪个服务。`JEV_API_URL` 指定兼容服务地址，未设置时使用 TypeSafe 官方 SDK。请根据 key 所属的服务选择一套配置。

**使用 `jevtypesafeai.com` 接口的 key：**

```bash
export TYPESAFE_API_KEY='your-jev-service-key'
export JEV_API_URL='https://jevtypesafeai.com/api/v1/decide'
export TYPESAFE_DEFAULT_MODEL='jev-1.13.0'
```

如果 `TYPESAFE_API_KEY` 已配置，保留原值，只需设置上面的地址和模型。使用这个服务时需保留 `JEV_API_URL`；unset 会将同一个 key 发往 TypeSafe 官方服务，可能报 `TypeSafeAuthenticationError`。`TYPESAFE_DEFAULT_MODEL` 只选择模型，不改变 key 所属的服务。

<details>
<summary>仅使用 TypeSafe 官方签发的 key 时，展开此配置</summary>

```bash
export TYPESAFE_API_KEY='your-typesafe-key'
export TYPESAFE_DEFAULT_MODEL='jev-latest'
unset JEV_API_URL TYPESAFE_BASE_URL
```

</details>

其他兼容服务应填写自己的完整请求地址和模型名。如果变量已配置在 `~/.bashrc`，可以先 `source ~/.bashrc`，再激活虚拟环境。程序不自动读取 `.env`。

### 3. 启动 agent

```bash
# 检查文本模型和 Jev 连接；会产生 API 调用。
python -m jev_tiermem.agent_loop --check-providers

# 启动交互式 agent，允许读取当前仓库文件。
python -m jev_tiermem.agent_loop \
  --workspace . --store jev_tiermem/.runs/memory --session my-project
```

输入任务即可使用，例如：

```text
读取 jev_tiermem/tests/test_memory.py，概述 RouterTests 覆盖了哪些行为。
/checkpoint
通过 recall 找回网络异常测试使用的异常类型，并引用原始证据。
/exit
```

`/checkpoint` 保存新增笔记并释放近期窗口；`/exit` 保存退出。再次启动时使用相同的 `--store` 和 `--session`，即可复用已有记忆。`--workspace` 指定可读取目录，文件需要实际读取后才进入历史。

## 记忆如何保存

每个 session 对应一个本地目录：

```text
<store>/<session hash>/
├── MEMORY.md          # 普通 summary，每条附原文 raw_ids
├── raw.sqlite3        # 原始消息、工具输出及全文索引
└── agent_trace.jsonl  # 普通 agent 的调用记录
```

普通 agent loop 会自动保存事件并增量生成笔记；MCP 接入由宿主选择需要保存的内容，并提交自己的摘要。两者都保存原文索引。Jev 在检索或条件写回时调用，deepsearch 搜索本地历史。

检索时，Jev 先判断相关 summary 是否足够（默认阈值 `0.8`）：足够就直接返回；不足才沿 `raw_ids` 和关键词查原文。找回原文后再判断一次，足够则停止，不足才继续 deepsearch，直到找到证据或达到检索预算。接口调用失败会单独显示为“判断失败”。

默认最多 **3 轮 raw 检索，包含首次**，最多再补搜两轮。普通文本模型负责生成补充查询，SQLite 执行本地检索，Jev 判断是否已有足够证据。具体输入、预算和输出含义见 [检索流程与 deepsearch](examples/README.md#检索流程与-deepsearch)。

查看交互式 agent 的笔记，或直接搜索原文，无需调用模型：

```bash
python -m jev_tiermem --store jev_tiermem/.runs/memory --session my-project show
python -m jev_tiermem --store jev_tiermem/.runs/memory --session my-project search 'test_network_failure'
```

## 接入现有 agent

已有 agent 可以通过 [Python API](examples/README.md#接入自己的-agent) 保存原始事件和带来源的 summary，再调用检索与条件写回接口。

OpenClaw、Codex 等支持 MCP 的客户端可以连接本地 stdio 服务。安装前面的 `live,mcp` 依赖后，按 [MCP 接入指南](integrations/README.md) 配置即可。MCP 提供记忆工具，记录与检索由宿主工作流调用。

喜欢这个方向，欢迎 [Star TierMem](https://github.com/FreedomIntelligence/Tiermem)、Watch 仓库更新，或 [Follow FreedomIntelligence](https://github.com/FreedomIntelligence)。使用反馈和接入案例欢迎提交到 [Issues](https://github.com/FreedomIntelligence/Tiermem/issues)。
