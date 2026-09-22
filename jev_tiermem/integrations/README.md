# OpenClaw 与 Codex

通过本地 stdio MCP 服务，为 OpenClaw 或 Codex 提供按需调用的记忆工具：值得保留时写入，需要回忆时检索，普通任务直接执行。

可以先运行[自己的 demo agent](../examples/README.md#demo-2agent-按需调用-mcp-记忆工具)，体验相同接口上的写入、跳过记忆和跨会话回查。

先在仓库根目录创建独立环境：

```bash
python -m venv jev_tiermem/.venv
jev_tiermem/.venv/bin/python -m pip install -e './jev_tiermem[live,mcp]'
```

先按 [README 的环境变量说明](../README.md#2-配置-api) 配置文本模型和 Jev。将下面的 `/absolute/path/Tiermem`、`/absolute/path/jev-memory` 替换成你的路径。API key 通过启动宿主进程的环境提供：`TYPESAFE_API_KEY`、`OPENAI_API_KEY`；文本接口读取 `OPENAI_BASE_URL`。本服务不读取 `.env` 或 bashrc。只调用 observe/add_summary/search/get 不需要 key。

以下模板使用兼容接口 `https://jevtypesafeai.com/api/v1/decide` 和 `jev-1.13.0`。MCP 显式传 `--model`、`--jev-model`、`--jev-api-url`，不会自动读取 `OPENAI_MODEL`、`TYPESAFE_DEFAULT_MODEL`、`JEV_API_URL`。使用 TypeSafe 官方 SDK 时，去掉完整 URL 参数，模型改成 `jev-latest`，并使用官方 key；只有 SDK 模式才使用可选的 `TYPESAFE_BASE_URL`。

## Codex

将 [codex.toml](codex.toml) 的条目合并到 `~/.codex/config.toml` 或受信任项目的 `.codex/config.toml`。`env_vars` 会转发列出的环境变量，无需把 key 写进配置。

也可以注册：

```bash
codex mcp add jev_tiermem -- \
  /absolute/path/Tiermem/jev_tiermem/.venv/bin/python \
  -m jev_tiermem.mcp_server \
  --store /absolute/path/jev-memory --session alice/project-a \
  --model gpt-4.1-mini --jev-model jev-1.13.0 \
  --jev-api-url https://jevtypesafeai.com/api/v1/decide
```

在配置里补充示例的 `env_vars` 和 `tool_timeout_sec`，按文本服务实际支持的名称调整 `--model`，然后从已导出变量的终端启动新的 Codex 会话并检查 `/mcp`。官方接口见 [Codex MCP](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。

## OpenClaw

支持原生 `mcp.servers` 的 OpenClaw 可将 [openclaw.json](openclaw.json) 的条目合并到当前配置，也可以运行：

```bash
openclaw mcp add jev_tiermem \
  --command /absolute/path/Tiermem/jev_tiermem/.venv/bin/python \
  --arg -m --arg jev_tiermem.mcp_server \
  --arg=--store --arg /absolute/path/jev-memory \
  --arg=--session --arg alice/project-a \
  --arg=--model --arg gpt-4.1-mini \
  --arg=--jev-model --arg jev-1.13.0 \
  --arg=--jev-api-url --arg https://jevtypesafeai.com/api/v1/decide

openclaw mcp doctor jev_tiermem --probe
```

配置中设置足够的 `requestTimeoutMs`，覆盖多轮模型请求。Gateway 所在进程需要能访问 Python 路径、数据目录与 provider 环境变量。官方接口见 [OpenClaw MCP](https://docs.openclaw.ai/tools/mcp)。较早版本如没有 `openclaw mcp add`，需要先确认其 MCP 支持。

## 按需使用

主要接入三个工具：`jev_memory_observe` 保存原文，`jev_memory_add_summary` 保存宿主摘要及来源索引，`jev_memory_retrieve` 返回历史证据。摘要和最终回答都可由宿主生成，无需为每轮任务额外调用服务端文本模型。

服务启动时会通过 MCP `instructions` 提供按需使用约定，工具描述也说明了输入与行为。Codex 支持读取该字段作为服务器级指引，见[官方 MCP 文档](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)。工具实际调用由宿主决定；本接入不注册自动压缩 hook。

接入后可依次尝试：

```text
运行 router 测试，把完整输出和一句话摘要保存到 Jev TierMem。
2 + 2 等于多少？
```

再用相同 store/session 开始新的宿主会话，提问：

```text
从 Jev TierMem 回查之前包含 network_failure 的完整测试名和结果，引用原始证据。
```

## 工具选择

| 时机 | MCP 工具 |
| --- | --- |
| 收到需要保存的消息或工具输出 | `jev_memory_observe`，保存返回的 raw IDs |
| 记录阶段进展 | `jev_memory_add_summary`，附上来源 IDs；或用 `jev_memory_compact` 自动生成笔记 |
| 回答历史问题 | `jev_memory_retrieve`，检查 `sufficient` 并使用返回的证据；需要时设置 `deepsearch=true` |
| 保留从原文恢复的新事实 | `jev_memory_promote`，提交事实与精确原文引文，按返回结果确认是否写入 |

`retrieve` 不额外生成最终答案，最适合宿主已有模型的场景。`recall` 才会调用服务端答案模型。deepsearch 的查询改写可能需要 `OPENAI_API_KEY`，即使最终答案由宿主生成。

`--session` 是固定的用户/项目记忆空间，应跨该项目的多次聊天复用。不同用户用不同 session/独立服务实例，避免在群聊中混用个人记忆。这是单用户本地工具，不是多租户 HTTP 服务。

连接后，可先用 observe → get → add_summary → retrieve 确认记录、来源索引和检索流程。
