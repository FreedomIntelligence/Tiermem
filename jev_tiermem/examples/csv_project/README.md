# CSV 导入修复小项目

这是一个为记忆 demo 准备的可复现项目：`contacts.py` 能读取普通 UTF-8 CSV，却无法读取带 UTF-8 BOM 的导出文件。数据为示例客户，所有错误与测试结果均由实际运行生成。

文件说明：

- `contacts.py`：待修复的导入函数。
- `fixtures/plain.csv`：普通 UTF-8 文件。
- `fixtures/excel_export.csv`：带 BOM 的同一份数据，前三个字节为 `ef bb bf`。
- `check_contacts.py`：检查两种编码的文件可读，以及客户编号前导零、姓名中的逗号得到保留。

从 Tiermem 仓库根目录运行完整 demo：

```bash
# 脚本修复，体验记忆写入与跨会话回查
bash jev_tiermem/run_demo.sh coding

# 模型自己读代码、复现、修复，并通过 MCP 记住过程
bash jev_tiermem/run_demo.sh mcp
```

两个入口都会将项目复制到本次运行的 `workspace/`，仅修改该副本。新 agent 回查时只收到问题与记忆工具，没有前一轮对话或文件访问工具。

只想手工复现 bug 时，无需 API 或额外依赖：

```bash
cd jev_tiermem/examples/csv_project
python -B -m unittest -v check_contacts
```

此处预期出现 `KeyError: 'customer_id'` 和 `FAILED (errors=1)`。完整 demo 会在副本中将文件读取编码改为 `utf-8-sig`，再得到三项测试通过。运行结束后的 `coding.log` 保留检查过程、失败输出、实际 diff 与回归结果。

详见[记忆写入、检索与实际回答](../README.md)。
