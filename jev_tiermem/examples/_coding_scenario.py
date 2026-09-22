"""Shared, isolated coding fixture. Every transcript comes from executed tools."""

import difflib
import re
import shutil
import subprocess
import sys
from pathlib import Path


QUESTION = (
    "上次 CSV 导入为什么不能直接用 utf-8？请从之前的记录找出：修复后用的编码、"
    "触发失败的 CSV 文件的前三个字节（十六进制），以及回归测试保留的客户编号和姓名。"
    "请引用记忆证据。"
)

CODING_TOOLS = [
    {"name": "inspect_project", "description": "Read the CSV importer, regression tests and fixture bytes in the demo workspace.",
     "arguments": {"type": "object", "properties": {}}},
    {"name": "run_import_tests", "description": "Run the real CSV import regression tests; returns exit code and the unabridged output.",
     "arguments": {"type": "object", "properties": {}}},
    {"name": "write_importer", "description": "Replace only contacts.py in the isolated demo workspace. Tests and fixtures stay fixed.",
     "arguments": {"type": "object", "properties": {"source": {"type": "string"}}, "required": ["source"]}},
    {"name": "coding_history", "description": "Get the complete coding tool transcript for saving to memory. Includes inspections, test runs and diffs.",
     "arguments": {"type": "object", "properties": {}}},
]
CODING_TOOL_NAMES = {tool["name"] for tool in CODING_TOOLS}


class CodingScenario:
    def __init__(self, directory):
        self.workspace = Path(directory) / "workspace"
        shutil.copytree(Path(__file__).with_name("csv_project"), self.workspace,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        self.transcript = []
        self.test_runs = []
        self.diffs = []

    def call(self, name, arguments):
        if name == "coding_history":
            return {"text": "\n\n".join(self.transcript)}
        if name == "inspect_project":
            sections = ["Task: repair CSV contact import without changing customer IDs or names."]
            for filename in ("contacts.py", "check_contacts.py"):
                sections.append(f"File: {filename}\n{(self.workspace / filename).read_text(encoding='utf-8')}")
            for path in sorted((self.workspace / "fixtures").glob("*.csv")):
                data = path.read_bytes()
                sections.append(f"Fixture: fixtures/{path.name}\nFirst three bytes (hex): {data[:3].hex(' ')}"
                                f"\nDecoded with utf-8 (repr): {data.decode('utf-8')!r}")
            output = {"text": "\n\n".join(sections)}
        elif name == "run_import_tests":
            completed = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "-v", "check_contacts"],
                cwd=self.workspace, capture_output=True, text=True, timeout=30,
            )
            output = {"exit_code": completed.returncode,
                      "text": "Command: python -B -m unittest -v check_contacts\n"
                              f"Exit code: {completed.returncode}\n" + completed.stdout + completed.stderr}
            self.test_runs.append(output)
        elif name == "write_importer":
            source = arguments["source"]
            if not isinstance(source, str) or not source.strip() or len(source) > 8000:
                raise ValueError("Expected a small Python source file")
            compile(source, "contacts.py", "exec")
            path = self.workspace / "contacts.py"
            before = path.read_text(encoding="utf-8")
            path.write_text(source, encoding="utf-8")
            diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True),
                                             fromfile="a/contacts.py", tofile="b/contacts.py"))
            self.diffs.append(diff)
            output = {"text": diff, "file": "contacts.py"}
        else:
            raise ValueError("Unknown coding tool")
        self.transcript.append(f"Tool: {name}\n{output['text']}")
        return output

    def checks(self):
        return {
            "bug_reproduced": bool(self.test_runs) and self.test_runs[0]["exit_code"] != 0
                              and "KeyError: 'customer_id'" in self.test_runs[0]["text"],
            "fix_applied": any(self.diffs),
            "regressions_passed": len(self.test_runs) >= 2 and self.test_runs[-1]["exit_code"] == 0
                                  and "Ran 3 tests" in self.test_runs[-1]["text"],
        }


def answer_has_details(answer):
    text = answer.lower()
    hex_text = re.sub(r"[\s`:,\-]", "", text.replace("0x", ""))
    return all(value in text for value in ("utf-8-sig", "00073", "chen, mei")) and "efbbbf" in hex_text
