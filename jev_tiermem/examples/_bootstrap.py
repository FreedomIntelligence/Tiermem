"""Select a usable Python environment for directly executed demo scripts."""

import importlib.util
import os
import sys
from pathlib import Path


def ensure_runtime(script, *, with_mcp=False):
    root = Path(script).resolve().parents[2]
    # Direct script execution puts examples/, rather than the checkout, on sys.path.
    sys.path.insert(0, str(root))
    modules = ["openai", "httpx2", "typesafe_sdk"]
    if with_mcp:
        modules.append("mcp")
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if not missing:
        return

    venv = root / "jev_tiermem" / ".venv"
    interpreter = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    marker = "_JEV_TIERMEM_DEMO_RUNTIME"
    if (interpreter.is_file() and Path(sys.prefix).resolve() != venv.resolve()
            and os.environ.get(marker) != str(venv)):
        print(f"当前 Python 缺少 {', '.join(missing)}；使用 jev_tiermem/.venv 运行。",
              file=sys.stderr, flush=True)
        os.execve(str(interpreter), [str(interpreter), str(Path(script).resolve()), *sys.argv[1:]],
                  {**os.environ, marker: str(venv)})

    demo = "mcp" if with_mcp else "coding"
    print(f"当前 Python 缺少 demo 依赖：{', '.join(missing)}。\n"
          "请用自动安装并启动的入口：\n"
          f"  bash jev_tiermem/run_demo.sh {demo}\n"
          "原有 demo 参数可继续附在命令后。", file=sys.stderr)
    raise SystemExit(1)
