#!/usr/bin/env bash
# Create a local environment from the published dependency declarations, then run a demo.
set -euo pipefail

tiermem_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
case "${1:-mcp}" in
  coding) tiermem_demo=coding_memory; tiermem_extras=live ;;
  mcp) tiermem_demo=mcp_agent; tiermem_extras=live,mcp ;;
  -h|--help)
    cat <<'USAGE'
Usage: bash jev_tiermem/run_demo.sh [coding|mcp] [demo arguments]

coding  Reproduce/fix a CSV import bug, write memory, then recall the details.
mcp     Let an agent fix the CSV importer and use memory tools on demand (default).

The first run creates jev_tiermem/.venv and installs the declared dependencies.
Configure the API environment variables described in README before running.
USAGE
    exit 0 ;;
  *) echo '请选择 coding 或 mcp，例如：bash jev_tiermem/run_demo.sh mcp' >&2; exit 2 ;;
esac
if (( $# > 0 )); then shift; fi
cd -- "$tiermem_root"

tiermem_python="$tiermem_root/jev_tiermem/.venv/bin/python"
if [[ ! -x "$tiermem_python" ]]; then
  echo '首次运行：创建 jev_tiermem/.venv'
  "${TIERMEM_PYTHON:-python}" -m venv "$tiermem_root/jev_tiermem/.venv"
fi

if ! "$tiermem_python" - "$tiermem_extras" <<'PY'
import importlib.metadata
import importlib.util
import sys

modules = ["openai", "httpx2", "typesafe_sdk"]
if "mcp" in sys.argv[1].split(","):
    modules.append("mcp")
try:
    importlib.metadata.version("jev-tiermem")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if all(importlib.util.find_spec(name) for name in modules) else 1)
PY
then
  echo "安装 demo 依赖：jev_tiermem[$tiermem_extras]"
  "$tiermem_python" -m pip install -e "$tiermem_root/jev_tiermem[$tiermem_extras]"
fi

exec "$tiermem_python" "$tiermem_root/jev_tiermem/examples/$tiermem_demo.py" "$@"
