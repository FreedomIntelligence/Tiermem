#!/usr/bin/env bash
# Load the user's existing bash environment in a child shell; never edit it.
set -euo pipefail
tiermem_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
exec bash -ic '
  set -e
  cd -- "$1"
  shift
  export JEV_API_URL="${JEV_API_URL:-https://jevtypesafeai.com/api/v1/decide}"
  export TYPESAFE_DEFAULT_MODEL="${TYPESAFE_DEFAULT_MODEL:-jev-1.13.0}"
  exec jev_tiermem/.venv/bin/python -m jev_tiermem.agent_loop --reasoning-effort low "$@"
' jev-tiermem-agent "$tiermem_root" "$@"
