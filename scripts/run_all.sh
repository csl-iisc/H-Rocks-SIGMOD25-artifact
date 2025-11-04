#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
SCRIPT_DIR="$ROOT/scripts"

declare -a RUNNERS=(
  "run_figure8a.sh"
  "run_figure8b.sh"
  "run_figure8c.sh"
  "run_figure8d.sh"
  "run_figure9a.sh"
  "run_figure10.sh"
  "run_figure12.sh"
  "run_figure13a.sh"
  "run_figure13b.sh"
  "run_figure13c.sh"
  "run_figure13d.sh"
)

for script in "${RUNNERS[@]}"; do
  path="$SCRIPT_DIR/$script"
  if [[ -x "$path" ]]; then
    echo "[run_all] ==> $script"
    "$path"
  else
    echo "[run_all] $script not found or not executable; skipping."
  fi
done
