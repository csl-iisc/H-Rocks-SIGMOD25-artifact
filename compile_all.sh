#!/usr/bin/env bash
# Minimal compile_all.sh — just call each subproject's compile.sh if present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 8)}"

PROJECTS=(
  "viper"
  "Plush"
  "pmem-rocksdb"
  "pmem-rocksdb/h-rocks"
)

for dir in "${PROJECTS[@]}"; do
  if [[ -f "$ROOT/$dir/compile.sh" ]]; then
    echo "==> Building $dir"
    ( cd "$ROOT/$dir" && JOBS="$JOBS" bash ./compile.sh )
  else
    echo "==> Skipping $dir (compile.sh not found)"
  fi
done

echo "All done."
