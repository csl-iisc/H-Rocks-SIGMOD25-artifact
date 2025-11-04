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

# pmem-rocksdb/benchmarks has no compile.sh; build via make if present.
BENCH_DIR="$ROOT/pmem-rocksdb/benchmarks"
if [[ -f "$BENCH_DIR/Makefile" ]]; then
  echo "==> Building pmem-rocksdb/benchmarks"
  ( cd "$BENCH_DIR" && make all )
else
  echo "==> Skipping pmem-rocksdb/benchmarks (Makefile not found)"
fi

# uTree (multiThread)
UTREE_DIR="$ROOT/utree/multiThread/utree"
if [[ -x "$UTREE_DIR/build.sh" ]]; then
  echo "==> Building utree/multiThread"
  ( cd "$UTREE_DIR" && bash ./build.sh )
else
  echo "==> Skipping utree/multiThread (build.sh not found or not executable)"
fi

echo "All done."
