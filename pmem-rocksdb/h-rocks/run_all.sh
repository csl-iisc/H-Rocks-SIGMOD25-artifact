#!/usr/bin/env bash
set -euo pipefail

RUNNERS=(
  "run_puts.sh"
  "run_gets.sh"
  "run_updates.sh"
  "run_deletes.sh"
  "run_ycsbA.sh"
  "run_ycsbB.sh"
  "run_ycsbC.sh"
  "run_ycsbD.sh"
  "run_diff_size.sh"
  "run_diff_size_gets.sh"
)

for s in "${RUNNERS[@]}"; do
  echo "=============================================================="
  echo "==> $s"
  echo "=============================================================="
  bash "./$s"
done

echo "All workloads completed."
