#!/usr/bin/env bash
set -euo pipefail

# Runs the viper/microbenchmarks workloads in sequence:
# put, get, update, delete, ycsbA, ycsbB, ycsbC, ycsbD

RUNNERS=(
  "run_prefill_put.sh"
  "run_prefill_get.sh"
  "run_update.sh"
  "run_prefill_delete.sh"
  "run_ycsbA.sh"
  "run_ycsbB.sh"
  "run_ycsbC.sh"
  "run_ycsbD.sh"
)

for s in "${RUNNERS[@]}"; do
  echo "=============================================================="
  echo "==> $s"
  echo "=============================================================="
  if [[ -f "./$s" ]]; then
    bash "./$s"
  else
    echo "WARNING: ./$s not found; skipping."
  fi
done

echo "All viper microbenchmarks completed."

