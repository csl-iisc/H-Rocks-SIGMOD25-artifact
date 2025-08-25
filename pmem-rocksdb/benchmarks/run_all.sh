#!/usr/bin/env bash
set -e

for s in \
  run_puts.sh \
  run_gets.sh \
  run_updates.sh \
  run_deletes.sh \
  run_ycsbA.sh \
  run_ycsbB.sh \
  run_ycsbC.sh \
  run_ycsbD.sh
do
  echo "==> $s"
  bash "./$s"
done

echo "All workloads completed."
