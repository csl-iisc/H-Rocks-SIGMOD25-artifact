# Create simple run scripts: puts, gets, updates, deletes, ycsbA-D, and run_all
import os, textwrap, json, pathlib

scripts = {}

scripts["run_puts.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_puts
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_puts_${N}"
  rm -rf "$DB"
  echo "==> puts N=$N (DB=$DB)"
  ./bin/test_puts -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/puts_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_gets.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_gets
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_gets_${N}"
  rm -rf "$DB"
  echo "==> gets N=$N (prefill=$N, DB=$DB)"
  ./bin/test_gets -p "$N" -g "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/gets_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_updates.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_updates
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_updates_${N}"
  rm -rf "$DB"
  echo "==> updates N=$N (prefill=$N, DB=$DB)"
  ./bin/test_updates -p "$N" -u "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/updates_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_deletes.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_deletes
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_deletes_${N}"
  rm -rf "$DB"
  echo "==> deletes N=$N (prefill=$N, DB=$DB)"
  ./bin/test_deletes -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/deletes_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_ycsbA.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbA
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbA_${N}"
  rm -rf "$DB"
  echo "==> YCSB-A ops=$N (reads=50%%/puts=50%%, prefill=$N, DB=$DB)"
  ./bin/test_ycsbA -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/ycsbA_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_ycsbB.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbB
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbB_${N}"
  rm -rf "$DB"
  echo "==> YCSB-B ops=$N (reads=90%%/puts=10%%, prefill=$N, DB=$DB)"
  ./bin/test_ycsbB -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/ycsbB_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_ycsbC.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbC
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbC_${N}"
  rm -rf "$DB"
  echo "==> YCSB-C ops=$N (reads=100%%, prefill=$N, DB=$DB)"
  ./bin/test_ycsbC -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/ycsbC_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_ycsbD.sh"] = """#!/usr/bin/env bash
set -e
OUT_DIR=output_ycsbD
mkdir -p "$OUT_DIR"
for N in 10000 1000000 10000000 25000000 50000000 75000000 100000000; do
  DB="/pmem/hrocks_ycsbD_${N}"
  rm -rf "$DB"
  echo "==> YCSB-D ops=$N (reads=90%%/inserts=10%%, prefill=$N, DB=$DB)"
  ./bin/test_ycsbD -p "$N" -n "$N" -k 8 -v 8 -t 32 -f "$DB" | tee "$OUT_DIR/ycsbD_${N}.log"
done
echo "done: $OUT_DIR"
"""

scripts["run_all.sh"] = """#!/usr/bin/env bash
set -e
bash ./run_puts.sh
bash ./run_gets.sh
bash ./run_updates.sh
bash ./run_deletes.sh
bash ./run_ycsbA.sh
bash ./run_ycsbB.sh
bash ./run_ycsbC.sh
bash ./run_ycsbD.sh
echo "All workloads completed."
"""

base = "./"
paths = []
for name, content in scripts.items():
    p = os.path.join(base, name)
    with open(p, "w") as f:
        f.write(content)
    os.chmod(p, 0o755)
    paths.append(p)

paths
