#!/usr/bin/env bash
set -Euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# Optional filters you can export before running:
#   SIZES="10000 100000 1000000"
#   VAL_SIZES="8 100"
# They will be forwarded to all figure scripts.

FIG_SCRIPTS=(
  "run_figure8a.sh"   # PUTs vs size
  "run_figure8b.sh"   # GETs vs size
  "run_figure8c.sh"   # DELETE vs size
  "run_figure8d.sh"   # UPDATE vs size
  "run_figure9a.sh"   # RANGE arrival vs sustained (Plush/RocksDB/H-Rocks)
  "run_figure10.sh"   # Var-KV PUT/GET (6 KV pairs)
  "run_figure12.sh"   # Latency graphs
  "run_figure13a.sh"  # YCSB-A
  "run_figure13b.sh"  # YCSB-B
  "run_figure13c.sh"  # YCSB-C
  "run_figure13d.sh"  # YCSB-D
)

ok=()
skipped=()
failed=()

log_dir="$ROOT/out/logs"
mkdir -p "$log_dir"

echo "== H-Rocks Artifact: run all figures =="
echo "Repo root: $ROOT"
echo "Logs:      $log_dir"
echo

start_ts=$(date +%s)

for s in "${FIG_SCRIPTS[@]}"; do
  if [[ -x "$ROOT/$s" ]]; then
    echo "---- Running $s ----"
    log="$log_dir/${s%.sh}.log"
    {
      echo "[ENV] SIZES=${SIZES:-<unset>}  VAL_SIZES=${VAL_SIZES:-<unset>}"
      echo "[START] $(date -Is)  script=$s"
      # Forward possible filters to child script
      SIZES="${SIZES:-}" VAL_SIZES="${VAL_SIZES:-}" "$ROOT/$s"
      echo "[END] $(date -Is)  script=$s"
    } > "$log" 2>&1 \
      && { echo "✅ OK: $s"; ok+=("$s"); } \
      || { echo "❌ FAIL: $s (see $log)"; failed+=("$s"); }
    echo
  else
    echo "⏭️  SKIP: $s (not found or not executable)"
    skipped+=("$s")
    echo
  fi
done

end_ts=$(date +%s)
dur=$(( end_ts - start_ts ))

echo "================ Summary ================"
[[ ${#ok[@]}      -gt 0 ]] && { echo "OK (${#ok[@]}):      ${ok[*]}"; }
[[ ${#skipped[@]} -gt 0 ]] && { echo "Skipped (${#skipped[@]}): ${skipped[*]}"; }
[[ ${#failed[@]}  -gt 0 ]] && { echo "Failed (${#failed[@]}):  ${failed[*]}"; }
echo "Total time: ${dur}s"
echo

# List produced figures (pdf/png) under out/
echo "Produced figures:"
find "$ROOT/out" -maxdepth 2 -type f \( -name '*.pdf' -o -name '*.png' \) -print | sort || true
echo
echo "Done."
