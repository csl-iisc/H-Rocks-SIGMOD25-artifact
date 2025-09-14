#!/usr/bin/env bash
set -euo pipefail

# Run this from anywhere; it finds the repo root.
ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
echo "[remove_min_suffix] repo root: $ROOT"

# 1) Rename every *_min.sh -> .sh (preserve exec bit; use git mv if available)
echo "[step 1/2] Renaming files…"
while IFS= read -r -d '' f; do
  new="${f%_min.sh}.sh"
  if [[ -e "$new" ]]; then
    echo "  [skip] $new already exists (leaving $f as-is)"
    continue
  fi
  if command -v git >/dev/null 2>&1 && git -C "$ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git mv "$f" "$new" || mv "$f" "$new"
  else
    mv "$f" "$new"
  fi
  chmod +x "$new" || true
  echo "  [mv]   $f -> $new"
done < <(find "$ROOT" -type f -name '*_min.sh' -print0)

# 2) Update references: “…_min.sh” -> “….sh” in all shell scripts
echo "[step 2/2] Updating references…"
mapfile -t files < <(grep -rl --include='*.sh' '_min\.sh' "$ROOT" || true)
for f in "${files[@]}"; do
  cp "$f" "$f.bak"
  sed -E -i 's/(_[A-Za-z0-9]+)_min\.sh/\1.sh/g' "$f"
  echo "  [sed]  $f (backup: $f.bak)"
done

echo "[done] Removed _min suffix from scripts and updated references."
