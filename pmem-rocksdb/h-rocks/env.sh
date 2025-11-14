#!/usr/bin/env bash

# Resolve script directory once.
HR_HROCKS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a writable PMEM directory. Prefer a user-provided override,
# fall back to /pmem if it exists and is writable, otherwise use a
# workspace-local directory so sandboxed runs keep working.
if [[ -z "${HR_PMEM_DIR:-}" ]]; then
  if [[ -d /pmem && -w /pmem ]]; then
    HR_PMEM_DIR="/pmem"
  else
    HR_PMEM_DIR="${HR_HROCKS_DIR}/tmp/pmem"
  fi
fi
mkdir -p "$HR_PMEM_DIR"
export HR_PMEM_DIR

# Similar logic for temporary files (default /dev/shm for performance).
if [[ -z "${HR_SHM_DIR:-}" ]]; then
  if [[ -d /dev/shm && -w /dev/shm ]]; then
    HR_SHM_DIR="/dev/shm"
  else
    HR_SHM_DIR="${HR_HROCKS_DIR}/tmp/shm"
  fi
fi
mkdir -p "$HR_SHM_DIR"
export HR_SHM_DIR

# Wrapper that retries rm -rf with sudo if permissions block deletion
hr_rm_rf() {
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  if rm -rf "$@" 2>/dev/null; then
    return 0
  fi
  sudo rm -rf "$@"
}
