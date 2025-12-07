#!/bin/bash
# ==============================================================
#  clean_memory.sh — Safely reclaim memory on Jetson (Nano/Xavier/Orin)
#  - Kills lingering Cursor/VS Code background processes
#  - Flushes filesystem buffers (sync)
#  - Drops page cache, dentries, and inodes
#  - (Optional) Refreshes swap/zram with --refresh-swap
#  Usage: sudo ./clean_memory.sh [--refresh-swap]
# ==============================================================

set -euo pipefail

# ---------- helpers ----------
log() { echo -e "[$(date '+%F %T')] $*"; }

mem_snapshot() {
  # prints: total_MB used_MB free_MB avail_MB
  free -m | awk 'NR==2 {print $2, $3, $4, $7}'
}

kill_by_pattern_for_user() {
  local user_regex="$1"    # user name
  local pattern="$2"       # grep pattern
  local pids
  pids="$(pgrep -u "$user_regex" -f "$pattern" || true)"
  [[ -z "$pids" ]] && return 0

  log "Found processes for pattern \"$pattern\": $pids"
  # Try graceful termination first
  kill $pids 2>/dev/null || true
  sleep 0.5
  # Force kill remaining
  pids="$(pgrep -u "$user_regex" -f "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    log "Forcing kill for: $pids"
    kill -9 $pids 2>/dev/null || true
  fi
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "Please run with sudo:  sudo $0 ${*:-}"
    exit 1
  fi
}

refresh_swap_if_requested() {
  if [[ "${REFRESH_SWAP}" == "1" ]]; then
    if swapon --show | grep -q .; then
      log "Refreshing swap (swapoff → swapon)…"
      swapoff -a || true
      sleep 1
      swapon -a || true
      log "Swap refreshed."
    else
      log "No active swap detected; skipping refresh."
    fi
  fi
}

# ---------- main ----------
REFRESH_SWAP=0
if [[ "${1:-}" == "--refresh-swap" ]]; then
  REFRESH_SWAP=1
fi

require_root

HOST="$(hostname)"
REAL_USER="${SUDO_USER:-$USER}"

echo "=============================================================="
echo " Jetson Memory Clean  |  Host: ${HOST}  |  User: ${REAL_USER}"
echo " Timestamp: $(date)"
echo "=============================================================="

read -r T0 U0 F0 A0 < <(mem_snapshot)
log "Before: total=${T0}MB, used=${U0}MB, free=${F0}MB, avail=${A0}MB"

# 0) Clean up Cursor / VS Code background processes
log "[1/4] Killing lingering Cursor / VS Code processes for user ${REAL_USER}…"
# Cursor IDE (cursor-server) and helpers
kill_by_pattern_for_user "$REAL_USER" "cursor-server"
kill_by_pattern_for_user "$REAL_USER" "/Cursor/"
kill_by_pattern_for_user "$REAL_USER" "cursor"
# VS Code variants
kill_by_pattern_for_user "$REAL_USER" "^code( |$|-.+)"
kill_by_pattern_for_user "$REAL_USER" "/.*/Code Helper"
kill_by_pattern_for_user "$REAL_USER" "code-oss"
# Language servers that often remain (only if launched by Code/Cursor)
kill_by_pattern_for_user "$REAL_USER" "typescript-language-features"
kill_by_pattern_for_user "$REAL_USER" "tsserver.js"
kill_by_pattern_for_user "$REAL_USER" "eslint.*server"
kill_by_pattern_for_user "$REAL_USER" "pyright-langserver"
# Node helpers spawned by the editors
kill_by_pattern_for_user "$REAL_USER" "node.*(cursor|code|Code)"

# 1) (Optional) Refresh swap
refresh_swap_if_requested

# 2) Flush filesystem buffers
log "[2/4] Flushing filesystem buffers (sync)…"
sync

# 3) Drop caches (pagecache + dentries + inodes)
log "[3/4] Dropping kernel caches (echo 3 > /proc/sys/vm/drop_caches)…"
if [[ -w /proc/sys/vm/drop_caches ]]; then
  echo 3 > /proc/sys/vm/drop_caches
else
  log "WARNING: Cannot write /proc/sys/vm/drop_caches (permissions)."
fi

# 4) One more quick sync just to be thorough
log "[4/4] Final sync…"
sync

# ---------- report ----------
read -r T1 U1 F1 A1 < <(mem_snapshot)
FREED_USED=$(( U0 > U1 ? U0 - U1 : 0 ))
GAIN_AVAIL=$(( A1 > A0 ? A1 - A0 : 0 ))

echo "--------------------------------------------------------------"
log "After : total=${T1}MB, used=${U1}MB, free=${F1}MB, avail=${A1}MB"
log "Freed (by used): ${FREED_USED} MB"
log "Gained (available): ${GAIN_AVAIL} MB"
echo "--------------------------------------------------------------"
echo "Done ✅"