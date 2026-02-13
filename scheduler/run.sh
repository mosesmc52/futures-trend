#!/usr/bin/env bash
# run.sh — cron entrypoint with Uptime Kuma heartbeat
# - Success: sends status=up with run duration
# - Failure: sends status=down with exit code

echo "[$(date)] Running ETF Trend Algo..."

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

# ----------------------------
# Load env (if present)
# ----------------------------
set -a
[ -f /app/.env ] && . /app/.env
set +a

# ----------------------------
# Sanity checks (don’t print secrets)
# ----------------------------
: "${ALPACA_KEY_ID:?ALPACA_KEY_ID not set}"
: "${ALPACA_SECRET_KEY:?ALPACA_SECRET_KEY not set}"

# Kuma config (same as yours)
# --- Kuma config / URL building ---
KUMA_URL="${KUMA_PUSH_URL:-}"

# Helper: pick first reachable base URL if none specified
pick_kuma_base() {
  # 200/OK or any HTTP response counts as reachable
  try() { curl -sfm 1 "http://$1/" >/dev/null 2>&1; }  # 1s timeout
  if [ -n "${KUMA_HOST:-}" ]; then
    echo "http://${KUMA_HOST}:3001"
    return
  fi
  if try "uptime-kuma:3001"; then echo "http://uptime-kuma:3001"; return; fi
  if try "host.docker.internal:3001"; then echo "http://host.docker.internal:3001"; return; fi
  if try "127.0.0.1:3001"; then echo "http://127.0.0.1:3001"; return; fi
  echo ""
}

if [ -z "${KUMA_URL}" ]; then
  if [ -n "${KUMA_TOKEN:-}" ]; then
    BASE="$(pick_kuma_base)"
    if [ -n "$BASE" ]; then
      KUMA_URL="${BASE%/}/api/push/${KUMA_TOKEN}"
    fi
  fi
fi

[ -z "${KUMA_URL}" ] && echo "[warn] No KUMA URL resolved (set KUMA_PUSH_URL or KUMA_HOST+KUMA_TOKEN)." >&2

CURL_BIN="$(command -v curl || true)"
WGET_BIN="$(command -v wget || true)"

kuma_send() {
  # usage: kuma_send <status up|down> <msg> <duration_ms or ''> <exit_code or ''>
  local status="$1"; shift
  local msg="$1"; shift
  local dur_ms="${1:-}"; shift || true
  local code="${1:-}"; shift || true
  [ -z "${KUMA_URL}" ] && return 0

  if [ -n "${CURL_BIN}" ]; then
    ${CURL_BIN} -fsS -G "${KUMA_URL}" \
      --data-urlencode "status=${status}" \
      --data-urlencode "msg=${msg}" \
      $( [ -n "${dur_ms}" ] && printf -- --data-urlencode "ping=%s" "${dur_ms}" ) \
      $( [ -n "${code}" ] && printf -- --data-urlencode "code=%s" "${code}" ) \
      >/dev/null || true
  elif [ -n "${WGET_BIN}" ]; then
    # Simple fallback encoding
    local qp="status=${status}"
    [ -n "${msg}" ] && qp="${qp}&msg=$(printf %s "${msg}" | tr ' ' '+')"
    [ -n "${dur_ms}" ] && qp="${qp}&ping=${dur_ms}"
    [ -n "${code}" ] && qp="${qp}&code=${code}"
    ${WGET_BIN} -qO- "${KUMA_URL}?${qp}" >/dev/null 2>&1 || true
  fi
}

# Millisecond timing (GNU date) with fallback to seconds * 1000
START_TS="$(date +%s%3N 2>/dev/null || date +%s)"
to_ms() {
  local now="$(date +%s%3N 2>/dev/null || date +%s)"
  if [ "${#START_TS}" -ge 13 ] && [ "${#now}" -ge 13 ]; then
    echo $(( now - START_TS ))
  else
    echo $(( (now - START_TS) * 1000 ))
  fi
}

on_exit() {
  rc=$?
  dur_ms="$(to_ms)"
  if [ $rc -ne 0 ]; then
    kuma_send "down" "${KUMA_MSG_FAIL:-failed}" "${dur_ms}" "${rc}"
  fi
  exit $rc
}
trap on_exit EXIT INT TERM

# ----------------------------
# Run your job
# ----------------------------
cd /app
poetry run python algo.py

# Success heartbeat
kuma_send "up" "${KUMA_MSG_OK:-ok}" "$(to_ms)" "0"
