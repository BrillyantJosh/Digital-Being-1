#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# HEARTBEAT RUNNER — Adaptivni ritem srca
#
# Namesto fiksnega cron intervala, bere interval iz heartbeat_rhythm tabele.
# Več nalog = hitrejši ritem, manj nalog = počasnejši ritem.
#
# Logika ritma:
#   0 nalog   → 300s (5 min)  — srce počiva
#   1-2 nalogi → 120s (2 min)  — mirno delo
#   3-5 nalog  → 60s  (1 min)  — normalen ritem
#   6+ nalog   → 30s           — intenzivno delo
# ══════════════════════════════════════════════════════════════════════════

DB_PATH="/opt/entity/state/memory.db"
HEARTBEAT_SCRIPT="/opt/entity/src/heartbeat.py"
DEFAULT_INTERVAL=60
LOG_FILE="/opt/entity/logs/heartbeat_runner.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_msg "Heartbeat runner started"

while true; do
    # Poženi heartbeat cikel
    python3 "$HEARTBEAT_SCRIPT" 2>&1 | while IFS= read -r line; do
        log_msg "HB: $line"
    done

    # Preberi interval iz DB
    INTERVAL=$(sqlite3 "$DB_PATH" \
        "SELECT current_interval_seconds FROM heartbeat_rhythm WHERE id=1" 2>/dev/null)

    # Fallback na default če DB ni dostopna
    if [ -z "$INTERVAL" ] || [ "$INTERVAL" -lt 10 ] 2>/dev/null; then
        INTERVAL=$DEFAULT_INTERVAL
    fi

    log_msg "Sleeping ${INTERVAL}s (next heartbeat)"
    sleep "$INTERVAL"
done
