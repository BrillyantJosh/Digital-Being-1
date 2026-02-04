#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════
# COMMUNICATION RUNNER — Komunikacijski cikel
#
# Teče vsakih 30s. Preverja neprebrana sporočila in odgovarja.
# Hitro izstopi če ni novih sporočil (brez Gemini klica).
# ══════════════════════════════════════════════════════════════════════════

COMM_SCRIPT="/opt/entity/src/communication_cycle.py"
INTERVAL=30
LOG_FILE="/opt/entity/logs/communication_runner.log"

log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$LOG_FILE"
}

log_msg "Communication runner started"

while true; do
    python3 "$COMM_SCRIPT" 2>&1 | while IFS= read -r line; do
        log_msg "COMM: $line"
    done

    sleep "$INTERVAL"
done
