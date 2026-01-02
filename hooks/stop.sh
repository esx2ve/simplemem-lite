#!/bin/bash
# SimpleMem Lite - Stop Hook
# Called by Claude Code at session end to trigger trace processing.
#
# Input: JSON on stdin with {cwd, session_id, transcript_path}
# Environment: CLAUDE_PROJECT_DIR (preferred over cwd)
# Output: None (silent operation - Stop hooks cannot block)
#
# Install: Add to ~/.claude/settings.json hooks.Stop

set -e

# Debug logging
LOG_FILE="${HOME}/.simplemem_lite/logs/stop-hook.log"
mkdir -p "$(dirname "$LOG_FILE")"
echo "$(date '+%Y-%m-%d %H:%M:%S') Stop hook triggered" >> "$LOG_FILE"

# Lock file location
LOCK_FILE="${HOME}/.simplemem_lite/server.lock"

# Read input from stdin
INPUT=$(cat)
echo "$(date '+%Y-%m-%d %H:%M:%S') Input: $INPUT" >> "$LOG_FILE"

# Check if lock file exists
if [[ ! -f "$LOCK_FILE" ]]; then
    # Server not running - silent exit
    exit 0
fi

# Check for jq availability
if ! command -v jq &> /dev/null; then
    # Can't parse JSON without jq - silent exit
    exit 0
fi

# Parse lock file
PORT=$(jq -r '.port' "$LOCK_FILE" 2>/dev/null)
TOKEN=$(jq -r '.token' "$LOCK_FILE" 2>/dev/null)
HOST=$(jq -r '.host // "127.0.0.1"' "$LOCK_FILE" 2>/dev/null)

if [[ -z "$PORT" || "$PORT" == "null" ]]; then
    exit 0
fi

# Extract fields from input - use CLAUDE_PROJECT_DIR if available
CWD="${CLAUDE_PROJECT_DIR:-$(echo "$INPUT" | jq -r '.cwd // empty')}"
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')

if [[ -z "$CWD" ]]; then
    exit 0
fi

# Construct JSON safely using jq to avoid injection
REQUEST_BODY=$(jq -n \
    --arg cwd "$CWD" \
    --arg session_id "$SESSION_ID" \
    --arg transcript_path "$TRANSCRIPT_PATH" \
    '{cwd: $cwd, session_id: $session_id, transcript_path: $transcript_path}')

# Make HTTP request to server (fire and forget)
# Stop hooks cannot block termination, so we don't wait for response
curl -s -X POST \
    "http://${HOST}:${PORT}/hook/stop" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" \
    --connect-timeout 2 \
    --max-time 10 \
    >/dev/null 2>&1 || true

exit 0
