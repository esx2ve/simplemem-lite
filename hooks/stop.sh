#!/bin/bash
# SimpleMem Lite - Stop Hook
# Called by Claude Code at session end to trigger trace processing.
#
# Input: JSON on stdin with {cwd, session_id, transcript_path}
# Output: None (silent operation)
#
# Install: Add to ~/.claude/settings.json hooks.Stop

set -e

# Lock file location
LOCK_FILE="${HOME}/.simplemem_lite/server.lock"

# Read input from stdin
INPUT=$(cat)

# Check if lock file exists
if [[ ! -f "$LOCK_FILE" ]]; then
    # Server not running - silent exit
    exit 0
fi

# Parse lock file
PORT=$(jq -r '.port' "$LOCK_FILE" 2>/dev/null)
TOKEN=$(jq -r '.token' "$LOCK_FILE" 2>/dev/null)
HOST=$(jq -r '.host // "127.0.0.1"' "$LOCK_FILE" 2>/dev/null)

if [[ -z "$PORT" || "$PORT" == "null" ]]; then
    exit 0
fi

# Extract fields from input
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')
TRANSCRIPT_PATH=$(echo "$INPUT" | jq -r '.transcript_path // empty')

if [[ -z "$CWD" ]]; then
    exit 0
fi

# Make HTTP request to server (fire and forget)
curl -s -X POST \
    "http://${HOST}:${PORT}/hook/stop" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"cwd\": \"${CWD}\", \"session_id\": \"${SESSION_ID}\", \"transcript_path\": \"${TRANSCRIPT_PATH}\"}" \
    --connect-timeout 2 \
    --max-time 10 \
    >/dev/null 2>&1 || true

exit 0
