#!/bin/bash
# SimpleMem Lite - Session Start Hook
# Called by Claude Code at session start to inject project context.
#
# Input: JSON on stdin with {cwd, session_id}
# Output: Context string for Claude (stdout)
#
# Install: Add to ~/.claude/settings.json hooks.SessionStart

set -e

# Lock file location
LOCK_FILE="${HOME}/.simplemem_lite/server.lock"

# Read input from stdin
INPUT=$(cat)

# Check if lock file exists
if [[ ! -f "$LOCK_FILE" ]]; then
    # Server not running - silent exit (don't break Claude)
    exit 0
fi

# Parse lock file
PORT=$(jq -r '.port' "$LOCK_FILE" 2>/dev/null)
TOKEN=$(jq -r '.token' "$LOCK_FILE" 2>/dev/null)
HOST=$(jq -r '.host // "127.0.0.1"' "$LOCK_FILE" 2>/dev/null)

if [[ -z "$PORT" || "$PORT" == "null" ]]; then
    exit 0
fi

# Extract cwd and session_id from input
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')
SESSION_ID=$(echo "$INPUT" | jq -r '.session_id // empty')

if [[ -z "$CWD" ]]; then
    exit 0
fi

# Make HTTP request to server
RESPONSE=$(curl -s -X POST \
    "http://${HOST}:${PORT}/hook/session-start" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"cwd\": \"${CWD}\", \"session_id\": \"${SESSION_ID}\"}" \
    --connect-timeout 2 \
    --max-time 5 \
    2>/dev/null) || exit 0

# Extract context from response and output it
CONTEXT=$(echo "$RESPONSE" | jq -r '.context // empty' 2>/dev/null)

if [[ -n "$CONTEXT" && "$CONTEXT" != "null" ]]; then
    echo "$CONTEXT"
fi
