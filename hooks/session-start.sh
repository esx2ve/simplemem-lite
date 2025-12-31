#!/bin/bash
# SimpleMem Lite - Session Start Hook
# Called by Claude Code at session start to inject project context.
#
# Input: JSON on stdin with {cwd, session_id, transcript_path}
# Environment: CLAUDE_PROJECT_DIR (preferred over cwd)
# Output: Context string for Claude (stdout)
#
# Install: Add to ~/.claude/settings.json hooks.SessionStart
#
# Race condition handling:
# If MCP server isn't ready yet (no lock file), we write a pending session
# file and output a CRITICAL message that instructs Claude to call
# get_project_status, which will process the pending session.

set -e

# DEBUG: Log that hook was called (remove after debugging)
echo "$(date): SessionStart hook invoked" >> /tmp/session-start-debug.log

# File locations
LOCK_FILE="${HOME}/.simplemem_lite/server.lock"
PENDING_FILE="${HOME}/.simplemem_lite/pending_session.json"
DATA_DIR="${HOME}/.simplemem_lite"

# Read input from stdin
INPUT=$(cat)

# Ensure data directory exists
mkdir -p "$DATA_DIR"

# Check for jq availability first (needed for lock file parsing)
if ! command -v jq &> /dev/null; then
    # Can't parse JSON without jq - silent exit
    exit 0
fi

# Function to handle deferred context (server not ready)
handle_deferred() {
    echo "$INPUT" > "$PENDING_FILE"
    echo "[SimpleMem] CRITICAL: MCP server starting. You MUST call get_project_status(project_root=\"\${cwd}\") immediately to load session context."
    exit 0
}

# Check if lock file exists
if [[ ! -f "$LOCK_FILE" ]]; then
    handle_deferred
fi

# Parse lock file
PORT=$(jq -r '.port' "$LOCK_FILE" 2>/dev/null)
TOKEN=$(jq -r '.token' "$LOCK_FILE" 2>/dev/null)
HOST=$(jq -r '.host // "127.0.0.1"' "$LOCK_FILE" 2>/dev/null)
PID=$(jq -r '.pid' "$LOCK_FILE" 2>/dev/null)

if [[ -z "$PORT" || "$PORT" == "null" ]]; then
    exit 0
fi

# Validate PID is still running (stale lock file check)
if [[ -n "$PID" && "$PID" != "null" && "$PID" =~ ^[0-9]+$ ]]; then
    if ! kill -0 "$PID" 2>/dev/null; then
        # PID not running - lock file is stale, use deferred path
        echo "$(date): Stale lock file detected (PID $PID not running)" >> /tmp/session-start-debug.log
        rm -f "$LOCK_FILE"
        handle_deferred
    fi
elif [[ -n "$PID" && "$PID" != "null" ]]; then
    # Invalid PID format - treat as stale
    echo "$(date): Invalid PID format in lock file: '$PID'" >> /tmp/session-start-debug.log
    rm -f "$LOCK_FILE"
    handle_deferred
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

# Make HTTP request to server
RESPONSE=$(curl -s -X POST \
    "http://${HOST}:${PORT}/hook/session-start" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" \
    --connect-timeout 2 \
    --max-time 5 \
    2>/dev/null) || exit 0

# Extract context from response and output it
CONTEXT=$(echo "$RESPONSE" | jq -r '.context // empty' 2>/dev/null)

if [[ -n "$CONTEXT" && "$CONTEXT" != "null" ]]; then
    echo "$CONTEXT"
fi
