#!/bin/bash
# SimpleMem Lite - Hook Installer
# Installs hook scripts into Claude Code settings.
#
# Usage: ./install.sh [--uninstall]

set -e

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="${HOME}/.claude/settings.json"

# Hook paths (absolute)
SESSION_START_HOOK="${SCRIPT_DIR}/session-start.sh"
STOP_HOOK="${SCRIPT_DIR}/stop.sh"

# Make hooks executable
chmod +x "$SESSION_START_HOOK" "$STOP_HOOK"

# Check for jq
if ! command -v jq &> /dev/null; then
    echo "Error: jq is required but not installed."
    echo "Install with: brew install jq (macOS) or apt install jq (Linux)"
    exit 1
fi

# Uninstall mode
if [[ "$1" == "--uninstall" ]]; then
    echo "Uninstalling SimpleMem hooks..."

    if [[ -f "$SETTINGS_FILE" ]]; then
        # Remove our hooks from the settings
        jq 'del(.hooks.SessionStart[] | select(contains("simplemem"))) |
            del(.hooks.Stop[] | select(contains("simplemem")))' \
            "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && \
            mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
        echo "Hooks removed from $SETTINGS_FILE"
    else
        echo "Settings file not found: $SETTINGS_FILE"
    fi
    exit 0
fi

# Install mode
echo "Installing SimpleMem hooks..."

# Create settings directory if needed
mkdir -p "$(dirname "$SETTINGS_FILE")"

# Create settings file if it doesn't exist
if [[ ! -f "$SETTINGS_FILE" ]]; then
    echo '{}' > "$SETTINGS_FILE"
fi

# Check if hooks already installed
if grep -q "simplemem" "$SETTINGS_FILE" 2>/dev/null; then
    echo "SimpleMem hooks already installed in $SETTINGS_FILE"
    echo "Use --uninstall to remove, then reinstall."
    exit 0
fi

# Add hooks to settings
# Uses jq to safely merge hook configuration
jq --arg session "$SESSION_START_HOOK" --arg stop "$STOP_HOOK" '
    .hooks //= {} |
    .hooks.SessionStart //= [] |
    .hooks.Stop //= [] |
    .hooks.SessionStart += [$session] |
    .hooks.Stop += [$stop]
' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"

echo "Hooks installed successfully!"
echo ""
echo "Session start hook: $SESSION_START_HOOK"
echo "Stop hook: $STOP_HOOK"
echo ""
echo "Settings file: $SETTINGS_FILE"
echo ""
echo "Note: Restart Claude Code for hooks to take effect."
echo "To uninstall: $0 --uninstall"
