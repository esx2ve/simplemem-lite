"""Time utilities for temporal filtering in SimpleMem.

Provides parsing for relative time specifications (e.g., "2d", "1w")
and ISO date strings for use in memory search filtering.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Optional


# Pattern for relative time: number + unit (h=hours, d=days, w=weeks, m=months)
RELATIVE_TIME_PATTERN = re.compile(r"^(\d+)([hdwm])$", re.IGNORECASE)

# Unit multipliers in seconds
TIME_UNITS = {
    "h": 3600,       # 1 hour
    "d": 86400,      # 1 day
    "w": 604800,     # 1 week
    "m": 2592000,    # 30 days (approximate month)
}


def parse_relative_time(spec: str) -> Optional[int]:
    """Parse a relative time specification into a Unix timestamp cutoff.

    Supported formats:
    - "2h" -> 2 hours ago
    - "3d" -> 3 days ago
    - "1w" -> 1 week ago
    - "2m" -> 2 months ago (60 days)

    Args:
        spec: Relative time specification string

    Returns:
        Unix timestamp for the cutoff, or None if parsing fails

    Examples:
        >>> parse_relative_time("2d")  # 2 days ago
        1704931200  # (example timestamp)
        >>> parse_relative_time("1w")  # 1 week ago
        1704326400  # (example timestamp)
    """
    match = RELATIVE_TIME_PATTERN.match(spec.strip())
    if not match:
        return None

    amount = int(match.group(1))
    unit = match.group(2).lower()

    if unit not in TIME_UNITS:
        return None

    seconds_ago = amount * TIME_UNITS[unit]
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)
    return int(cutoff.timestamp())


def parse_iso_date(spec: str) -> Optional[int]:
    """Parse an ISO date string into a Unix timestamp.

    Supported formats:
    - "2024-01-15" (date only, assumes start of day UTC)
    - "2024-01-15T10:30:00" (datetime, assumes UTC)
    - "2024-01-15T10:30:00Z" (datetime with Z)
    - "2024-01-15T10:30:00+00:00" (datetime with timezone)

    Args:
        spec: ISO date/datetime string

    Returns:
        Unix timestamp, or None if parsing fails
    """
    spec = spec.strip()

    # Try various ISO formats
    formats = [
        "%Y-%m-%d",                    # Date only
        "%Y-%m-%dT%H:%M:%S",           # Datetime without TZ
        "%Y-%m-%dT%H:%M:%SZ",          # Datetime with Z
        "%Y-%m-%dT%H:%M:%S%z",         # Datetime with timezone
        "%Y-%m-%dT%H:%M:%S.%f",        # Datetime with microseconds
        "%Y-%m-%dT%H:%M:%S.%fZ",       # Datetime with microseconds and Z
        "%Y-%m-%dT%H:%M:%S.%f%z",      # Datetime with microseconds and TZ
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(spec, fmt)
            # If no timezone, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())
        except ValueError:
            continue

    return None


def parse_time_spec(spec: Optional[str]) -> Optional[int]:
    """Parse a time specification (relative or ISO) into a Unix timestamp.

    This is the main entry point for time parsing. It handles:
    - Relative times: "2d", "1w", "12h", "3m"
    - ISO dates: "2024-01-15", "2024-01-15T10:30:00Z"

    Args:
        spec: Time specification string, or None

    Returns:
        Unix timestamp, or None if spec is None or parsing fails

    Examples:
        >>> parse_time_spec("2d")      # 2 days ago
        >>> parse_time_spec("2024-01-15")  # Specific date
        >>> parse_time_spec(None)      # Returns None
    """
    if spec is None:
        return None

    spec = spec.strip()
    if not spec:
        return None

    # Try relative time first (more common use case)
    result = parse_relative_time(spec)
    if result is not None:
        return result

    # Try ISO date
    result = parse_iso_date(spec)
    if result is not None:
        return result

    return None


def timestamp_to_iso(timestamp: int) -> str:
    """Convert a Unix timestamp to ISO format string.

    Args:
        timestamp: Unix timestamp

    Returns:
        ISO format datetime string (UTC)
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.isoformat()
