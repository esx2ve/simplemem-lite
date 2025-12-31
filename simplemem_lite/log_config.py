"""Logging configuration for SimpleMem Lite.

Uses loguru with automatic rotation and structured logging.
Logs are stored in ~/.simplemem_lite/logs/ with:
- Rotation at 10 MB per file
- Retention of 7 days
- Compression of old logs
"""

import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Determine log directory
_log_dir = Path.home() / ".simplemem_lite" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

# Console handler - INFO level, colored
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    colorize=True,
)

# File handler - DEBUG level, structured JSON, with rotation
logger.add(
    _log_dir / "simplemem_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="10 MB",      # Rotate at 10 MB
    retention="7 days",    # Keep 7 days of logs
    compression="zip",     # Compress old logs
    enqueue=True,          # Thread-safe
)

# Also create a symlink to latest log for easy access
_latest_log = _log_dir / "latest.log"
logger.add(
    _latest_log,
    level="TRACE",         # Most verbose for latest
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
    rotation="5 MB",
    retention=1,           # Only keep current
)


def get_logger(name: str):
    """Get a logger with the given name bound to context.

    Args:
        name: Module or component name

    Returns:
        Logger instance with name bound
    """
    return logger.bind(name=name)


# Export configured logger
__all__ = ["logger", "get_logger"]
