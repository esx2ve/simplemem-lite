"""Logging configuration for SimpleMem Lite.

Uses loguru with automatic rotation and structured logging.
Logs are stored in ~/.simplemem_lite/logs/ with:
- Rotation at 10 MB per file
- Retention of 7 days
- Compression of old logs

Environment variables for log level control:
- SIMPLEMEM_LITE_LOG_LEVEL: Global log level (default: INFO)
- SIMPLEMEM_LITE_LOG_AST: AST chunker log level
- SIMPLEMEM_LITE_LOG_EMBEDDINGS: Embedding provider log level
- SIMPLEMEM_LITE_LOG_INDEXER: Code indexer log level
"""

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

from loguru import logger

# Get global log level from environment
_global_log_level = os.getenv("SIMPLEMEM_LITE_LOG_LEVEL", "INFO").upper()

# Component-specific log level overrides
_component_log_levels: dict[str, str] = {
    "ast_chunker": os.getenv("SIMPLEMEM_LITE_LOG_AST", "").upper(),
    "embeddings": os.getenv("SIMPLEMEM_LITE_LOG_EMBEDDINGS", "").upper(),
    "code_index": os.getenv("SIMPLEMEM_LITE_LOG_INDEXER", "").upper(),
}


def _log_filter(record) -> bool:
    """Filter log records based on global and component-specific log levels.

    Allows component-specific log level overrides while respecting global level.
    """
    name = record["extra"].get("name", "")

    # Check component overrides first
    for component, level in _component_log_levels.items():
        if level and component in name:
            try:
                return record["level"].no >= logger.level(level).no
            except ValueError:
                pass  # Invalid level, fall through to global

    # Fallback to global level
    try:
        return record["level"].no >= logger.level(_global_log_level).no
    except ValueError:
        return True  # If level parsing fails, allow the message


# Remove default handler
logger.remove()

# Determine log directory
_log_dir = Path.home() / ".simplemem_lite" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)

# Console handler - uses filter for level control (allows component overrides)
logger.add(
    sys.stderr,
    level=0,  # Accept all, let filter decide
    filter=_log_filter,
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


@contextmanager
def log_timing(operation: str, log_instance=None, level: str = "debug"):
    """Context manager for timing operations with automatic logging.

    Args:
        operation: Description of the operation being timed
        log_instance: Logger instance (uses global logger if None)
        level: Log level for the timing message (default: debug)

    Yields:
        dict with 'elapsed_ms' key (populated after context exits)

    Example:
        with log_timing("AST parsing", log) as timing:
            tree = parser.parse(content)
        # timing['elapsed_ms'] now contains the elapsed time
    """
    log_fn = log_instance or logger
    timing = {"elapsed_ms": 0.0}
    start = perf_counter()
    try:
        yield timing
    finally:
        timing["elapsed_ms"] = (perf_counter() - start) * 1000
        getattr(log_fn, level)(f"{operation}: {timing['elapsed_ms']:.1f}ms")


# Export configured logger and utilities
__all__ = ["logger", "get_logger", "log_timing"]
