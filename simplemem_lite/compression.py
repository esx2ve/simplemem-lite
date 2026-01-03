"""Compression utilities for SimpleMem-Lite.

Provides gzip compression for large payloads (code files, traces)
sent between MCP thin layer and backend API.
"""

import base64
import gzip
import json
from typing import Any

# Default maximum decompressed size: 100MB
# Can be overridden via BackendConfig.max_decompressed_size_mb
DEFAULT_MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024


class DecompressionLimitExceeded(ValueError):
    """Raised when decompressed data exceeds size limit."""

    def __init__(self, actual_size: int, max_size: int):
        self.actual_size = actual_size
        self.max_size = max_size
        super().__init__(
            f"Decompressed payload ({actual_size:,} bytes) exceeds limit ({max_size:,} bytes)"
        )


def compress_payload(data: Any) -> str:
    """Gzip compress and base64 encode data for JSON transport.

    Args:
        data: Any JSON-serializable data (dict, list, str, etc.)

    Returns:
        Base64-encoded gzip-compressed string
    """
    json_bytes = json.dumps(data).encode("utf-8")
    compressed = gzip.compress(json_bytes, compresslevel=6)
    return base64.b64encode(compressed).decode("ascii")


def decompress_payload(
    data: str,
    max_size: int = DEFAULT_MAX_DECOMPRESSED_SIZE,
) -> Any:
    """Decode base64 and gunzip data with size limit.

    Args:
        data: Base64-encoded gzip-compressed string
        max_size: Maximum allowed decompressed size in bytes (default: 100MB)

    Returns:
        Original JSON-deserialized data

    Raises:
        DecompressionLimitExceeded: If decompressed data exceeds max_size
    """
    compressed = base64.b64decode(data.encode("ascii"))
    json_bytes = gzip.decompress(compressed)

    # Enforce size limit to prevent compression bombs
    if len(json_bytes) > max_size:
        raise DecompressionLimitExceeded(len(json_bytes), max_size)

    return json.loads(json_bytes.decode("utf-8"))


def compress_if_large(data: Any, threshold_bytes: int = 1024) -> tuple[Any, bool]:
    """Compress data only if it exceeds threshold size.

    Args:
        data: Any JSON-serializable data
        threshold_bytes: Minimum size to trigger compression (default 1KB)

    Returns:
        Tuple of (data or compressed_data, was_compressed)
    """
    json_bytes = json.dumps(data).encode("utf-8")
    if len(json_bytes) > threshold_bytes:
        return compress_payload(data), True
    return data, False


def get_compression_ratio(original: Any, compressed: str) -> float:
    """Calculate compression ratio for logging/debugging.

    Args:
        original: Original data
        compressed: Compressed string from compress_payload()

    Returns:
        Ratio of compressed/original size (lower is better)
    """
    original_size = len(json.dumps(original).encode("utf-8"))
    compressed_size = len(compressed.encode("ascii"))
    return compressed_size / original_size if original_size > 0 else 1.0
