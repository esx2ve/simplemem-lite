"""Daemon lifecycle manager for SimpleMem Lite.

Handles starting, stopping, and health-checking the singleton daemon.
Ensures only one daemon runs at a time via lock file.

Usage:
    from simplemem_lite.daemon_manager import DaemonManager

    manager = DaemonManager()
    if not manager.is_running():
        manager.start_daemon()

    # Or use ensure_running() for auto-start
    manager.ensure_running()
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from simplemem_lite.log_config import get_logger

log = get_logger("daemon_manager")

# Paths (must match daemon.py)
DAEMON_SOCKET_PATH = Path.home() / ".simplemem_lite" / "daemon.sock"
DAEMON_LOCK_PATH = Path.home() / ".simplemem_lite" / "daemon.lock"
DAEMON_LOG_PATH = Path.home() / ".simplemem_lite" / "logs" / "daemon.log"


class DaemonManager:
    """Manages the SimpleMem daemon lifecycle.

    Provides methods to:
    - Check if daemon is running
    - Start daemon in background
    - Stop daemon gracefully
    - Ensure daemon is running (auto-start if needed)
    - Get daemon status and health
    """

    def __init__(self):
        """Initialize the daemon manager."""
        log.debug("DaemonManager initialized")
        self._startup_timeout = 10.0  # seconds to wait for daemon startup
        self._ping_timeout = 2.0  # seconds to wait for ping response

    def is_running(self) -> bool:
        """Check if daemon is currently running.

        Verifies:
        1. Lock file exists with valid PID
        2. Socket file exists
        3. Process with PID is alive
        4. Daemon responds to ping

        Returns:
            True if daemon is running and healthy
        """
        log.trace("Checking if daemon is running...")

        # Check lock file
        if not DAEMON_LOCK_PATH.exists():
            log.trace("No lock file, daemon not running")
            return False

        try:
            lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
            pid = lock_data.get("pid")
            if not pid:
                log.debug("Lock file missing PID")
                return False
        except (json.JSONDecodeError, OSError) as e:
            log.debug(f"Failed to read lock file: {e}")
            return False

        # Check if process exists
        if not self._process_exists(pid):
            log.debug(f"Process {pid} no longer exists, cleaning stale lock")
            self._cleanup_stale()
            return False

        # Check socket exists
        if not DAEMON_SOCKET_PATH.exists():
            log.debug("Socket file missing, daemon may be starting")
            return False

        # Try ping for definitive check
        try:
            result = self._ping_daemon()
            if result:
                log.trace(f"Daemon is running (PID={pid})")
                return True
        except Exception as e:
            log.debug(f"Ping failed: {e}")

        return False

    def _process_exists(self, pid: int) -> bool:
        """Check if a process with given PID exists."""
        try:
            os.kill(pid, 0)  # Signal 0 = just check existence
            return True
        except OSError:
            return False

    def _cleanup_stale(self) -> None:
        """Clean up stale lock and socket files."""
        log.debug("Cleaning up stale daemon files...")

        if DAEMON_LOCK_PATH.exists():
            try:
                DAEMON_LOCK_PATH.unlink()
                log.debug("Removed stale lock file")
            except OSError as e:
                log.warning(f"Failed to remove lock file: {e}")

        if DAEMON_SOCKET_PATH.exists():
            try:
                DAEMON_SOCKET_PATH.unlink()
                log.debug("Removed stale socket file")
            except OSError as e:
                log.warning(f"Failed to remove socket file: {e}")

    def _ping_daemon(self) -> dict | None:
        """Send ping to daemon and return response.

        Returns:
            Response dict or None if ping failed
        """
        import asyncio
        from simplemem_lite.client import DaemonClient

        async def do_ping():
            client = DaemonClient()
            try:
                return await asyncio.wait_for(
                    client.ping(),
                    timeout=self._ping_timeout
                )
            finally:
                await client.close()

        try:
            return asyncio.run(do_ping())
        except Exception:
            return None

    def start_daemon(self, wait: bool = True) -> dict[str, Any]:
        """Start the daemon in background.

        Args:
            wait: If True, wait for daemon to be ready before returning

        Returns:
            Status dict with pid, socket path, etc.
        """
        log.info("Starting SimpleMem daemon...")

        # Check if already running
        if self.is_running():
            lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
            log.info(f"Daemon already running (PID={lock_data.get('pid')})")
            return {
                "status": "already_running",
                "pid": lock_data.get("pid"),
                "socket": str(DAEMON_SOCKET_PATH),
            }

        # Clean up any stale files
        self._cleanup_stale()

        # Create log directory
        DAEMON_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

        # Find daemon module path
        daemon_module = Path(__file__).parent / "daemon.py"
        if not daemon_module.exists():
            log.error(f"Daemon module not found: {daemon_module}")
            return {"error": f"Daemon module not found: {daemon_module}"}

        # Start daemon process
        log.debug(f"Spawning daemon: {sys.executable} {daemon_module}")

        try:
            # Open log file for daemon output
            with open(DAEMON_LOG_PATH, "a") as log_file:
                process = subprocess.Popen(
                    [sys.executable, str(daemon_module)],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from parent
                    cwd=str(Path.home()),
                )

            log.info(f"Daemon process started (PID={process.pid})")

            # Wait for daemon to be ready
            if wait:
                ready = self._wait_for_ready()
                if not ready:
                    log.error("Daemon failed to start within timeout")
                    return {
                        "error": "Daemon failed to start within timeout",
                        "pid": process.pid,
                        "log": str(DAEMON_LOG_PATH),
                    }

            # Read final lock data
            if DAEMON_LOCK_PATH.exists():
                lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
                return {
                    "status": "started",
                    "pid": lock_data.get("pid"),
                    "socket": str(DAEMON_SOCKET_PATH),
                    "log": str(DAEMON_LOG_PATH),
                }
            else:
                return {
                    "status": "started",
                    "pid": process.pid,
                    "socket": str(DAEMON_SOCKET_PATH),
                    "log": str(DAEMON_LOG_PATH),
                }

        except Exception as e:
            log.error(f"Failed to start daemon: {e}")
            return {"error": str(e)}

    def _wait_for_ready(self) -> bool:
        """Wait for daemon to become ready.

        Returns:
            True if daemon is ready, False if timeout
        """
        log.debug(f"Waiting for daemon to be ready (timeout={self._startup_timeout}s)...")

        start_time = time.time()
        poll_interval = 0.2

        while time.time() - start_time < self._startup_timeout:
            # Check if socket exists and daemon responds
            if DAEMON_SOCKET_PATH.exists():
                try:
                    result = self._ping_daemon()
                    if result and result.get("status") == "ok":
                        elapsed = time.time() - start_time
                        log.info(f"Daemon ready after {elapsed:.1f}s")
                        return True
                except Exception:
                    pass  # Keep trying

            time.sleep(poll_interval)

        return False

    def stop_daemon(self, timeout: float = 5.0) -> dict[str, Any]:
        """Stop the daemon gracefully.

        Args:
            timeout: Seconds to wait for graceful shutdown

        Returns:
            Status dict
        """
        log.info("Stopping SimpleMem daemon...")

        # Check if running
        if not DAEMON_LOCK_PATH.exists():
            log.info("Daemon not running (no lock file)")
            return {"status": "not_running"}

        try:
            lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
            pid = lock_data.get("pid")
        except (json.JSONDecodeError, OSError) as e:
            log.warning(f"Failed to read lock file: {e}")
            self._cleanup_stale()
            return {"status": "cleaned_stale"}

        if not pid:
            log.warning("Lock file missing PID")
            self._cleanup_stale()
            return {"status": "cleaned_stale"}

        # Check if process exists
        if not self._process_exists(pid):
            log.info(f"Process {pid} already dead, cleaning up")
            self._cleanup_stale()
            return {"status": "already_dead", "pid": pid}

        # Send SIGTERM for graceful shutdown
        log.debug(f"Sending SIGTERM to PID {pid}")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            log.error(f"Failed to send SIGTERM: {e}")
            return {"error": str(e), "pid": pid}

        # Wait for process to exit
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._process_exists(pid):
                elapsed = time.time() - start_time
                log.info(f"Daemon stopped gracefully after {elapsed:.1f}s")
                self._cleanup_stale()
                return {"status": "stopped", "pid": pid}
            time.sleep(0.1)

        # Force kill if still running
        log.warning(f"Daemon didn't stop gracefully, sending SIGKILL")
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            self._cleanup_stale()
            return {"status": "killed", "pid": pid}
        except OSError as e:
            log.error(f"Failed to kill daemon: {e}")
            return {"error": str(e), "pid": pid}

    def ensure_running(self) -> dict[str, Any]:
        """Ensure daemon is running, starting it if necessary.

        This is the primary method for MCP servers to use.
        It's idempotent and safe to call multiple times.

        Returns:
            Status dict with daemon info
        """
        log.trace("Ensuring daemon is running...")

        if self.is_running():
            lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
            log.trace(f"Daemon already running (PID={lock_data.get('pid')})")
            return {
                "status": "running",
                "pid": lock_data.get("pid"),
                "socket": str(DAEMON_SOCKET_PATH),
            }

        return self.start_daemon(wait=True)

    def get_status(self) -> dict[str, Any]:
        """Get detailed daemon status.

        Returns:
            Dict with running status, PID, uptime, etc.
        """
        log.trace("Getting daemon status...")

        if not DAEMON_LOCK_PATH.exists():
            return {
                "running": False,
                "reason": "no_lock_file",
            }

        try:
            lock_data = json.loads(DAEMON_LOCK_PATH.read_text())
        except (json.JSONDecodeError, OSError) as e:
            return {
                "running": False,
                "reason": "invalid_lock_file",
                "error": str(e),
            }

        pid = lock_data.get("pid")
        started_at = lock_data.get("started_at", 0)

        if not self._process_exists(pid):
            return {
                "running": False,
                "reason": "process_dead",
                "pid": pid,
            }

        if not DAEMON_SOCKET_PATH.exists():
            return {
                "running": False,
                "reason": "socket_missing",
                "pid": pid,
            }

        # Try to get extended status from daemon
        try:
            result = self._get_daemon_status()
            if result:
                return {
                    "running": True,
                    "pid": pid,
                    "socket": str(DAEMON_SOCKET_PATH),
                    "started_at": started_at,
                    "uptime": result.get("uptime", time.time() - started_at),
                    "client_count": result.get("client_count", 0),
                    "request_count": result.get("request_count", 0),
                    "handlers": result.get("handlers", []),
                }
        except Exception as e:
            log.debug(f"Failed to get daemon status: {e}")

        # Fallback to basic status
        return {
            "running": True,
            "pid": pid,
            "socket": str(DAEMON_SOCKET_PATH),
            "started_at": started_at,
            "uptime": time.time() - started_at,
        }

    def _get_daemon_status(self) -> dict | None:
        """Get extended status from daemon via RPC."""
        import asyncio
        from simplemem_lite.client import DaemonClient

        async def do_status():
            client = DaemonClient()
            try:
                return await asyncio.wait_for(
                    client.call("daemon_status", {}),
                    timeout=self._ping_timeout
                )
            finally:
                await client.close()

        try:
            return asyncio.run(do_status())
        except Exception:
            return None

    def restart_daemon(self) -> dict[str, Any]:
        """Restart the daemon.

        Returns:
            Status dict from start operation
        """
        log.info("Restarting SimpleMem daemon...")

        # Stop if running
        stop_result = self.stop_daemon()
        log.debug(f"Stop result: {stop_result}")

        # Start fresh
        return self.start_daemon(wait=True)


# Module-level singleton for convenience
_manager: DaemonManager | None = None


def get_manager() -> DaemonManager:
    """Get the singleton daemon manager."""
    global _manager
    if _manager is None:
        _manager = DaemonManager()
    return _manager


def ensure_daemon_running() -> dict[str, Any]:
    """Convenience function to ensure daemon is running.

    Returns:
        Status dict with daemon info
    """
    return get_manager().ensure_running()
