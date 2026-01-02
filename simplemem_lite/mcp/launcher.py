"""Backend launcher for auto-starting local backend from MCP.

When the MCP server runs locally, it can auto-start the backend API
if it's not already running. This provides a seamless local experience.
"""

import asyncio
import os
import subprocess
import sys
import time

import httpx

from simplemem_lite.log_config import get_logger

log = get_logger("mcp.launcher")

# Default settings
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8420
HEALTH_CHECK_TIMEOUT = 2.0  # seconds
STARTUP_TIMEOUT = 30.0  # seconds
STARTUP_POLL_INTERVAL = 0.5  # seconds


class BackendLauncher:
    """Manages auto-starting the local backend server.

    The launcher:
    1. Checks if backend is already running via health check
    2. If not, starts it as a subprocess
    3. Waits for it to become healthy
    4. Tracks the subprocess for cleanup on shutdown
    """

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
    ):
        """Initialize the launcher.

        Args:
            host: Backend host (default: 127.0.0.1)
            port: Backend port (default: 8420)
        """
        self.host = host or os.environ.get("SIMPLEMEM_BACKEND_HOST", DEFAULT_HOST)
        # Prefer explicit port parameter over environment variable
        if port is not None:
            self.port = port
        else:
            port_env = os.environ.get("PORT")
            self.port = int(port_env) if port_env else DEFAULT_PORT
        self.base_url = f"http://{self.host}:{self.port}"

        self._process: subprocess.Popen | None = None
        self._we_started_it = False

    @property
    def health_url(self) -> str:
        """URL for health check endpoint."""
        return f"{self.base_url}/health"

    async def is_healthy(self) -> bool:
        """Check if the backend is healthy and reachable.

        Returns:
            True if backend responds to health check
        """
        try:
            async with httpx.AsyncClient(timeout=HEALTH_CHECK_TIMEOUT) as client:
                response = await client.get(self.health_url)
                if response.status_code == 200:
                    data = response.json()
                    return data.get("status") == "healthy"
                return False
        except Exception:
            return False

    async def ensure_running(self) -> bool:
        """Ensure the backend is running, starting it if necessary.

        Returns:
            True if backend is running (either already was or we started it)
        """
        # First check if already running
        if await self.is_healthy():
            log.info(f"Backend already running at {self.base_url}")
            return True

        # Try to start it
        log.info(f"Backend not running, attempting to start at {self.base_url}")
        started = await self._start_backend()

        if not started:
            log.error("Failed to start backend")
            return False

        # Wait for it to become healthy
        healthy = await self._wait_for_healthy()
        if healthy:
            log.info(f"Backend started successfully at {self.base_url}")
            self._we_started_it = True
        else:
            log.error("Backend started but never became healthy")
            await self.stop()

        return healthy

    async def _start_backend(self) -> bool:
        """Start the backend as a subprocess.

        Returns:
            True if process started successfully
        """
        if self._process is not None and self._process.poll() is None:
            # Already have a running process
            return True

        try:
            # Set environment variables for the backend
            env = os.environ.copy()
            env["HOST"] = self.host
            env["PORT"] = str(self.port)

            # Find the backend module path
            # We run: python -m simplemem_lite.backend.main
            cmd = [
                sys.executable,
                "-m",
                "simplemem_lite.backend.main",
            ]

            log.debug(f"Starting backend with command: {' '.join(cmd)}")

            # Start as subprocess with output discarded to prevent buffer deadlock.
            # The backend logs to its own file, so we don't need stdout/stderr here.
            self._process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Don't create a new process group - we want to be able to kill it
                start_new_session=False,
            )

            # Brief wait to check for immediate failures
            await asyncio.sleep(0.1)

            if self._process.poll() is not None:
                # Process exited immediately - something went wrong
                exit_code = self._process.returncode
                log.error(f"Backend process exited immediately with code {exit_code}")
                self._process = None
                return False

            return True

        except Exception as e:
            log.error(f"Failed to start backend process: {e}")
            self._process = None
            return False

    async def _wait_for_healthy(self) -> bool:
        """Wait for backend to become healthy.

        Returns:
            True if backend becomes healthy within timeout
        """
        start_time = time.time()

        while time.time() - start_time < STARTUP_TIMEOUT:
            # Check if process is still running
            if self._process is not None and self._process.poll() is not None:
                log.error("Backend process exited unexpectedly")
                return False

            # Check health
            if await self.is_healthy():
                return True

            await asyncio.sleep(STARTUP_POLL_INTERVAL)

        return False

    async def stop(self) -> None:
        """Stop the backend if we started it."""
        if self._process is None:
            return

        if not self._we_started_it:
            log.debug("Not stopping backend - we didn't start it")
            return

        log.info("Stopping backend server...")

        try:
            # Send SIGTERM for graceful shutdown
            self._process.terminate()

            # Wait for graceful shutdown (non-blocking)
            try:
                await asyncio.to_thread(self._process.wait, timeout=5.0)
                log.info("Backend stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                log.warning("Backend not responding to SIGTERM, killing...")
                self._process.kill()
                await asyncio.to_thread(self._process.wait, timeout=2.0)
                log.info("Backend killed")

        except Exception as e:
            log.error(f"Error stopping backend: {e}")

        finally:
            self._process = None
            self._we_started_it = False

    def is_running(self) -> bool:
        """Check if we have a running subprocess."""
        return self._process is not None and self._process.poll() is None


# Global launcher instance
_launcher: BackendLauncher | None = None
_launcher_lock = asyncio.Lock()


async def get_launcher() -> BackendLauncher:
    """Get or create the global launcher instance (thread-safe)."""
    global _launcher
    if _launcher is None:
        async with _launcher_lock:
            if _launcher is None:
                _launcher = BackendLauncher()
    return _launcher


async def ensure_backend_running() -> bool:
    """Ensure the backend is running, auto-starting if needed.

    This is the main entry point for MCP server to use.

    Returns:
        True if backend is available
    """
    launcher = await get_launcher()
    return await launcher.ensure_running()


async def stop_backend() -> None:
    """Stop the backend if we started it.

    Called during MCP server shutdown.
    """
    global _launcher
    if _launcher is not None:
        await _launcher.stop()
        _launcher = None
