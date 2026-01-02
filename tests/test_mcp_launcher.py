"""Tests for MCP backend launcher."""

import asyncio
import subprocess
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from simplemem_lite.mcp.launcher import (
    BackendLauncher,
    ensure_backend_running,
    get_launcher,
    stop_backend,
)


@pytest.fixture(autouse=True)
def reset_globals():
    """Reset global state between tests."""
    import simplemem_lite.mcp.launcher as launcher_module

    launcher_module._launcher = None
    yield
    launcher_module._launcher = None


class TestBackendLauncher:
    """Tests for BackendLauncher class."""

    def test_init_defaults(self):
        """Launcher should use default host and port."""
        launcher = BackendLauncher()
        assert launcher.host == "127.0.0.1"
        assert launcher.port == 8420
        assert launcher.base_url == "http://127.0.0.1:8420"

    def test_init_custom(self):
        """Launcher should accept custom host and port."""
        launcher = BackendLauncher(host="0.0.0.0", port=9000)
        assert launcher.host == "0.0.0.0"
        assert launcher.port == 9000
        assert launcher.base_url == "http://0.0.0.0:9000"

    def test_health_url(self):
        """health_url should return correct endpoint."""
        launcher = BackendLauncher()
        assert launcher.health_url == "http://127.0.0.1:8420/health"

    @pytest.mark.asyncio
    async def test_is_healthy_success(self):
        """is_healthy should return True when backend responds."""
        launcher = BackendLauncher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await launcher.is_healthy()

        assert result is True
        mock_client.get.assert_called_once_with(launcher.health_url)

    @pytest.mark.asyncio
    async def test_is_healthy_not_healthy(self):
        """is_healthy should return False when status is not healthy."""
        launcher = BackendLauncher()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "degraded"}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await launcher.is_healthy()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_healthy_connection_error(self):
        """is_healthy should return False on connection error."""
        launcher = BackendLauncher()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Connection refused")
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            MockClient.return_value = mock_client

            result = await launcher.is_healthy()

        assert result is False

    @pytest.mark.asyncio
    async def test_ensure_running_already_healthy(self):
        """ensure_running should return True if backend is already healthy."""
        launcher = BackendLauncher()

        with patch.object(launcher, "is_healthy", return_value=True):
            result = await launcher.ensure_running()

        assert result is True
        assert not launcher._we_started_it

    @pytest.mark.asyncio
    async def test_ensure_running_starts_backend(self):
        """ensure_running should start backend if not healthy."""
        launcher = BackendLauncher()

        # First call returns False (not running), subsequent calls return True
        health_calls = [False, True]

        with patch.object(
            launcher, "is_healthy", side_effect=lambda: health_calls.pop(0)
        ):
            with patch.object(launcher, "_start_backend", return_value=True):
                result = await launcher.ensure_running()

        assert result is True
        assert launcher._we_started_it

    @pytest.mark.asyncio
    async def test_ensure_running_start_fails(self):
        """ensure_running should return False if start fails."""
        launcher = BackendLauncher()

        with patch.object(launcher, "is_healthy", return_value=False):
            with patch.object(launcher, "_start_backend", return_value=False):
                result = await launcher.ensure_running()

        assert result is False
        assert not launcher._we_started_it

    @pytest.mark.asyncio
    async def test_start_backend_success(self):
        """_start_backend should start subprocess successfully."""
        launcher = BackendLauncher()

        mock_process = MagicMock()
        mock_process.poll.return_value = None  # Process still running

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("asyncio.sleep", return_value=None):
                result = await launcher._start_backend()

        assert result is True
        assert launcher._process is mock_process

    @pytest.mark.asyncio
    async def test_start_backend_immediate_exit(self):
        """_start_backend should return False if process exits immediately."""
        launcher = BackendLauncher()

        mock_process = MagicMock()
        mock_process.poll.return_value = 1  # Process exited
        mock_process.returncode = 1

        with patch("subprocess.Popen", return_value=mock_process):
            with patch("asyncio.sleep", return_value=None):
                result = await launcher._start_backend()

        assert result is False
        assert launcher._process is None

    @pytest.mark.asyncio
    async def test_stop_does_nothing_if_not_started(self):
        """stop should do nothing if we didn't start the backend."""
        launcher = BackendLauncher()
        launcher._process = MagicMock()
        launcher._we_started_it = False

        await launcher.stop()

        # Process should not be terminated
        launcher._process.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_terminates_process(self):
        """stop should terminate the process if we started it."""
        launcher = BackendLauncher()

        mock_process = MagicMock()
        mock_process.wait.return_value = None
        launcher._process = mock_process
        launcher._we_started_it = True

        with patch("asyncio.to_thread", side_effect=lambda f, **kwargs: f(**kwargs)):
            await launcher.stop()

        mock_process.terminate.assert_called_once()
        mock_process.wait.assert_called()
        assert launcher._process is None
        assert not launcher._we_started_it

    @pytest.mark.asyncio
    async def test_stop_kills_on_timeout(self):
        """stop should kill process if it doesn't respond to SIGTERM."""
        launcher = BackendLauncher()

        mock_process = MagicMock()
        mock_process.wait.side_effect = [
            subprocess.TimeoutExpired("cmd", 5.0),  # First wait times out
            None,  # Second wait succeeds after kill
        ]
        launcher._process = mock_process
        launcher._we_started_it = True

        with patch("asyncio.to_thread", side_effect=lambda f, **kwargs: f(**kwargs)):
            await launcher.stop()

        mock_process.terminate.assert_called_once()
        mock_process.kill.assert_called_once()

    def test_is_running_no_process(self):
        """is_running should return False if no process."""
        launcher = BackendLauncher()
        assert launcher.is_running() is False

    def test_is_running_process_running(self):
        """is_running should return True if process is running."""
        launcher = BackendLauncher()
        launcher._process = MagicMock()
        launcher._process.poll.return_value = None
        assert launcher.is_running() is True

    def test_is_running_process_exited(self):
        """is_running should return False if process exited."""
        launcher = BackendLauncher()
        launcher._process = MagicMock()
        launcher._process.poll.return_value = 0
        assert launcher.is_running() is False


class TestGlobalFunctions:
    """Tests for global launcher functions."""

    @pytest.mark.asyncio
    async def test_get_launcher_creates_once(self):
        """get_launcher should create launcher only once."""
        launcher1 = await get_launcher()
        launcher2 = await get_launcher()

        assert launcher1 is launcher2

    @pytest.mark.asyncio
    async def test_ensure_backend_running_delegates(self):
        """ensure_backend_running should delegate to launcher."""
        mock_launcher = AsyncMock()
        mock_launcher.ensure_running.return_value = True

        with patch(
            "simplemem_lite.mcp.launcher.get_launcher",
            return_value=mock_launcher,
        ):
            result = await ensure_backend_running()

        assert result is True
        mock_launcher.ensure_running.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_backend_delegates(self):
        """stop_backend should delegate to launcher."""
        import simplemem_lite.mcp.launcher as launcher_module

        mock_launcher = AsyncMock()
        launcher_module._launcher = mock_launcher

        await stop_backend()

        mock_launcher.stop.assert_called_once()
        assert launcher_module._launcher is None


class TestEnvironmentConfig:
    """Tests for environment variable configuration."""

    def test_host_from_env(self):
        """Launcher should read host from environment."""
        with patch.dict(
            "os.environ", {"SIMPLEMEM_BACKEND_HOST": "192.168.1.1"}, clear=False
        ):
            launcher = BackendLauncher()
            assert launcher.host == "192.168.1.1"

    def test_port_from_env(self):
        """Launcher should read port from environment."""
        with patch.dict("os.environ", {"PORT": "9999"}, clear=False):
            launcher = BackendLauncher()
            assert launcher.port == 9999
