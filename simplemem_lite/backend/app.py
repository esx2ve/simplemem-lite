"""FastAPI application factory for SimpleMem-Lite backend."""

import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette import status

from simplemem_lite.backend.config import SecurityMode, get_config

log = logging.getLogger("simplemem_lite.backend.app")

# API Key header for optional authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str | None:
    """Verify API key if authentication is required.

    Returns the API key if valid, raises 401 if invalid when auth is required.
    """
    config = get_config()

    # If auth not required, skip validation
    if not config.require_auth:
        return None

    # Auth required but no key provided
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Verify the key using constant-time comparison to prevent timing attacks
    if not config.api_key or not secrets.compare_digest(api_key, config.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


def _log_security_mode() -> None:
    """Log security mode and settings at startup."""
    config = get_config()

    if config.mode == SecurityMode.DEV:
        log.warning("=" * 70)
        log.warning("  RUNNING IN DEV MODE - SECURITY CONTROLS RELAXED")
        log.warning("=" * 70)
        log.warning("  - Authentication:      DISABLED")
        log.warning("  - Project isolation:   DISABLED")
        log.warning("  - Arbitrary Cypher:    ALLOWED")
        log.warning("  - Verbose errors:      ENABLED")
        log.warning("  - Host binding:        %s (localhost only)", config.host)
        log.warning("")
        log.warning("  Set SIMPLEMEM_MODE=prod for production deployments")
        log.warning("=" * 70)
    else:
        log.info("=" * 70)
        log.info("  RUNNING IN PROD MODE - SECURITY CONTROLS ACTIVE")
        log.info("=" * 70)
        log.info("  - Authentication:      %s", "REQUIRED" if config.require_auth else "disabled")
        log.info("  - Project isolation:   %s", "REQUIRED" if config.require_project_id else "disabled")
        log.info("  - Arbitrary Cypher:    %s", "allowed" if config.allow_arbitrary_cypher else "BLOCKED")
        log.info("  - Verbose errors:      %s", "enabled" if config.verbose_errors else "DISABLED")
        log.info("  - Host binding:        %s", config.host)
        log.info("=" * 70)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    config = get_config()

    # Log security mode at startup
    _log_security_mode()

    # Startup: Initialize services
    # These will be lazily initialized when first accessed
    app.state.config = config

    yield

    # Shutdown: Cleanup resources
    # Services handle their own cleanup


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    config = get_config()

    app = FastAPI(
        title="SimpleMem-Lite Backend",
        description="Backend API for SimpleMem-Lite memory system",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Add gzip middleware for response compression
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Global exception handler for sanitized error messages
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions with mode-aware verbosity."""
        cfg = get_config()
        if cfg.verbose_errors:
            # Dev mode: include full error details
            log.exception("Unhandled exception: %s", exc)
            return JSONResponse(
                status_code=500,
                content={
                    "detail": str(exc),
                    "type": type(exc).__name__,
                },
            )
        else:
            # Prod mode: sanitize error messages
            log.error("Internal error (sanitized): %s: %s", type(exc).__name__, exc)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

    # Import and include routers with optional auth
    from simplemem_lite.backend.api import router as api_router

    # Add auth dependency if configured
    dependencies = [Depends(verify_api_key)] if config.require_auth else []
    app.include_router(api_router, prefix="/api/v1", dependencies=dependencies)

    # Health check endpoint at root
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint for load balancers."""
        cfg = get_config()
        return {
            "status": "healthy",
            "service": "simplemem-lite-backend",
            "mode": cfg.mode.value,
        }

    return app
