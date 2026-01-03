"""FastAPI application factory for SimpleMem-Lite backend."""

import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import APIKeyHeader
from starlette import status

from simplemem_lite.backend.config import get_config

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
    if not secrets.compare_digest(api_key, config.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return api_key


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup/shutdown."""
    config = get_config()

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

    # Import and include routers with optional auth
    from simplemem_lite.backend.api import router as api_router

    # Add auth dependency if configured
    dependencies = [Depends(verify_api_key)] if config.require_auth else []
    app.include_router(api_router, prefix="/api/v1", dependencies=dependencies)

    # Health check endpoint at root
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint for load balancers."""
        return {"status": "healthy", "service": "simplemem-lite-backend"}

    return app
