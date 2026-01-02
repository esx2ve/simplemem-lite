"""Entry point for running the SimpleMem-Lite backend server."""

import uvicorn

from simplemem_lite.backend.app import create_app
from simplemem_lite.backend.config import get_config


def main() -> None:
    """Run the backend server with uvicorn."""
    config = get_config()

    app = create_app()

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
