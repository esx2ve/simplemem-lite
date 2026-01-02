# SimpleMem-Lite MCP Server
# Fly.io deployment with SSE transport for remote MCP

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies
RUN uv pip install --system -e ".[falkordb]"

# Copy application code
COPY simplemem_lite/ ./simplemem_lite/
COPY hooks/ ./hooks/

# Create data directory for persistence
RUN mkdir -p /data

# Environment variables
ENV SIMPLEMEM_DATA_DIR=/data
ENV SIMPLEMEM_TRANSPORT=sse
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose MCP SSE port
EXPOSE 8000

# Health check - just verify port is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -sf http://localhost:8000/sse --max-time 5 -o /dev/null || exit 1

# Run the MCP server with SSE transport
CMD ["python", "-m", "simplemem_lite.server"]
