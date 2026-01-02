# SimpleMem-Lite Backend API
# Fly.io deployment for cloud backend
#
# Architecture:
#   MCP (local, always) → Backend API (this, cloud or local)
#   MCP handles: stdio protocol, local file access, compression
#   Backend handles: LanceDB vectors, KuzuDB graph, LLM calls

FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast package management (pinned for reproducibility)
RUN pip install uv==0.5.14

# Copy dependency files first (better caching)
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies (no editable install, no cache for smaller image)
RUN uv pip install --system --no-cache "."

# Copy application code
COPY simplemem_lite/ ./simplemem_lite/

# Create data directory for persistence
RUN mkdir -p /data

# Environment variables
ENV SIMPLEMEM_DATA_DIR=/data
ENV HOST=0.0.0.0
ENV PORT=8420

# Expose backend API port
EXPOSE 8420

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8420/health --max-time 5 || exit 1

# Run the backend API server
CMD ["python", "-m", "simplemem_lite.backend.main"]
