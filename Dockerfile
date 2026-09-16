# ============================================================================
# AgentX - Production Docker image
#   - Stage 1: install dependencies
#   - Stage 2: slim runtime with the API server
# Build:   docker build -t agentx .
# Run:     docker run -p 8000:8000 agentx
# ============================================================================

FROM python:3.11-slim AS builder

WORKDIR /build

# System libraries needed by scipy / matplotlib wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ----------------------------------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Runtime system dependencies (fonts for matplotlib reliability plots)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "agentx.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
