# ── Hermes Router Proxy — Docker ─────────────────────────────────────────────
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY server.py router_config.example.yaml ./
RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx pyyaml

# ── Runtime ──────────────────────────────────────────────────────────────────
FROM python:3.13-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app/server.py /app/server.py

# Default config (users should mount their own router_config.yaml over this)
COPY router_config.example.yaml /app/router_config.yaml

EXPOSE 8766

ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python3 -c "import httpx; httpx.get('http://localhost:8766/health', timeout=5).raise_for_status()" || exit 1

CMD ["python3", "/app/server.py"]
