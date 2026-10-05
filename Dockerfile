# syntax=docker/dockerfile:1
# Optimized multi-stage image for the Advanced RAG API.
#
# Build:  docker build -t advanced-rag .
# Run:    docker run --env-file .env -p 8000:8000 advanced-rag
# Notes:
#   - Chroma Cloud is used when CHROMA_API_KEY is set in .env (no local
#     ./vector_db needed inside the container).
#   - Ollama must be reachable: run Ollama on the host and set
#     OLLAMA_HOST=http://host.docker.internal:11434 (or your Ollama URL).
#     langchain-ollama honors OLLAMA_HOST.
#   - Embedding/reranker models download from HuggingFace on first use into
#     /home/app/.cache (mount a volume to persist it across restarts).

# ---------- Stage 1: resolve + install dependencies ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALLS_ONLY=1

# CPU-only PyTorch: the container has no GPU, so resolve torch from the CPU
# index (PyPI as fallback for everything else). Avoids multi-GB CUDA wheels.
# Runtime-only env; the repo lockfile stays untouched.
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app

# Dependency metadata first for layer caching.
COPY pyproject.toml uv.lock .python-version ./

# Install runtime deps only (no dev group) into a venv.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project \
      --default-index https://download.pytorch.org/whl/cpu \
      --index https://pypi.org/simple

# ---------- Stage 2: lean runtime ----------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_OFFLINE=0

# Non-root user + tini-less signal handling via exec-form CMD.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app /home/app/.cache \
    && chown -R app:app /app /home/app/.cache

WORKDIR /app

# Virtualenv from builder, then the application source.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml uv.lock README.md ./

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Pre-compile imports so first request is fast (models still lazy-load).
RUN python -c "import advanced_rag.api; print('app imports ok')"

CMD ["uvicorn", "advanced_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
