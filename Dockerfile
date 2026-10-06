# syntax=docker/dockerfile:1
# Optimized backend image (torch-free).
#
# Stack: Voyage AI embeddings + Cohere rerank (APIs), fastembed (ONNX)
# generation. No torch / transformers / CUDA wheels anywhere, so the image
# stays small (~300-600MB) with plain PyPI resolution.
# Models on volumes: fastembed's ONNX cache (~200MB on first chunking use)
# lives on /models, not in image layers. (LLM weights stay in Ollama;
# Cohere is API-side.) Local ChromaDB (when CHROMA_API_KEY is empty) on /data.
#
# Build:  podman build -t advanced-rag-backend .
# Run:    podman run -d --name rag-api -p 8000:8000 --env-file .env \
#           -v rag-models:/models -v rag-data:/data \
#           -e OLLAMA_HOST=http://host.containers.internal:11434 \
#           advanced-rag-backend
# NOTE: langchain-ollama honors OLLAMA_HOST to reach host-side Ollama.
# NOTE: COHERE_API_KEY must be in .env. Re-ingest after switching stacks:
# the embedding space changed, so old Nomic vectors are not comparable.

# ---------- Stage 1: resolve + install dependencies ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

# Dependency metadata first for layer caching.
COPY pyproject.toml uv.lock .python-version ./

# Runtime deps only (no dev group). Plain PyPI: no torch/CUDA in the tree.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# ---------- Stage 2: lean runtime ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PATH="/app/.venv/bin:$PATH" \
    # Model + data locations live on volumes (see VOLUME below).
    HF_HOME=/models/hf \
    TRANSFORMERS_CACHE=/models/hf \
    SENTENCE_TRANSFORMERS_HOME=/models/st \
    VECTOR_DB_DIR=/data/vectordb

# Non-root user; volume mount points owned by it.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app /models /data \
    && chown -R app:app /app /models /data

WORKDIR /app

# Virtualenv from builder, then the application source.
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app src/ ./src/
COPY --chown=app:app pyproject.toml uv.lock README.md ./

USER app

# Model weights (downloaded from HuggingFace on first use) and the local
# ChromaDB (when not using Chroma Cloud) persist here across rebuilds.
VOLUME ["/models", "/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# Pre-compile imports so first request is fast (models still lazy-load).
RUN python -c "import advanced_rag.api; print('app imports ok')"

CMD ["uvicorn", "advanced_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
