# Advanced RAG

A grounded Retrieval-Augmented Generation system over your own documents:
ingest → hybrid retrieval (BM25 + dense) → Cohere rerank → strict grounded
generation (Ollama dev / Groq prod) → claim-level LLM evaluation. Ships with a
React chat UI, FastAPI backend, and container images for both.

```
docs/ ──ingest──▶ chunks ──embed (Voyage)──▶ Chroma (local / Cloud)
                                                        │
query ──▶ hybrid search ──▶ Cohere rerank ──▶ augment ──▶ LLM ──▶ answer + sources
                                                        │
                                                        ▼
                                              evaluator (faith / relevance)
```

## Features

- **Ingestion**: PDF/DOCX loading, encoding + PDF-artifact cleaning, exact + fuzzy dedup, token / recursive / semantic / hierarchical chunking, batched embedding with % progress and a single-writer lock.
- **Retrieval**: BM25 (regex + stopword tokenizer) fused with dense search via RRF, Cohere Rerank on top, acronym query expansion (`JDK`, `oops`, …).
- **Generation**: strict context-grounded prompt with deterministic fallback (`"I cannot answer…"`) when context is missing or irrelevant.
- **Evaluation**: claim-level faithfulness (atomic claims + verbatim-quote verification with fuzzy matching), quote-evidenced relevance, separate in-scope vs fallback reporting, batch runner with restart-safe `--ids` subsets.
- **Serving**: FastAPI (`/query`, `/health`, chat UI at `/`, OpenAI-compatible `/v1/*` stub), React frontend, multi-stage container images, CI + pre-commit gates.

## Prerequisites

| Need | Notes |
|------|-------|
| Python 3.13 + `uv` | `uv sync --group dev` |
| Ollama (`ollama serve`) | any chat model, e.g. `ollama pull llama3.2` |
| Node 20 + npm | only for frontend dev (`npm run dev`) |
| Podman (or Docker) | to run the images |
| API keys | `VOYAGE_API_KEY` (embeddings), `COHERE_API_KEY` (rerank); `CHROMA_*` for Chroma Cloud, `GROQ_API_KEY` for prod generation |

## Configuration

```bash
cp .env.example .env   # then fill in keys
```

| Variable | Default | Purpose |
|----------|---------|---------|
| `LLM_PROVIDER` | `ollama` | `ollama` (local dev) or `groq` (prod) |
| `OLLAMA_MODEL` / `GROQ_MODEL` | `llama3.2` / `llama-3.3-70b-versatile` | model ids per provider |
| `VOYAGE_API_KEY` / `VOYAGE_EMBED_MODEL` | `voyage-3` | shared embedding space (ingest + query) |
| `COHERE_API_KEY` / `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | reranker (text-based) |
| `FASTEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | local ONNX, chunk boundaries only |
| `CHROMA_API_KEY/_TENANT/_DATABASE` | empty (= local ChromaDB) | set all three for Chroma Cloud |
| `RAG_TOP_K` / `RAG_MIN_RERANK_SCORE` | `4` / `0.01` | retrieval breadth / relevance floor |
| `DOCS_DIR` | `./docs` | ingest source |

> Changing the embedding model invalidates the index — re-ingest afterwards.
> `.env` edits need a **full** uvicorn restart (`--reload` only hot-reloads code).

## Local development

```bash
uv sync --group dev

# 1. Ingest (chunk → embed → index). Rebuilds the collection.
uv run python -m advanced_rag.app ingest ./docs --strategy recursive
# strategies: recursive | token | semantic | hierarchical

# 2. Verify the index
uv run python -m advanced_rag.check_db

# 3. Ask questions
uv run python -m advanced_rag.app query "What is bytecode in Java?"
uv run python -m advanced_rag.app query "..." --top-k 4 --min-score 0.01 --evaluate

# 4. Backend API (http://localhost:8000, chat UI at /)
uv run uvicorn advanced_rag.api:app --host 0.0.0.0 --port 8000 --reload

# 5. Frontend dev server (http://localhost:5173, proxies /api → :8000)
cd frontend && npm install && npm run dev

# 6. Evaluate (10 queries; use --ids subsets for restart-safe chunks)
uv run python -m advanced_rag.Evaluation.run_batch --ids 1,2,3,4
```

API surface: `GET /health`, `POST /query {query, top_k, min_rerank_score?, model?, evaluate?}`,
`GET /v1/models`, `POST /v1/chat/completions` (OpenAI-compatible stub).

## Running the container images

Images (build with podman; Dockerfiles avoid docker.io-only bases):

```bash
podman build -t advanced-rag-backend .
podman build -t advanced-rag-frontend ./frontend
# published tags on main: <dockerhub-user>/advanced-rag[:latest,:sha]
#                         <dockerhub-user>/advanced-rag-frontend[:latest,:sha]
```

Backend (needs `.env`; models/cache on volumes so rebuilds stay small):

```bash
podman run -d --name rag-api -p 8000:8000 \
  --env-file .env \
  -e OLLAMA_HOST=http://host.containers.internal:11434 \
  -v rag-models:/models \
  -v rag-data:/data \
  advanced-rag-backend
curl http://localhost:8000/health
```

Frontend (port 3000, proxies `/api/*` to the backend):

```bash
podman run -d --name rag-frontend -p 3000:3000 \
  -e BACKEND_URL=http://host.containers.internal:8000 \
  advanced-rag-frontend
# open http://localhost:3000
```

Or run both in one pod (containers then reach each other on localhost):

```bash
podman pod create --name rag -p 8000:8000 -p 3000:3000
podman run -d --pod rag --env-file .env -e OLLAMA_HOST=http://host.containers.internal:11434 \
  -v rag-models:/models -v rag-data:/data --name rag-api advanced-rag-backend
podman run -d --pod rag -e BACKEND_URL=http://127.0.0.1:8000 --name rag-frontend advanced-rag-frontend
```

Ingest from inside the backend image (docs + caches mounted):

```bash
podman run --rm --env-file .env \
  -v ./docs:/app/docs:ro -e DOCS_DIR=/app/docs \
  -v rag-models:/models -v rag-data:/data \
  advanced-rag-backend python -m advanced_rag.app ingest /app/docs --strategy recursive
```

Housekeeping: `podman logs rag-api`, `podman stop rag-api rag-frontend`, `podman rm rag-api rag-frontend`.
`host.containers.internal` needs a backend/Ollama listening on all interfaces (`--host 0.0.0.0`).

## Checks, CI, git

```bash
uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/
uv run pytest tests/ -q            # 18 offline unit tests (no models/network)
pre-commit install                 # ruff + prettier + pytest + large-file guard
```

- `ci.yml`: lint → test on every push/PR.
- `docker.yml`: builds both images, smoke-tests the running containers, pushes to Docker Hub on `main` (needs `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` secrets).
- First push: `git add -A && git commit -m "…" && git push` (DB dumps, logs and `.env` are gitignored).

## Project layout

```
src/advanced_rag/   Loaders/ Chunker/ Vector/ Retrieval/ Augmentor/ Generator/ Evaluation/
frontend/           React + Vite app, Node static/proxy server, own Dockerfile
tests/              offline unit tests (tokenizer, augmentor, evaluator, backend routing)
docs/               source PDFs (gitignored)
```

## Last measured baseline (re-measure after stack changes)

In-scope (n=9): faithfulness **0.46** · context relevance **0.75** · answer relevance **1.00** · overall **0.74** · fallback accuracy **1.00**. Measured pre-Voyage (Nomic + bge + Ollama judges).

## Troubleshooting

- **Refusals with good sources shown**: `RAG_MIN_RERANK_SCORE` too high for the score scale — recalibrate from a batch run.
- **`Collection … does not exist` during ingest**: two concurrent ingests — the `.ingest.lock` now blocks the second one; wait and retry once.
- **Stale behavior after `.env` edits**: full uvicorn restart (reload doesn't re-read env values).
- **Index wiped (e.g. host restart)**: restore `vector_db_backup.tar.gz` or re-ingest; Chroma Cloud avoids this entirely.
- **`/v1/models` 404 spam in logs**: some local tool probing for an OpenAI API — harmless; the stub route answers it.
