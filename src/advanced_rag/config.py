"""Central configuration for the advanced-rag pipeline (env-driven)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _clean(value: str) -> str:
    """Strip whitespace + one layer of surrounding quotes.

    python-dotenv strips quotes when reading .env, but container runtimes
    (docker/podman --env-file) pass values through verbatim, so
    LLM_PROVIDER='"ollama"' would otherwise fail provider matching (and
    int()/float() parsing for numeric settings).
    """
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return value


def _env(name: str, default: str = "") -> str:
    """Env reader that tolerates quoted values from container --env-file."""
    raw = os.getenv(name, default)
    return _clean(raw) if isinstance(raw, str) else default


# Normalize already-exported variables too (container --env-file case):
# load_dotenv() never overrides existing environ, so scrub in place.
for _key in (
    "LLM_PROVIDER",
    "OLLAMA_MODEL",
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "VOYAGE_API_KEY",
    "VOYAGE_EMBED_MODEL",
    "COHERE_API_KEY",
    "COHERE_RERANK_MODEL",
    "FASTEMBED_MODEL",
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "VECTOR_DB_DIR",
    "RAG_COLLECTION",
    "RAG_TOP_K",
    "RAG_MIN_RERANK_SCORE",
    "DOCS_DIR",
):
    if _key in os.environ:
        os.environ[_key] = _clean(os.environ[_key])

PACKAGE_DIR = Path(__file__).resolve().parent
# Canonical vector DB location (used when Chroma runs locally).
VECTOR_DB_DIR = _env("VECTOR_DB_DIR", str(PACKAGE_DIR / "vector_db"))
COLLECTION_NAME = _env("RAG_COLLECTION", "rag_collection")

# Chroma Cloud (https://cloud.trychroma.com). When CHROMA_API_KEY is set,
# the pipeline uses Chroma Cloud; otherwise it falls back to a local
# persistent ChromaDB at VECTOR_DB_DIR. Tenant/database come from .env.
CHROMA_API_KEY = _env("CHROMA_API_KEY", "")
CHROMA_TENANT = _env("CHROMA_TENANT", "")
CHROMA_DATABASE = _env("CHROMA_DATABASE", "")
USE_CHROMA_CLOUD = bool(CHROMA_API_KEY)

# Embeddings: Voyage AI API for the shared vector space (ingest + retrieval
# MUST use the same model). Rerank stays Cohere (text-based, needs no shared
# space). fastembed (local ONNX) is only for semantic-chunk boundaries.
VOYAGE_API_KEY = _env("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = _env("VOYAGE_EMBED_MODEL", "voyage-3")
COHERE_API_KEY = _env("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = _env("COHERE_RERANK_MODEL", "rerank-english-v3.0")
FASTEMBED_MODEL = _env("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Generation: Ollama for local dev, Groq for prod (LLM_PROVIDER picks) ---
LLM_PROVIDER = _env("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = _env("GROQ_API_KEY", "")
GROQ_MODEL = _env("GROQ_MODEL", "openai/gpt-oss-20b")

# Retrieval / augmentation defaults.
DEFAULT_TOP_K = int(_env("RAG_TOP_K", "4"))
# Rerank-score floor (Cohere relevance 0-1). Recalibrate against a labeled
# batch if you change COHERE_RERANK_MODEL; 0.01 admits relevant-but-weak
# short-query hits while empty/junk retrieval still falls to fallback.
DEFAULT_MIN_RERANK_SCORE = float(_env("RAG_MIN_RERANK_SCORE", "0.01"))

DOCS_DIR = _env("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "docs"))
