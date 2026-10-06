"""Central configuration for the advanced-rag pipeline (env-driven)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PACKAGE_DIR = Path(__file__).resolve().parent
# Canonical vector DB location (used when Chroma runs locally).
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(PACKAGE_DIR / "vector_db"))
COLLECTION_NAME = os.getenv("RAG_COLLECTION", "rag_collection")

# Chroma Cloud (https://cloud.trychroma.com). When CHROMA_API_KEY is set,
# the pipeline uses Chroma Cloud; otherwise it falls back to a local
# persistent ChromaDB at VECTOR_DB_DIR. Tenant/database come from .env.
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY", "")
CHROMA_TENANT = os.getenv("CHROMA_TENANT", "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
USE_CHROMA_CLOUD = bool(CHROMA_API_KEY)

# Embeddings: Voyage AI API for the shared vector space (ingest + retrieval
# MUST use the same model). Rerank stays Cohere (text-based, needs no shared
# space). fastembed (local ONNX) is only for semantic-chunk boundaries.
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
COHERE_RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-english-v3.0")
FASTEMBED_MODEL = os.getenv("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Generation: Ollama for local dev, Groq for prod (LLM_PROVIDER picks) ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

# Retrieval / augmentation defaults.
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# Rerank-score floor (Cohere relevance 0-1). Recalibrate against a labeled
# batch if you change COHERE_RERANK_MODEL; 0.01 admits relevant-but-weak
# short-query hits while empty/junk retrieval still falls to fallback.
DEFAULT_MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.01"))

DOCS_DIR = os.getenv("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "docs"))
