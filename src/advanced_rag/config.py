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

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")

# Local generation via Ollama (`ollama serve` daemon, no API key needed).
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")

# Retrieval / augmentation defaults.
DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# Rerank-score floor: bge-reranker-base scores run LOW on this corpus
# (relevant hits ~0.02-0.7, out-of-scope junk ~0.001). Calibrated floor 0.01
# keeps junk out while admitting relevant-but-low short-query hits.
DEFAULT_MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.01"))

DOCS_DIR = os.getenv("DOCS_DIR", str(Path(__file__).resolve().parents[2] / "docs"))
