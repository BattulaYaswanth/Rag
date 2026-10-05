"""
retriever.py
Module for retrieving context using Vector Similarity Search,
Hybrid Search (BM25 + Dense Vectors), and Cross-Encoder Reranking.
"""

import os
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# Prevent HuggingFace from attempting online checks if cached locally
os.environ["HF_HUB_OFFLINE"] = "1"

# Word tokens (alphanumeric runs) — punctuation-attached forms like
# "language." and "language" must match the same term.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Minimal English stopword list (dependency-free). These dominate BM25
# scores without carrying topical meaning.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "else",
        "when",
        "at",
        "by",
        "for",
        "with",
        "about",
        "into",
        "through",
        "during",
        "of",
        "to",
        "in",
        "on",
        "is",
        "it",
        "its",
        "as",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "can",
        "could",
        "should",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "how",
        "why",
        "where",
        "there",
        "here",
        "from",
        "not",
        "no",
        "so",
        "than",
        "too",
        "very",
        "you",
        "your",
        "they",
        "them",
        "their",
        "his",
        "her",
        "our",
        "we",
        "he",
        "she",
        "i",
    }
)

# Locate vector_db inside src/advanced_rag/
RETRIEVAL_DIR = Path(__file__).resolve().parent  # .../src/advanced_rag/Retrieval
ADVANCED_RAG_DIR = RETRIEVAL_DIR.parent  # .../src/advanced_rag
DEFAULT_DB_DIR = str(ADVANCED_RAG_DIR / "vector_db")


class AdvancedRetriever:
    """Handles query embedding, similarity search, hybrid search, and reranking."""

    def __init__(
        self,
        persist_directory: str = DEFAULT_DB_DIR,
        collection_name: str = "rag_collection",
        embedding_model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        reranker_model_name: str = "BAAI/bge-reranker-base",
    ):
        print(f"Loading retriever embedding model '{embedding_model_name}'...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )

        from advanced_rag.Vector.chroma_client import describe_backend, get_chroma_client

        print(f"Connecting to ChromaDB ({describe_backend()})")
        client, _ = get_chroma_client(local_path=persist_directory)
        self.vector_store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=self.embeddings,
        )

        self.reranker_model_name = reranker_model_name
        self._reranker = None
        self._bm25 = None
        self._cleaned_corpus = None
        self._all_metas = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            print(f"Loading Cross-Encoder Reranker '{self.reranker_model_name}'...")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def invalidate_bm25_cache(self) -> None:
        """Clear cached BM25 index to force re-indexing on next search."""
        self._bm25 = None
        self._cleaned_corpus = None
        self._all_metas = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokens with stopwords removed.

        Used for BOTH the BM25 corpus and queries so term matching is
        consistent (previously plain ``str.split()``, where "language."
        and "language" were different terms and stopwords dominated scores).
        """
        return [
            tok for tok in _TOKEN_RE.findall(text.lower()) if tok not in _STOPWORDS and len(tok) > 1
        ]

    def _get_bm25_index(self):
        """Lazy-loads and caches BM25 corpus to avoid rebuilding on every query."""
        if self._bm25 is None:
            raw_data = self.vector_store.get(include=["documents", "metadatas"])
            all_docs = raw_data.get("documents", [])
            self._all_metas = raw_data.get("metadatas", [])
            self._cleaned_corpus = [self._clean_doc_text(doc) for doc in all_docs]
            tokenized_corpus = [self._tokenize(doc) for doc in self._cleaned_corpus]
            if tokenized_corpus:
                self._bm25 = BM25Okapi(tokenized_corpus)
            else:
                self._bm25 = None
        return self._bm25, self._cleaned_corpus, self._all_metas

    @staticmethod
    def expand_query(query: str) -> str:
        """Enriches queries with acronym expansions and canonical terms for higher recall."""
        expanded = query
        expansions = {
            r"\bJDK\b": "JDK Java Development Kit",
            r"\bJRE\b": "JRE Java Runtime Environment",
            r"\bJVM\b": "JVM Java Virtual Machine",
            r"\bOOP\b": "OOP Object-Oriented Programming",
            r"\boops\b": "oops OOP Object-Oriented Programming",
            r"\bAPI\b": "API Application Programming Interface",
            r"\bcompiler vs interpreter\b": "difference between Java compiler and interpreter",
        }
        for pattern, replacement in expansions.items():
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        return expanded.strip()

    def _format_query(self, query: str) -> str:
        """Ensures query carries search_query: prefix for Nomic models."""
        clean = query.replace("search_query: ", "").replace("search_document: ", "").strip()
        return f"search_query: {clean}"

    def _clean_doc_text(self, text: str) -> str:
        """Strips search_document: prefix from stored document text."""
        if text.startswith("search_document: "):
            return text[len("search_document: ") :]
        return text

    # -------------------------------------------------------------------------
    # 1. Standard Vector Similarity Search
    # -------------------------------------------------------------------------
    def vector_search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Converts query to embedding and performs similarity search in ChromaDB."""
        formatted_query = self._format_query(query)
        results_with_scores = self.vector_store.similarity_search_with_score(
            formatted_query, k=top_k
        )

        formatted_results = []
        for doc, score in results_with_scores:
            formatted_results.append(
                {
                    "content": self._clean_doc_text(doc.page_content),
                    "raw_content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                }
            )
        return formatted_results

    # -------------------------------------------------------------------------
    # 2. Hybrid Search (BM25 Keyword + Vector Embeddings)
    # -------------------------------------------------------------------------
    def hybrid_search(
        self, query: str, top_k: int = 5, vector_weight: float = 0.6
    ) -> list[dict[str, Any]]:
        """Combines BM25 keyword search and vector retrieval via RRF with in-memory caching."""
        bm25, cleaned_corpus, all_metas = self._get_bm25_index()

        if not cleaned_corpus or bm25 is None:
            return []

        clean_query = query.replace("search_query: ", "").replace("search_document: ", "").strip()
        expanded_query = self.expand_query(clean_query)

        # Tokenize expanded query for BM25 (same tokenizer as the corpus)
        tokenized_query = self._tokenize(expanded_query)
        bm25_scores = bm25.get_scores(tokenized_query)

        # Dense retrieval uses both clean and expanded search
        vector_results = self.vector_search(expanded_query, top_k=top_k * 2)

        rrf_scores: dict[str, float] = {}
        content_map: dict[str, dict[str, Any]] = {}

        for rank, res in enumerate(vector_results, start=1):
            text = res["content"]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + (vector_weight / (60 + rank))
            content_map[text] = res

        bm25_top_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: top_k * 2]

        for rank, idx in enumerate(bm25_top_indices, start=1):
            text = cleaned_corpus[idx]
            rrf_scores[text] = rrf_scores.get(text, 0.0) + ((1.0 - vector_weight) / (60 + rank))
            if text not in content_map:
                content_map[text] = {
                    "content": text,
                    "metadata": all_metas[idx] if idx < len(all_metas) else {},
                    "similarity_score": float(bm25_scores[idx]),
                }

        sorted_texts = sorted(rrf_scores.keys(), key=lambda t: rrf_scores[t], reverse=True)[:top_k]

        return [content_map[t] for t in sorted_texts]

    # -------------------------------------------------------------------------
    # 3. Hybrid Search + Cross-Encoder Reranking
    # -------------------------------------------------------------------------
    def rerank_search(
        self, query: str, top_k: int = 4, initial_fetch_k: int = 25
    ) -> list[dict[str, Any]]:
        """Retrieves candidate chunks via Hybrid Search and rescores via Cross-Encoder."""
        clean_query = query.replace("search_query: ", "").replace("search_document: ", "").strip()
        expanded_query = self.expand_query(clean_query)
        candidates = self.hybrid_search(expanded_query, top_k=initial_fetch_k)

        if not candidates:
            return []

        # Cross-encoder pairs between original question and chunk content
        pairs = [[expanded_query, c["content"]] for c in candidates]
        scores = self.reranker.predict(pairs)

        for candidate, score in zip(candidates, scores, strict=True):
            candidate["rerank_score"] = float(score)

        reranked_results = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

        return reranked_results
