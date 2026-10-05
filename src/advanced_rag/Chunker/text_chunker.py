"""
text_chunker.py
Provides token-based, semantic, and hierarchical chunking strategies
with automated document prefix formatting for Nomic models.
"""

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class TextChunker:
    """Provides chunking capabilities using open-source models."""

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        print(f"Initializing HuggingFace embedding model '{model_name}'...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
            },
            encode_kwargs={"normalize_embeddings": True},
        )

    def _format_document(self, text: str) -> str:
        """Appends 'search_document: ' prefix required by nomic-embed-text models."""
        if not text.startswith("search_document: "):
            return f"search_document: {text}"
        return text

    def chunk_by_tokens(
        self, text: str, chunk_size: int = 256, chunk_overlap: int = 32
    ) -> list[str]:
        """Token-based chunking with window overlap."""
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(text)
        return [self._format_document(c) for c in chunks]

    def chunk_recursive(
        self,
        text: str,
        chunk_size: int = 750,
        chunk_overlap: int = 120,
        min_chunk_len: int = 100,
    ) -> list[str]:
        """
        Recursive character chunking respecting paragraph and sentence boundaries.
        Filters out micro-chunks and non-informative fragments.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        raw_chunks = splitter.split_text(text)
        filtered = []
        for c in raw_chunks:
            c_str = c.strip()
            if len(c_str) < min_chunk_len:
                continue
            # Filter chunks that are primarily dots and numbers (TOC lines)
            alnum_ratio = sum(1 for ch in c_str if ch.isalnum()) / max(len(c_str), 1)
            if alnum_ratio < 0.4:
                continue
            filtered.append(self._format_document(c_str))
        return filtered

    def chunk_semantic(
        self,
        text: str,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 90.0,
        min_chunk_len: int = 100,
    ) -> list[str]:
        """Semantic chunking based on embedding distance shifts."""
        semantic_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount,
        )
        docs = semantic_splitter.create_documents([text])
        filtered = []
        for doc in docs:
            c_str = doc.page_content.strip()
            if len(c_str) >= min_chunk_len:
                filtered.append(self._format_document(c_str))
        return filtered

    def chunk_hierarchical(
        self,
        text: str,
        parent_chunk_size: int = 1000,
        parent_overlap: int = 100,
        child_chunk_size: int = 256,
        child_overlap: int = 32,
    ) -> list[dict[str, str | list[str]]]:
        """Hierarchical (Parent-Child) chunking."""
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_overlap
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_overlap
        )

        parent_docs = parent_splitter.split_text(text)
        hierarchical_structure = []

        for p_idx, parent_text in enumerate(parent_docs):
            children = child_splitter.split_text(parent_text)
            hierarchical_structure.append(
                {
                    "parent_id": f"parent_{p_idx}",
                    "parent_chunk": self._format_document(parent_text),
                    "child_chunks": [self._format_document(c) for c in children],
                }
            )

        return hierarchical_structure
