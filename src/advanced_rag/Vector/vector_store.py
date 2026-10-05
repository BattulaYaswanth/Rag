"""
vector_store.py
Module for generating embeddings and storing/indexing chunks in ChromaDB or FAISS.
Supports standard chunks as well as Parent-Child hierarchical metadata.
"""

from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DIR = Path(__file__).resolve().parent
DEFAULT_PERSIST_DIR = str(VECTOR_DIR.parent / "vector_db")


class VectorStoreManager:
    """Handles embedding generation and vector database operations."""

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        persist_directory: str = DEFAULT_PERSIST_DIR,
    ):
        print(f"Initializing Embedding Model ({model_name})...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cpu",
                "trust_remote_code": True,
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )
        self.persist_directory = persist_directory

    def _recreate_empty_collection(self, collection_name: str) -> Chroma:
        """Delete the collection if present and return a fresh empty handle."""
        from advanced_rag.Vector.chroma_client import get_chroma_client

        client, _ = get_chroma_client(local_path=self.persist_directory)
        try:
            names = [c.name for c in client.list_collections()]
        except Exception:
            names = []
        if collection_name in names:
            try:
                client.delete_collection(collection_name)
            except Exception as e:
                print(f"  (delete notice: {e})")
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            client=client,
        )

    @staticmethod
    def _add_in_batches(store: Chroma, documents: list[Document], batch_size: int) -> None:
        """Embed + store one batch at a time, printing % progress."""
        total = len(documents)
        for start in range(0, total, batch_size):
            batch = documents[start : start + batch_size]
            store.add_documents(batch)
            done = min(start + len(batch), total)
            print(f"  [{done}/{total}] {done / total * 100:.0f}% embedded + stored")

    def build_chroma_db(
        self,
        chunks: list[str | dict[str, Any]],
        collection_name: str = "rag_collection",
        batch_size: int = 100,
    ) -> Chroma:
        """
        Converts text chunks or hierarchical dictionaries into vector embeddings
        and stores them in a local Chroma vector database.

        Writes in batches (progress printed per batch). Retries once against a
        freshly recreated collection if the handle goes stale mid-write.
        """
        try:
            from chromadb.errors import NotFoundError
        except Exception:
            NotFoundError = Exception  # type: ignore[no-redef]

        documents = self._prepare_documents(chunks)
        total = len(documents)
        if total == 0:
            raise ValueError("No documents to index — nothing was stored.")

        print(
            f"Embedding + storing {total} chunk vector(s) in ChromaDB (batch size {batch_size})..."
        )
        store = self._recreate_empty_collection(collection_name)
        try:
            self._add_in_batches(store, documents, batch_size)
        except NotFoundError:
            print("  Collection handle went stale mid-write — recreating once and retrying...")
            store = self._recreate_empty_collection(collection_name)
            self._add_in_batches(store, documents, batch_size)

        print(f"✓ ChromaDB persisted to '{self.persist_directory}' ({total} vectors)")
        return store

    def build_faiss_index(
        self, chunks: list[str | dict[str, Any]], save_path: str = "./faiss_index"
    ) -> FAISS:
        """
        Converts text chunks into vector embeddings and indexes them using FAISS.
        """
        documents = self._prepare_documents(chunks)

        print(f"Embedding and indexing {len(documents)} vector(s) using FAISS...")
        vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        vector_store.save_local(save_path)
        print(f"✓ FAISS index saved to '{save_path}'")
        return vector_store

    def _prepare_documents(self, chunks: list[str | dict[str, Any]]) -> list[Document]:
        """
        Converts raw strings or hierarchical chunk dicts into standard
        LangChain Document objects with appropriate metadata.
        """
        documents = []

        for idx, chunk in enumerate(chunks):
            if isinstance(chunk, str):
                # Standard text chunk (token/semantic/recursive)
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"chunk_id": idx, "char_len": len(chunk)},
                    )
                )
            elif isinstance(chunk, dict) and "content" in chunk:
                meta = chunk.get("metadata", {}).copy()
                meta.setdefault("chunk_id", idx)
                meta.setdefault("char_len", len(chunk["content"]))
                documents.append(Document(page_content=chunk["content"], metadata=meta))
            elif isinstance(chunk, dict) and "child_chunks" in chunk:
                # Hierarchical chunking: Index the small child chunk for vector search,
                # but attach the parent chunk text in metadata for retrieval context.
                parent_id = chunk.get("parent_id")
                parent_chunk = chunk.get("parent_chunk")

                for child_idx, child_text in enumerate(chunk.get("child_chunks", [])):
                    documents.append(
                        Document(
                            page_content=child_text,
                            metadata={
                                "parent_id": parent_id,
                                "parent_chunk": parent_chunk,
                                "child_index": child_idx,
                            },
                        )
                    )

        return documents


if __name__ == "__main__":
    # Smoke Test
    test_chunks = [
        "Vector databases allow high-speed similarity search over dense embeddings.",
        "FAISS and ChromaDB are popular choices for local retrieval augmented generation.",
    ]

    vsm = VectorStoreManager()
    chroma_db = vsm.build_chroma_db(test_chunks)

    # Perform similarity search test
    results = chroma_db.similarity_search("How to store vector embeddings?", k=1)
    print(f"\nTop Similarity Result: {results[0].page_content}")
