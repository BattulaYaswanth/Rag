"""
main.py
Complete RAG Pipeline Entrypoint:
1. Load documents recursively.
2. Clean and deduplicate content.
3. Split content into chunks.
4. Convert chunks to embeddings & store in Vector Database (ChromaDB/FAISS).
"""

import argparse
import atexit
import os
import sys
from pathlib import Path
from typing import Any

from advanced_rag.Chunker.text_chunker import TextChunker
from advanced_rag.Loaders.data_cleaner import DataCleaner
from advanced_rag.Loaders.data_loader import DataLoader
from advanced_rag.Vector.vector_store import VectorStoreManager


def _claim_ingest_lock(persist_directory: str) -> Path:
    """Single-writer guard: two concurrent ingests corrupt Chroma's SQLite.

    Exits if another LIVE ingest holds the lock; clears stale locks whose
    PID is dead (e.g. after a crash or server restart).
    """
    lock_path = Path(persist_directory) / ".ingest.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
        except Exception:
            old_pid = -1
        alive = False
        if old_pid > 0:
            try:
                os.kill(old_pid, 0)
                alive = True
            except Exception:
                alive = False
        if alive:
            print(f"ERROR: another ingestion (PID {old_pid}) is already running.")
            print("Running two ingests at once corrupts the index — wait for it to finish.")
            sys.exit(2)
        print(f"Clearing stale ingest lock from dead PID {old_pid}.")
    lock_path.write_text(str(os.getpid()))
    atexit.register(lambda: lock_path.unlink(missing_ok=True))
    return lock_path


def run_pipeline(
    docs_folder: str,
    strategy: str = "recursive",
    vector_db_type: str = "chroma",
    reset_collection: bool = True,
    batch_size: int = 100,
) -> Any:
    target_dir = Path(docs_folder).resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Error: Path '{target_dir}' is invalid or does not exist.")
        sys.exit(1)

    loader = DataLoader()
    cleaner = DataCleaner()
    chunker = TextChunker()
    vector_mgr = VectorStoreManager()

    if vector_db_type == "chroma" and reset_collection:
        import tempfile

        from advanced_rag import config as _cfg
        from advanced_rag.Vector.chroma_client import describe_backend, get_chroma_client

        print(f"Vector backend: {describe_backend()}")
        lock_dir = (
            vector_mgr.persist_directory if not _cfg.USE_CHROMA_CLOUD else tempfile.gettempdir()
        )
        _claim_ingest_lock(lock_dir)
        try:
            client, _ = get_chroma_client(local_path=vector_mgr.persist_directory)
            if "rag_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection("rag_collection")
                print("Reset existing Chroma collection 'rag_collection'")
            else:
                print("No existing 'rag_collection' to reset — building fresh.")
        except Exception as e:
            print(f"Notice during collection reset: {e}")

    # Phase 1: Loading
    print(f"\n=== Phase 1: Ingesting Documents from '{target_dir}' ===")
    extension_map = {".pdf": loader.load_pdf, ".docx": loader.load_docx}
    files = [f for f in target_dir.rglob("*") if f.suffix.lower() in extension_map]

    if not files:
        print("No supported files found.")
        return

    raw_records: list[tuple[Path, str]] = []
    for file_path in files:
        ext = file_path.suffix.lower()
        try:
            print(f"  └─ Loading [{ext.upper()}] {file_path.name}...")
            text_content = extension_map[ext](file_path)
            if text_content and text_content.strip():
                raw_records.append((file_path, text_content))
        except Exception as e:
            print(f"  └─ Error loading {file_path.name}: {e}")

    # Phase 2: Cleaning
    print("\n=== Phase 2: Cleaning & Deduplication ===")
    cleaned_docs = [cleaner.clean_document(txt, lowercase=False) for _, txt in raw_records]
    cleaned_docs = [c for c in cleaned_docs if c]

    exact_deduped = cleaner.deduplicate_exact(cleaned_docs)
    final_docs = cleaner.deduplicate_fuzzy(exact_deduped, similarity_threshold=85.0)
    print(f"Final clean unique documents: {len(final_docs)}")

    # Phase 3: Chunking
    print(f"\n=== Phase 3: Chunking (Strategy: '{strategy}') ===")
    all_chunks = []
    for doc_idx, doc in enumerate(final_docs, start=1):
        before = len(all_chunks)
        if strategy == "token":
            all_chunks.extend(chunker.chunk_by_tokens(doc))
        elif strategy == "semantic":
            all_chunks.extend(chunker.chunk_semantic(doc))
        elif strategy == "hierarchical":
            all_chunks.extend(chunker.chunk_hierarchical(doc))
        elif strategy == "recursive":
            all_chunks.extend(chunker.chunk_recursive(doc))
        made = len(all_chunks) - before
        print(f"  [{doc_idx}/{len(final_docs)}] +{made} chunks (total so far: {len(all_chunks)})")

    print(f"Generated {len(all_chunks)} chunk units.")

    # Phase 4: Embeddings & Vector Storage
    print(f"\n=== Phase 4: Generating Embeddings & Storing in {vector_db_type.upper()} ===")
    if vector_db_type == "chroma":
        vector_store = vector_mgr.build_chroma_db(all_chunks, batch_size=batch_size)
    elif vector_db_type == "faiss":
        vector_store = vector_mgr.build_faiss_index(all_chunks)
    else:
        raise ValueError(f"Unsupported DB type: {vector_db_type}")

    print("\n=== Pipeline Complete! Ready for Similarity Queries ===")
    return vector_store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-End RAG Ingestion Pipeline")
    parser.add_argument("folder", type=str, help="Directory containing documents")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["token", "semantic", "hierarchical", "recursive"],
        default="recursive",
        help="Chunking strategy",
    )
    parser.add_argument(
        "--db",
        type=str,
        choices=["chroma", "faiss"],
        default="chroma",
        help="Vector Store target",
    )

    args = parser.parse_args()
    run_pipeline(docs_folder=args.folder, strategy=args.strategy, vector_db_type=args.db)
