"""
chunker.py (backwards-compatibility shim)

The canonical implementation lives in
:mod:`advanced_rag.Chunker.text_chunker`. This module re-exports it so
older imports (``from advanced_rag.Chunker.chunker import TextChunker``)
keep working, including the legacy ``chunk_semantically`` method name.
"""

from advanced_rag.Chunker.text_chunker import TextChunker

# Legacy alias: older code called `chunk_semantically(...)`, the canonical
# method is now `chunk_semantic(...)`.
if not hasattr(TextChunker, "chunk_semantically"):

    def _chunk_semantically(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return self.chunk_semantic(*args, **kwargs)

    TextChunker.chunk_semantically = _chunk_semantically  # type: ignore[attr-defined]

__all__ = ["TextChunker"]
