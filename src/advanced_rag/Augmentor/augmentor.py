"""
augmentor.py
Module for Context Preparation & Augmentation:
- Filtering retrieved chunks based on thresholds
- Combining and structuring context with citations/metadata
- Building the final LLM prompt payload
"""

from typing import Any


class ContextAugmentor:
    """Handles context filtering, structuring, and LLM prompt assembly."""

    def __init__(self, default_max_context_length: int | None = None):
        self.default_max_context_length = default_max_context_length

    # -------------------------------------------------------------------------
    # 1. Filtering Low-Quality / Low-Score Chunks
    # -------------------------------------------------------------------------
    def filter_chunks(
        self,
        retrieved_chunks: list[dict[str, Any]],
        min_rerank_score: float = 0.0,
        max_distance_score: float = 0.6,
    ) -> list[dict[str, Any]]:
        """
        Filters out retrieved chunks that do not satisfy quality or similarity thresholds.

        - For Rerank mode: higher rerank_score is better (filters out < min_rerank_score).
        - For Vector mode: lower distance_score is better (filters out > max_distance_score).
        """
        filtered = []
        for chunk in retrieved_chunks:
            # Rerank score filtering (Cross-Encoder logits/probabilities)
            if "rerank_score" in chunk:
                if chunk["rerank_score"] >= min_rerank_score:
                    filtered.append(chunk)
            # Vector similarity distance filtering
            elif "similarity_score" in chunk:
                if chunk["similarity_score"] <= max_distance_score:
                    filtered.append(chunk)
            else:
                # If no score exists, retain chunk by default
                filtered.append(chunk)

        return filtered

    # -------------------------------------------------------------------------
    # 2. Structuring and Combining Context with Citations
    # -------------------------------------------------------------------------
    def format_context_block(
        self,
        chunks: list[dict[str, Any]],
        include_metadata: bool = True,
    ) -> str:
        """
        Formats retrieved chunks into a clean, structured context string
        with source citations for the LLM.
        """
        if not chunks:
            return "No relevant context found."

        context_blocks = []
        for idx, item in enumerate(chunks, start=1):
            content = item.get("content", "").strip()
            metadata = item.get("metadata", {})
            chunk_id = metadata.get("chunk_id", f"ref_{idx}")

            # Build metadata header/citation tag
            if include_metadata:
                header = f"[Source #{idx} | Chunk ID: {chunk_id}]"
                formatted_chunk = f"{header}\n{content}"
            else:
                formatted_chunk = content

            context_blocks.append(formatted_chunk)

        # Join formatted chunks with distinct block separators
        combined_context = "\n\n---\n\n".join(context_blocks)

        # Optional length guard rail
        if (
            self.default_max_context_length
            and len(combined_context) > self.default_max_context_length
        ):
            combined_context = (
                combined_context[: self.default_max_context_length] + "\n...[Truncated]"
            )

        return combined_context

    # -------------------------------------------------------------------------
    # 3. Assembling Final LLM Prompt
    # -------------------------------------------------------------------------
    def build_augmented_prompt(
        self,
        user_query: str,
        retrieved_chunks: list[dict[str, Any]],
        system_instruction: str | None = None,
        min_rerank_score: float = 0.20,
    ) -> dict[str, Any]:
        # Filter out low-scoring chunks
        valid_chunks = self.filter_chunks(retrieved_chunks, min_rerank_score=min_rerank_score)

        # Short-circuit if no relevant chunks survived filtering
        if not valid_chunks:
            return {
                "system_prompt": system_instruction or "",
                "user_prompt": f"User Question: {user_query}",
                "structured_context": "No relevant context found.",
                "context_used_count": 0,
                "has_relevant_context": False,
            }

        structured_context = self.format_context_block(valid_chunks)

        user_prompt = (
            f"Context Information:\n"
            f"====================\n"
            f"{structured_context}\n"
            f"====================\n\n"
            f"User Question: {user_query}\n\n"
            f"Instructions:\n"
            f"- Answer the user question using ONLY the facts explicitly stated in the Context Information above.\n"
            f"- Cite your sources using [Source #X] when appropriate.\n"
            f"- Do NOT use external analogies or outside knowledge not present in the context.\n"
            f"- If the context does not fully answer the question, state what is missing rather than inventing facts.\n\n"
            f"Answer:"
        )

        return {
            "system_prompt": system_instruction,
            "user_prompt": user_prompt,
            "structured_context": structured_context,
            "context_used_count": len(valid_chunks),
            "has_relevant_context": True,
        }
