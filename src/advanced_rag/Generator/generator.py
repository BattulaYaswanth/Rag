"""
generator.py
Module for generating grounded responses using a local Ollama LLM
augmented with structured RAG context.
"""

import os
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


DEFAULT_STRICT_SYSTEM_PROMPT = (
    "You are a strict retrieval-grounded assistant. Answer the user query using ONLY the facts explicitly stated in the provided context.\n\n"
    "CRITICAL GROUNDING RULES:\n"
    "1. Answer concisely and factually based strictly on the provided context.\n"
    "2. Do NOT use outside knowledge, metaphors, analogies, or speculate.\n"
    "3. If the provided context is insufficient or missing to answer the question, state explicitly:\n"
    '   "I cannot answer this question based on the provided context."'
)


class RAGGenerator:
    """Handles sending augmented prompts to the LLM and returning grounded responses."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float = 0.0,
        default_system_prompt: str | None = None,
    ):
        """
        Initializes the LLM generator (local Ollama, no API key needed).

        :param model_name: Ollama model id. Defaults to OLLAMA_MODEL env (or qwen2.5-coder:3b).
        Set temperature=0.0 by default for maximum precision and strict grounding.
        """
        model = model_name or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
        self.llm = ChatOllama(model=model, temperature=temperature)
        self.llm_id = f"ollama:{model}"
        self.default_system_prompt = default_system_prompt or DEFAULT_STRICT_SYSTEM_PROMPT

    def generate_response(self, augmented_payload: dict[str, Any]) -> str:
        """
        Generates an LLM response from an augmented payload dictionary.

        Expected keys in augmented_payload:
        - 'system_prompt' (optional): Custom system instructions.
        - 'user_prompt' or 'query' (required): The user question or formatted prompt.
        - 'context' (optional): Raw context string if prompts aren't pre-assembled.
        - 'has_relevant_context' (optional): Boolean indicating if relevant context was found.
        """
        # Fast deterministic fallback if no relevant context survived filtering
        if not augmented_payload.get("has_relevant_context", True):
            return "I cannot answer this question based on the provided context."
        system_prompt = augmented_payload.get("system_prompt") or self.default_system_prompt

        # Retrieve user prompt, falling back to query or context wrapping if necessary
        user_prompt = augmented_payload.get("user_prompt")
        if not user_prompt:
            query = augmented_payload.get("query", "")
            context = augmented_payload.get("context", "")
            user_prompt = (
                f"RETRIEVED CONTEXT:\n{context}\n\nUSER QUERY: {query}" if context else query
            )

        messages = []

        # Only append system instruction if content is a valid non-empty string
        if isinstance(system_prompt, str) and system_prompt.strip():
            messages.append(("system", system_prompt.strip()))

        messages.append(("human", user_prompt.strip()))

        response = self.llm.invoke(messages)
        return str(response.content)
