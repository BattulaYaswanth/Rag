"""
generator.py
Module for generating grounded responses augmented with structured RAG context.
Provider is env-driven: Ollama for local dev, Groq for prod (LLM_PROVIDER).
Model ids come from .env (OLLAMA_MODEL / GROQ_MODEL).
"""

import os
from typing import Any

from dotenv import load_dotenv

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
        llm_provider: str | None = None,
    ):
        """
        Initializes the LLM generator.

        :param llm_provider: "ollama" (local dev) or "groq" (prod).
            Defaults to LLM_PROVIDER env (or "ollama").
        :param model_name: Model id. Defaults to OLLAMA_MODEL / GROQ_MODEL
            env for the active provider.
        Set temperature=0.0 by default for maximum precision and strict grounding.
        """
        from advanced_rag import config as _config

        provider = (llm_provider or os.getenv("LLM_PROVIDER", _config.LLM_PROVIDER)).lower()
        if provider == "groq":
            from langchain_groq import ChatGroq

            api_key = os.getenv("GROQ_API_KEY", _config.GROQ_API_KEY)
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing. Add it to .env.")
            model = model_name or os.getenv("GROQ_MODEL", _config.GROQ_MODEL)
            self.llm = ChatGroq(model=model, temperature=temperature, api_key=api_key)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            model = model_name or os.getenv("OLLAMA_MODEL", _config.OLLAMA_MODEL)
            self.llm = ChatOllama(model=model, temperature=temperature)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use ollama|groq).")
        self.llm_provider = provider
        self.llm_id = f"{provider}:{model}"
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
