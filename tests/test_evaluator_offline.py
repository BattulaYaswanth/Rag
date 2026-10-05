"""Offline unit tests: evaluator helpers (no LLM calls)."""

from advanced_rag.Evaluation.evaluator import (
    NO_CONTEXT_MARKER,
    RAGEvaluator,
    _quote_supported,
    is_refusal,
)


def test_is_refusal():
    assert is_refusal("I cannot answer this question based on the provided context.")
    assert not is_refusal("Java bytecode is portable across platforms.")


def test_quote_supported_exact_and_truncated():
    ctx = "The designers of Java chose a combination of compilation and interpretation."
    assert _quote_supported("combination of compilation and interpretation", ctx)
    # Judge-truncated quote still verifies (fuzzy longest-match cover).
    assert _quote_supported(
        "The designers of Java chose a combination of compilation and interpretati", ctx
    )


def test_quote_supported_rejects_fabrication():
    ctx = "Java bytecode runs on the JVM."
    assert not _quote_supported("Java was invented by aliens in 1995", ctx)
    assert not _quote_supported("", ctx)


def test_faithfulness_valid_refusal_needs_no_llm():
    ev = RAGEvaluator.__new__(RAGEvaluator)  # skip ChatOllama init
    out = ev.evaluate_faithfulness(
        NO_CONTEXT_MARKER, "I cannot answer this question based on the provided context."
    )
    assert out["score"] == 1.0 and out["total_count"] == 0


def test_context_relevance_empty_context_needs_no_llm():
    ev = RAGEvaluator.__new__(RAGEvaluator)
    out = ev.evaluate_context_relevance("What is X?", NO_CONTEXT_MARKER)
    assert out["score"] == 0.0 and out["quote_verified"] is False
