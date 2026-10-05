"""Offline unit tests: context augmentation (no models needed)."""

from advanced_rag.Augmentor.augmentor import ContextAugmentor


def _chunks():
    return [
        {"content": "Java bytecode is portable.", "metadata": {"chunk_id": 1}, "rerank_score": 0.8},
        {"content": "Unrelated filler.", "metadata": {"chunk_id": 2}, "rerank_score": 0.01},
        {"content": "No score here.", "metadata": {"chunk_id": 3}},
    ]


def test_filter_chunks_rerank_threshold():
    aug = ContextAugmentor()
    kept = aug.filter_chunks(_chunks(), min_rerank_score=0.05)
    assert {c["metadata"]["chunk_id"] for c in kept} == {1, 3}  # weak filtered, scoreless kept


def test_filter_chunks_distance_threshold():
    aug = ContextAugmentor()
    vec = [{"content": "a", "metadata": {}, "similarity_score": 0.5}]
    assert aug.filter_chunks(vec, max_distance_score=0.6) == vec
    assert aug.filter_chunks(vec, max_distance_score=0.1) == []


def test_format_context_block_citations_and_empty():
    aug = ContextAugmentor()
    assert aug.format_context_block([]) == "No relevant context found."
    block = aug.format_context_block(_chunks()[:1])
    assert "[Source #1" in block and "Java bytecode is portable." in block


def test_build_augmented_prompt_short_circuits_without_context():
    aug = ContextAugmentor()
    payload = aug.build_augmented_prompt(
        user_query="What is X?",
        retrieved_chunks=[{"content": "junk", "metadata": {}, "rerank_score": 0.0}],
        min_rerank_score=0.5,
    )
    assert payload["has_relevant_context"] is False
    assert payload["context_used_count"] == 0


def test_build_augmented_prompt_grounded_shape():
    aug = ContextAugmentor()
    payload = aug.build_augmented_prompt(
        user_query="What is bytecode?",
        retrieved_chunks=_chunks(),
        min_rerank_score=0.05,
    )
    assert payload["has_relevant_context"] is True
    assert payload["context_used_count"] == 2
    assert "ONLY the facts" in payload["user_prompt"]
