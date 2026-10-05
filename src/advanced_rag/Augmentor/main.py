"""Demo: Retrieval → Augmentation (run with `python -m advanced_rag.Augmentor.main`)."""

from advanced_rag.Augmentor.augmentor import ContextAugmentor
from advanced_rag.Retrieval.retriever import AdvancedRetriever


def main(query: str = "Explain the Java") -> None:
    # 1. Retrieve
    retriever = AdvancedRetriever()
    raw_results = retriever.rerank_search(query, top_k=5)

    # 2. Augment (Filter & Structure Context)
    augmentor = ContextAugmentor()
    augmented_payload = augmentor.build_augmented_prompt(
        user_query=query,
        retrieved_chunks=raw_results,
        min_rerank_score=0.20,  # Filter out weakly relevant chunks
    )

    print(f"System Prompt:\n{augmented_payload['system_prompt']}\n")
    print(f"User Prompt:\n{augmented_payload['user_prompt']}\n")
    print(f"Chunks retained for LLM: {augmented_payload['context_used_count']}")


if __name__ == "__main__":
    main()
