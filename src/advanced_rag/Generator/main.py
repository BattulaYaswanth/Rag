"""
test_generation.py
End-to-end integration test from Retrieval -> Augmentation -> Generation.
"""

from advanced_rag.Augmentor.augmentor import ContextAugmentor
from advanced_rag.Generator.generator import RAGGenerator
from advanced_rag.Retrieval.retriever import AdvancedRetriever


def run_rag_pipeline(query: str):
    print(f"\n--- Testing RAG Pipeline for Query: '{query}' ---")

    # 1. Retrieval Stage
    print("\n1. Running Retrieval & Reranking...")
    retriever = AdvancedRetriever()
    raw_results = retriever.rerank_search(query, top_k=5)

    # 2. Augmentation Stage
    print("\n2. Augmenting Context & Structuring Prompt...")
    augmentor = ContextAugmentor()
    augmented_payload = augmentor.build_augmented_prompt(
        user_query=query,
        retrieved_chunks=raw_results,
        min_rerank_score=0.1,  # Adjust threshold based on your needs
    )

    print(f"   -> Retained {augmented_payload['context_used_count']} chunks for LLM.")

    # 3. Generation Stage
    print("\n3. Generating Response via LLM (Ollama)...")
    generator = RAGGenerator(model_name="qwen2.5-coder:3b", temperature=0.1)
    answer = generator.generate_response(augmented_payload)

    print("\n" + "=" * 50)
    print("FINAL LLM ANSWER:")
    print("=" * 50)
    print(answer)
    print("=" * 50)


if __name__ == "__main__":
    test_query = "Explain what Java is and what compiler does"
    run_rag_pipeline(test_query)
