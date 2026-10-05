"""
query.py
CLI tool to test retrieval modes against your stored vector database.
"""

import argparse

from advanced_rag.Retrieval.retriever import AdvancedRetriever


def query_pipeline():
    parser = argparse.ArgumentParser(description="Query RAG Context Store")
    parser.add_argument("query", type=str, help="Question/Search query")
    parser.add_argument(
        "--mode",
        choices=["vector", "hybrid", "rerank"],
        default="rerank",
        help="Retrieval mode (default: rerank)",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of chunks to return")

    args = parser.parse_args()
    retriever = AdvancedRetriever()

    print(f"\nExecuting '{args.mode.upper()}' search for: \"{args.query}\"\n")

    # Initialize results to satisfy static type checkers / linters
    results = []

    if args.mode == "vector":
        results = retriever.vector_search(args.query, top_k=args.top_k)
    elif args.mode == "hybrid":
        results = retriever.hybrid_search(args.query, top_k=args.top_k)
    elif args.mode == "rerank":
        results = retriever.rerank_search(args.query, top_k=args.top_k)

    if not results:
        print("⚠️ No results returned from vector store.")
        return

    for idx, item in enumerate(results, start=1):
        print(f"=== Match #{idx} ===")
        if "rerank_score" in item:
            print(f"Rerank Score : {item['rerank_score']:.4f}")
        elif "similarity_score" in item:
            print(f"Distance Score: {item['similarity_score']:.4f}")
        print(f"Metadata     : {item['metadata']}")
        print(f"Content      :\n{item['content']}\n")


if __name__ == "__main__":
    query_pipeline()
