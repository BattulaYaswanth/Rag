"""
app.py — Unified RAG application (Ollama-only generation).

Ties together Retrieval → Augmentation → Generation (→ optional Evaluation)
in one reusable class plus a CLI:

    python -m advanced_rag.app query "What is bytecode?" --model qwen2.5-coder:3b
    python -m advanced_rag.app query "What is bytecode?" --evaluate
    python -m advanced_rag.app ingest ./docs --strategy recursive
"""

import argparse
import sys
from typing import Any

from advanced_rag import config
from advanced_rag.Augmentor.augmentor import ContextAugmentor
from advanced_rag.Evaluation.evaluator import RAGEvaluator
from advanced_rag.Generator.generator import RAGGenerator
from advanced_rag.Retrieval.retriever import AdvancedRetriever


class RAGPipeline:
    """Single entrypoint for the full RAG loop."""

    def __init__(
        self,
        llm_model: str | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        min_rerank_score: float | None = None,
        enable_eval: bool = False,
    ):
        self.top_k = top_k or config.DEFAULT_TOP_K
        self.min_rerank_score = (
            config.DEFAULT_MIN_RERANK_SCORE if min_rerank_score is None else min_rerank_score
        )
        self.retriever = AdvancedRetriever(
            persist_directory=config.VECTOR_DB_DIR,
            collection_name=config.COLLECTION_NAME,
        )
        self.augmentor = ContextAugmentor()
        self.generator = RAGGenerator(
            model_name=llm_model or config.OLLAMA_MODEL,
            temperature=temperature,
        )
        self.enable_eval = enable_eval
        self.evaluator = RAGEvaluator() if enable_eval else None

    def ask(
        self,
        query: str,
        top_k: int | None = None,
        min_rerank_score: float | None = None,
    ) -> dict[str, Any]:
        """Run retrieve → augment → generate (→ evaluate) and return everything."""
        k = top_k or self.top_k
        threshold = self.min_rerank_score if min_rerank_score is None else min_rerank_score

        raw_results = self.retriever.rerank_search(query, top_k=k)
        payload = self.augmentor.build_augmented_prompt(
            user_query=query,
            retrieved_chunks=raw_results,
            min_rerank_score=threshold,
        )
        answer = self.generator.generate_response(payload)

        result: dict[str, Any] = {
            "query": query,
            "answer": answer,
            "llm": self.generator.llm_id,
            "context_used_count": payload.get("context_used_count", 0),
            "has_relevant_context": payload.get("has_relevant_context", False),
            "structured_context": payload.get("structured_context", ""),
            "sources": [
                {"content": c.get("content", ""), "metadata": c.get("metadata", {})}
                for c in raw_results[:k]
            ],
        }
        if self.evaluator is not None:
            result["evaluation"] = self.evaluator.run_full_evaluation(
                query=query,
                context=result["structured_context"],
                response=answer,
            )
        return result

    def ingest(self, docs_folder: str | None = None, strategy: str = "recursive") -> Any:
        """Run the ingestion pipeline into the canonical vector DB."""
        from advanced_rag.main import run_pipeline

        return run_pipeline(
            docs_folder=docs_folder or config.DOCS_DIR,
            strategy=strategy,
            vector_db_type="chroma",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified RAG pipeline (Ollama)")
    sub = parser.add_subparsers(dest="command", required=True)

    q = sub.add_parser("query", help="Ask a question")
    q.add_argument("question", help="Question to ask")
    q.add_argument(
        "--model", default=None, help="Ollama model id (default: OLLAMA_MODEL env or llama3.2)"
    )
    q.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    q.add_argument("--min-score", type=float, default=config.DEFAULT_MIN_RERANK_SCORE)
    q.add_argument("--evaluate", action="store_true", help="Run LLM-as-judge metrics")

    ing = sub.add_parser("ingest", help="Ingest documents into ChromaDB")
    ing.add_argument("folder", nargs="?", default=config.DOCS_DIR)
    ing.add_argument(
        "--strategy",
        choices=["token", "semantic", "hierarchical", "recursive"],
        default="recursive",
    )
    return parser


def main(argv: list | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "ingest":
        pipe = RAGPipeline()
        pipe.ingest(docs_folder=args.folder, strategy=args.strategy)
        return

    pipe = RAGPipeline(
        llm_model=args.model,
        top_k=args.top_k,
        min_rerank_score=args.min_score,
        enable_eval=args.evaluate,
    )
    result = pipe.ask(args.question)
    print(f"\nQ: {result['query']}\n")
    print(f"A ({result['llm']}): {result['answer']}\n")
    print(f"Context chunks used: {result['context_used_count']}")
    for i, src in enumerate(result["sources"], start=1):
        preview = src["content"][:200].replace("\n", " ")
        print(f"  [Source #{i}] {preview}...")
    if "evaluation" in result:
        ev = result["evaluation"]
        print(f"\nEval overall: {ev['overall_score']}")


if __name__ == "__main__":
    # Standard Python idiom: pass sys.argv[1:] directly
    main(sys.argv[1:])
