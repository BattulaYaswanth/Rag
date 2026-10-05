"""
run_batch.py
Runs batch evaluation with SEPARATE reporting tracks:

- IN-SCOPE queries → RAG triad means (claim-level faithfulness, context
  relevance, answer relevance). Each query swings the mean by ±1/n.
- FALLBACK queries (expect_fallback / "Out of Scope" category) → fallback
  accuracy (correct refusals / total). NEVER averaged with in-scope quality.
"""

import argparse
import json
from pathlib import Path

from advanced_rag.Augmentor.augmentor import ContextAugmentor
from advanced_rag.Evaluation.evaluator import RAGEvaluator
from advanced_rag.Generator.generator import RAGGenerator
from advanced_rag.Retrieval.retriever import AdvancedRetriever


def is_fallback_test(test: dict) -> bool:
    """Explicit flag wins; otherwise infer from the category name."""
    if "expect_fallback" in test:
        return bool(test["expect_fallback"])
    return test.get("category", "").lower().startswith("out of scope")


def run_batch_test():
    eval_dir = Path(__file__).resolve().parent
    queries_file = eval_dir / "test_queries.json"
    if not queries_file.exists():
        # Fallback to local working directory
        queries_file = Path("test_queries.json")
    if not queries_file.exists():
        print(f"Error: test_queries.json not found at {queries_file}.")
        return

    with open(queries_file, encoding="utf-8") as f:
        test_cases = json.load(f)

    args = argparse.ArgumentParser(description="Batch RAG evaluation")
    args.add_argument(
        "--ids",
        default=None,
        help="Comma-separated query ids to run, e.g. --ids 2,3,4 (default: all)",
    )
    only_ids = args.parse_args().ids
    if only_ids:
        only_ids = set(only_ids.split(","))
        test_cases = [t for t in test_cases if str(t["id"]) in only_ids]
        print(f"Running subset: ids={sorted(only_ids)}")

    log_path = eval_dir / "evaluation_logs.jsonl"
    retriever = AdvancedRetriever()
    augmentor = ContextAugmentor()
    generator = RAGGenerator(temperature=0.0)
    evaluator = RAGEvaluator(log_file_path=str(log_path))

    in_scope_rows = []
    fallback_rows = []

    print(f"\n--- Starting Evaluation for {len(test_cases)} Queries ---\n")

    for test in test_cases:
        query_id = test["id"]
        category = test["category"]
        query = test["query"]
        expect_fallback = is_fallback_test(test)

        print(f"[{query_id}/{len(test_cases)}] Testing: '{query}' ({category})")

        # 1. Retrieve (top_k=4 with reranking)
        raw_results = retriever.rerank_search(query, top_k=4)

        # 2. Augment with confidence threshold
        from advanced_rag import config as _config

        payload = augmentor.build_augmented_prompt(
            user_query=query,
            retrieved_chunks=raw_results,
            min_rerank_score=_config.DEFAULT_MIN_RERANK_SCORE,
        )

        # 3. Generate grounded response
        answer = generator.generate_response(payload)

        # 4. Evaluate grounded context vs response
        eval_context = payload.get("structured_context") or payload["user_prompt"]
        eval_res = evaluator.run_full_evaluation(query=query, context=eval_context, response=answer)

        # Log trace
        evaluator.log_interaction(
            query=query,
            retrieved_context=eval_context,
            generated_response=answer,
            eval_metrics={**eval_res, "expect_fallback": expect_fallback},
        )

        row = {
            "id": query_id,
            "category": category,
            "query": query,
            "track": eval_res["track"],
            "overall_score": eval_res["overall_score"],
            "faithfulness": eval_res["faithfulness"]["score"],
            "context_relevance": eval_res["context_relevance"]["score"],
            "answer_relevance": eval_res["answer_relevance"]["score"],
            "claims": f"{eval_res['faithfulness'].get('supported_count', '?')}"
            f"/{eval_res['faithfulness'].get('total_count', '?')}",
        }
        if eval_res["track"] == "fallback":
            row["fallback_correct"] = eval_res["fallback_correct"]
            fallback_rows.append(row)
            print(f"   └─ FALLBACK track: correct={eval_res['fallback_correct']}")
        else:
            in_scope_rows.append(row)
            print(
                f"   └─ overall={eval_res['overall_score']:.2f} | "
                f"faith={row['faithfulness']:.2f} ({row['claims']} claims) | "
                f"crel={row['context_relevance']:.2f} | arel={row['answer_relevance']:.2f}"
            )

    # ---------------- Summary: tracks reported SEPARATELY ----------------
    print("\n" + "=" * 62)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 62)

    if in_scope_rows:
        n = len(in_scope_rows)

        def _mean(key: str) -> float:
            return sum(r[key] for r in in_scope_rows) / n

        print(f"IN-SCOPE QUALITY (n={n}; each query swings a mean by ±{1 / n:.2f})")
        print(f"  Average Faithfulness (claim-level) : {_mean('faithfulness'):.2f} / 1.00")
        print(f"  Average Context Relevance          : {_mean('context_relevance'):.2f} / 1.00")
        print(f"  Average Answer Relevance           : {_mean('answer_relevance'):.2f} / 1.00")
        print(f"  Average Overall RAG Score          : {_mean('overall_score'):.2f} / 1.00")

    if fallback_rows:
        n = len(fallback_rows)
        correct = sum(1 for r in fallback_rows if r.get("fallback_correct"))
        print(f"FALLBACK BEHAVIOR (n={n}, NOT averaged with quality above)")
        print(f"  Fallback accuracy (correct refusals): {correct}/{n} = {correct / n:.2f}")

    print("\nPer-query breakdown:")
    for r in in_scope_rows + fallback_rows:
        tag = "FALLBACK" if r["track"] == "fallback" else f"overall={r['overall_score']:.2f}"
        print(
            f"  [{r['id']}] {tag} | faith={r['faithfulness']:.2f} ({r['claims']}) | "
            f"crel={r['context_relevance']:.2f} | arel={r['answer_relevance']:.2f} | {r['query'][:55]}"
        )

    print(f"Full logs saved to: {log_path}")
    print("=" * 62)


if __name__ == "__main__":
    run_batch_test()
