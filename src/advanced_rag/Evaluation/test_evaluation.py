"""
test_evaluation.py
Tests claim-level evaluation metrics and feedback logging.
"""

from advanced_rag.Evaluation.evaluator import RAGEvaluator


def test_eval_pipeline():
    evaluator = RAGEvaluator()

    # Sample inputs
    query = "What does the Java compiler do?"
    context = "[Source #1] The Java compiler translates source code (.java) into bytecode (.class)."
    response = "The Java compiler translates source code into bytecode [Source #1]."

    print("\n--- Running Evaluation Metrics ---")
    metrics = evaluator.run_full_evaluation(query, context, response)

    print(f"Track          : {metrics['track']}")
    print(f"Overall Score  : {metrics['overall_score']}")
    faith = metrics["faithfulness"]
    print(
        f"Faithfulness   : {faith['score']} ({faith.get('supported_count')}/{faith.get('total_count')} claims)"
    )
    for claim in faith.get("claims", []):
        status = "SUPPORTED" if claim["verdict"] == "supported" else "UNSUPPORTED"
        print(f"  [{status} | quote_verified={claim['quote_verified']}] {claim['claim']}")
    print(f"Context Rel.   : {metrics['context_relevance']}")
    print(f"Answer Rel.    : {metrics['answer_relevance']}")

    print("\n--- Logging Feedback ---")
    evaluator.log_interaction(
        query=query,
        retrieved_context=context,
        generated_response=response,
        eval_metrics=metrics,
        user_feedback="thumbs_up",
        user_correction=None,
    )
    print("Log recorded in 'evaluation_logs.jsonl'")


if __name__ == "__main__":
    test_eval_pipeline()
