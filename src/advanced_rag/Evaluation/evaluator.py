"""
evaluator.py
Module for RAG Evaluation & Feedback (claim-level judging):

- Faithfulness: response is split into atomic claims; EACH claim is verified
  against the context with a required verbatim quote. Quotes are validated
  programmatically — fabricated evidence flips the verdict to unsupported.
- Context relevance: judge must quote the most relevant span (also validated).
- Tracks: "in_scope" queries get the RAG triad mean; "fallback" queries
  (no context retrieved) are scored on fallback_correct and NEVER averaged
  with in-scope quality metrics.
- Explicit user feedback logging (thumbs up/down, corrections).
"""

import json
import os
import re
from datetime import UTC, datetime
from typing import Any

REFUSAL_PHRASES = (
    "cannot answer",
    "not enough information",
    "no relevant context",
    "based on the provided context",
    "not contain sufficient information",
    "does not provide",
    "do not have enough information",
)

NO_CONTEXT_MARKER = "No relevant context found."


def is_refusal(text: str) -> bool:
    """Heuristic: does the response decline to answer from context?"""
    lowered = text.lower()
    return any(phrase in lowered for phrase in REFUSAL_PHRASES)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _quote_supported(quote: str, context: str, threshold: float = 0.8) -> bool:
    """Is the quoted evidence really in the context?

    Exact substring match first; otherwise fuzzy longest-match cover so that
    judge-truncated quotes ("interpretati..." for "interpretation...") still
    verify, while fabricated quotes fail.
    """
    import difflib

    nq, nc = _normalize(quote), _normalize(context)
    if not nq:
        return False
    if nq in nc:
        return True
    matcher = difflib.SequenceMatcher(None, nq, nc, autojunk=False)
    covered = sum(block.size for block in matcher.get_matching_blocks())
    return covered / max(len(nq), 1) >= threshold


class RAGEvaluator:
    """Evaluates pipeline outputs using LLM-as-a-judge and records feedback."""

    def __init__(
        self,
        eval_model_name: str | None = None,
        log_file_path: str = "evaluation_logs.jsonl",
        llm_provider: str | None = None,
    ):
        """LLM-as-a-judge. Provider follows LLM_PROVIDER env (ollama|groq)."""
        from advanced_rag import config as _config

        provider = (llm_provider or os.getenv("LLM_PROVIDER", _config.LLM_PROVIDER)).lower()
        if provider == "groq":
            from langchain_groq import ChatGroq

            api_key = os.getenv("GROQ_API_KEY", _config.GROQ_API_KEY)
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing. Add it to .env.")
            model = eval_model_name or os.getenv("GROQ_MODEL", _config.GROQ_MODEL)
            self.llm = ChatGroq(model=model, temperature=0.0, api_key=api_key)
        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            model = eval_model_name or os.getenv("OLLAMA_MODEL", _config.OLLAMA_MODEL)
            self.llm = ChatOllama(model=model, temperature=0.0)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER '{provider}' (use ollama|groq).")
        self.log_file_path = log_file_path

    # -------------------------------------------------------------------------
    # 1. Claim-level faithfulness (RAGAS-style)
    # -------------------------------------------------------------------------
    def _extract_claims(self, response: str) -> dict[str, Any]:
        """Split a response into atomic, verifiable factual claims.

        The model's own "refusal" flag is NOT trusted (small judges flip it
        arbitrarily); refusal is decided in code via is_refusal(). If the
        model returns no claims for a substantive response, retry once, then
        fall back to naive sentence splitting — a long factual answer must
        never score a free 1.0 from zero extracted claims.
        """
        if is_refusal(response):
            return {"refusal": True, "claims": [], "extraction_fallback": False}

        prompt = (
            "List every factual statement asserted in the RESPONSE below. "
            "Copy each fact as its own item. Ignore citation tags like [Source #1]. "
            'Always set "refusal" to false.\n\n'
            f"RESPONSE:\n{response}\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"refusal": false, "claims": ["fact 1", "fact 2"]}'
        )
        for attempt in range(2):
            parsed = self._run_json_evaluation(
                prompt if attempt == 0 else prompt + "\nReturn ONLY the JSON object."
            )
            claims = parsed.get("claims", [])
            if not isinstance(claims, list):
                claims = []
            claims = [str(c).strip() for c in claims if str(c).strip()]
            if claims:
                return {"refusal": False, "claims": claims, "extraction_fallback": False}

        # Fallback: naive sentence split so substantive answers are still checked.
        cleaned = re.sub(r"\[Source #\d+[^\]]*\]", " ", response)
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if len(s.strip()) >= 15]
        return {"refusal": False, "claims": sentences, "extraction_fallback": True}

    def evaluate_faithfulness(self, context: str, response: str) -> dict[str, Any]:
        """
        Claim-level faithfulness: every atomic claim must be supported by the
        context, with a verbatim quote as evidence. Fabricated quotes are
        caught programmatically and flip the verdict to unsupported.
        """
        # Valid refusal with no invented facts → perfectly faithful.
        if is_refusal(response) and (NO_CONTEXT_MARKER in context or not context.strip()):
            return {
                "score": 1.0,
                "reason": "Valid refusal: no facts asserted, nothing to hallucinate.",
                "claims": [],
                "supported_count": 0,
                "total_count": 0,
            }

        extracted = self._extract_claims(response)
        claims: list[str] = extracted["claims"]
        if not claims:
            return {
                "score": 1.0,
                "reason": "No verifiable factual claims asserted; nothing to hallucinate.",
                "claims": [],
                "supported_count": 0,
                "total_count": 0,
            }

        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
        prompt = (
            "You are a strict fact-checker. For EACH numbered claim, decide whether it is "
            "directly supported by the CONTEXT. A claim is SUPPORTED only if the context states "
            "the same fact (paraphrase allowed, but no new details). Copy the exact supporting "
            'sentence from the CONTEXT verbatim into "quote". If no sentence supports the claim, '
            'use verdict "unsupported" and an empty quote.\n\n'
            f"CONTEXT:\n{context}\n\n"
            f"CLAIMS:\n{numbered}\n\n"
            "Respond with ONLY a JSON object, no other text:\n"
            '{"judgments": [{"claim_id": 1, "verdict": "supported", "quote": "exact sentence"}, '
            '{"claim_id": 2, "verdict": "unsupported", "quote": ""}]}'
        )
        parsed = self._run_json_evaluation(prompt)
        judgments = parsed.get("judgments", [])
        if not isinstance(judgments, list):
            judgments = []

        results = []
        for idx, claim in enumerate(claims, start=1):
            match = next(
                (j for j in judgments if isinstance(j, dict) and j.get("claim_id") == idx),
                None,
            )
            verdict = (match or {}).get("verdict", "unsupported")
            quote = str((match or {}).get("quote", "")).strip()
            # Hard validation: quoted evidence must actually occur in the context
            # (fuzzy: tolerates judge-truncated quotes, catches fabrications).
            quote_verified = _quote_supported(quote, context)
            if verdict == "supported" and not quote_verified:
                verdict = "unsupported"  # judge fabricated its evidence
            results.append(
                {
                    "claim": claim,
                    "verdict": verdict,
                    "quote": quote,
                    "quote_verified": quote_verified,
                }
            )

        supported = sum(1 for r in results if r["verdict"] == "supported")
        total = len(results)
        score = round(supported / total, 2) if total else 1.0
        return {
            "score": score,
            "reason": f"{supported}/{total} claims supported by context.",
            "claims": results,
            "supported_count": supported,
            "total_count": total,
        }

    # -------------------------------------------------------------------------
    # 2. Context relevance (quote-evidenced)
    # -------------------------------------------------------------------------
    def evaluate_context_relevance(self, query: str, context: str) -> dict[str, Any]:
        """Is the retrieved context relevant? Judge must cite a verbatim span."""
        if NO_CONTEXT_MARKER in context or not context.strip():
            return {
                "score": 0.0,
                "reason": "No relevant context was retrieved for this query.",
                "quote": "",
                "quote_verified": False,
            }

        prompt = (
            f"Question: {query}\n"
            f"Context: {context}\n\n"
            "Does the Context contain information that helps answer the Question? "
            "Reply with ONLY a JSON object, no other text, using keys "
            '"score" (1.0 clearly helpful, 0.5 partly helpful, 0.0 irrelevant), '
            '"reason" (explain in your own words, never copy these instructions), '
            '"quote" (copy the single most helpful sentence from the Context word '
            "for word, or an empty string if none is helpful)."
        )
        parsed = self._run_json_evaluation(prompt)
        score = float(parsed.get("score", 0.0))
        quote = str(parsed.get("quote", "")).strip()
        quote_verified = _quote_supported(quote, context)
        if score > 0 and not quote_verified:
            # Judge claims relevance but cannot point at real evidence → discount.
            score = round(score * 0.5, 2)
            reason = (
                parsed.get("reason", "") + " [discounted: quoted span not found in context.]"
            ).strip()
        else:
            reason = str(parsed.get("reason", ""))
        return {
            "score": min(max(score, 0.0), 1.0),
            "reason": reason,
            "quote": quote,
            "quote_verified": quote_verified,
        }

    def evaluate_answer_relevance(self, query: str, context: str, response: str) -> dict[str, Any]:
        """Does the response address the query (or correctly refuse)?"""
        prompt = (
            f"Question: {query}\n"
            f"Context: {context}\n"
            f"Answer: {response}\n\n"
            "Does the Answer address the Question? Reply with ONLY a JSON object, "
            "no other text, using keys "
            '"score" (1.0 if it answers directly, or correctly says the Context '
            "lacks the information without inventing facts; 0.5 if partial; "
            '0.0 if off-topic), "reason" (explain in your own words, never copy '
            "these instructions)."
        )
        parsed = self._run_json_evaluation(prompt)
        score = min(max(float(parsed.get("score", 0.0)), 0.0), 1.0)
        result = {"score": score, "reason": str(parsed.get("reason", ""))}

        # Guardrail: a valid refusal on empty context must never score 0.
        if is_refusal(response) and (NO_CONTEXT_MARKER in context or not context.strip()):
            result["score"] = 1.0
            result["reason"] = "Response correctly refused when context was insufficient."
        return result

    def run_full_evaluation(self, query: str, context: str, response: str) -> dict[str, Any]:
        """
        Runs quality metrics on a separate track per query type:

        - "in_scope": normal RAG triad mean (faithfulness + context + answer) / 3.
        - "fallback": no context was retrieved; scored ONLY on fallback_correct
          (did the model refuse instead of hallucinating?). Never averaged with
          in-scope quality — aggregate separately in run_batch.py.
        """
        no_context = NO_CONTEXT_MARKER in context or not context.strip()
        faithfulness = self.evaluate_faithfulness(context, response)

        if no_context:
            fallback_correct = bool(is_refusal(response)) and faithfulness["score"] >= 0.8
            # Empty-context non-refusal with claims scores via claim check above.
            answer_rel = self.evaluate_answer_relevance(query, context, response)
            return {
                "track": "fallback",
                "overall_score": 1.0 if fallback_correct else 0.0,
                "fallback_correct": fallback_correct,
                "faithfulness": faithfulness,
                "context_relevance": {
                    "score": 0.0,
                    "reason": "No context retrieved (fallback track).",
                    "quote": "",
                    "quote_verified": False,
                },
                "answer_relevance": answer_rel,
            }

        context_rel = self.evaluate_context_relevance(query, context)
        answer_rel = self.evaluate_answer_relevance(query, context, response)

        # Guardrail: refusing is CORRECT when the retrieved context is junk —
        # e.g. Q4 retrieved primitive-types for a bytecode question. A refusal
        # grounded in irrelevant context must not score 0 on answer relevance.
        if is_refusal(response) and context_rel["score"] < 0.3:
            answer_rel = {
                "score": 1.0,
                "reason": "Correct refusal: retrieved context was irrelevant/insufficient.",
            }

        overall_score = round(
            (faithfulness["score"] + context_rel["score"] + answer_rel["score"]) / 3.0,
            2,
        )
        return {
            "track": "in_scope",
            "overall_score": overall_score,
            "fallback_correct": None,
            "faithfulness": faithfulness,
            "context_relevance": context_rel,
            "answer_relevance": answer_rel,
        }

    # -------------------------------------------------------------------------
    # 3. Track User Feedback & Log Traces
    # -------------------------------------------------------------------------
    def log_interaction(
        self,
        query: str,
        retrieved_context: str,
        generated_response: str,
        eval_metrics: dict[str, Any] | None = None,
        user_feedback: str | None = None,  # "thumbs_up", "thumbs_down"
        user_correction: str | None = None,
    ) -> dict[str, Any]:
        """Logs query trace, metrics, and explicit user feedback for model improvement."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "query": query,
            "retrieved_context": retrieved_context,
            "generated_response": generated_response,
            "evaluation_metrics": eval_metrics or {},
            "user_feedback": user_feedback,
            "user_correction": user_correction,
        }

        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return record

    def _run_json_evaluation(self, prompt: str) -> dict[str, Any]:
        """Executes prompt against LLM and parses JSON output with multi-tier fallbacks."""
        try:
            res = self.llm.invoke(prompt).content.strip()

            # 1. Direct JSON parse
            try:
                parsed = json.loads(res)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                pass

            # 2. Extract from markdown code fences
            fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", res)
            if fence_match:
                try:
                    parsed = json.loads(fence_match.group(1).strip())
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    pass

            # 3. Extract outermost { ... }
            start = res.find("{")
            end = res.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    parsed = json.loads(res[start : end + 1])
                    return parsed if isinstance(parsed, dict) else {}
                except Exception:
                    pass

            return {}
        except Exception as e:
            return {"_error": f"Evaluation invocation error: {e!s}"}
