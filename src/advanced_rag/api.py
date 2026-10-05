"""
api.py — FastAPI service + minimal chat UI for the RAG pipeline (Ollama).

Run:
    uv run uvicorn advanced_rag.api:app --host 0.0.0.0 --port 8000
Then open http://localhost:8000 in a browser.

Endpoints:
    GET  /            → chat UI
    GET  /health      → DB + LLM status
    POST /query       → {"query": ..., "top_k": 4} → answer + sources
"""

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from advanced_rag import config
from advanced_rag.app import RAGPipeline

app = FastAPI(title="Advanced RAG API")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = 4
    min_rerank_score: float | None = None
    model: str | None = None
    evaluate: bool = False


class SourceItem(BaseModel):
    content: str
    metadata: dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    llm: str
    context_used_count: int
    has_relevant_context: bool
    sources: list[SourceItem] = []
    evaluation: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# OpenAI-compatible stub (silences /v1/* probers; routes chat via the RAG pipe)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = []


@app.get("/v1/models")
def openai_models() -> dict[str, Any]:
    """Minimal model list so OpenAI-protocol clients stop 404-spamming."""
    import time as _time

    return {
        "object": "list",
        "data": [
            {
                "id": config.OLLAMA_MODEL,
                "object": "model",
                "created": int(_time.time()),
                "owned_by": "ollama",
            }
        ],
    }


@app.post("/v1/chat/completions")
def openai_chat_completion(req: ChatCompletionRequest) -> dict[str, Any]:
    """Non-streaming chat completion: last user message → RAG answer."""
    import time as _time
    import uuid as _uuid

    query = next(
        (m.content for m in reversed(req.messages) if m.role == "user" and m.content.strip()),
        "",
    )
    if not query:
        raise HTTPException(status_code=400, detail="No user message found.")

    pipe = _get_pipeline(req.model, False)
    result = pipe.ask(query)
    answer = result["answer"]
    model_id = req.model or config.OLLAMA_MODEL
    return {
        "id": f"chatcmpl-{_uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(_time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": max(1, len(query) // 4),
            "completion_tokens": max(1, len(answer) // 4),
            "total_tokens": max(2, (len(query) + len(answer)) // 4),
        },
    }


@lru_cache(maxsize=4)
def _get_pipeline(model: str | None, evaluate: bool) -> RAGPipeline:
    return RAGPipeline(llm_model=model or config.OLLAMA_MODEL, enable_eval=evaluate)


@app.get("/health")
def health() -> dict[str, Any]:
    from advanced_rag.Vector.chroma_client import describe_backend, get_chroma_client

    status: dict[str, Any] = {
        "llm": {"provider": "ollama", "model": config.OLLAMA_MODEL},
    }
    try:
        client, mode = get_chroma_client()
        cols = client.list_collections()
        status["vector_db"] = {
            "mode": mode,
            "backend": describe_backend(),
            "collections": [
                {"name": c.name, "count": client.get_collection(c.name).count()} for c in cols
            ],
        }
    except Exception as e:
        status["vector_db"] = {"error": str(e)}
    return status


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipe = _get_pipeline(req.model, req.evaluate)
    result = pipe.ask(req.query, top_k=req.top_k, min_rerank_score=req.min_rerank_score)
    return QueryResponse(
        query=result["query"],
        answer=result["answer"],
        llm=result["llm"],
        context_used_count=result["context_used_count"],
        has_relevant_context=result["has_relevant_context"],
        sources=result["sources"],
        evaluation=result.get("evaluation"),
    )


@app.get("/", response_class=HTMLResponse)
def chat_ui() -> str:
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Advanced RAG Chat</title>
<style>
body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem}
#log{border:1px solid #ccc;border-radius:8px;padding:1rem;min-height:300px;margin-bottom:1rem}
.q{font-weight:bold;margin-top:1rem}.a{white-space:pre-wrap;background:#f6f6f6;padding:.6rem;border-radius:6px}
.row{display:flex;gap:.5rem}input{flex:1;padding:.6rem}button{padding:.6rem 1rem}
.src{font-size:.8em;color:#555}
</style></head><body>
<h2>Advanced RAG Chat</h2>
<div id="log"></div>
<div class="row"><input id="q" placeholder="Ask about your documents...">
<button onclick="ask()">Ask</button></div>
<script>
async function ask(){
  const q=document.getElementById('q').value; if(!q) return;
  const log=document.getElementById('log');
  log.innerHTML+=`<div class="q">You: ${q}</div>`;
  document.getElementById('q').value='';
  const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({query:q})});
  const j=await r.json();
  log.innerHTML+=`<div class="a">${j.answer}</div>
    <div class="src">via ${j.llm} · ${j.context_used_count} chunks</div>`;
  log.scrollTop=log.scrollHeight;
}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter')ask()});
</script></body></html>"""
