import { useCallback, useEffect, useRef, useState } from "react";
import { fetchHealth, postQuery } from "./api.js";

function HealthBadge({ health }) {
  if (health === undefined)
    return <span className="badge idle">checking…</span>;
  if (health === null)
    return <span className="badge down">backend unreachable</span>;
  const db = health.vector_db || {};
  const col = (db.collections || [])[0];
  return (
    <span className="badge up" title={JSON.stringify(health)}>
      {health.llm?.provider}:{health.llm?.model} ·{" "}
      {col ? `${col.count} chunks` : "no index"}
    </span>
  );
}

function Sources({ sources }) {
  const [open, setOpen] = useState(false);
  if (!sources?.length) return null;
  return (
    <div className="sources">
      <button className="link" onClick={() => setOpen((v) => !v)}>
        {open ? "▾" : "▸"} {sources.length} source
        {sources.length > 1 ? "s" : ""}
      </button>
      {open &&
        sources.map((s, i) => (
          <blockquote key={i}>
            <div className="meta">
              #{i + 1}
              {s.metadata?.chunk_id !== undefined &&
                ` · chunk ${s.metadata.chunk_id}`}
            </div>
            {s.content}
          </blockquote>
        ))}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState(undefined);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const logRef = useRef(null);

  useEffect(() => {
    const ctrl = new AbortController();
    fetchHealth(ctrl.signal)
      .then(setHealth)
      .catch(() => setHealth(null));
    return () => ctrl.abort();
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo({
      top: logRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, busy]);

  const ask = useCallback(async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setError("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setBusy(true);
    try {
      const res = await postQuery({ query: q });
      setMessages((m) => [...m, { role: "assistant", ...res }]);
    } catch (e) {
      setError(`Query failed: ${e.message}. Is the backend running?`);
    } finally {
      setBusy(false);
    }
  }, [input, busy]);

  return (
    <div className="page">
      <header>
        <div>
          <h1>Advanced RAG Chat</h1>
          <p className="sub">Grounded answers from your document index</p>
        </div>
        <HealthBadge health={health} />
      </header>

      <div className="log" ref={logRef}>
        {messages.length === 0 && (
          <p className="hint">
            Ask anything about your documents, e.g. “What is bytecode in Java?”
          </p>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div className="q" key={i}>
              {m.text}
            </div>
          ) : (
            <div className="a-wrap" key={i}>
              <div className="a">{m.answer}</div>
              <div className="src">
                via {m.llm} · {m.context_used_count} chunk
                {m.context_used_count === 1 ? "" : "s"}
                {m.evaluation && ` · eval ${m.evaluation.overall_score}`}
              </div>
              <Sources sources={m.sources} />
            </div>
          ),
        )}
        {busy && <div className="a thinking">Thinking…</div>}
      </div>

      {error && <div className="error">{error}</div>}

      <div className="row">
        <input
          value={input}
          placeholder="Ask about your documents…"
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          disabled={busy}
        />
        <button onClick={ask} disabled={busy || !input.trim()}>
          Ask
        </button>
      </div>
    </div>
  );
}
