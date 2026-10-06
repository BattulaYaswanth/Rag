const BASE = ""; // same-origin; nginx proxies /api/* to the backend

async function getJSON(path, options) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

export function fetchHealth(signal) {
  return getJSON("/api/health", { signal });
}

export function postQuery({ query, top_k, evaluate }) {
  return getJSON("/api/query", {
    method: "POST",
    body: JSON.stringify({ query, top_k, evaluate: !!evaluate }),
  });
}
