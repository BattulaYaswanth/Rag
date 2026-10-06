// Minimal static-file + API-proxy server (Node stdlib only).
// Serves the Vite build in ./dist, falls back to index.html (SPA),
// and proxies /api/* to the RAG backend.
//
// Env:
//   PORT        - listen port (default 3000)
//   BACKEND_URL - RAG API origin, e.g. http://host.containers.internal:8000
"use strict";

const fs = require("node:fs");
const http = require("node:http");
const path = require("node:path");

const PORT = Number(process.env.PORT || "3000");
const BACKEND_URL = (
  process.env.BACKEND_URL || "http://host.containers.internal:8000"
).replace(/\/+$/, "");
const DIST = path.join(__dirname, "dist");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

function serveFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end("Not found");
      return;
    }
    res.writeHead(200, {
      "Content-Type":
        TYPES[path.extname(filePath)] || "application/octet-stream",
    });
    res.end(data);
  });
}

function proxyApi(req, res) {
  const target = new URL(BACKEND_URL + req.url.replace(/^\/api/, "") || "/");
  const options = {
    protocol: target.protocol,
    hostname: target.hostname,
    port: target.port,
    path: target.pathname + target.search,
    method: req.method,
    headers: { ...req.headers, host: target.host },
  };
  const proxy = http.request(options, (backRes) => {
    res.writeHead(backRes.statusCode, backRes.headers);
    backRes.pipe(res);
  });
  proxy.on("error", () => {
    res.writeHead(502, { "Content-Type": "application/json" });
    res.end(
      JSON.stringify({ detail: `Backend unreachable at ${BACKEND_URL}` }),
    );
  });
  proxy.setTimeout(300000);
  req.pipe(proxy);
}

const server = http.createServer((req, res) => {
  const urlPath = (req.url || "/").split("?")[0];
  if (urlPath === "/api" || urlPath.startsWith("/api/"))
    return proxyApi(req, res);

  const filePath = path.normalize(
    path.join(DIST, decodeURIComponent(urlPath.slice(1))),
  );
  if (!filePath.startsWith(DIST)) {
    res.writeHead(403, { "Content-Type": "text/plain" });
    res.end("Forbidden");
    return;
  }
  fs.stat(filePath, (err, stat) => {
    if (!err && stat.isFile()) return serveFile(res, filePath);
    serveFile(res, path.join(DIST, "index.html")); // SPA fallback
  });
});

server.timeout = 0; // LLM answers can take minutes; rely on proxy timeout instead
server.listen(PORT, () =>
  console.log(`frontend on :${PORT}, backend ${BACKEND_URL}`),
);
