# syntax=docker/dockerfile:1
# Frontend: Vite build -> Node static server with /api proxy to the backend.
# Registry-safe bases (registry.access.redhat.com; no docker.io).
# NOTE: the ubi nodejs image runs as non-root `default`; build steps run as
# root and the runtime drops back to `default`.
#
# Build:  podman build -t advanced-rag-frontend ./frontend
# Run:    podman run -p 3000:3000 -e BACKEND_URL=http://host.containers.internal:8000 advanced-rag-frontend
# Browse: http://localhost:3000

# ---------- Stage 1: build the static bundle ----------
FROM registry.access.redhat.com/ubi9/nodejs-20:latest AS builder

USER 0
WORKDIR /app/build

# Manifest first for layer caching.
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY index.html vite.config.js ./
COPY src/ ./src/
RUN npm run build

# ---------- Stage 2: serve + proxy via Node stdlib ----------
FROM registry.access.redhat.com/ubi9/nodejs-20:latest AS runtime

# BACKEND_URL: RAG API origin (no trailing slash). PORT: this server's port.
ENV BACKEND_URL=http://host.containers.internal:8000 \
    PORT=3000

USER 0
WORKDIR /srv/app
COPY --from=builder /app/build/dist ./dist
COPY server.js ./
RUN chmod -R a+rX /srv/app
USER default

EXPOSE 3000

CMD ["node", "server.js"]
