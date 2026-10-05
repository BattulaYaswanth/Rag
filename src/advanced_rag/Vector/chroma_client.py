"""Single place that decides HOW we talk to Chroma.

Cloud (:class:`chromadb.CloudClient`) when CHROMA_API_KEY is set in .env,
otherwise a local :class:`chromadb.PersistentClient` at VECTOR_DB_DIR.
Every module (ingest, retrieval, health checks) must come through here so
local-vs-cloud can never disagree.
"""

import chromadb

from advanced_rag import config


def get_chroma_client(local_path: str | None = None) -> tuple[object, str]:
    """Return (client, mode) where mode is 'cloud' or 'local'.

    :param local_path: persist directory for local mode (defaults to config).
    :raises RuntimeError: if cloud is requested but tenant/database are missing.
    """
    if config.USE_CHROMA_CLOUD:
        if not config.CHROMA_TENANT or not config.CHROMA_DATABASE:
            raise RuntimeError(
                "CHROMA_API_KEY is set but CHROMA_TENANT/CHROMA_DATABASE are missing. "
                "Add all three to .env (see .env.example)."
            )
        client = chromadb.CloudClient(
            api_key=config.CHROMA_API_KEY,
            tenant=config.CHROMA_TENANT,
            database=config.CHROMA_DATABASE,
        )
        return client, "cloud"
    return chromadb.PersistentClient(path=local_path or config.VECTOR_DB_DIR), "local"


def describe_backend() -> str:
    """Human-readable backend label for logs and /health."""
    if config.USE_CHROMA_CLOUD:
        return f"cloud (database={config.CHROMA_DATABASE})"
    return f"local ({config.VECTOR_DB_DIR})"
