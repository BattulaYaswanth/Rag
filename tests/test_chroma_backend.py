"""Offline unit tests: backend selection (no network calls)."""

from advanced_rag import config
from advanced_rag.Vector import chroma_client
from advanced_rag.Vector.chroma_client import describe_backend, get_chroma_client


def test_local_backend_by_default(monkeypatch):
    monkeypatch.setattr(config, "CHROMA_API_KEY", "")
    monkeypatch.setattr(config, "USE_CHROMA_CLOUD", False)
    monkeypatch.setattr(config, "VECTOR_DB_DIR", "/tmp/opencode-chroma-test")
    _client, mode = get_chroma_client()
    assert mode == "local"
    assert describe_backend().startswith("local")


def test_cloud_missing_tenant_raises(monkeypatch):
    monkeypatch.setattr(config, "CHROMA_API_KEY", "dummy")
    monkeypatch.setattr(config, "USE_CHROMA_CLOUD", True)
    monkeypatch.setattr(config, "CHROMA_TENANT", "")
    monkeypatch.setattr(config, "CHROMA_DATABASE", "")
    try:
        get_chroma_client()
    except RuntimeError as e:
        assert "CHROMA_TENANT" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_chroma_client_module_imports():
    assert callable(chroma_client.get_chroma_client)
