"""Offline unit tests: BM25 tokenizer + query expansion (no models needed)."""

from advanced_rag.Retrieval.retriever import _STOPWORDS, _TOKEN_RE, AdvancedRetriever


def test_tokenize_splits_punctuation():
    toks = AdvancedRetriever._tokenize("Java language. Platform-independent!")
    assert "java" in toks
    assert "language" in toks  # "language." must match "language"
    assert "platform" in toks and "independent" in toks


def test_tokenize_removes_stopwords():
    toks = AdvancedRetriever._tokenize("What does the Java compiler do?")
    for stop in ("what", "does", "the"):
        assert stop not in toks
    assert "java" in toks and "compiler" in toks


def test_tokenize_drops_single_chars():
    # Letter-spaced PDF artifacts ("J a v a") contribute nothing.
    assert AdvancedRetriever._tokenize("J a v a") == []
    assert _TOKEN_RE is not None and _STOPWORDS


def test_expand_query_acronyms():
    assert "Java Development Kit" in AdvancedRetriever.expand_query("Explain JDK")
    assert "Object-Oriented Programming" in AdvancedRetriever.expand_query("What is oops?")


def test_format_query_prefix():
    r = AdvancedRetriever.__new__(AdvancedRetriever)  # no DB connection
    assert r._format_query("hello") == "search_query: hello"
    assert r._format_query("search_document: hello") == "search_query: hello"
    assert r._clean_doc_text("search_document: hi") == "hi"
