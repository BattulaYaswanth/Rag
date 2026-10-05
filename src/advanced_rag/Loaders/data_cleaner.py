"""
data_cleaner.py
Module for cleaning, normalizing, and deduplicating raw textual data.
"""

import re
import unicodedata

import ftfy
from rapidfuzz import fuzz


class DataCleaner:
    """Methods for normalizing, cleaning noise, and deduplicating text data."""

    @staticmethod
    def fix_encoding(raw_text: str) -> str:
        """Fix broken unicode, mojibake, and encoding anomalies."""
        # ftfy fixes bad encoding such as HTML entities and mojibake.
        return ftfy.fix_text(raw_text)

    @staticmethod
    def fix_pdf_formatting(raw_text: str) -> str:
        """Fix PDF artifacts like broken hyphenation, TOC dot leaders, and running headers."""
        text = raw_text
        # Re-join hyphenated words split across line breaks (e.g. 'synchro-\n nized' -> 'synchronized')
        text = re.sub(r"(\b[a-zA-Z]+)-\s*\n\s*([a-zA-Z]+\b)", r"\1\2", text)
        # Remove Table of Contents dot leaders (e.g. '. . . . . . . . 135')
        text = re.sub(r"\.{2,}", " ", text)
        # Remove common running headers/footers (e.g. '1.2. ASYNCHRONOUS EVENTS 5')
        text = re.sub(
            r"(?m)^\s*(?:\d+\s+CHAPTER\s+\d+.*|CONTENTS\s+[ivxlcdm\d]+.*|\d+\.\d+\..*\d+|\d+\s*)$",
            "",
            text,
        )
        return text

    @staticmethod
    def remove_noise(
        raw_text: str,
        remove_html: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        remove_special_chars: bool = False,
    ) -> str:
        """Remove unwanted noise such as HTML tags, URLs, and specific patterns."""
        text = raw_text

        # Strip remnant HTML tags
        if remove_html:
            text = re.sub(r"<[^>]+>", " ", text)

        # Remove URLs
        if remove_urls:
            text = re.sub(r"https?://\S+|www\.\S+", "", text)

        # Remove Email addresses
        if remove_emails:
            text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "", text)

        # Optional: Keep only alphanumeric characters and standard punctuation
        if remove_special_chars:
            text = re.sub(r"[^\w\s.,!?-]", "", text)

        return text

    @staticmethod
    def normalize_text(
        text: str,
        lowercase: bool = False,
        normalize_unicode: str = "NFKC",
        preserve_paragraphs: bool = True,
    ) -> str:
        """
        Normalize Unicode formats, whitespace, and case.

        normalize_unicode modes: 'NFC', 'NFKC', 'NFD', 'NFKD'
        """
        # Standardize Unicode characters
        text = unicodedata.normalize(normalize_unicode, text)

        if preserve_paragraphs:
            # Standardize horizontal whitespace
            text = re.sub(r"[ \t\r\f\v]+", " ", text)
            # Compress excessive newlines to maximum of 2 (paragraph break)
            text = re.sub(r"\n\s*\n+", "\n\n", text)
            # Remove line breaks that occur mid-sentence
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
            text = text.strip()
        else:
            text = re.sub(r"\s+", " ", text).strip()

        if lowercase:
            text = text.lower()

        return text

    @classmethod
    def clean_document(
        cls,
        raw_text: str,
        lowercase: bool = False,
        preserve_paragraphs: bool = True,
    ) -> str:
        """Full pipeline execution for a single string or document."""
        text = cls.fix_encoding(raw_text)
        text = cls.fix_pdf_formatting(text)
        text = cls.remove_noise(text)
        text = cls.normalize_text(
            text, lowercase=lowercase, preserve_paragraphs=preserve_paragraphs
        )
        return text

    @staticmethod
    def deduplicate_exact(documents: list[str]) -> list[str]:
        """Remove exact duplicate entries while preserving order."""
        seen = set()
        deduped = []
        for doc in documents:
            if doc not in seen:
                seen.add(doc)
                deduped.append(doc)
        return deduped

    @staticmethod
    def deduplicate_fuzzy(documents: list[str], similarity_threshold: float = 85.0) -> list[str]:
        """
        Remove near-duplicate strings using RapidFuzz string distance matching.

        :param documents: List of cleaned text strings
        :param similarity_threshold: Score (0-100) above which two items are duplicates
        """
        unique_docs = []

        for doc in documents:
            is_duplicate = False
            for existing_doc in unique_docs:
                # Compare similarity ratio
                score = fuzz.ratio(doc, existing_doc)
                if score >= similarity_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_docs.append(doc)

        return unique_docs


if __name__ == "__main__":
    # Integration Example
    raw_samples = [
        "   Hello   World! Check out https://example.com &amp; test.  ",
        "Hello World! Check out test.",  # Near duplicate of #1 after noise removal
        "   Hello   World! Check out https://example.com &amp; test.  ",  # Exact duplicate
        "Different content altogether.",
    ]

    cleaner = DataCleaner()

    print("--- 1. Single Document Cleaning ---")
    cleaned_sample = cleaner.clean_document(raw_samples[0])
    print(f"Original : {raw_samples[0]!r}")
    print(f"Cleaned  : {cleaned_sample!r}\n")

    print("--- 2. Batch Processing & Deduplication ---")
    cleaned_batch = [cleaner.clean_document(doc) for doc in raw_samples]

    print(f"Before Deduplication ({len(cleaned_batch)} items):")
    for i, item in enumerate(cleaned_batch):
        print(f"  [{i}] {item}")

    deduped_exact = cleaner.deduplicate_exact(cleaned_batch)
    print(f"\nAfter Exact Deduplication ({len(deduped_exact)} items):")
    for item in deduped_exact:
        print(f"  - {item}")

    deduped_fuzzy = cleaner.deduplicate_fuzzy(deduped_exact, similarity_threshold=85.0)
    print(f"\nAfter Fuzzy Deduplication ({len(deduped_fuzzy)} items):")
    for item in deduped_fuzzy:
        print(f"  - {item}")
