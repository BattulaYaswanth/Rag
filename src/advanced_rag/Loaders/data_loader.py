"""
data_loader.py
Module for loading raw text/data from various input sources.
"""

from pathlib import Path
from typing import Any

import docx
import pypdf
import requests
from bs4 import BeautifulSoup
from sqlalchemy import create_engine, text


class DataLoader:
    """Unified loader interface for multiple data sources."""

    @staticmethod
    def load_pdf(file_path: str | Path, space_width: float = 300.0) -> str:
        """Extract text content from a PDF file."""
        reader = pypdf.PdfReader(file_path)
        extracted_text = []
        for page in reader.pages:
            try:
                text = page.extract_text(space_width=space_width)
            except Exception:
                text = page.extract_text()
            if text:
                extracted_text.append(text)
        return "\n\n".join(extracted_text)

    @staticmethod
    def load_docx(file_path: str | Path) -> str:
        """Extract text content from a Microsoft Word DOCX file."""
        doc = docx.Document(file_path)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n".join(paragraphs)

    @staticmethod
    def load_web(url: str, timeout: int = 10) -> str:
        """Scrape main text body from a web URL."""
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove non-content script and style tags
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()

        return soup.get_text(separator="\n")

    @staticmethod
    def load_api(
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fetch JSON payload from a REST API endpoint."""
        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=params,
            json=payload,
            timeout=10,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def load_database(connection_string: str, query: str) -> list[dict[str, Any]]:
        """
        Query a relational database using SQLAlchemy.

        Example connection_string:
        - SQLite: 'sqlite:///example.db'
        - PostgreSQL: 'postgresql://user:password@localhost:5432/dbname'
        """
        engine = create_engine(connection_string)
        with engine.connect() as connection:
            result = connection.execute(text(query))
            return [dict(row._mapping) for row in result]


if __name__ == "__main__":
    # Example usage smoke test
    print("DataLoader module ready.")
