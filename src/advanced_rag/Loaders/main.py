"""
main.py
CLI entrypoint pipeline that scans a specified folder for supported documents,
routes each file based on its extension, and cleans/deduplicates the extracted text.
"""

import argparse
import sys
from pathlib import Path

from advanced_rag.Loaders.data_cleaner import DataCleaner
from advanced_rag.Loaders.data_loader import DataLoader


def process_docs_folder(folder_path: str) -> list[str]:
    """
    Scans a directory for supported document extensions, executes specific
    loaders, and cleans/deduplicates all gathered text.
    """
    target_dir = Path(folder_path).resolve()

    if not target_dir.exists():
        print(f"Error: Directory '{target_dir}' does not exist.")
        sys.exit(1)

    if not target_dir.is_dir():
        print(f"Error: Path '{target_dir}' is not a directory.")
        sys.exit(1)

    loader = DataLoader()
    cleaner = DataCleaner()

    raw_records: list[tuple[Path, str]] = []

    print(f"=== Step 1: Scanning '{target_dir}' for Supported Files ===")

    # Define file extension router mappings
    extension_map = {
        ".pdf": loader.load_pdf,
        ".docx": loader.load_docx,
    }

    # Gather all supported files recursively
    files_to_process = [
        f for f in target_dir.rglob("*") if f.is_file() and f.suffix.lower() in extension_map
    ]

    if not files_to_process:
        print("No supported files (.pdf, .docx) found in the specified directory.")
        return []

    print(f"Found {len(files_to_process)} document(s) to process.\n")

    # Step 1: Load documents based on file extension
    for file_path in files_to_process:
        ext = file_path.suffix.lower()
        load_func = extension_map.get(ext)

        try:
            print(f"Loading [{ext.upper()}] {file_path.name}...")
            text_content = load_func(file_path)
            if text_content.strip():
                raw_records.append((file_path, text_content))
            else:
                print(f"  └─ Warning: No text extracted from {file_path.name}")
        except Exception as e:
            print(f"  └─ Error reading {file_path.name}: {e}")

    print(f"\nSuccessfully loaded content from {len(raw_records)} file(s).")

    # Step 2: Cleaning and Normalization
    print("\n=== Step 2: Cleaning and Normalizing ===")
    cleaned_documents = []

    for _path, raw_text in raw_records:
        cleaned_text = cleaner.clean_document(raw_text, lowercase=True)
        if cleaned_text:
            cleaned_documents.append(cleaned_text)

    print(f"Cleaned {len(cleaned_documents)} non-empty document string(s).")

    # Step 3: Deduplication
    print("\n=== Step 3: Deduplication ===")
    exact_deduped = cleaner.deduplicate_exact(cleaned_documents)
    print(f"Count after Exact Deduplication: {len(exact_deduped)}")

    final_dataset = cleaner.deduplicate_fuzzy(exact_deduped, similarity_threshold=85.0)
    print(f"Count after Fuzzy Deduplication: {len(final_dataset)}")

    print("\n=== Sample Processed Output ===")
    for idx, doc in enumerate(final_dataset, start=1):
        preview = doc[:120] + "..." if len(doc) > 120 else doc
        print(f"[{idx}] {preview}")

    return final_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest, clean, and deduplicate documents from a target folder."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to the directory containing documents (e.g., ./docs or /path/to/documents)",
    )

    args = parser.parse_args()
    process_docs_folder(args.folder)
