"""check_db.py — inspect the Chroma collection used by this project."""

import argparse

from advanced_rag.Vector.chroma_client import describe_backend, get_chroma_client


def inspect() -> None:
    client, _mode = get_chroma_client()
    print(f"Backend: {describe_backend()}")
    collections = client.list_collections()
    print("Collections found:", [c.name for c in collections])
    for c in collections:
        col = client.get_collection(c.name)
        n = col.count()
        print(f"  - {c.name}: {n} documents")
        if n > 0:
            sample = col.get(limit=1)
            print("    Sample Metadata:", sample["metadatas"])
            doc = (sample["documents"] or [""])[0]
            print("    Sample Chunk:", doc[:200])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect Chroma vector DB")
    parser.parse_args()
    inspect()
