"""Ingest RSS feeds -> chunk -> embed -> ChromaDB.

Run `python rag/indexer.py --full` once to build the index, then let the
scheduler call `refresh()` every 5 minutes. Both paths go through
`index_documents`, which is idempotent: chunk IDs derive from the stable
document ID, so anything Chroma already holds is skipped without touching
Ollama. On a quiet 5-minute window that makes a refresh nearly free.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # allow `python rag/indexer.py`

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from config import settings, configure_logging
from ingestion.feeds import ALL_FEEDS
from ingestion.rss_loader import load_all_feeds
from rag.retriever import get_vector_store, get_chroma_client

EMBED_BATCH = 64


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size_chars,
        chunk_overlap=settings.chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(docs: list[Document]) -> tuple[list[Document], list[str]]:
    """Split docs into chunks and give each chunk an ID of the form <doc_id>#<n>."""
    splitter = _splitter()
    chunks, ids = [], []
    seen: set[str] = set()
    for doc in docs:
        # Defensive: a feed may still repeat an item verbatim; Chroma refuses duplicate IDs in one call.
        if doc.metadata["doc_id"] in seen:
            continue
        seen.add(doc.metadata["doc_id"])
        pieces = splitter.split_text(doc.page_content) or [doc.page_content]
        for index, text in enumerate(pieces):
            metadata = {**doc.metadata, "chunk_index": index, "chunk_total": len(pieces)}
            chunks.append(Document(page_content=text, metadata=metadata))
            ids.append(f"{doc.metadata['doc_id']}#{index}")
    return chunks, ids


def _existing_ids(candidate_ids: list[str]) -> set[str]:
    collection = get_chroma_client().get_or_create_collection(settings.chroma_collection)
    found: set[str] = set()
    for start in range(0, len(candidate_ids), 500):
        batch = candidate_ids[start:start + 500]
        found.update(collection.get(ids=batch, include=[])["ids"])
    return found


def index_documents(docs: list[Document]) -> dict:
    """Embed and upsert only chunks Chroma doesn't already hold. Returns counts."""
    started = time.time()
    chunks, ids = chunk_documents(docs)
    already = _existing_ids(ids)
    new_pairs = [(c, i) for c, i in zip(chunks, ids) if i not in already]
    if not new_pairs:
        logger.info(f"Index up to date ({len(ids)} chunks already present)")
        return {"documents": len(docs), "chunks": len(ids), "added": 0, "seconds": round(time.time() - started, 1)}

    store = get_vector_store()
    for start in range(0, len(new_pairs), EMBED_BATCH):
        batch = new_pairs[start:start + EMBED_BATCH]
        store.add_documents([c for c, _ in batch], ids=[i for _, i in batch])
        logger.info(f"Embedded {min(start + EMBED_BATCH, len(new_pairs))}/{len(new_pairs)} new chunks")
    return {"documents": len(docs), "chunks": len(ids), "added": len(new_pairs), "seconds": round(time.time() - started, 1)}


def refresh(categories: list[str] | None = None, limit_per_feed: int | None = None) -> dict:
    """Fetch feeds and index whatever is new. Safe to call as often as you like."""
    docs = load_all_feeds(categories, limit_per_feed or settings.max_items_per_feed)
    result = index_documents(docs)
    logger.info(f"Refresh complete: {result}")
    return result


def reset_index() -> None:
    client = get_chroma_client()
    try:
        client.delete_collection(settings.chroma_collection)
        logger.warning(f"Deleted collection {settings.chroma_collection}")
    except Exception:
        pass
    get_vector_store.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or refresh the NSE RAG index")
    parser.add_argument("--full", action="store_true", help="ingest every feed up to MAX_ITEMS_PER_FEED")
    parser.add_argument("--reset", action="store_true", help="drop the existing collection first")
    parser.add_argument("--feeds", nargs="*", choices=list(ALL_FEEDS), help="limit to these feed categories")
    parser.add_argument("--limit", type=int, help="items per feed (default: 50, or MAX_ITEMS_PER_FEED with --full)")
    args = parser.parse_args()

    configure_logging("indexer")
    if args.reset:
        reset_index()
    limit = args.limit or (settings.max_items_per_feed if args.full else 50)
    result = refresh(args.feeds, limit)
    print(result)


if __name__ == "__main__":
    main()
