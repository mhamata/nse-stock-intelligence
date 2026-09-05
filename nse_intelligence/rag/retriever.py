"""ChromaDB access: one persistent collection, metadata-filtered similarity search.

ChromaDB stores three things per chunk: the embedding, the raw text, and a
flat metadata dict. We lean on metadata heavily - the system prompt wants
"only RELIANCE documents", "only financial results", "nothing older than 24h",
and each of those becomes a `where` clause here instead of a post-filter, so
we never waste the top-k budget on chunks we would throw away.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from rag.embedder import get_langchain_embeddings


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.ClientAPI:
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    return Chroma(
        client=get_chroma_client(),
        collection_name=settings.chroma_collection,
        embedding_function=get_langchain_embeddings(),
        # Cosine distance suits normalised text embeddings better than the L2 default.
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_where(symbol: str | None = None, category: str | None = None, max_age_hours: float | None = None,
                source: str | None = None) -> dict | None:
    clauses: list[dict] = []
    if symbol:
        clauses.append({"symbol": symbol.strip().upper()})
    if category:
        clauses.append({"category": category.strip().lower()})
    if source:
        clauses.append({"source": source})
    if max_age_hours:
        clauses.append({"published_ts": {"$gte": int(time.time() - max_age_hours * 3600)}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


@dataclass
class Hit:
    document: Document
    score: float  # cosine *similarity* in [0, 1]; higher is better

    @property
    def citation(self) -> str:
        m = self.document.metadata
        when = m.get("published", "")[:16].replace("T", " ") or "date unknown"
        return f"{m.get('feed_label', m.get('category'))} | {m.get('company') or m.get('title')} | {when} | {m.get('link', '')}"


def search(query: str, k: int = 5, symbol: str | None = None, category: str | None = None,
           max_age_hours: float | None = None, source: str | None = None) -> list[Hit]:
    where = build_where(symbol, category, max_age_hours, source)
    store = get_vector_store()
    results = store.similarity_search_with_relevance_scores(query, k=k, filter=where)
    return [Hit(doc, round(score, 4)) for doc, score in results]


def keyword_filter(hits: list[Hit], keyword: str | None) -> list[Hit]:
    """Cheap lexical post-filter for tool arguments like keyword='dividend'."""
    if not keyword:
        return hits
    needle = keyword.lower()
    return [h for h in hits if needle in h.document.page_content.lower()]


def collection_stats() -> dict:
    collection = get_chroma_client().get_or_create_collection(settings.chroma_collection)
    count = collection.count()
    latest = 0
    if count:
        sample = collection.get(limit=min(count, 2000), include=["metadatas"])
        latest = max((m.get("published_ts", 0) for m in sample["metadatas"]), default=0)
    return {"chunks": count, "latest_published_ts": latest, "persist_dir": str(settings.chroma_persist_dir)}


def latest_documents(category: str | None = None, symbol: str | None = None, limit: int = 20) -> list[Document]:
    """Newest ingested items by metadata, no embedding needed (for UI feeds)."""
    collection = get_chroma_client().get_or_create_collection(settings.chroma_collection)
    where = build_where(symbol, category)
    raw = collection.get(where=where, limit=2000, include=["metadatas", "documents"]) if where else \
        collection.get(limit=2000, include=["metadatas", "documents"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(raw["documents"], raw["metadatas"]) if m.get("chunk_index", 0) == 0]
    docs.sort(key=lambda d: d.metadata.get("published_ts", 0), reverse=True)
    return docs[:limit]
