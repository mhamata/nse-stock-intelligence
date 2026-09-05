"""Ollama embedding factories.

Both frameworks in this project need an embedding object: LangChain writes
vectors into ChromaDB during ingestion, LlamaIndex reads them back for the
citation query engine. They MUST use the same model (nomic-embed-text) or the
query vector would live in a different space from the stored vectors and
similarity search would return noise. Building both here guarantees that.
"""
from __future__ import annotations

from functools import lru_cache

from config import settings


@lru_cache(maxsize=1)
def get_langchain_embeddings():
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=settings.ollama_embed_model, base_url=settings.ollama_base_url)


@lru_cache(maxsize=1)
def get_llamaindex_embedding():
    from llama_index.embeddings.ollama import OllamaEmbedding

    return OllamaEmbedding(model_name=settings.ollama_embed_model, base_url=settings.ollama_base_url)


def embed_query(text: str) -> list[float]:
    return get_langchain_embeddings().embed_query(text)
