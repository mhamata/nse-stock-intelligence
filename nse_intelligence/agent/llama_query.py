"""LlamaIndex CitationQueryEngine over the shared ChromaDB collection.

Why a custom retriever instead of LlamaIndex's ChromaVectorStore? The
collection is written by LangChain, so its metadata isn't in LlamaIndex's
native node format. Wrapping our own `rag.retriever.search` keeps ONE index
with TWO read paths (agent tools and this engine) and zero format coupling.

The CitationQueryEngine numbers each retrieved chunk ([1], [2], ...) and
instructs the LLM to cite those numbers inline - exactly what the design
document asks for in the RAG CONTEXT panel.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from llama_index.core import Settings as LISettings
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from config import settings
from rag.embedder import get_llamaindex_embedding
from rag.retriever import search


class ChromaSearchRetriever(BaseRetriever):
    def __init__(self, k: int = 5, symbol: str | None = None, category: str | None = None,
                 max_age_hours: float | None = None):
        super().__init__()
        self.k, self.symbol, self.category, self.max_age_hours = k, symbol, category, max_age_hours

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        hits = search(query_bundle.query_str, k=self.k, symbol=self.symbol,
                      category=self.category, max_age_hours=self.max_age_hours)
        nodes = []
        for hit in hits:
            m = hit.document.metadata
            node = TextNode(text=hit.document.page_content, id_=m.get("doc_id", ""), metadata={
                "citation": hit.citation, "category": m.get("category"), "symbol": m.get("symbol"),
                "published": m.get("published"), "link": m.get("link"), "title": m.get("title"),
            })
            nodes.append(NodeWithScore(node=node, score=hit.score))
        return nodes


@lru_cache(maxsize=1)
def _llm():
    from llama_index.llms.ollama import Ollama

    return Ollama(model=settings.ollama_model, base_url=settings.ollama_base_url,
                  request_timeout=180.0, context_window=settings.ollama_num_ctx, temperature=0.1)


@dataclass
class CitedAnswer:
    answer: str
    citations: list[dict]


def citation_query(question: str, k: int = 5, symbol: str | None = None,
                   category: str | None = None, max_age_hours: float | None = None) -> CitedAnswer:
    LISettings.llm = _llm()
    LISettings.embed_model = get_llamaindex_embedding()
    engine = CitationQueryEngine(
        retriever=ChromaSearchRetriever(k, symbol, category, max_age_hours),
        llm=_llm(),
        citation_chunk_size=1024,  # our chunks are already small; don't re-split them
    )
    response = engine.query(question)
    citations = [
        {"n": i + 1, "score": round(sn.score or 0, 3), **{key: sn.node.metadata.get(key) for key in ("citation", "title", "published", "link", "symbol", "category")}}
        for i, sn in enumerate(response.source_nodes)
    ]
    return CitedAnswer(str(response), citations)
