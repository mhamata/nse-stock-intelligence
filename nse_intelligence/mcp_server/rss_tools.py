"""FastMCP server exposing NSE data tools to the LLM agent.

Run with:  python mcp_server/rss_tools.py     (stdio transport)

Every tool returns plain JSON-serialisable data with explicit provenance
(`source`, `fetched_at`, `feed_url`, `link`). The agent's system prompt
forbids uncited numbers, so we make citing easy by never returning a bare
figure without where it came from.

IMPORTANT: never print() to stdout in this process. stdout is the MCP wire.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from mcp.server.fastmcp import FastMCP

from config import configure_logging
from ingestion.feeds import ALL_FEEDS, NSE_FEEDS
from ingestion.rss_loader import load_feed
from ingestion.price_fetcher import get_quote as _get_quote, get_corporate_actions as _corp_actions
from ingestion.symbols import get_universe, INDEX_CSVS
from ingestion.financials import fetch_results_from_xbrl

mcp = FastMCP("nse-intelligence")

DISCLAIMER = "This is informational only, not financial advice."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _doc_summary(doc, max_chars: int = 400) -> dict:
    m = doc.metadata
    return {
        "title": m.get("title"), "company": m.get("company"), "symbol": m.get("symbol") or None,
        "category": m.get("category"), "pub_date": m.get("published") or m.get("pub_date_raw"),
        "summary": doc.page_content.split("\n", 1)[-1][:max_chars],
        "link": m.get("link"), "feed_url": m.get("feed_url"), "doc_id": m.get("doc_id"),
    }


def _filter_by_symbol(docs, symbol: str):
    """Keep items about `symbol` without substring false positives.

    'RELIANCE' must not match 'Reliance Mutual Fund' NAV notices filed by
    another company. We match the resolved symbol exactly, or the company's
    normalised legal name; news items (no symbol) fall back to a whole-word
    match on the headline.
    """
    import re
    from ingestion.symbols import normalise_name

    wanted = symbol.strip().upper()
    universe = get_universe()
    try:
        target_name = normalise_name(universe.master[wanted].name) if universe.is_symbol(wanted) else ""
    except Exception:
        target_name = ""
    kept = []
    for doc in docs:
        m = doc.metadata
        if m.get("symbol") == wanted:
            kept.append(doc)
        elif m.get("source") == "nse":
            if target_name and normalise_name(m.get("company", "")) == target_name:
                kept.append(doc)
        elif re.search(rf"\b{re.escape(wanted)}\b", m.get("title", ""), re.IGNORECASE) or (
                target_name and target_name.split()[0] in normalise_name(m.get("title", ""))):
            kept.append(doc)
    return kept


@mcp.tool()
def list_feeds() -> dict:
    """List every RSS feed category this system can fetch, with its URL and label.
    Call this if you are unsure which category name to pass to fetch_rss."""
    return {k: {"label": f.label, "source": f.source, "url": f.url} for k, f in ALL_FEEDS.items()}


@mcp.tool()
def fetch_rss(category: str = "corporate_announcement", limit: int = 10, symbol: str = "") -> dict:
    """Fetch an NSE or news RSS feed LIVE and return its newest items.

    category: one of corporate_announcement, financial_results, insider_trading,
      corporate_actions, board_meetings, investor_complaints, circulars, buyback,
      news_et_markets, news_moneycontrol, news_business_standard (or a full feed URL).
    symbol: optional NSE symbol (e.g. RELIANCE) to keep only that company's items.
    Use this FIRST for any question about latest announcements, filings or news.
    """
    try:
        docs = load_feed(category)
    except Exception as exc:
        return {"error": f"Feed '{category}' unavailable: {exc}", "fetched_at": _now()}
    if symbol:
        docs = _filter_by_symbol(docs, symbol)
    items = [_doc_summary(d) for d in docs[: max(1, min(limit, 50))]]
    hint = None
    if not items:
        hint = (f"No items in '{category}' matched symbol '{symbol}'. Try fetch_rss with category='corporate_announcement', "
                "search_announcements(symbol=...), or rag_query(question, symbol=...) before concluding there is no news.")
    return {"category": category, "hint": hint, "feed_url": ALL_FEEDS[category].url if category in ALL_FEEDS else category,
            "total_items_in_feed": len(docs), "returned": len(items), "fetched_at": _now(), "items": items}


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """Get the current price snapshot for an NSE stock symbol (RELIANCE, TCS, INFY)
    or index (NIFTY50, BANKNIFTY, NIFTYIT). Returns last price, change %, day range,
    volume, 52-week high/low, the data source and the fetch timestamp.
    If `error` is set, tell the user 'Price unavailable - check NSE directly.'"""
    quote = _get_quote(symbol).to_dict()
    quote["disclaimer"] = DISCLAIMER
    return quote


@mcp.tool()
def list_symbols(index: str = "NIFTY50") -> dict:
    """List constituent symbols of an NSE index. Supported: NIFTY50, NIFTYBANK,
    NIFTYIT, NIFTYNEXT50, NIFTY100, NIFTYFMCG, NIFTYPHARMA, NIFTYAUTO."""
    try:
        companies = get_universe().index_constituents(index)
    except KeyError as exc:
        return {"error": str(exc), "supported": list(INDEX_CSVS)}
    return {"index": index.upper(), "count": len(companies), "fetched_at": _now(),
            "symbols": [{"symbol": c.symbol, "name": c.name, "industry": c.industry} for c in companies]}


@mcp.tool()
def search_announcements(symbol: str = "", keyword: str = "", category: str = "", limit: int = 5) -> dict:
    """Semantic search over the ChromaDB corpus of ingested NSE announcements and
    news. Filter by symbol and/or feed category; keyword is used both as the search
    query and as a lexical filter. Good for 'has X announced a dividend recently?'."""
    from rag.retriever import search, keyword_filter

    query = " ".join(part for part in (symbol, keyword) if part) or "latest corporate announcement"
    hits = search(query, k=max(limit * 3, 10), symbol=symbol or None, category=category or None)
    hits = keyword_filter(hits, keyword)[:limit]
    hint = None if hits else "Nothing indexed matches. Try rag_query with a fuller question, or fetch_rss for live items."
    return {"query": query, "returned": len(hits), "hint": hint, "fetched_at": _now(),
            "items": [{**_doc_summary(h.document), "similarity": h.score} for h in hits]}


@mcp.tool()
def get_corporate_actions(symbol: str) -> dict:
    """Dividends, splits, bonus issues and rights for a symbol, with ex-date and
    record date. Tries NSE's corporate-actions API, then the corporate actions RSS feed."""
    symbol = symbol.strip().upper()
    try:
        rows = _corp_actions(symbol)
        return {"symbol": symbol, "source": "nse_api", "fetched_at": _now(), "count": len(rows), "actions": rows[:20]}
    except Exception as exc:
        logger.warning(f"NSE corporate actions API failed for {symbol}: {exc}")
    feed = fetch_rss("corporate_actions", limit=50, symbol=symbol)
    return {"symbol": symbol, "source": "rss_corporate_actions", "fetched_at": _now(),
            "count": feed.get("returned", 0), "actions": feed.get("items", [])}


@mcp.tool()
def get_financial_results(symbol: str) -> dict:
    """Latest quarterly/annual results (revenue, profit, EPS per period) for a symbol,
    parsed from NSE's XBRL filing. Falls back to related documents from the RAG index."""
    symbol = symbol.strip().upper()
    try:
        docs = [d for d in load_feed("financial_results") if d.metadata.get("symbol") == symbol]
    except Exception as exc:
        docs = []
        logger.warning(f"financial_results feed failed: {exc}")
    for doc in docs:
        link = doc.metadata.get("link", "")
        if link.endswith(".xml"):
            try:
                results = fetch_results_from_xbrl(link).to_dict()
                results.update({"symbol": symbol, "source": "nse_xbrl", "fetched_at": _now(),
                                "filed": doc.metadata.get("published"), "disclaimer": DISCLAIMER})
                return results
            except Exception as exc:
                logger.warning(f"XBRL parse failed for {link}: {exc}")
    rag = search_announcements(symbol=symbol, keyword="results", limit=5)
    return {"symbol": symbol, "source": "rag_fallback", "fetched_at": _now(),
            "note": "No XBRL filing for this symbol in the current results feed; showing related indexed documents.",
            "items": rag["items"]}


@mcp.tool()
def rag_query(question: str, k: int = 5, symbol: str = "", max_age_hours: float = 0) -> dict:
    """Query the LlamaIndex/ChromaDB RAG index over all ingested NSE documents and news.
    Returns the top-k chunks with source citations. Set max_age_hours=24 for
    price-sensitive questions to exclude stale material. Prefer this for historical
    questions; prefer fetch_rss for 'latest' questions."""
    from rag.retriever import search

    hits = search(question, k=max(1, min(k, 10)), symbol=symbol or None, max_age_hours=max_age_hours or None)
    return {"question": question, "returned": len(hits), "fetched_at": _now(),
            "chunks": [{"text": h.document.page_content[:600], "similarity": h.score, "citation": h.citation,
                        "doc_id": h.document.metadata.get("doc_id"), "pub_date": h.document.metadata.get("published")}
                       for h in hits]}


if __name__ == "__main__":
    configure_logging("mcp_server")
    logger.info("Starting NSE MCP server on stdio")
    mcp.run(transport="stdio")
