"""LangChain data loader for NSE and news RSS feeds.

Responsibilities:
  1. Download and parse a feed (feedparser handles RSS/Atom quirks).
  2. Normalise every item into a LangChain `Document` with consistent metadata
     so the RAG layer, the MCP tools and the UI all speak the same shape.
  3. Give every item a deterministic ID so re-ingesting a feed every 5 minutes
     is idempotent - ChromaDB upserts, it never duplicates.
"""
from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Iterable

import feedparser
from langchain_core.documents import Document
from loguru import logger

from ingestion.feeds import Feed, ALL_FEEDS, NSE_FEEDS, resolve_feed
from ingestion.http import fetch_bytes

IST = timezone(timedelta(hours=5, minutes=30))

# NSE mixes several date layouts across feeds; try them in order.
_DATE_FORMATS = (
    "%d-%b-%Y %H:%M:%S",   # 05-Sep-2026 21:19:51
    "%d-%b-%Y %H:%M",      # 02-May-2026 16:46
    "%d-%b-%Y",            # 08-Sep-2026
    "%a, %d %b %Y %H:%M:%S %z",  # Fri, 4 Sep 2026 00:00:00 +0530 (RFC 822)
)


def parse_pub_date(raw: str | None, parsed_struct=None) -> datetime | None:
    """Return an aware datetime (IST) or None. Never raises."""
    if raw:
        for fmt in _DATE_FORMATS:
            try:
                dt = datetime.strptime(raw.strip(), fmt)
                return dt if dt.tzinfo else dt.replace(tzinfo=IST)
            except ValueError:
                continue
    if parsed_struct:  # feedparser already parsed it (news feeds usually)
        try:
            return datetime(*parsed_struct[:6], tzinfo=timezone.utc).astimezone(IST)
        except Exception:
            return None
    return None


def parse_pipe_fields(summary: str) -> dict[str, str]:
    """NSE summaries look like 'SERIES:EQ |PURPOSE:DIVIDEND ... |RECORD DATE:08-Sep-2026'.

    Turn them into {'series': 'EQ', 'purpose': '...', 'record_date': '...'}.
    Free-text summaries (no pipes/colons) come back as an empty dict.
    """
    fields: dict[str, str] = {}
    for part in summary.split("|"):
        if ":" not in part:
            continue
        key, _, value = part.partition(":")
        key = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")
        if key and value.strip():
            fields[key] = value.strip()
    return fields


def symbol_from_link(link: str) -> str | None:
    """Announcement PDFs are named '<SYMBOL>_<timestamp>_....pdf'."""
    match = re.search(r"/corporate/([A-Z0-9&\-]+)_\d{8,}", link or "")
    return match.group(1) if match else None


def stable_id(feed: Feed, title: str, link: str, published: str, summary: str = "") -> str:
    """Deterministic ID so re-ingestion upserts instead of duplicating.

    The summary is part of the hash because some feeds (insider trading,
    investor complaints) publish several items with identical title, link and
    timestamp that differ only in their summary text.
    """
    digest = hashlib.sha1(f"{feed.category}|{title}|{link}|{published}|{summary}".encode()).hexdigest()
    return f"{feed.category}:{digest[:16]}"


def entry_to_document(feed: Feed, entry, resolve_symbol=None) -> Document:
    title = (entry.get("title") or "").strip()
    link = (entry.get("link") or "").strip()
    summary_raw = (entry.get("summary") or "").strip()
    published_raw = entry.get("published") or entry.get("updated") or ""
    published = parse_pub_date(published_raw, entry.get("published_parsed"))
    fields = parse_pipe_fields(summary_raw) if feed.source == "nse" else {}

    symbol = symbol_from_link(link)
    company = title
    if feed.source == "nse":
        # Corporate-action titles embed the ex-date: "Foo Ltd - Ex-Date: 08-Sep-2026".
        company = re.split(r"\s+-\s+Ex-Date", title)[0].strip()
        if symbol is None and resolve_symbol is not None:
            symbol = resolve_symbol(company)

    # The page_content is what gets embedded. We prepend the feed label and
    # company so the vector captures *what kind* of document this is, which
    # markedly improves retrieval for queries like "TCS dividend".
    lines = [f"[{feed.label}] {company}"]
    if symbol:
        lines[0] += f" ({symbol})"
    if summary_raw:
        lines.append(summary_raw.replace(" |", "\n"))
    if published:
        lines.append(f"Published: {published.strftime('%d-%b-%Y %H:%M')} IST")

    metadata = {
        "doc_id": stable_id(feed, title, link, published_raw, summary_raw),
        "category": feed.category,
        "feed_label": feed.label,
        "source": feed.source,
        "feed_url": feed.url,
        "title": title,
        "company": company,
        "symbol": symbol or "",
        "link": link,
        "published": published.isoformat() if published else "",
        "published_ts": int(published.timestamp()) if published else 0,
        "pub_date_raw": published_raw,
    }
    # Chroma metadata must be flat scalars; flatten the parsed fields with a prefix.
    for key, value in fields.items():
        metadata[f"f_{key}"] = value[:500]
    return Document(page_content="\n".join(lines), metadata=metadata)


def load_feed(category_or_url: str, limit: int | None = None) -> list[Document]:
    """Fetch one feed and return its items as LangChain Documents, newest first."""
    feed = resolve_feed(category_or_url)
    parsed = feedparser.parse(fetch_bytes(feed.url))
    if parsed.bozo and not parsed.entries:
        raise RuntimeError(f"Feed {feed.url} could not be parsed: {parsed.bozo_exception}")

    resolver = None
    if feed.source == "nse":
        try:
            from ingestion.symbols import get_universe
            resolver = get_universe().resolve
        except Exception as exc:
            logger.warning(f"Symbol universe unavailable, skipping name resolution: {exc}")

    docs = [entry_to_document(feed, e, resolver) for e in parsed.entries]
    docs.sort(key=lambda d: d.metadata["published_ts"], reverse=True)
    if limit:
        docs = docs[:limit]
    logger.info(f"Loaded {len(docs)} items from {feed.category}")
    return docs


def load_all_feeds(categories: Iterable[str] | None = None, limit_per_feed: int | None = None) -> list[Document]:
    """Load several feeds, skipping any that fail so one outage doesn't block ingestion."""
    docs: list[Document] = []
    for category in (categories or ALL_FEEDS):
        try:
            docs.extend(load_feed(category, limit_per_feed))
        except Exception as exc:
            logger.error(f"Feed {category} failed: {exc}")
    return docs


def is_nse_category(category: str) -> bool:
    return category in NSE_FEEDS
