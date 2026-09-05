# NSE Live Stock Intelligence System

A fully local, privacy-first chatbot over National Stock Exchange (India) data.
It ingests NSE's RSS feeds and financial-news feeds every 5 minutes into a
ChromaDB vector store, and answers questions through a ReAct agent whose tools
are served over the Model Context Protocol. No cloud APIs; the LLM and the
embedding model both run in Ollama on your machine.

**Informational only, not financial advice.** See the Legal notice at the end.

## Architecture

```
┌──────────────────────── LAYER 1: DATA INGESTION ────────────────────────┐
│ NSE RSS (8 feeds)   NSE JSON API (indices, corp. actions)   News RSS (3) │
│ yfinance (prices)   NSE XBRL (quarterly results)   NSE CSVs (symbols)    │
│            ▼ mcp_server/rss_tools.py  (FastMCP, stdio, 8 tools)          │
├──────────────────────── LAYER 2: RAG PIPELINE ──────────────────────────┤
│ ingestion/rss_loader.py → LangChain Documents (+ stable IDs, metadata)   │
│ rag/indexer.py → RecursiveCharacterTextSplitter → nomic-embed-text       │
│ rag/retriever.py → ChromaDB (persistent, cosine, metadata filters)       │
├──────────────────────── LAYER 3: LLM REASONING ─────────────────────────┤
│ agent/langchain_agent.py → LangChain create_agent (ReAct) + MCP tools    │
│ agent/llama_query.py → LlamaIndex CitationQueryEngine (same collection)  │
│ agent/memory.py → last 10 exchanges per session                          │
│ Ollama :11434 → gemma4:e4b (chat) · nomic-embed-text (embeddings)        │
├──────────────────────── LAYER 4: UI ────────────────────────────────────┤
│ ui/streamlit_app.py (primary) · ui/gradio_app.py (alternative)           │
│ scheduler/feed_scheduler.py → APScheduler refresh every 5 min            │
└──────────────────────────────────────────────────────────────────────────┘
```

## Quick start

Prerequisites: [Ollama](https://ollama.com), [uv](https://docs.astral.sh/uv/) (or Python 3.12+ with pip).

```bash
ollama pull gemma4:e4b && ollama pull nomic-embed-text  # ~10 GB total; llama3.2 (2 GB) works too
cd nse_intelligence
cp .env.example .env
uv sync                                                 # or: pip install -r requirements.txt
uv run python rag/indexer.py --full                     # build the vector store (~10 s)
uv run streamlit run ui/streamlit_app.py                # open http://localhost:8501
```

Other entry points:

| Command | What it does |
|---|---|
| `uv run python ui/gradio_app.py` | Gradio chat UI on http://localhost:7860 |
| `uv run python scheduler/feed_scheduler.py` | Standalone 5-minute refresh loop (the Streamlit app already runs one in-process) |
| `uv run python mcp_server/rss_tools.py` | MCP server on stdio, for use from any MCP client (Claude Desktop, etc.) |
| `uv run python agent/langchain_agent.py "What's happening with TCS?"` | One-shot agent run in the terminal, with the tool trace |
| `uv run python rag/indexer.py --reset --full` | Drop and rebuild the index |
| `uv run pytest` | Offline unit tests (no network, no Ollama) |

## Configuration (`.env`)

| Key | Default | Notes |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:e4b` | Plans multi-tool calls well. `llama3.2` (3B) is ~10x faster but shallower; auto-context keeps it usable |
| `OLLAMA_NUM_CTX` | `8192` | Context window. Ollama's default is too small for the prompt plus several tool results |
| `AGENT_AUTO_CONTEXT` | `true` | Pre-fetch quote, filings and RAG chunks for symbols named in a question before the agent runs |
| `MAX_ITEMS_PER_FEED` | `300` | Announcements feed alone holds 1000+ items |
| `RAG_REFRESH_INTERVAL_MINUTES` | `5` | |
| `NSE_SESSION_REFRESH_MINUTES` | `15` | Cookie rotation for the NSE JSON API |

## MCP tools

| Tool | Source | Notes |
|---|---|---|
| `fetch_rss(category, limit, symbol)` | live RSS | 8 NSE categories + 3 news feeds |
| `get_quote(symbol)` | NSE `allIndices` for indices; NSE `quote-equity` then yfinance for stocks | NSE blocks `quote-equity` with 403 at the time of writing, so yfinance (15-min delay) does the work. A circuit breaker stops retrying NSE for the refresh window |
| `list_symbols(index)` | NSE index CSVs | NIFTY50, NIFTYBANK, NIFTYIT, NIFTYNEXT50, NIFTY100, FMCG, PHARMA, AUTO |
| `search_announcements(symbol, keyword, category, limit)` | ChromaDB | semantic + lexical filter |
| `get_corporate_actions(symbol)` | NSE API, RSS fallback | dividends, splits, bonus, buybacks |
| `get_financial_results(symbol)` | NSE XBRL | revenue, PBT, net profit, EPS per period |
| `rag_query(question, k, symbol, max_age_hours)` | ChromaDB | top-k chunks with citations |
| `list_feeds()` | registry | |

## Data sources: what changed versus the design document

The RSS URLs in the design document returned 404 when this was built
(September 2026). Verified replacements live in `ingestion/feeds.py`:

| Design doc | Reality |
|---|---|
| `.../RSS/corporate_announcement.xml` | `.../RSS/Online_announcements.xml` |
| `.../RSS/financial_results.xml` | `.../RSS/Financial_Results.xml` |
| `.../RSS/insider_trading.xml` | `.../RSS/Insider_Trading.xml` |
| `.../RSS/corporate_actions.xml` | `.../RSS/Corporate_action.xml` |
| `.../RSS/press_release.xml` | **Gone.** No replacement feed exists |
| `.../RSS/investor_complaints.xml` | `.../RSS/Investor_Complaints.xml` |
| `www.nseindia.com/content/RSS/circulars.xml` | `nsearchives.nseindia.com/content/RSS/Circulars.xml` |
| (not listed) | Added `Board_Meetings.xml`, `Daily_Buyback.xml`, and Economic Times / Moneycontrol / Business Standard market news |

Feeds identify companies by name, not symbol. `ingestion/symbols.py` maps
names to symbols using NSE's public equity master (`EQUITY_L.csv`, ~2,570
companies) and the announcement PDF filename, which starts with the symbol.

## Design notes

- **Idempotent ingestion.** Every feed item gets a SHA-1 ID from its feed,
  title, link, date and summary; chunk IDs are `<doc_id>#<n>`. The indexer asks
  Chroma which IDs already exist and embeds only the rest, so a 5-minute
  refresh on a quiet feed costs milliseconds.
- **One index, two read paths.** LangChain writes the collection. The agent's
  MCP tools and the LlamaIndex `CitationQueryEngine` both read it, the latter
  through a small custom retriever rather than LlamaIndex's own Chroma wrapper,
  so there is no metadata-format coupling.
- **Long-lived MCP session.** `AgentRuntime` runs an asyncio loop on a daemon
  thread and holds one stdio session to the MCP server for the life of the
  process. Streamlit and Gradio call `ask()` synchronously.
- **Auto-context.** Small local models tend to stop after one tool call. When a
  question names a known symbol, the runtime calls `get_quote`, `fetch_rss` and
  `rag_query` first and passes the results with the question. The agent still
  runs its ReAct loop and can call more tools.
- **Version drift.** The design document pins late-2024 versions that no longer
  install together. This project uses LangChain 1.x (`create_agent`), `mcp`
  1.x (FastMCP inside the SDK), LlamaIndex 0.14 and ChromaDB 1.5, pinned in
  `uv.lock` / `requirements.txt`.

## Project layout

```
nse_intelligence/
  config.py                  settings from .env, logging
  ingestion/
    feeds.py                 feed registry (verified URLs)
    http.py                  browser-like headers NSE requires
    rss_loader.py            RSS -> LangChain Documents
    symbols.py               index constituents, name -> symbol, detection
    price_fetcher.py         NSE JSON API + yfinance, circuit breaker
    financials.py            XBRL quarterly results parser
  rag/
    embedder.py              Ollama embeddings (LangChain + LlamaIndex)
    indexer.py               chunk, embed, upsert; CLI
    retriever.py             ChromaDB search with metadata filters
  mcp_server/rss_tools.py    FastMCP server (8 tools)
  agent/
    prompts.py               master system prompt
    memory.py                 conversation buffer
    langchain_agent.py       ReAct agent + MCP client runtime
    llama_query.py           CitationQueryEngine
  scheduler/feed_scheduler.py
  ui/streamlit_app.py, ui/gradio_app.py
  tests/                     offline unit tests
  chroma_db/, logs/          created at runtime
```

## Legal

This system is for informational and educational purposes only. It is not
financial advice or a recommendation to buy or sell any security. NSE data is
fetched from publicly available feeds; for production use obtain proper data
licensing from NSE via authorised vendors and comply with SEBI regulations and
NSE terms of use.
