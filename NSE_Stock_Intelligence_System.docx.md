**NSE LIVE STOCK INTELLIGENCE SYSTEM**

Complete System Design, Architecture & Master Prompt

*LangChain  |  LlamaIndex  |  MCP  |  Ollama  |  NSE RSS  |  ChromaDB  |  Streamlit*

| LangChain v0.3 | LlamaIndex v0.10 | Ollama Local LLM | FastMCP Server |
| :---: | :---: | :---: | :---: |

# **1\. Executive Summary**

This document provides the complete specification for building an NSE Live Stock Intelligence chatbot — a fully local, privacy-first AI application that ingests National Stock Exchange data in real-time, builds a RAG (Retrieval-Augmented Generation) knowledge base, and answers natural language queries through an interactive Streamlit or Gradio interface.

The system is designed to run 100% on-premises using Ollama as the local LLM runtime — no OpenAI, no cloud APIs, no data leaves the machine. Every component is open-source and production-grade.

[https://www.nseindia.com/content/nsccl/rss.xml](https://www.nseindia.com/content/nsccl/rss.xml) 

| 100% Local No cloud LLM | 4 Tech Stacks LangChain+LlamaIndex+MCP+Ollama | Real-time RAG Updates every 5 min | NSE \+ News 7 RSS feed categories |
| :---: | :---: | :---: | :---: |

# **2\. System Architecture**

The system is organized into four layers. Data flows top-to-bottom from ingestion through processing, reasoning, and user interaction. All inter-layer communication is handled via well-defined Python interfaces or the MCP protocol.

| NSE LIVE STOCK INTELLIGENCE SYSTEM — ARCHITECTURE |  |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **LAYER 1: DATA INGESTION** |  |  |  |  |  |
| **NSE RSS Feed(Circulars)** | **NSE JSON API(Prices)** | **CorporateFilings RSS** | **PressReleases RSS** | **FinancialResults RSS** | **News RSS(ET/MC)** |
| **MCP SERVER  —  Model Context Protocol** Tools: fetch\_rss | parse\_feed | get\_quote | list\_symbols | search\_announcements |  |  |  |  |  |
| **LAYER 2: PROCESSING & RAG PIPELINE** |  |  |  |  |  |
| **LangChainData Loader** | **LlamaIndexDocument Parser** | **Chunking& Splitter** | **Embedding(nomic-embed)** | **ChromaDBVector Store** | **RAGRetriever** |
| **LAYER 3: LLM REASONING (Ollama — Local)** |  |  |  |  |  |
| **LangChainAgent** | **LlamaIndexQuery Engine** | **Ollama API:11434** | **llama3.2 /mistral** | **ToolCalling** | **Memory \&Context** |
| **LAYER 4: UI — Streamlit / Gradio Chatbot** |  |  |  |  |  |
| **ChatInterface** | **Live PriceWidget** | **AnnounceFeed** | **RAGContext Panel** | **StockCharts** | **SessionMemory** |

# **3\. NSE Data Sources & RSS Feed Details**

## **3.1  What NSE RSS Feeds Actually Provide**

NSE's official RSS feeds do NOT provide live tick prices — they deliver structured XML for regulatory and corporate events. This is a critical distinction:

| WARNING | NSE RSS \= Corporate events only. For live prices, use NSE's internal JSON endpoints (nseindia.com/api/\*) with session cookie handling, or a licensed broker API like Zerodha Kite Connect. |
| :---: | :---- |

## **3.2  Official NSE RSS Feed URLs**

| Feed | URL |
| :---- | :---- |
| **Corporate Announcements** | https://nsearchives.nseindia.com/content/RSS/corporate\_announcement.xml |
| **Financial Results** | https://nsearchives.nseindia.com/content/RSS/financial\_results.xml |
| **Insider Trading** | https://nsearchives.nseindia.com/content/RSS/insider\_trading.xml |
| **Corporate Actions** | https://nsearchives.nseindia.com/content/RSS/corporate\_actions.xml |
| **Press Releases** | https://nsearchives.nseindia.com/content/RSS/press\_release.xml |
| **Investor Complaints** | https://nsearchives.nseindia.com/content/RSS/investor\_complaints.xml |
| **Circulars** | https://www.nseindia.com/content/RSS/circulars.xml |

## **3.3  Price Data Strategy**

Since RSS does not carry prices, the system uses a two-tier price strategy:

* Primary: NSE internal JSON API — nseindia.com/api/quote-equity?symbol=RELIANCE (requires session cookie rotation via requests.Session)

* Fallback: Yahoo Finance Python library (yfinance) for delayed price data — free, reliable, 15-min delay

* For production algo-trading: integrate Zerodha Kite Connect or Upstox API (requires broker account)

# **4\. Technology Stack**

| Component | Technology | Role in System |
| :---- | :---- | :---- |
| **Data Feed** | **NSE RSS \+ JSON API** | Real-time announcements, corporate filings, price snapshots |
| **MCP Server** | **FastMCP (Python)** | Exposes NSE tools to LLM agent — fetch\_rss, get\_quote, search\_filings |
| **Data Loader** | **LangChain WebBaseLoader** | Fetches and parses RSS XML into LangChain Document objects |
| **Doc Parser** | **LlamaIndex SimpleDirectoryReader** | Structures parsed data, adds metadata (symbol, timestamp, category) |
| **Chunking** | **RecursiveCharacterTextSplitter** | Splits docs into 512-token chunks with 50-token overlap |
| **Embeddings** | **Ollama nomic-embed-text** | Converts text chunks to dense vectors — fully local |
| **Vector DB** | **ChromaDB (persistent)** | Stores and retrieves embeddings by similarity search |
| **LLM Runtime** | **Ollama local API :11434** | Runs llama3.2 / mistral / gemma3 — no cloud dependency |
| **Agent Chain** | **LangChain ReAct Agent** | Decides when to call MCP tools vs. query RAG vs. answer from memory |
| **Query Engine** | **LlamaIndex CitationQueryEngine** | RAG retrieval with source citations per answer |
| **Chat Memory** | **LangChain ConversationBufferMemory** | Maintains multi-turn conversation context |
| **UI Framework** | **Streamlit (primary)** | Chat UI with live panels, price widgets, announcement feeds |
| **Alt UI** | **Gradio ChatInterface** | Alternative lightweight chat frontend |

# **5\. Project File Structure**

| File / Folder | Purpose |
| :---- | :---- |
| nse\_intelligence/ | Root project folder |
|   .env | All credentials & config |
|   requirements.txt | Python dependencies |
|   mcp\_server/rss\_tools.py | MCP server — NSE RSS \+ price tools |
|   ingestion/rss\_loader.py | LangChain RSS data loader |
|   ingestion/price\_fetcher.py | NSE JSON / finance price client |
|   rag/indexer.py | LlamaIndex document indexer |
|   rag/embedder.py | Ollama nomic-embed-text wrapper |
|   rag/retriever.py | ChromaDB similarity search |
|   agent/langchain\_agent.py | ReAct agent with MCP tool binding |
|   agent/llama\_query.py | LlamaIndex CitationQueryEngine |
|   agent/memory.py | Conversation buffer \+ context mgmt |
|   ui/streamlit\_app.py | Primary Streamlit chat UI |
|   ui/gradio\_app.py | Alternative Gradio interface |
|   scheduler/feed\_scheduler.py | APScheduler — refreshes RAG every 5 min |
|   chroma\_db/ | Persistent vector store (auto-created) |
|   logs/ | Ingestion \+ query logs |

# **6\. Master System Prompt**

Copy this prompt as the SYSTEM message for your LLM. It configures the agent's persona, tools, RAG behavior, and output format.

| SYSTEM PROMPT — NSE Stock Intelligence Agent |
| :---- |
| *You are NSE-GPT, a specialized financial intelligence assistant for the National Stock Exchange of India. You operate on live NSE data fetched in real-time via RSS feeds, corporate filing APIs, and price endpoints. You run fully locally using Ollama — all processing is private and on-premises.* **IDENTITY & CAPABILITIES** You are an expert in Indian equity markets, NSE-listed companies, SEBI regulations, F\&O (Futures & Options), currency derivatives, indices (NIFTY 50, NIFTY Bank, NIFTY IT), and corporate events. You speak plainly, cite sources, and never hallucinate financial data. **AVAILABLE TOOLS (via MCP)** fetch\_rss(feed\_url, category) — Fetches and parses an NSE RSS feed. Returns list of items with title, pubDate, description, link. Call this FIRST for any question about announcements, filings, or news. get\_quote(symbol) — Fetches current price snapshot for an NSE symbol (e.g. RELIANCE, TCS, INFY). Returns LTP, change%, volume, 52w high/low. search\_announcements(symbol, keyword) — Searches historical RSS corpus in ChromaDB for announcements mentioning the symbol or keyword. list\_symbols(index) — Returns all symbols in a given index (NIFTY50, NIFTYBANK, NIFTYIT, etc.). get\_corporate\_actions(symbol) — Returns upcoming dividends, splits, bonus issues, rights for a given symbol. rag\_query(question) — Queries the LlamaIndex ChromaDB RAG index over all ingested NSE documents. Returns top-k relevant chunks with source citations. get\_financial\_results(symbol) — Fetches latest quarterly P\&L, EPS, revenue from NSE financial results RSS. **REASONING PROTOCOL (ReAct)** Think: What does the user actually want? Price? News? Filing? Comparison? Analysis? Decide: Which tool(s) will get the freshest, most relevant data? Act: Call tools in logical sequence. Call get\_quote AND fetch\_rss together for holistic stock questions. Observe: Read tool output carefully. If RSS returns no items, say so explicitly — do not fabricate. Synthesize: Combine live data with RAG context. Always cite your sources (feed URL, pubDate, article title). Respond: Give a clear, structured answer in the format specified below. **OUTPUT FORMAT** Always structure your answer as: SUMMARY — 1-3 sentence direct answer to the question LIVE DATA — Price, volume, change% if relevant (from get\_quote) LATEST NEWS — Top 3 announcements from RSS (title, date, one-line summary) RAG CONTEXT — Supporting background from the vector store (cite source chunk) ANALYSIS — Your synthesis, observations, or cautions SOURCES — List all feed URLs, pub dates, and document IDs referenced **HARD RULES — NEVER BREAK THESE** NEVER invent or estimate stock prices. If get\_quote fails, say 'Price unavailable — check NSE directly.' NEVER provide buy/sell recommendations. Always add: 'This is informational only, not financial advice.' NEVER use stale data without disclosing the timestamp. Always show pubDate from RSS items. NEVER skip source citations. Every factual claim must trace to a tool call result. NEVER hallucinate financial results, EPS, or revenue. Only report data returned by tools. If a question is outside NSE/Indian markets scope, say so and suggest the user check NSE website directly. **RAG BEHAVIOR** The RAG index is refreshed every 5 minutes from all active RSS feeds. If a user asks about 'latest news', always call fetch\_rss live rather than relying solely on RAG — the vector store may be minutes behind. When using rag\_query, retrieve top-5 chunks. Discard chunks older than 24 hours for price-sensitive queries. For historical questions (e.g., 'What did TCS announce last quarter?'), RAG is preferred over live RSS. Always show the chunk source (feed category, pub date, document title) alongside RAG-derived information. **CONVERSATION MEMORY** You maintain a conversation buffer of the last 10 exchanges. Use this to: (1) avoid re-fetching data already shown in this session, (2) resolve pronouns like 'it', 'that stock', 'the company' from context, (3) build on prior analysis rather than repeating it. If the buffer window is exhausted, ask the user to clarify the subject. **EXAMPLE INTERACTIONS** **User: What's happening with Reliance today?** *NSE-GPT: \[Calls get\_quote(RELIANCE) \+ fetch\_rss(corporate\_announcement) \+ rag\_query(Reliance latest)\] → Structured response with live price, latest announcements, and background context.* **User: Show me all NIFTY 50 stocks that declared dividends this week.** *NSE-GPT: \[Calls list\_symbols(NIFTY50) then get\_corporate\_actions() for each\] → Table of dividend declarations with ex-date, amount, record date.* **User: Summarize TCS's last quarterly results.** *NSE-GPT: \[Calls get\_financial\_results(TCS) \+ rag\_query(TCS quarterly results)\] → Revenue, PAT, EPS, YoY comparison with source citations.* |

# **7\. Step-by-Step Setup Guide**

## **Step 1 — Install Prerequisites**

* Python 3.11+ and pip

* Ollama — install from ollama.com, then: ollama pull llama3.2 && ollama pull nomic-embed-text

* MySQL 8.x (already set up from previous pipeline)

* ChromaDB will be installed automatically via pip

## **Step 2 — Install Python Packages**

| CMD | pip  install langchain langchain-community langchain-ollama llama-index llama-index-llms-ollama llama-index-embeddings-ollama chromadb streamlit gradio feedparser beautifulsoup4 requests yfinance apscheduler mcp fastmcp python-dotenv pydantic |
| :---: | :---- |

## **Step 3 — Configure .env**

* OLLAMA\_BASE\_URL=http://localhost:11434

* OLLAMA\_MODEL=llama3.2   (or mistral, gemma3:12b)

* OLLAMA\_EMBED\_MODEL=nomic-embed-text

* CHROMA\_PERSIST\_DIR=./chroma\_db

* RAG\_REFRESH\_INTERVAL\_MINUTES=5

* NSE\_SESSION\_REFRESH\_MINUTES=15   (for price API cookie rotation)

## **Step 4 — Start Ollama**

| CMD | ollama serve   \# runs on localhost:11434 |
| :---: | :---- |

## **Step 5 — Start MCP Server**

| CMD | python mcp\_server/rss\_tools.py   \# starts FastMCP on stdio |
| :---: | :---- |

## **Step 6 — Run Initial RAG Indexing**

| CMD | python rag/indexer.py \--full   \# fetches all feeds, embeds, stores in ChromaDB |
| :---: | :---- |

## **Step 7 — Launch Streamlit UI**

| CMD | streamlit run ui/streamlit\_app.py |
| :---: | :---- |

## **Step 8 — Or Launch Gradio UI**

| CMD | python ui/gradio\_app.py |
| :---: | :---- |

# **8\. requirements.txt**

| \# Core LLM frameworks langchain==0.3.7 langchain-community==0.3.7 langchain-ollama==0.2.1 llama-index==0.10.67 llama-index-llms-ollama==0.3.4 llama-index-embeddings-ollama==0.3.1 llama-index-vector-stores-chroma==0.2.1 \# Vector store chromadb==0.5.23 \# MCP mcp==1.2.0 fastmcp==0.4.1 \# Data ingestion feedparser==6.0.11 beautifulsoup4==4.12.3 requests==2.32.3 yfinance==0.2.48 \# UI streamlit==1.40.2 gradio==5.9.1 plotly==5.24.1 \# Scheduling apscheduler==3.10.4 \# Utils python-dotenv==1.0.1 pydantic==2.10.3 pandas==2.2.3 loguru==0.7.3 |
| :---- |

# **9\. Legal & Compliance Notice**

| DISCLAIMER | This system is for informational and educational purposes only. It does not constitute financial advice, investment advice, or a recommendation to buy or sell any security. NSE data is fetched from publicly available RSS feeds. For production trading systems, obtain proper data licensing from NSE via authorized vendors. Always comply with SEBI regulations and NSE terms of use. |
| :---: | :---- |

*Document prepared for: NSE Live Stock Intelligence System*