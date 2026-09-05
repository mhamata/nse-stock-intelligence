"""Streamlit chat UI for NSE-GPT.

Layout (per the design doc's Layer 4):
    left  : chat interface + per-answer tool trace ("RAG context panel")
    right : live price widget, stock chart, announcement feed
    sidebar: model/index status, manual refresh, session memory controls

Run:  streamlit run ui/streamlit_app.py
"""
from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from config import settings, configure_logging
from ingestion.feeds import NSE_FEEDS, NEWS_FEEDS
from ingestion.price_fetcher import get_quote, get_history
from ingestion.symbols import detect_symbols
from rag.retriever import collection_stats, latest_documents

st.set_page_config(page_title="NSE Stock Intelligence", page_icon="📈", layout="wide")


# ---------- shared, process-wide resources ----------
@st.cache_resource(show_spinner="Starting local agent (Ollama + MCP server)...")
def agent_runtime():
    configure_logging("streamlit")
    from agent.langchain_agent import get_runtime

    return get_runtime()


@st.cache_resource
def background_scheduler():
    from scheduler.feed_scheduler import start_background_scheduler

    return start_background_scheduler()


@st.cache_data(ttl=60, show_spinner=False)
def cached_quote(symbol: str) -> dict:
    return get_quote(symbol).to_dict()


@st.cache_data(ttl=300, show_spinner=False)
def cached_history(symbol: str, period: str):
    return get_history(symbol, period)


@st.cache_data(ttl=60, show_spinner=False)
def cached_stats() -> dict:
    return collection_stats()


# ---------- session state ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "watch_symbol" not in st.session_state:
    st.session_state.watch_symbol = "RELIANCE"

runtime = agent_runtime()
background_scheduler()


def detect_symbol(text: str) -> str | None:
    """First NSE symbol mentioned in the question, so the price widget follows the chat."""
    found = detect_symbols(text)
    return found[0] if found else None


# ---------- sidebar ----------
with st.sidebar:
    st.title("📈 NSE-GPT")
    st.caption("100% local · Ollama · LangChain · LlamaIndex · MCP · ChromaDB")
    stats = cached_stats()
    latest = datetime.fromtimestamp(stats["latest_published_ts"]).strftime("%d %b %H:%M") if stats["latest_published_ts"] else "-"
    st.metric("Indexed chunks", stats["chunks"], help=f"Newest item: {latest}")
    st.write(f"**Model:** `{settings.ollama_model}`  \n**Embeddings:** `{settings.ollama_embed_model}`  \n**Refresh:** every {settings.rag_refresh_interval_minutes} min")
    if st.button("🔄 Refresh index now", use_container_width=True):
        from rag.indexer import refresh

        with st.spinner("Fetching feeds and embedding new items..."):
            result = refresh()
        cached_stats.clear()
        st.success(f"Added {result['added']} new chunks in {result['seconds']}s")
    st.divider()
    st.session_state.watch_symbol = st.text_input("Watch symbol", st.session_state.watch_symbol).strip().upper() or "RELIANCE"
    chart_period = st.selectbox("Chart period", ["1mo", "3mo", "6mo", "1y"], index=2)
    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        runtime.memory.clear(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()
    with st.expander("Session memory"):
        exchanges = runtime.memory.exchanges(st.session_state.session_id)
        st.caption(f"{len(exchanges)}/{runtime.memory.window} exchanges buffered")
        for i, ex in enumerate(exchanges, 1):
            st.markdown(f"**{i}.** {ex.question[:80]}")
    st.divider()
    st.caption("Informational only, not financial advice.")

# ---------- main layout ----------
chat_col, side_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader("Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tools"):
                with st.expander(f"🔧 {len(message['tools'])} tool call(s) · RAG context"):
                    for event in message["tools"]:
                        st.markdown(f"**{event['name']}** `{event['args']}`")
                        if event["result"]:
                            st.code(event["result"], language="json")

    if prompt := st.chat_input("Ask about any NSE stock, filing, index or announcement..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        if symbol := detect_symbol(prompt):
            st.session_state.watch_symbol = symbol

        with st.chat_message("assistant"):
            with st.spinner("Thinking, calling tools, retrieving context..."):
                reply = runtime.ask(prompt, session_id=st.session_state.session_id)
            content = reply.answer or f"⚠️ {reply.error}"
            st.markdown(content)
            tools = [{"name": e.name, "args": e.args, "result": e.result_preview} for e in reply.tool_events]
            if tools:
                with st.expander(f"🔧 {len(tools)} tool call(s) · RAG context"):
                    for event in tools:
                        st.markdown(f"**{event['name']}** `{event['args']}`")
                        if event["result"]:
                            st.code(event["result"], language="json")
        st.session_state.messages.append({"role": "assistant", "content": content, "tools": tools})
        st.rerun()

with side_col:
    symbol = st.session_state.watch_symbol
    st.subheader(f"Live price · {symbol}")
    quote = cached_quote(symbol)
    if quote.get("error"):
        st.warning(quote["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Last", f"{quote['last_price']:,.2f}", f"{quote['change_pct']:+.2f}%" if quote.get("change_pct") is not None else None)
        c2.metric("Day range", f"{quote['day_low'] or 0:,.0f}–{quote['day_high'] or 0:,.0f}")
        c3.metric("52w range", f"{quote['week52_low'] or 0:,.0f}–{quote['week52_high'] or 0:,.0f}")
        st.caption(f"Source: {quote['source']} · fetched {quote['fetched_at'][11:19]} UTC")

    if not symbol.startswith(("NIFTY", "BANKNIFTY")):
        try:
            history = cached_history(symbol, chart_period)
            if len(history):
                fig = go.Figure(go.Candlestick(x=history.index, open=history["Open"], high=history["High"],
                                               low=history["Low"], close=history["Close"]))
                fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.caption(f"Chart unavailable: {exc}")

    st.subheader("Announcement feed")
    category = st.selectbox("Feed", list(NSE_FEEDS) + list(NEWS_FEEDS), format_func=lambda k: {**NSE_FEEDS, **NEWS_FEEDS}[k].label, label_visibility="collapsed")
    only_symbol = st.checkbox(f"Only {symbol}", value=False)
    for doc in latest_documents(category, symbol if only_symbol else None, limit=8):
        m = doc.metadata
        when = (m.get("published") or "")[:16].replace("T", " ")
        title = f"{m.get('company') or m.get('title')}" + (f" ({m['symbol']})" if m.get("symbol") else "")
        with st.expander(f"{when} · {title[:70]}"):
            st.write(doc.page_content.split("\n", 1)[-1])
            if m.get("link"):
                st.markdown(f"[Source document]({m['link']})")
