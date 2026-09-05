"""Gradio alternative front-end. Run:  python ui/gradio_app.py

Lighter than the Streamlit app: a chat panel with the agent's tool trace
folded into each answer, plus a quick quote box.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr

from config import configure_logging
from ingestion.price_fetcher import get_quote


def respond(message: str, history: list, request: gr.Request) -> str:
    """Gradio's request carries a per-browser-tab session hash; that becomes our memory key."""
    from agent.langchain_agent import get_runtime

    session_id = getattr(request, "session_hash", None) or "gradio-default"
    reply = get_runtime().ask(message, session_id=session_id)
    if reply.error:
        return f"⚠️ {reply.error}"
    trace = "\n".join(f"- `{e.name}({e.args})`" for e in reply.tool_events)
    return reply.answer + (f"\n\n<details><summary>🔧 Tool calls</summary>\n\n{trace}\n</details>" if trace else "")


def quote_box(symbol: str) -> str:
    q = get_quote(symbol)
    if q.error:
        return q.error
    return (f"**{q.symbol}** {q.last_price:,.2f} ({q.change_pct:+.2f}%)  \n"
            f"52w: {q.week52_low:,.0f}–{q.week52_high:,.0f} · source {q.source} · {q.fetched_at}")


def build() -> gr.Blocks:
    with gr.Blocks(title="NSE-GPT") as demo:
        gr.Markdown("# 📈 NSE-GPT (Gradio)\n100% local NSE intelligence. Informational only, not financial advice.")
        with gr.Row():
            with gr.Column(scale=3):
                gr.ChatInterface(fn=respond,
                                 examples=["What's happening with Reliance today?", "Which NIFTY 50 stocks declared dividends recently?",
                                           "Summarize the latest insider trading filings"])
            with gr.Column(scale=1):
                symbol = gr.Textbox(label="Symbol", value="RELIANCE")
                out = gr.Markdown()
                gr.Button("Get quote").click(quote_box, symbol, out)
    return demo


if __name__ == "__main__":
    configure_logging("gradio")
    build().launch()
