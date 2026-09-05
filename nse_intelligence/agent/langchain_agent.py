"""ReAct agent (LangChain 1.x `create_agent`) with tools served over MCP.

Architecture:
    Streamlit / Gradio (sync)  -->  AgentRuntime.ask()  -->  background asyncio loop
                                                              |-- MCP stdio session (one subprocess, long-lived)
                                                              |-- ChatOllama(llama3.2)
                                                              `-- create_agent(...)  (ReAct loop: model <-> tools)

The MCP client is async and the server is a child process. Opening a session
per question would re-spawn that process (and re-import ChromaDB) every time.
So the runtime owns one event loop on a daemon thread, opens the session once,
and every UI call is marshalled onto that loop with run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_ollama import ChatOllama
from loguru import logger

from agent.memory import ConversationMemory
from agent.prompts import build_system_prompt
from config import settings
from ingestion.symbols import detect_symbols

MCP_SERVER_PATH = Path(__file__).resolve().parents[1] / "mcp_server" / "rss_tools.py"

# Intent routing for auto-context: question keywords -> the feed most likely to
# hold the answer. A 3B model picks categories poorly; this seeds the right one.
TOPIC_FEEDS: list[tuple[tuple[str, ...], str]] = [
    (("dividend", "record date", "ex-date", "ex date", "bonus", "split", "rights issue", "corporate action"), "corporate_actions"),
    (("result", "quarter", "earnings", "revenue", "profit", "eps", "q1", "q2", "q3", "q4"), "financial_results"),
    (("insider", "promoter", "pledge"), "insider_trading"),
    (("buyback", "buy-back", "buy back"), "buyback"),
    (("circular", "sebi", "regulation", "compliance"), "circulars"),
    (("board meeting", "board meetings", "agm", "egm"), "board_meetings"),
    (("complaint", "grievance"), "investor_complaints"),
    (("market", "sensex", "nifty", "news", "rally", "fell", "crash", "fii", "dii"), "news_et_markets"),
]
DISCLAIMER = "This is informational only, not financial advice."


@dataclass
class ToolEvent:
    name: str
    args: dict[str, Any]
    result_preview: str = ""


@dataclass
class AgentReply:
    answer: str
    tool_events: list[ToolEvent] = field(default_factory=list)
    error: str | None = None


def build_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_ctx=settings.ollama_num_ctx,  # default 4k is too small for prompt + tool JSON
    )


class AgentRuntime:
    """Long-lived agent: one MCP session, one model, many chat sessions."""

    def __init__(self, memory: ConversationMemory | None = None):
        self.memory = memory or ConversationMemory()
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="agent-loop", daemon=True)
        self._thread.start()
        self._ready = threading.Event()
        self._shutdown = asyncio.Event()
        self.tools: list = []
        self.agent = None
        self._startup_error: Exception | None = None
        asyncio.run_coroutine_threadsafe(self._serve(), self._loop)
        self._ready.wait(timeout=120)
        if self._startup_error:
            raise self._startup_error

    async def _serve(self) -> None:
        client = MultiServerMCPClient({
            "nse": {"transport": "stdio", "command": sys.executable, "args": [str(MCP_SERVER_PATH)]},
        })
        try:
            async with client.session("nse") as session:
                self.tools = await load_mcp_tools(session)
                logger.info(f"MCP tools loaded: {[t.name for t in self.tools]}")
                self.agent = create_agent(model=build_llm(), tools=self.tools, system_prompt=build_system_prompt())
                self._ready.set()
                await self._shutdown.wait()  # keep the session open for the app's lifetime
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()

    # ----- public sync API -----
    def ask(self, question: str, session_id: str = "default",
            on_event: Callable[[ToolEvent], None] | None = None) -> AgentReply:
        future = asyncio.run_coroutine_threadsafe(self._ask(question, session_id, on_event), self._loop)
        return future.result(timeout=600)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._shutdown.set)

    # ----- internals -----
    async def _auto_context(self, question: str, events: list[ToolEvent], on_event) -> str:
        """Pre-fetch grounding data for symbols named in the question.

        Small models often stop after one tool call. Seeding the turn with the
        quote, the latest filings and the top RAG chunks means the answer can
        always populate LIVE DATA / LATEST NEWS / RAG CONTEXT from real tool
        output. The agent may still call more tools on top of this.
        """
        symbols = detect_symbols(question)[:2]
        lowered = question.lower()
        topics = [category for keywords, category in TOPIC_FEEDS if any(k in lowered for k in keywords)][:2]
        if not symbols and not topics:
            return ""
        by_name = {t.name: t for t in self.tools}
        calls: list[tuple[str, dict]] = []
        for symbol in symbols:
            calls.append(("get_quote", {"symbol": symbol}))
            if not symbol.startswith(("NIFTY", "BANKNIFTY")):
                calls.append(("fetch_rss", {"category": "corporate_announcement", "symbol": symbol, "limit": 5}))
                calls.append(("rag_query", {"question": question, "symbol": symbol, "k": 4}))
        for category in topics:
            args = {"category": category, "limit": 8}
            if symbols and not symbols[0].startswith(("NIFTY", "BANKNIFTY")):
                args["symbol"] = symbols[0]
            calls.append(("fetch_rss", args))
        if not symbols:
            calls.append(("rag_query", {"question": question, "k": 5}))
        blocks = []
        for name, args in calls:
            tool = by_name.get(name)
            if tool is None:
                continue
            event = ToolEvent(f"auto:{name}", args)
            events.append(event)
            if on_event:
                on_event(event)
            try:
                result = await tool.ainvoke(args)
            except Exception as exc:
                result = f'{{"error": "{type(exc).__name__}: {exc}"}}'
            text = result if isinstance(result, str) else str(result)
            event.result_preview = text[:300]
            if on_event:
                on_event(event)
            blocks.append(f"<{name} args={args}>\n{text[:3000]}\n</{name}>")
        return "\n\n".join(blocks)

    async def _ask(self, question: str, session_id: str, on_event) -> AgentReply:
        events: list[ToolEvent] = []
        pending: dict[str, ToolEvent] = {}
        final_text = ""
        content = question
        if settings.agent_auto_context:
            context = await self._auto_context(question, events, on_event)
            if context:
                content = (f"{question}\n\n[Pre-fetched tool results for this question - cite them, and call more tools if needed]\n"
                           f"{context}")
        messages = [*self.memory.history(session_id), HumanMessage(content=content)]
        try:
            async for update in self.agent.astream({"messages": messages}, stream_mode="updates"):
                for node_output in update.values():
                    for message in (node_output or {}).get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                event = ToolEvent(call["name"], call.get("args", {}))
                                pending[call["id"]] = event
                                events.append(event)
                                if on_event:
                                    on_event(event)
                            if message.content and not message.tool_calls:
                                final_text = message.content if isinstance(message.content, str) else str(message.content)
                        elif isinstance(message, ToolMessage):
                            event = pending.get(message.tool_call_id)
                            if event:
                                event.result_preview = str(message.content)[:300]
                                if on_event:
                                    on_event(event)
        except Exception as exc:
            logger.exception("Agent run failed")
            return AgentReply(answer="", tool_events=events, error=f"{type(exc).__name__}: {exc}")

        if not final_text:
            final_text = "I could not produce an answer from the tool results. Please rephrase or try again."
        # Hard rule 3 from the system prompt, enforced in code rather than trusted to the model.
        if DISCLAIMER.lower() not in final_text.lower():
            final_text = f"{final_text.rstrip()}\n\n_{DISCLAIMER}_"
        self.memory.remember(session_id, question, final_text)
        return AgentReply(answer=final_text, tool_events=events)


_runtime: AgentRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> AgentRuntime:
    """Process-wide singleton so every UI session shares one MCP subprocess."""
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = AgentRuntime()
        return _runtime


if __name__ == "__main__":  # quick manual smoke test:  python agent/langchain_agent.py "What's happening with TCS?"
    from config import configure_logging

    configure_logging("agent")
    runtime = get_runtime()
    query = " ".join(sys.argv[1:]) or "What is the NIFTY 50 at right now?"
    reply = runtime.ask(query, on_event=lambda e: print(f"  -> tool {e.name}({e.args})" if not e.result_preview else f"     <- {e.result_preview[:120]}"))
    print("\n" + (reply.answer or reply.error))
    runtime.close()
