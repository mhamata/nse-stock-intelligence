"""Conversation memory: the last N exchanges per chat session.

LangChain 1.x removed ConversationBufferMemory. What we want is simple enough
to own outright: for each session keep the human/assistant turns, drop the
oldest once we exceed the window, and hand them to the agent as message
objects. Tool-call chatter from previous turns is deliberately NOT replayed -
it would flood a small model's context with stale JSON.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

DEFAULT_WINDOW = 10  # exchanges, as required by the system prompt


@dataclass
class Exchange:
    question: str
    answer: str


@dataclass
class ConversationMemory:
    window: int = DEFAULT_WINDOW
    _sessions: dict[str, deque[Exchange]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def history(self, session_id: str) -> list[BaseMessage]:
        with self._lock:
            exchanges = list(self._sessions.get(session_id, ()))
        messages: list[BaseMessage] = []
        for exchange in exchanges:
            messages.append(HumanMessage(content=exchange.question))
            messages.append(AIMessage(content=exchange.answer))
        return messages

    def remember(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            bucket = self._sessions.setdefault(session_id, deque(maxlen=self.window))
            bucket.append(Exchange(question, answer))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def exchanges(self, session_id: str) -> list[Exchange]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))
