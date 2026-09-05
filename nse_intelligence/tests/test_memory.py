from langchain_core.messages import AIMessage, HumanMessage

from agent.memory import ConversationMemory


def test_history_alternates_human_and_ai_messages():
    memory = ConversationMemory(window=10)
    memory.remember("s1", "price of TCS?", "TCS is ...")
    history = memory.history("s1")
    assert isinstance(history[0], HumanMessage) and isinstance(history[1], AIMessage)


def test_window_keeps_only_the_last_n_exchanges():
    memory = ConversationMemory(window=3)
    for i in range(5):
        memory.remember("s1", f"q{i}", f"a{i}")
    assert [m.content for m in memory.history("s1")][::2] == ["q2", "q3", "q4"]


def test_sessions_are_isolated_and_clearable():
    memory = ConversationMemory()
    memory.remember("a", "q", "a"); memory.remember("b", "q", "a")
    memory.clear("a")
    assert memory.history("a") == [] and len(memory.history("b")) == 2
