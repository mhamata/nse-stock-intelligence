from ingestion.symbols import COMMON_ALIASES, detect_symbols, normalise_name


class FakeUniverse:
    def is_symbol(self, token: str) -> bool:
        return token in {"TCS", "INFY", "MARINE"}


def test_normalise_strips_noise_words_and_punctuation():
    assert normalise_name("Reliance Industries Limited") == "reliance industries"
    assert normalise_name("Marine Electricals (India) Ltd.") == "marine electricals"


def test_detects_aliases_and_symbols_in_order():
    assert detect_symbols("What's happening with Reliance and TCS today?", FakeUniverse()) == ["RELIANCE", "TCS"]


def test_detects_indices_and_ignores_stopwords():
    assert detect_symbols("Show me the latest news about Nifty", FakeUniverse()) == ["NIFTY50"]
    assert detect_symbols("what does this stock price tell", FakeUniverse()) == []


def test_alias_table_maps_to_upper_case_symbols():
    assert all(v == v.upper() for v in COMMON_ALIASES.values())
