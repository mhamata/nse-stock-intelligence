import time

from rag.retriever import build_where


def test_no_filters_returns_none_so_chroma_skips_where():
    assert build_where() is None


def test_single_filter_is_not_wrapped_in_and():
    assert build_where(symbol="tcs") == {"symbol": "TCS"}


def test_multiple_filters_are_combined_with_and_and_age_is_a_timestamp_bound():
    where = build_where(symbol="INFY", category="Corporate_Actions", max_age_hours=24)
    assert set(where) == {"$and"} and len(where["$and"]) == 3
    bound = where["$and"][2]["published_ts"]["$gte"]
    assert abs(bound - (time.time() - 86400)) < 5
