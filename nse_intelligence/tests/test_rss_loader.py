"""Pure unit tests for feed parsing - no network, no Ollama."""
from datetime import datetime

from ingestion.feeds import NSE_FEEDS, resolve_feed
from ingestion.rss_loader import entry_to_document, parse_pipe_fields, parse_pub_date, stable_id, symbol_from_link


def test_parses_nse_datetime_with_seconds():
    dt = parse_pub_date("05-Sep-2026 21:19:51")
    assert dt == datetime(2026, 9, 5, 21, 19, 51, tzinfo=dt.tzinfo)
    assert dt.utcoffset().total_seconds() == 5.5 * 3600


def test_parses_nse_datetime_without_seconds_and_rfc822():
    assert parse_pub_date("02-May-2026 16:46").hour == 16
    assert parse_pub_date("Fri, 4 Sep 2026 00:00:00 +0530").day == 4


def test_unparseable_date_returns_none_instead_of_raising():
    assert parse_pub_date("yesterday-ish") is None


def test_pipe_fields_become_snake_case_keys():
    fields = parse_pipe_fields("SERIES:EQ |PURPOSE:DIVIDEND - RE 0.01 PER SHARE |RECORD DATE:08-Sep-2026 |BOOK CLOSURE START DATE:-")
    assert fields == {"series": "EQ", "purpose": "DIVIDEND - RE 0.01 PER SHARE", "record_date": "08-Sep-2026", "book_closure_start_date": "-"}


def test_free_text_summary_yields_no_fields():
    assert parse_pipe_fields("Tata Motors launches tender offer for Iveco Group") == {}


def test_symbol_is_read_from_announcement_pdf_filename():
    assert symbol_from_link("https://nsearchives.nseindia.com/corporate/KSOLVES_05092026212507_Notice.pdf") == "KSOLVES"
    assert symbol_from_link("https://www.nseindia.com/companies-listing/corporate-filings-actions") is None


def test_stable_id_changes_when_summary_changes():
    feed = NSE_FEEDS["insider_trading"]
    a = stable_id(feed, "Acme Ltd", "http://x", "01-Jan-2026", "TYPE: Equity")
    b = stable_id(feed, "Acme Ltd", "http://x", "01-Jan-2026", "TYPE: Warrants")
    assert a != b and a.startswith("insider_trading:")


def test_entry_to_document_builds_flat_metadata():
    feed = NSE_FEEDS["corporate_actions"]
    entry = {"title": "Foo Ltd - Ex-Date: 08-Sep-2026", "link": "https://www.nseindia.com/companies-listing/x",
             "summary": "SERIES:EQ |PURPOSE:DIVIDEND - RS 2 PER SHARE |RECORD DATE:08-Sep-2026", "published": "04-Sep-2026 06:09:04"}
    doc = entry_to_document(feed, entry, resolve_symbol=lambda name: "FOO" if name == "Foo Ltd" else None)
    assert doc.metadata["company"] == "Foo Ltd"
    assert doc.metadata["symbol"] == "FOO"
    assert doc.metadata["f_record_date"] == "08-Sep-2026"
    assert doc.metadata["published_ts"] > 0
    assert all(isinstance(v, (str, int, float, bool)) for v in doc.metadata.values()), "Chroma needs flat scalar metadata"
    assert doc.page_content.startswith("[Corporate Actions] Foo Ltd (FOO)")


def test_resolve_feed_accepts_key_or_url_or_custom_url():
    assert resolve_feed("circulars").category == "circulars"
    assert resolve_feed(NSE_FEEDS["buyback"].url).category == "buyback"
    assert resolve_feed("https://example.com/feed.xml").category == "custom"
