"""Registry of every RSS feed the system ingests.

The URLs in the original design document returned 404 when this project was
built (September 2026): NSE moved its feeds to capitalised filenames on the
nsearchives host and dropped the press-release feed entirely. Everything
below was verified live at build time. Keep this file as the single source
of truth so a URL change only has to be fixed in one place.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feed:
    category: str      # short machine key used in metadata and tool arguments
    url: str
    label: str         # human-readable name for the UI
    source: str        # "nse" for exchange filings, "news" for press coverage


NSE_FEEDS: dict[str, Feed] = {
    "corporate_announcement": Feed(
        "corporate_announcement",
        "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml",
        "Corporate Announcements", "nse",
    ),
    "financial_results": Feed(
        "financial_results",
        "https://nsearchives.nseindia.com/content/RSS/Financial_Results.xml",
        "Financial Results", "nse",
    ),
    "insider_trading": Feed(
        "insider_trading",
        "https://nsearchives.nseindia.com/content/RSS/Insider_Trading.xml",
        "Insider Trading", "nse",
    ),
    "corporate_actions": Feed(
        "corporate_actions",
        "https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml",
        "Corporate Actions", "nse",
    ),
    "board_meetings": Feed(
        "board_meetings",
        "https://nsearchives.nseindia.com/content/RSS/Board_Meetings.xml",
        "Board Meetings", "nse",
    ),
    "investor_complaints": Feed(
        "investor_complaints",
        "https://nsearchives.nseindia.com/content/RSS/Investor_Complaints.xml",
        "Investor Complaints", "nse",
    ),
    "circulars": Feed(
        "circulars",
        "https://nsearchives.nseindia.com/content/RSS/Circulars.xml",
        "Circulars", "nse",
    ),
    "buyback": Feed(
        "buyback",
        "https://nsearchives.nseindia.com/content/RSS/Daily_Buyback.xml",
        "Daily Buyback", "nse",
    ),
}

NEWS_FEEDS: dict[str, Feed] = {
    "news_et_markets": Feed(
        "news_et_markets",
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "Economic Times - Markets", "news",
    ),
    "news_moneycontrol": Feed(
        "news_moneycontrol",
        "https://www.moneycontrol.com/rss/latestnews.xml",
        "Moneycontrol - Latest", "news",
    ),
    "news_business_standard": Feed(
        "news_business_standard",
        "https://www.business-standard.com/rss/markets-106.rss",
        "Business Standard - Markets", "news",
    ),
}

ALL_FEEDS: dict[str, Feed] = {**NSE_FEEDS, **NEWS_FEEDS}


def resolve_feed(category_or_url: str) -> Feed:
    """Accept either a category key ("financial_results") or a full URL."""
    key = category_or_url.strip().lower()
    if key in ALL_FEEDS:
        return ALL_FEEDS[key]
    for feed in ALL_FEEDS.values():
        if feed.url == category_or_url.strip():
            return feed
    if category_or_url.startswith("http"):
        return Feed("custom", category_or_url.strip(), "Custom feed", "custom")
    raise KeyError(f"Unknown feed '{category_or_url}'. Known: {', '.join(ALL_FEEDS)}")
