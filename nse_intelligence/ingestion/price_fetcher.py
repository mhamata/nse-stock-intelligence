"""Two-tier price client.

Tier 1 - NSE's own JSON API (nseindia.com/api/quote-equity). It needs the
         cookies a browser gets from loading a page first, and those cookies
         expire, so we keep a Session and re-warm it on a timer.
Tier 2 - Yahoo Finance via yfinance (symbol + ".NS"). Delayed ~15 minutes but
         dependable. At build time NSE returned 403 for the quote endpoint
         even with cookies, so in practice this tier does most of the work.

Both tiers return the same `Quote` shape so callers never care which one
answered - but the `source` field always says, because the system prompt
requires timestamps and provenance on every number.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import requests
from loguru import logger

from config import settings
from ingestion.http import BROWSER_HEADERS

NSE_HOME = "https://www.nseindia.com/get-quotes/equity?symbol=RELIANCE"
NSE_QUOTE = "https://www.nseindia.com/api/quote-equity?symbol={symbol}"
NSE_ALL_INDICES = "https://www.nseindia.com/api/allIndices"
NSE_CORP_ACTIONS = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&symbol={symbol}"

INDEX_ALIASES = {
    "NIFTY": "NIFTY 50", "NIFTY50": "NIFTY 50", "NIFTY 50": "NIFTY 50",
    "BANKNIFTY": "NIFTY BANK", "NIFTYBANK": "NIFTY BANK", "NIFTY BANK": "NIFTY BANK",
    "NIFTYIT": "NIFTY IT", "NIFTY IT": "NIFTY IT",
    "SENSEX": None,  # BSE index, not on NSE
}


@dataclass
class Quote:
    symbol: str
    last_price: float | None
    previous_close: float | None
    change: float | None
    change_pct: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    week52_high: float | None
    week52_low: float | None
    source: str            # "nse" | "yfinance" | "nse_index"
    fetched_at: str        # ISO-8601 UTC
    company_name: str = ""
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class NSESession:
    """requests.Session with browser cookies that are refreshed periodically."""

    def __init__(self, refresh_minutes: int | None = None):
        self.refresh_seconds = 60 * (refresh_minutes or settings.nse_session_refresh_minutes)
        self._session: requests.Session | None = None
        self._warmed_at = 0.0
        # Circuit breaker: once NSE has denied us this many times in a row we
        # stop trying until the next cookie-refresh window, so callers fall
        # straight through to yfinance instead of paying a 2s penalty per call.
        self._consecutive_denials = 0
        self._tripped_at = 0.0
        self.max_denials = 3

    @property
    def tripped(self) -> bool:
        if self._consecutive_denials < self.max_denials:
            return False
        if time.time() - self._tripped_at > self.refresh_seconds:
            self._consecutive_denials = 0  # half-open: allow a retry
            return False
        return True

    def _record(self, denied: bool) -> None:
        if denied:
            self._consecutive_denials += 1
            if self._consecutive_denials == self.max_denials:
                self._tripped_at = time.time()
                logger.warning("NSE API circuit breaker tripped; using yfinance until next refresh window")
        else:
            self._consecutive_denials = 0

    def _warm(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({**BROWSER_HEADERS, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8"})
        session.get(NSE_HOME, timeout=15)  # sets nsit / nseappid / akamai cookies
        self._session, self._warmed_at = session, time.time()
        return session

    @property
    def session(self) -> requests.Session:
        if self._session is None or time.time() - self._warmed_at > self.refresh_seconds:
            return self._warm()
        return self._session

    def get_json(self, url: str) -> dict | list:
        headers = {"Accept": "application/json, text/plain, */*", "Referer": NSE_HOME, "X-Requested-With": "XMLHttpRequest"}
        if self.tripped:
            raise PermissionError("NSE API circuit breaker open")
        response = self.session.get(url, headers=headers, timeout=15)
        if response.status_code == 403:
            # Cookies rejected: warm once more, then give up to the caller.
            response = self._warm().get(url, headers=headers, timeout=15)
        self._record(denied=response.status_code == 403)
        response.raise_for_status()
        return response.json()


_nse = NSESession()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _f(value) -> float | None:
    try:
        return float(value) if value not in (None, "", "-") else None
    except (TypeError, ValueError):
        return None


def _quote_from_nse(symbol: str) -> Quote:
    data = _nse.get_json(NSE_QUOTE.format(symbol=symbol))
    price = data.get("priceInfo", {})
    info = data.get("info", {})
    hl = price.get("weekHighLow", {})
    return Quote(
        symbol=symbol, last_price=_f(price.get("lastPrice")), previous_close=_f(price.get("previousClose")),
        change=_f(price.get("change")), change_pct=_f(price.get("pChange")),
        day_high=_f(price.get("intraDayHighLow", {}).get("max")), day_low=_f(price.get("intraDayHighLow", {}).get("min")),
        volume=None, week52_high=_f(hl.get("max")), week52_low=_f(hl.get("min")),
        source="nse", fetched_at=_now(), company_name=info.get("companyName", ""),
    )


def _quote_from_yfinance(symbol: str) -> Quote:
    import yfinance as yf

    ticker = yf.Ticker(f"{symbol}.NS")
    fi = ticker.fast_info
    last, prev = _f(fi.last_price), _f(fi.previous_close)
    if last is None:
        raise ValueError(f"yfinance returned no price for {symbol}.NS")
    change = (last - prev) if prev else None
    return Quote(
        symbol=symbol, last_price=last, previous_close=prev, change=change,
        change_pct=(change / prev * 100) if (change is not None and prev) else None,
        day_high=_f(fi.day_high), day_low=_f(fi.day_low), volume=int(fi.last_volume or 0) or None,
        week52_high=_f(fi.year_high), week52_low=_f(fi.year_low),
        source="yfinance", fetched_at=_now(),
    )


def _quote_for_index(index_name: str) -> Quote:
    payload = _nse.get_json(NSE_ALL_INDICES)
    for row in payload.get("data", []):
        if row.get("indexSymbol", "").upper() == index_name:
            return Quote(
                symbol=index_name, last_price=_f(row.get("last")), previous_close=_f(row.get("previousClose")),
                change=_f(row.get("variation")), change_pct=_f(row.get("percentChange")),
                day_high=_f(row.get("high")), day_low=_f(row.get("low")), volume=None,
                week52_high=_f(row.get("yearHigh")), week52_low=_f(row.get("yearLow")),
                source="nse_index", fetched_at=_now(), company_name=row.get("index", ""),
            )
    raise ValueError(f"Index {index_name} not found in NSE allIndices")


def get_quote(symbol: str) -> Quote:
    """Public entry point. Never raises: failures are reported in Quote.error."""
    symbol = symbol.strip().upper().removesuffix(".NS")
    index_name = INDEX_ALIASES.get(symbol)
    if symbol in INDEX_ALIASES:
        if index_name is None:
            return Quote(symbol, *([None] * 9), source="none", fetched_at=_now(), error=f"{symbol} is not an NSE index")
        try:
            return _quote_for_index(index_name)
        except Exception as exc:
            logger.warning(f"Index quote failed for {symbol}: {exc}")

    try:
        return _quote_from_nse(symbol)
    except Exception as exc:
        logger.debug(f"NSE quote unavailable for {symbol} ({type(exc).__name__}); trying yfinance")
    try:
        return _quote_from_yfinance(symbol)
    except Exception as exc:
        logger.warning(f"All price sources failed for {symbol}: {exc}")
        return Quote(symbol, *([None] * 9), source="none", fetched_at=_now(), error="Price unavailable - check NSE directly.")


def get_corporate_actions(symbol: str) -> list[dict]:
    """Upcoming/recent dividends, splits, bonuses from NSE's corporate-actions API."""
    symbol = symbol.strip().upper()
    rows = _nse.get_json(NSE_CORP_ACTIONS.format(symbol=symbol))
    return [
        {
            "symbol": r.get("symbol"), "company": r.get("comp"), "purpose": r.get("subject"),
            "ex_date": r.get("exDate"), "record_date": r.get("recDate"), "series": r.get("series"),
        }
        for r in (rows if isinstance(rows, list) else [])
    ]


def get_history(symbol: str, period: str = "6mo"):
    """Daily OHLCV DataFrame for charts (yfinance only; NSE has no free history API)."""
    import yfinance as yf

    return yf.Ticker(f"{symbol.strip().upper()}.NS").history(period=period, auto_adjust=False)
