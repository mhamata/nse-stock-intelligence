"""Symbol universe: index constituents and company-name -> symbol resolution.

NSE feeds identify companies by their legal name ("Reliance Industries
Limited") while the LLM and price APIs work in ticker symbols ("RELIANCE").
This module bridges the two using NSE's own public CSV files, cached locally
for 24h so the system keeps working when NSE is slow or blocked.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from config import settings
from ingestion.http import fetch_text

INDEX_CSVS: dict[str, str] = {
    "NIFTY50": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
    "NIFTYBANK": "https://nsearchives.nseindia.com/content/indices/ind_niftybanklist.csv",
    "NIFTYIT": "https://nsearchives.nseindia.com/content/indices/ind_niftyitlist.csv",
    "NIFTYNEXT50": "https://nsearchives.nseindia.com/content/indices/ind_niftynext50list.csv",
    "NIFTY100": "https://nsearchives.nseindia.com/content/indices/ind_nifty100list.csv",
    "NIFTYFMCG": "https://nsearchives.nseindia.com/content/indices/ind_niftyfmcglist.csv",
    "NIFTYPHARMA": "https://nsearchives.nseindia.com/content/indices/ind_niftypharmalist.csv",
    "NIFTYAUTO": "https://nsearchives.nseindia.com/content/indices/ind_niftyautolist.csv",
}
EQUITY_MASTER_CSV = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
CACHE_TTL_SECONDS = 24 * 3600

# Words that add nothing to identity and vary between sources.
_NOISE = re.compile(r"\b(limited|ltd\.?|the|of|india|&|and)\b", re.IGNORECASE)


def normalise_name(name: str) -> str:
    """Reduce a company name to a comparable key: lowercase, no noise words."""
    cleaned = _NOISE.sub(" ", name)
    cleaned = re.sub(r"[^a-z0-9 ]", " ", cleaned.lower())
    return " ".join(cleaned.split())


@dataclass(frozen=True)
class Company:
    symbol: str
    name: str
    industry: str = ""
    isin: str = ""


class SymbolUniverse:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or (settings.chroma_persist_dir.parent / ".cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._master: dict[str, Company] | None = None
        self._by_name: dict[str, str] | None = None

    # ---------- caching ----------
    def _cached_text(self, key: str, url: str) -> str:
        path = self.cache_dir / f"{key}.csv"
        if path.exists() and time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS:
            return path.read_text()
        try:
            text = fetch_text(url)
            path.write_text(text)
            return text
        except Exception as exc:  # network failure: fall back to stale cache if any
            if path.exists():
                logger.warning(f"Using stale cache for {key}: {exc}")
                return path.read_text()
            raise

    # ---------- index constituents ----------
    def index_constituents(self, index: str) -> list[Company]:
        key = re.sub(r"[^A-Z0-9]", "", index.upper())
        if key not in INDEX_CSVS:
            raise KeyError(f"Unknown index '{index}'. Known: {', '.join(INDEX_CSVS)}")
        rows = csv.DictReader(io.StringIO(self._cached_text(key, INDEX_CSVS[key])))
        return [
            Company(r["Symbol"].strip(), r["Company Name"].strip(), r.get("Industry", "").strip(), r.get("ISIN Code", "").strip())
            for r in rows
        ]

    # ---------- full equity master ----------
    def _load_master(self) -> None:
        text = self._cached_text("EQUITY_L", EQUITY_MASTER_CSV)
        rows = csv.DictReader(io.StringIO(text))
        master: dict[str, Company] = {}
        for r in rows:
            r = {k.strip(): (v or "").strip() for k, v in r.items() if k}
            symbol = r.get("SYMBOL", "")
            if symbol:
                master[symbol] = Company(symbol, r.get("NAME OF COMPANY", ""), "", r.get("ISIN NUMBER", ""))
        self._master = master
        self._by_name = {normalise_name(c.name): c.symbol for c in master.values()}

    @property
    def master(self) -> dict[str, Company]:
        if self._master is None:
            self._load_master()
        return self._master  # type: ignore[return-value]

    def is_symbol(self, candidate: str) -> bool:
        return candidate.upper() in self.master

    def resolve(self, company_name: str) -> str | None:
        """Best-effort company name -> symbol. Returns None when unsure."""
        if self._by_name is None:
            self._load_master()
        key = normalise_name(company_name)
        if key in self._by_name:  # type: ignore[operator]
            return self._by_name[key]  # type: ignore[index]
        # Fall back to prefix matching: "Reliance Industries" vs "Reliance Industries Limited".
        for name_key, symbol in self._by_name.items():  # type: ignore[union-attr]
            if name_key.startswith(key) or key.startswith(name_key):
                return symbol
        return None


_universe: SymbolUniverse | None = None


def get_universe() -> SymbolUniverse:
    global _universe
    if _universe is None:
        _universe = SymbolUniverse()
    return _universe


# Everyday names people type that aren't literal symbols. The universe handles
# exact symbols; this handles "Reliance", "Infosys", "HDFC Bank", "Nifty".
COMMON_ALIASES: dict[str, str] = {
    "RELIANCE": "RELIANCE", "RIL": "RELIANCE", "INFOSYS": "INFY", "TCS": "TCS", "WIPRO": "WIPRO",
    "HDFC BANK": "HDFCBANK", "HDFCBANK": "HDFCBANK", "ICICI BANK": "ICICIBANK", "ICICI": "ICICIBANK",
    "SBI": "SBIN", "STATE BANK": "SBIN", "AXIS BANK": "AXISBANK", "KOTAK": "KOTAKBANK",
    "BHARTI AIRTEL": "BHARTIARTL", "AIRTEL": "BHARTIARTL", "ITC": "ITC", "L&T": "LT", "LARSEN": "LT",
    "MARUTI": "MARUTI", "TATA MOTORS": "TATAMOTORS", "TATA STEEL": "TATASTEEL", "TITAN": "TITAN",
    "ASIAN PAINTS": "ASIANPAINT", "SUN PHARMA": "SUNPHARMA", "BAJAJ FINANCE": "BAJFINANCE",
    "HUL": "HINDUNILVR", "HINDUSTAN UNILEVER": "HINDUNILVR", "ADANI ENTERPRISES": "ADANIENT",
    "ADANI PORTS": "ADANIPORTS", "ONGC": "ONGC", "NTPC": "NTPC", "POWER GRID": "POWERGRID",
    "NIFTY": "NIFTY50", "NIFTY 50": "NIFTY50", "NIFTY50": "NIFTY50", "BANK NIFTY": "BANKNIFTY",
    "BANKNIFTY": "BANKNIFTY", "NIFTY BANK": "BANKNIFTY", "NIFTY IT": "NIFTYIT", "NIFTYIT": "NIFTYIT",
}
_STOPWORDS = {"THE", "AND", "FOR", "WITH", "WHAT", "TODAY", "NEWS", "ABOUT", "SHOW", "LATEST", "THIS", "THAT", "FROM", "HAVE", "DOES", "STOCK", "PRICE", "TELL", "GIVE", "LAST", "WEEK", "WHICH", "THEIR", "THEY", "IT", "ANY"}


def detect_symbols(text: str, universe: "SymbolUniverse | None" = None) -> list[str]:
    """Return NSE symbols/indices mentioned in free text, in order of appearance."""
    universe = universe or get_universe()
    cleaned = re.sub(r"[^A-Za-z0-9& ]", " ", text).upper()
    found: list[str] = []
    for phrase, symbol in COMMON_ALIASES.items():
        if re.search(rf"\b{re.escape(phrase)}\b", cleaned) and symbol not in found:
            found.append(symbol)
    try:
        for token in cleaned.split():
            if len(token) >= 3 and token not in _STOPWORDS and token not in found and universe.is_symbol(token):
                found.append(token)
    except Exception:
        pass  # master list unavailable offline: aliases still work
    return found
