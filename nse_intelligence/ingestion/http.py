"""Shared HTTP helpers.

NSE's CDN (Akamai) rejects requests that don't look like a real browser, so
every outbound request in the project goes through this module to pick up the
same header set. Centralising it also means one place to add retries/timeouts.
"""
from __future__ import annotations

import requests

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

DEFAULT_TIMEOUT = 20


def fetch_bytes(url: str, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=timeout)
    response.raise_for_status()
    return response.content


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT) -> str:
    return fetch_bytes(url, timeout).decode("utf-8", errors="replace")
