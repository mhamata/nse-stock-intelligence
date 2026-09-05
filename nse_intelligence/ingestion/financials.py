"""Quarterly financial results from NSE's XBRL filings.

The financial-results RSS feed only carries a headline and a link to an XBRL
XML file. The XML is where the numbers live, tagged with a stable vocabulary
(`in-bse-fin:RevenueFromOperations`, `...:ProfitLossForPeriod`, ...) and a
`contextRef` that identifies the period. We extract a handful of headline
metrics per period so the LLM can report real figures with a source URL.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from loguru import logger

from ingestion.http import fetch_text

# Element name -> friendly label. Order is the display order.
KEY_METRICS: dict[str, str] = {
    "RevenueFromOperations": "revenue_from_operations",
    "OtherIncome": "other_income",
    "Income": "total_income",
    "Expenses": "total_expenses",
    "ProfitBeforeTax": "profit_before_tax",
    "TaxExpense": "tax_expense",
    "ProfitLossForPeriod": "net_profit",
    "ComprehensiveIncomeForThePeriod": "total_comprehensive_income",
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_basic",
    "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations": "eps_diluted",
}
HEADER_FIELDS = {
    "Symbol": "symbol",
    "NameOfTheCompany": "company",
    "NatureOfReportStandaloneConsolidated": "nature",
    "DateOfBoardMeetingWhenFinancialResultsWereApproved": "board_meeting_date",
    "WhetherResultsAreAuditedOrUnaudited": "audited",
}

_FACT = re.compile(r"<in-bse-fin:([A-Za-z0-9]+)\s+[^>]*contextRef=\"([^\"]+)\"[^>]*>([^<]*)</")
_CONTEXT = re.compile(
    r"<xbrli:context id=\"([^\"]+)\">.*?<xbrli:startDate>([^<]+)</xbrli:startDate>\s*<xbrli:endDate>([^<]+)</xbrli:endDate>",
    re.S,
)


@dataclass
class ResultsPeriod:
    context: str
    start: str
    end: str
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def months(self) -> int:
        y1, m1 = int(self.start[:4]), int(self.start[5:7])
        y2, m2 = int(self.end[:4]), int(self.end[5:7])
        return (y2 - y1) * 12 + (m2 - m1) + 1


@dataclass
class FinancialResults:
    source_url: str
    header: dict[str, str]
    periods: list[ResultsPeriod]

    def to_dict(self) -> dict:
        return {
            "source_url": self.source_url,
            **self.header,
            "units": "INR (as filed; usually absolute rupees or lakhs - see filing)",
            "periods": [
                {"context": p.context, "period_start": p.start, "period_end": p.end, "months": p.months, **p.metrics}
                for p in self.periods
            ],
        }


def parse_xbrl(xml: str, source_url: str = "") -> FinancialResults:
    contexts = {cid: (start, end) for cid, start, end in _CONTEXT.findall(xml)}
    header: dict[str, str] = {}
    periods: dict[str, ResultsPeriod] = {}

    for element, context_id, raw in _FACT.findall(xml):
        value = raw.strip()
        if element in HEADER_FIELDS and value and HEADER_FIELDS[element] not in header:
            header[HEADER_FIELDS[element]] = value
        if element in KEY_METRICS and context_id in contexts:
            try:
                number = float(value)
            except ValueError:
                continue
            start, end = contexts[context_id]
            period = periods.setdefault(context_id, ResultsPeriod(context_id, start, end))
            period.metrics.setdefault(KEY_METRICS[element], number)

    # Keep only the headline period contexts (OneD = latest quarter, etc.); the
    # many "...Expenses01D" contexts are line-item breakdowns we don't need.
    headline = [p for cid, p in periods.items() if re.fullmatch(r"(One|Two|Three|Four|Five|Six)D", cid)]
    headline.sort(key=lambda p: (p.months, p.end), reverse=False)
    return FinancialResults(source_url, header, headline)


def fetch_results_from_xbrl(url: str) -> FinancialResults:
    logger.info(f"Parsing XBRL results {url}")
    return parse_xbrl(fetch_text(url), url)
