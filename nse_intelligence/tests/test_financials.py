from ingestion.financials import parse_xbrl

SAMPLE = """<xbrl>
<xbrli:context id="OneD"><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="FourD"><xbrli:period><xbrli:startDate>2025-04-01</xbrli:startDate><xbrli:endDate>2026-03-31</xbrli:endDate></xbrli:period></xbrli:context>
<xbrli:context id="OneOperatingExpenses01D"><xbrli:period><xbrli:startDate>2026-04-01</xbrli:startDate><xbrli:endDate>2026-06-30</xbrli:endDate></xbrli:period></xbrli:context>
<in-bse-fin:Symbol contextRef="OneD">ACME</in-bse-fin:Symbol>
<in-bse-fin:NameOfTheCompany contextRef="OneD">Acme Limited</in-bse-fin:NameOfTheCompany>
<in-bse-fin:RevenueFromOperations contextRef="OneD" unitRef="INR" decimals="2">1000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:ProfitLossForPeriod contextRef="OneD" unitRef="INR" decimals="2">-50.5</in-bse-fin:ProfitLossForPeriod>
<in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations contextRef="OneD" unitRef="INRPerShare">-1.25</in-bse-fin:BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations>
<in-bse-fin:RevenueFromOperations contextRef="FourD" unitRef="INR">4000.00</in-bse-fin:RevenueFromOperations>
<in-bse-fin:RevenueFromOperations contextRef="OneOperatingExpenses01D" unitRef="INR">999</in-bse-fin:RevenueFromOperations>
</xbrl>"""


def test_extracts_header_and_headline_periods_only():
    results = parse_xbrl(SAMPLE, "http://src")
    assert results.header == {"symbol": "ACME", "company": "Acme Limited"}
    assert [p.context for p in results.periods] == ["OneD", "FourD"], "line-item contexts must be dropped"


def test_quarter_metrics_are_numeric_and_months_computed():
    quarter = parse_xbrl(SAMPLE).periods[0]
    assert quarter.months == 3
    assert quarter.metrics == {"revenue_from_operations": 1000.0, "net_profit": -50.5, "eps_basic": -1.25}


def test_to_dict_is_json_friendly():
    payload = parse_xbrl(SAMPLE, "http://src").to_dict()
    assert payload["source_url"] == "http://src"
    assert payload["periods"][1]["months"] == 12
