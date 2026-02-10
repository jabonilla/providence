"""Filing parser for SEC EDGAR XBRL data.

Extracts key financial metrics from XBRL company facts and converts
them into FilingPayload format. Handles missing/inconsistent XBRL
tags gracefully.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-FILING)

Classification: FROZEN — zero LLM calls. Pure data extraction.
"""

from datetime import date
from typing import Any, Optional

import structlog

from providence.schemas.payloads import FilingPayload, FilingType

logger = structlog.get_logger()

# Mapping of our field names to common XBRL tags (US-GAAP taxonomy)
XBRL_TAG_MAP: dict[str, list[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "eps": [
        "EarningsPerShareBasic",
        "EarningsPerShareDiluted",
    ],
    "total_assets": [
        "Assets",
    ],
    "total_liabilities": [
        "Liabilities",
        "LiabilitiesAndStockholdersEquity",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
}

# Additional ratio-related XBRL tags
RATIO_TAGS: dict[str, list[str]] = {
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "stockholders_equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent"],
}


def parse_financial_filing(
    xbrl_data: dict[str, Any],
    filing_type: FilingType,
    ticker: str,
    company_name: str,
    cik: str,
    filed_date: date,
    period_of_report: date,
    raw_text_excerpt: str = "",
) -> FilingPayload:
    """Parse XBRL data into a FilingPayload for 10-K or 10-Q filings.

    Args:
        xbrl_data: Raw XBRL company facts data from EDGAR.
        filing_type: FORM_10K or FORM_10Q.
        ticker: Stock ticker.
        company_name: Company legal name.
        cik: SEC CIK.
        filed_date: Filing date.
        period_of_report: Reporting period end date.
        raw_text_excerpt: Text excerpt from filing.

    Returns:
        FilingPayload with extracted financial metrics.
    """
    facts = _extract_us_gaap_facts(xbrl_data)

    # Extract core financial metrics
    revenue = _find_metric(facts, XBRL_TAG_MAP["revenue"], period_of_report)
    net_income = _find_metric(facts, XBRL_TAG_MAP["net_income"], period_of_report)
    eps = _find_metric(facts, XBRL_TAG_MAP["eps"], period_of_report)
    total_assets = _find_metric(facts, XBRL_TAG_MAP["total_assets"], period_of_report)
    total_liabilities = _find_metric(facts, XBRL_TAG_MAP["total_liabilities"], period_of_report)
    operating_cash_flow = _find_metric(facts, XBRL_TAG_MAP["operating_cash_flow"], period_of_report)

    # Compute key ratios where possible
    key_ratios = _compute_ratios(facts, period_of_report)

    return FilingPayload(
        filing_type=filing_type,
        company_name=company_name,
        cik=cik,
        ticker=ticker,
        filed_date=filed_date,
        period_of_report=period_of_report,
        revenue=revenue,
        net_income=net_income,
        eps=eps,
        total_assets=total_assets,
        total_liabilities=total_liabilities,
        operating_cash_flow=operating_cash_flow,
        key_ratios=key_ratios,
        raw_text_excerpt=raw_text_excerpt,
    )


def parse_event_filing(
    event_data: dict[str, Any],
    ticker: str,
    company_name: str,
    cik: str,
    filed_date: date,
    period_of_report: date,
    raw_text_excerpt: str = "",
) -> FilingPayload:
    """Parse 8-K event filing data into a FilingPayload.

    Args:
        event_data: Parsed event data (items, description, etc.).
        ticker: Stock ticker.
        company_name: Company legal name.
        cik: SEC CIK.
        filed_date: Filing date.
        period_of_report: Event date.
        raw_text_excerpt: Text excerpt from filing.

    Returns:
        FilingPayload with event fields populated.
    """
    event_type = event_data.get("event_type", "Unknown")
    event_description = event_data.get("event_description", "")
    material_impact = event_data.get("material_impact", False)

    return FilingPayload(
        filing_type=FilingType.FORM_8K,
        company_name=company_name,
        cik=cik,
        ticker=ticker,
        filed_date=filed_date,
        period_of_report=period_of_report,
        event_type=event_type,
        event_description=event_description,
        material_impact=material_impact,
        raw_text_excerpt=raw_text_excerpt,
    )


def _extract_us_gaap_facts(xbrl_data: dict[str, Any]) -> dict[str, Any]:
    """Extract US-GAAP facts from XBRL company facts structure.

    EDGAR structure: {facts: {us-gaap: {TagName: {units: {USD: [...]}}}}}
    """
    return xbrl_data.get("facts", {}).get("us-gaap", {})


def _find_metric(
    facts: dict[str, Any],
    tag_names: list[str],
    period_end: date,
) -> Optional[float]:
    """Find a metric value by trying multiple XBRL tag names.

    Searches for the most recent value matching the period end date.
    Falls back to the latest available value if no exact match.

    Args:
        facts: US-GAAP facts dictionary.
        tag_names: List of XBRL tag names to try (in priority order).
        period_end: Target period end date.

    Returns:
        Metric value as float, or None if not found.
    """
    period_str = period_end.isoformat()

    for tag in tag_names:
        tag_data = facts.get(tag, {})
        units = tag_data.get("units", {})

        # Try USD first, then USD/shares for per-share metrics
        for unit_key in ["USD", "USD/shares", "pure"]:
            entries = units.get(unit_key, [])
            if not entries:
                continue

            # Look for exact period match
            for entry in entries:
                end = entry.get("end", "")
                if end == period_str:
                    try:
                        return float(entry["val"])
                    except (ValueError, KeyError, TypeError):
                        continue

            # Fallback: most recent entry
            if entries:
                try:
                    latest = sorted(entries, key=lambda e: e.get("end", ""), reverse=True)[0]
                    return float(latest["val"])
                except (ValueError, KeyError, TypeError, IndexError):
                    continue

    return None


def _compute_ratios(facts: dict[str, Any], period_end: date) -> dict[str, float]:
    """Compute key financial ratios from available XBRL data.

    Returns:
        Dictionary of ratio_name → ratio_value.
    """
    ratios: dict[str, float] = {}

    current_assets = _find_metric(facts, RATIO_TAGS["current_assets"], period_end)
    current_liabilities = _find_metric(facts, RATIO_TAGS["current_liabilities"], period_end)
    equity = _find_metric(facts, RATIO_TAGS["stockholders_equity"], period_end)
    long_term_debt = _find_metric(facts, RATIO_TAGS["long_term_debt"], period_end)

    if current_assets is not None and current_liabilities is not None and current_liabilities != 0:
        ratios["current_ratio"] = round(current_assets / current_liabilities, 4)

    if long_term_debt is not None and equity is not None and equity != 0:
        ratios["debt_to_equity"] = round(long_term_debt / equity, 4)

    return ratios
