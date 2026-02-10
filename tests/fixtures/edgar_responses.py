"""Sample SEC EDGAR API responses for testing PERCEPT-FILING.

Replicates the structure of real EDGAR responses so tests
can run without making actual API calls.
"""


def filing_10q_aapl() -> dict:
    """Sample EDGAR filing metadata + XBRL data for AAPL 10-Q."""
    return {
        "filed_date": "2026-01-30",
        "period_of_report": "2025-12-31",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "accession_number": "0000320193-26-000012",
        "event_type": None,
        "raw_text_excerpt": "Apple Inc. reported record revenue of $124.3 billion for the quarter.",
        "xbrl_data": {
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 124_300_000_000, "form": "10-Q"},
                                {"end": "2025-09-30", "val": 94_900_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 33_900_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "EarningsPerShareDiluted": {
                        "units": {
                            "USD/shares": [
                                {"end": "2025-12-31", "val": 2.18, "form": "10-Q"},
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 352_600_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "Liabilities": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 274_800_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 39_100_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "AssetsCurrent": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 143_600_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                    "LiabilitiesCurrent": {
                        "units": {
                            "USD": [
                                {"end": "2025-12-31", "val": 153_900_000_000, "form": "10-Q"},
                            ]
                        }
                    },
                }
            }
        },
    }


def filing_10k_aapl() -> dict:
    """Sample EDGAR filing metadata + XBRL data for AAPL 10-K."""
    return {
        "filed_date": "2025-10-31",
        "period_of_report": "2025-09-30",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "accession_number": "0000320193-25-000098",
        "event_type": None,
        "raw_text_excerpt": "Apple Inc. annual report for fiscal year 2025.",
        "xbrl_data": {
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {"end": "2025-09-30", "val": 391_000_000_000, "form": "10-K"},
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {"end": "2025-09-30", "val": 97_000_000_000, "form": "10-K"},
                            ]
                        }
                    },
                }
            }
        },
    }


def filing_8k_aapl() -> dict:
    """Sample EDGAR 8-K event filing for AAPL."""
    return {
        "filed_date": "2026-01-15",
        "period_of_report": "2026-01-15",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "accession_number": "0000320193-26-000005",
        "event_type": "CEO_TRANSITION",
        "event_description": "Apple announces leadership transition in services division.",
        "material_impact": True,
        "raw_text_excerpt": "Item 5.02 - Departure of Directors or Certain Officers.",
    }


def filing_missing_xbrl() -> dict:
    """Filing with metadata but no XBRL data (PARTIAL status expected)."""
    return {
        "filed_date": "2026-01-30",
        "period_of_report": "2025-12-31",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "xbrl_data": {
            "facts": {
                "us-gaap": {}
            }
        },
    }


def filing_empty() -> dict:
    """Completely empty filing data (QUARANTINED expected)."""
    return {}


def filing_8k_no_event_type() -> dict:
    """8-K filing without event_type (PARTIAL expected)."""
    return {
        "filed_date": "2026-01-15",
        "period_of_report": "2026-01-15",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
    }
