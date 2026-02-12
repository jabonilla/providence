"""Realistic sample MarketStateFragments for integration testing.

10 fragments across 5 stocks (AAPL, NVDA, JPM, JNJ, TSLA) with a mix
of PRICE_OHLCV, FILING_10Q, and FILING_8K data types.

These serve as the canonical reference fixtures for all future sessions.
"""

from datetime import datetime, timedelta, timezone
from uuid import UUID

from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment

NOW = datetime(2026, 2, 9, 16, 0, 0, tzinfo=timezone.utc)

# ---------------------------------------------------------------------------
# Fixed UUIDs for deterministic cross-referencing
# ---------------------------------------------------------------------------
AAPL_PRICE_ID = UUID("a0000001-0001-0001-0001-000000000001")
AAPL_FILING_ID = UUID("a0000001-0001-0001-0001-000000000002")
NVDA_PRICE_ID = UUID("b0000002-0002-0002-0002-000000000001")
NVDA_FILING_ID = UUID("b0000002-0002-0002-0002-000000000002")
JPM_PRICE_ID = UUID("c0000003-0003-0003-0003-000000000001")
JPM_FILING_ID = UUID("c0000003-0003-0003-0003-000000000002")
JNJ_PRICE_ID = UUID("d0000004-0004-0004-0004-000000000001")
JNJ_FILING_ID = UUID("d0000004-0004-0004-0004-000000000002")
TSLA_PRICE_ID = UUID("e0000005-0005-0005-0005-000000000001")
TSLA_8K_ID = UUID("e0000005-0005-0005-0005-000000000002")


# ---------------------------------------------------------------------------
# Price fragments
# ---------------------------------------------------------------------------
AAPL_PRICE_FRAGMENT = MarketStateFragment(
    fragment_id=AAPL_PRICE_ID,
    agent_id="PERCEPT-PRICE",
    timestamp=NOW - timedelta(hours=1),
    source_timestamp=NOW - timedelta(hours=2),
    entity="AAPL",
    data_type=DataType.PRICE_OHLCV,
    schema_version="1.0.0",
    source_hash="polygon-aapl-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "open": 235.50,
        "high": 238.20,
        "low": 234.10,
        "close": 237.80,
        "volume": 62_345_678,
        "vwap": 236.45,
        "num_trades": 485_231,
        "timeframe": "1D",
    },
)

NVDA_PRICE_FRAGMENT = MarketStateFragment(
    fragment_id=NVDA_PRICE_ID,
    agent_id="PERCEPT-PRICE",
    timestamp=NOW - timedelta(hours=1),
    source_timestamp=NOW - timedelta(hours=2),
    entity="NVDA",
    data_type=DataType.PRICE_OHLCV,
    schema_version="1.0.0",
    source_hash="polygon-nvda-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "open": 892.30,
        "high": 915.70,
        "low": 888.10,
        "close": 912.45,
        "volume": 45_678_123,
        "vwap": 901.20,
        "num_trades": 612_844,
        "timeframe": "1D",
    },
)

JPM_PRICE_FRAGMENT = MarketStateFragment(
    fragment_id=JPM_PRICE_ID,
    agent_id="PERCEPT-PRICE",
    timestamp=NOW - timedelta(hours=1),
    source_timestamp=NOW - timedelta(hours=2),
    entity="JPM",
    data_type=DataType.PRICE_OHLCV,
    schema_version="1.0.0",
    source_hash="polygon-jpm-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "open": 215.80,
        "high": 218.40,
        "low": 214.90,
        "close": 217.65,
        "volume": 12_456_789,
        "vwap": 216.70,
        "num_trades": 198_432,
        "timeframe": "1D",
    },
)

JNJ_PRICE_FRAGMENT = MarketStateFragment(
    fragment_id=JNJ_PRICE_ID,
    agent_id="PERCEPT-PRICE",
    timestamp=NOW - timedelta(hours=1),
    source_timestamp=NOW - timedelta(hours=2),
    entity="JNJ",
    data_type=DataType.PRICE_OHLCV,
    schema_version="1.0.0",
    source_hash="polygon-jnj-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "open": 162.40,
        "high": 163.80,
        "low": 161.50,
        "close": 163.25,
        "volume": 8_234_567,
        "vwap": 162.90,
        "num_trades": 124_567,
        "timeframe": "1D",
    },
)

TSLA_PRICE_FRAGMENT = MarketStateFragment(
    fragment_id=TSLA_PRICE_ID,
    agent_id="PERCEPT-PRICE",
    timestamp=NOW - timedelta(hours=1),
    source_timestamp=NOW - timedelta(hours=2),
    entity="TSLA",
    data_type=DataType.PRICE_OHLCV,
    schema_version="1.0.0",
    source_hash="polygon-tsla-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "open": 385.20,
        "high": 398.90,
        "low": 380.10,
        "close": 395.60,
        "volume": 78_901_234,
        "vwap": 390.45,
        "num_trades": 892_345,
        "timeframe": "1D",
    },
)


# ---------------------------------------------------------------------------
# Filing fragments (10-Q for most, 8-K for TSLA)
# ---------------------------------------------------------------------------
AAPL_FILING_FRAGMENT = MarketStateFragment(
    fragment_id=AAPL_FILING_ID,
    agent_id="PERCEPT-FILING",
    timestamp=NOW - timedelta(hours=12),
    source_timestamp=NOW - timedelta(days=3),
    entity="AAPL",
    data_type=DataType.FILING_10Q,
    schema_version="1.0.0",
    source_hash="edgar-aapl-10q-2025q4",
    validation_status=ValidationStatus.VALID,
    payload={
        "filing_type": "10-Q",
        "company_name": "Apple Inc.",
        "cik": "0000320193",
        "ticker": "AAPL",
        "filed_date": "2026-01-30",
        "period_of_report": "2025-12-31",
        "revenue": 124_300_000_000,
        "net_income": 33_900_000_000,
        "eps": 2.18,
        "total_assets": 352_600_000_000,
        "total_liabilities": 274_800_000_000,
        "operating_cash_flow": 39_100_000_000,
        "gross_margin_pct": 46.5,
        "operating_margin_pct": 31.2,
        "revenue_growth_yoy": 0.12,
        "services_revenue": 28_400_000_000,
        "services_revenue_growth_yoy": 0.24,
    },
)

NVDA_FILING_FRAGMENT = MarketStateFragment(
    fragment_id=NVDA_FILING_ID,
    agent_id="PERCEPT-FILING",
    timestamp=NOW - timedelta(hours=24),
    source_timestamp=NOW - timedelta(days=7),
    entity="NVDA",
    data_type=DataType.FILING_10Q,
    schema_version="1.0.0",
    source_hash="edgar-nvda-10q-2025q4",
    validation_status=ValidationStatus.VALID,
    payload={
        "filing_type": "10-Q",
        "company_name": "NVIDIA Corporation",
        "cik": "0001045810",
        "ticker": "NVDA",
        "filed_date": "2026-01-25",
        "period_of_report": "2025-12-31",
        "revenue": 38_500_000_000,
        "net_income": 19_200_000_000,
        "eps": 7.84,
        "total_assets": 85_200_000_000,
        "total_liabilities": 28_400_000_000,
        "operating_cash_flow": 21_300_000_000,
        "gross_margin_pct": 74.8,
        "operating_margin_pct": 62.1,
        "revenue_growth_yoy": 0.94,
        "dc_revenue": 33_100_000_000,
        "dc_revenue_growth_yoy": 1.10,
    },
)

JPM_FILING_FRAGMENT = MarketStateFragment(
    fragment_id=JPM_FILING_ID,
    agent_id="PERCEPT-FILING",
    timestamp=NOW - timedelta(hours=36),
    source_timestamp=NOW - timedelta(days=10),
    entity="JPM",
    data_type=DataType.FILING_10Q,
    schema_version="1.0.0",
    source_hash="edgar-jpm-10q-2025q4",
    validation_status=ValidationStatus.VALID,
    payload={
        "filing_type": "10-Q",
        "company_name": "JPMorgan Chase & Co.",
        "cik": "0000019617",
        "ticker": "JPM",
        "filed_date": "2026-01-15",
        "period_of_report": "2025-12-31",
        "revenue": 44_800_000_000,
        "net_income": 13_200_000_000,
        "eps": 4.48,
        "total_assets": 4_120_000_000_000,
        "total_liabilities": 3_780_000_000_000,
        "operating_cash_flow": 18_900_000_000,
        "net_interest_income": 23_500_000_000,
        "provision_for_credit_losses": 2_100_000_000,
        "return_on_equity": 0.175,
        "tier1_capital_ratio": 0.152,
    },
)

JNJ_FILING_FRAGMENT = MarketStateFragment(
    fragment_id=JNJ_FILING_ID,
    agent_id="PERCEPT-FILING",
    timestamp=NOW - timedelta(hours=48),
    source_timestamp=NOW - timedelta(days=14),
    entity="JNJ",
    data_type=DataType.FILING_10Q,
    schema_version="1.0.0",
    source_hash="edgar-jnj-10q-2025q4",
    validation_status=ValidationStatus.VALID,
    payload={
        "filing_type": "10-Q",
        "company_name": "Johnson & Johnson",
        "cik": "0000200406",
        "ticker": "JNJ",
        "filed_date": "2026-01-22",
        "period_of_report": "2025-12-31",
        "revenue": 22_500_000_000,
        "net_income": 5_100_000_000,
        "eps": 2.12,
        "total_assets": 178_400_000_000,
        "total_liabilities": 102_300_000_000,
        "operating_cash_flow": 6_800_000_000,
        "gross_margin_pct": 69.2,
        "operating_margin_pct": 28.4,
        "revenue_growth_yoy": 0.04,
        "rd_expense": 3_800_000_000,
        "pipeline_drugs_phase3": 12,
    },
)

TSLA_8K_FRAGMENT = MarketStateFragment(
    fragment_id=TSLA_8K_ID,
    agent_id="PERCEPT-FILING",
    timestamp=NOW - timedelta(hours=6),
    source_timestamp=NOW - timedelta(hours=8),
    entity="TSLA",
    data_type=DataType.FILING_8K,
    schema_version="1.0.0",
    source_hash="edgar-tsla-8k-20260209",
    validation_status=ValidationStatus.VALID,
    payload={
        "filing_type": "8-K",
        "company_name": "Tesla, Inc.",
        "cik": "0001318605",
        "ticker": "TSLA",
        "filed_date": "2026-02-09",
        "event_type": "earnings_release",
        "event_description": "Tesla Q4 2025 earnings: Revenue $27.8B (+18% YoY), EPS $0.95 vs $0.82 consensus. Automotive margins 18.2%, energy storage revenue +120% YoY.",
        "material_impact": True,
        "revenue": 27_800_000_000,
        "eps": 0.95,
        "automotive_margin_pct": 18.2,
        "energy_revenue": 3_200_000_000,
        "deliveries": 512_000,
    },
)


# ---------------------------------------------------------------------------
# Convenience collections
# ---------------------------------------------------------------------------
ALL_PRICE_FRAGMENTS = [
    AAPL_PRICE_FRAGMENT,
    NVDA_PRICE_FRAGMENT,
    JPM_PRICE_FRAGMENT,
    JNJ_PRICE_FRAGMENT,
    TSLA_PRICE_FRAGMENT,
]

ALL_FILING_FRAGMENTS = [
    AAPL_FILING_FRAGMENT,
    NVDA_FILING_FRAGMENT,
    JPM_FILING_FRAGMENT,
    JNJ_FILING_FRAGMENT,
    TSLA_8K_FRAGMENT,
]

ALL_FRAGMENTS = ALL_PRICE_FRAGMENTS + ALL_FILING_FRAGMENTS

# Fragments grouped by ticker for easy access
FRAGMENTS_BY_TICKER = {
    "AAPL": [AAPL_PRICE_FRAGMENT, AAPL_FILING_FRAGMENT],
    "NVDA": [NVDA_PRICE_FRAGMENT, NVDA_FILING_FRAGMENT],
    "JPM": [JPM_PRICE_FRAGMENT, JPM_FILING_FRAGMENT],
    "JNJ": [JNJ_PRICE_FRAGMENT, JNJ_FILING_FRAGMENT],
    "TSLA": [TSLA_PRICE_FRAGMENT, TSLA_8K_FRAGMENT],
}

# Fragment ID maps for evidence linking in tests
FRAGMENT_ID_MAP = {
    "AAPL_PRICE": AAPL_PRICE_ID,
    "AAPL_FILING": AAPL_FILING_ID,
    "NVDA_PRICE": NVDA_PRICE_ID,
    "NVDA_FILING": NVDA_FILING_ID,
    "JPM_PRICE": JPM_PRICE_ID,
    "JPM_FILING": JPM_FILING_ID,
    "JNJ_PRICE": JNJ_PRICE_ID,
    "JNJ_FILING": JNJ_FILING_ID,
    "TSLA_PRICE": TSLA_PRICE_ID,
    "TSLA_8K": TSLA_8K_ID,
}
