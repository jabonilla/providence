"""Golden reference BeliefObjects for integration testing.

5 realistic BeliefObjects — one per stock (AAPL, NVDA, JPM, JNJ, TSLA)
— representing the kind of output COGNIT-FUNDAMENTAL should produce.

These serve as the canonical reference for all future sessions.
"""

from tests.fixtures.sample_fragments import (
    AAPL_FILING_ID,
    AAPL_PRICE_ID,
    JPM_FILING_ID,
    JPM_PRICE_ID,
    JNJ_FILING_ID,
    JNJ_PRICE_ID,
    NVDA_FILING_ID,
    NVDA_PRICE_ID,
    TSLA_8K_ID,
    TSLA_PRICE_ID,
)

# ---------------------------------------------------------------------------
# Raw LLM response dicts — used as mock return values
# ---------------------------------------------------------------------------

AAPL_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-AAPL-2026Q1-MARGIN-EXPANSION",
            "ticker": "AAPL",
            "thesis_summary": "Apple services mix shift accelerating, driving operating margin toward 33%+ as services revenue grows 24% YoY vs 12% overall",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.72,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": str(AAPL_FILING_ID),
                    "field_path": "payload.services_revenue_growth_yoy",
                    "observation": "Services revenue growing 24% YoY — 2x overall growth rate, highest margin segment",
                    "weight": 0.85,
                },
                {
                    "source_fragment_id": str(AAPL_FILING_ID),
                    "field_path": "payload.gross_margin_pct",
                    "observation": "Gross margin at 46.5%, up from 44.1% a year ago, driven by services mix",
                    "weight": 0.7,
                },
                {
                    "source_fragment_id": str(AAPL_PRICE_ID),
                    "field_path": "payload.close",
                    "observation": "Stock at $237.80, trading at 27x forward P/E — below 5-year average of 29x",
                    "weight": 0.4,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Gross margin drops below 44%, indicating services mix shift stalling",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 44.0,
                    "current_value": 46.5,
                },
                {
                    "description": "Services revenue growth decelerates below 15% YoY",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "services_revenue_growth_yoy",
                    "operator": "LT",
                    "threshold": 0.15,
                    "current_value": 0.24,
                },
                {
                    "description": "Stock price drops below $195 — 18% drawdown from current",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 195.0,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        }
    ]
}

NVDA_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-NVDA-2026Q1-AI-CAPEX-CYCLE",
            "ticker": "NVDA",
            "thesis_summary": "NVIDIA data center revenue acceleration sustained as hyperscaler AI capex increases 40%+ YoY with no signs of deceleration",
            "direction": "LONG",
            "magnitude": "LARGE",
            "raw_confidence": 0.78,
            "time_horizon_days": 120,
            "evidence": [
                {
                    "source_fragment_id": str(NVDA_FILING_ID),
                    "field_path": "payload.dc_revenue_growth_yoy",
                    "observation": "Data center revenue up 110% YoY, representing 86% of total revenue",
                    "weight": 0.9,
                },
                {
                    "source_fragment_id": str(NVDA_FILING_ID),
                    "field_path": "payload.gross_margin_pct",
                    "observation": "Gross margins at 74.8% — demonstrates pricing power and competitive moat",
                    "weight": 0.7,
                },
                {
                    "source_fragment_id": str(NVDA_PRICE_ID),
                    "field_path": "payload.volume",
                    "observation": "Elevated volume at 45.7M shares suggests strong institutional interest",
                    "weight": 0.35,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Data center revenue growth drops below 50% YoY",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "dc_revenue_growth_yoy",
                    "operator": "LT",
                    "threshold": 0.50,
                    "current_value": 1.10,
                },
                {
                    "description": "Gross margin falls below 70% indicating pricing pressure",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 70.0,
                    "current_value": 74.8,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        }
    ]
}

JPM_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-JPM-2026Q1-NII-PEAK",
            "ticker": "JPM",
            "thesis_summary": "JPMorgan net interest income may have peaked as Fed rate cuts compress margins, offset partially by strong trading revenue",
            "direction": "NEUTRAL",
            "magnitude": "SMALL",
            "raw_confidence": 0.58,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": str(JPM_FILING_ID),
                    "field_path": "payload.net_interest_income",
                    "observation": "NII at $23.5B, strong but likely to decline as rates fall through 2026",
                    "weight": 0.75,
                },
                {
                    "source_fragment_id": str(JPM_FILING_ID),
                    "field_path": "payload.provision_for_credit_losses",
                    "observation": "PCL at $2.1B — elevated but stable, suggesting credit quality holding",
                    "weight": 0.6,
                },
                {
                    "source_fragment_id": str(JPM_PRICE_ID),
                    "field_path": "payload.close",
                    "observation": "Stock at $217.65, near all-time highs — limited margin of safety",
                    "weight": 0.5,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Net interest income drops below $20B quarterly",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "net_interest_income",
                    "operator": "LT",
                    "threshold": 20_000_000_000,
                    "current_value": 23_500_000_000,
                },
                {
                    "description": "Tier 1 capital ratio falls below 13%",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "tier1_capital_ratio",
                    "operator": "LT",
                    "threshold": 0.13,
                    "current_value": 0.152,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Financials",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "MACRO",
            },
        }
    ]
}

JNJ_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-JNJ-2026Q1-PIPELINE-VALUE",
            "ticker": "JNJ",
            "thesis_summary": "JNJ Phase 3 pipeline undervalued with 12 drugs in late stage; R&D spend of $3.8B/quarter positions for accelerating revenue growth in 2027-2028",
            "direction": "LONG",
            "magnitude": "SMALL",
            "raw_confidence": 0.61,
            "time_horizon_days": 180,
            "evidence": [
                {
                    "source_fragment_id": str(JNJ_FILING_ID),
                    "field_path": "payload.pipeline_drugs_phase3",
                    "observation": "12 Phase 3 drugs — each approval could add $1-3B annual revenue",
                    "weight": 0.8,
                },
                {
                    "source_fragment_id": str(JNJ_FILING_ID),
                    "field_path": "payload.gross_margin_pct",
                    "observation": "Gross margin at 69.2% — strong pricing power supports R&D investment",
                    "weight": 0.5,
                },
                {
                    "source_fragment_id": str(JNJ_PRICE_ID),
                    "field_path": "payload.close",
                    "observation": "Stock at $163.25, trading at 15x forward P/E — discount to pharma peers",
                    "weight": 0.45,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Revenue growth turns negative (below 0%)",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "revenue_growth_yoy",
                    "operator": "LT",
                    "threshold": 0.0,
                    "current_value": 0.04,
                },
                {
                    "description": "R&D expense drops below $3B/quarter — signals pipeline cuts",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "rd_expense",
                    "operator": "LT",
                    "threshold": 3_000_000_000,
                    "current_value": 3_800_000_000,
                },
                {
                    "description": "Stock price drops below $140 — 14% drawdown",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 140.0,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Healthcare",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "NONE",
            },
        }
    ]
}

TSLA_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-TSLA-2026Q1-ENERGY-INFLECTION",
            "ticker": "TSLA",
            "thesis_summary": "Tesla energy storage business at inflection point with 120% YoY revenue growth; market ignoring high-margin energy segment worth $80-120B standalone",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.65,
            "time_horizon_days": 120,
            "evidence": [
                {
                    "source_fragment_id": str(TSLA_8K_ID),
                    "field_path": "payload.energy_revenue",
                    "observation": "Energy storage revenue $3.2B (+120% YoY), approaching 12% of total revenue",
                    "weight": 0.85,
                },
                {
                    "source_fragment_id": str(TSLA_8K_ID),
                    "field_path": "payload.automotive_margin_pct",
                    "observation": "Automotive margins at 18.2%, stabilizing after 2024-2025 price cuts",
                    "weight": 0.6,
                },
                {
                    "source_fragment_id": str(TSLA_PRICE_ID),
                    "field_path": "payload.volume",
                    "observation": "Volume at 78.9M shares — 2x average, strong post-earnings momentum",
                    "weight": 0.4,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Automotive margins drop below 15%",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "automotive_margin_pct",
                    "operator": "LT",
                    "threshold": 15.0,
                    "current_value": 18.2,
                },
                {
                    "description": "Energy storage revenue growth decelerates below 50% YoY",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "energy_revenue_growth_yoy",
                    "operator": "LT",
                    "threshold": 0.50,
                },
                {
                    "description": "Stock price drops below $320 — 19% drawdown",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 320.0,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Consumer Discretionary",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# Convenience: all responses indexed by ticker
# ---------------------------------------------------------------------------
BELIEF_RESPONSES_BY_TICKER = {
    "AAPL": AAPL_BELIEF_RESPONSE,
    "NVDA": NVDA_BELIEF_RESPONSE,
    "JPM": JPM_BELIEF_RESPONSE,
    "JNJ": JNJ_BELIEF_RESPONSE,
    "TSLA": TSLA_BELIEF_RESPONSE,
}
