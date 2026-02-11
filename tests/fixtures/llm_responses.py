"""Sample LLM responses for mocking COGNIT-FUNDAMENTAL tests.

Contains realistic JSON outputs that the Anthropic API would return,
as well as malformed responses for error-handling tests.
"""

# A well-formed, complete response from the LLM
VALID_FUNDAMENTAL_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-AAPL-2026Q1-MARGIN-EXPANSION",
            "ticker": "AAPL",
            "thesis_summary": "Apple services revenue mix shift driving operating margin expansion to 32%+ over next 2 quarters",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.72,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": "12345678-1234-5678-1234-567812345678",
                    "field_path": "payload.revenue",
                    "observation": "Revenue grew 12% YoY, exceeding consensus by 3%, with services growing 24%",
                    "weight": 0.8,
                },
                {
                    "source_fragment_id": "eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee",
                    "field_path": "payload.close",
                    "observation": "Stock trading at 26x forward P/E, below 5-year average of 28x despite improving margins",
                    "weight": 0.5,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Gross margin drops below 44% indicating services mix shift stalling",
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
                    "description": "Stock price drops below $165 (20% drawdown from current)",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 165.0,
                    "current_value": None,
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

# Response with multiple beliefs for the same ticker
MULTI_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-NVDA-2026Q1-AI-CAPEX",
            "ticker": "NVDA",
            "thesis_summary": "NVIDIA AI data center revenue acceleration as hyperscaler capex increases 40% YoY",
            "direction": "LONG",
            "magnitude": "LARGE",
            "raw_confidence": 0.82,
            "time_horizon_days": 120,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.revenue",
                    "observation": "Data center revenue up 110% YoY, exceeding all estimates",
                    "weight": 0.9,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Data center revenue growth drops below 50% YoY",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "dc_revenue_growth_yoy",
                    "operator": "LT",
                    "threshold": 0.50,
                },
                {
                    "description": "Gross margin falls below 70%",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 70.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        },
        {
            "thesis_id": "FUND-NVDA-2026Q1-SUPPLY-RISK",
            "ticker": "NVDA",
            "thesis_summary": "Supply constraint risk for H200/B100 GPUs could limit near-term upside",
            "direction": "NEUTRAL",
            "magnitude": "SMALL",
            "raw_confidence": 0.45,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.eps",
                    "observation": "Despite strong revenue, lead times extended to 6+ months",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Inventory-to-revenue ratio drops below 0.15",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "inventory_to_revenue",
                    "operator": "LT",
                    "threshold": 0.15,
                },
                {
                    "description": "Close price exceeds $1000",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "GT",
                    "threshold": 1000.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "NONE",
            },
        },
    ]
}

# Malformed response: missing required fields
MALFORMED_RESPONSE_NO_BELIEFS = {
    "analysis": "Apple is doing well",
    "recommendation": "Buy",
}

# Malformed response: beliefs array is empty
EMPTY_BELIEFS_RESPONSE = {
    "beliefs": [],
}

# Response with vague, non-machine-evaluable invalidation conditions
VAGUE_INVALIDATION_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-AAPL-2026Q1-VAGUE",
            "ticker": "AAPL",
            "thesis_summary": "Apple may face headwinds",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.55,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "12345678-1234-5678-1234-567812345678",
                    "field_path": "payload.revenue",
                    "observation": "Revenue growth slowing",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    # Has metric but missing operator — should be skipped
                    "description": "If fundamentals improve",
                    "metric": "revenue_growth",
                    "threshold": 0.10,
                },
                {
                    # Has operator but missing metric — should be skipped
                    "description": "If stock goes up a lot",
                    "operator": "GT",
                    "threshold": 200.0,
                },
                {
                    # Valid condition — should be kept
                    "description": "Revenue growth exceeds 10%",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "revenue_growth_yoy",
                    "operator": "GT",
                    "threshold": 0.10,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "NONE",
            },
        }
    ]
}

# Response with invalid evidence fragment IDs
BAD_EVIDENCE_REFS_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-AAPL-2026Q1-BADREF",
            "ticker": "AAPL",
            "thesis_summary": "Test thesis with bad evidence refs",
            "direction": "LONG",
            "magnitude": "SMALL",
            "raw_confidence": 0.6,
            "time_horizon_days": 90,
            "evidence": [
                {
                    # Not a valid UUID
                    "source_fragment_id": "not-a-uuid",
                    "field_path": "payload.close",
                    "observation": "Something",
                    "weight": 0.5,
                },
                {
                    # Valid UUID format
                    "source_fragment_id": "12345678-1234-5678-1234-567812345678",
                    "field_path": "payload.revenue",
                    "observation": "Revenue looks good",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Price drops below 150",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 150.0,
                },
                {
                    "description": "EPS drops below 5.0",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "eps",
                    "operator": "LT",
                    "threshold": 5.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        }
    ]
}

# Response with out-of-range confidence
OUT_OF_RANGE_CONFIDENCE_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "FUND-AAPL-2026Q1-HIGHCONF",
            "ticker": "AAPL",
            "thesis_summary": "Extremely confident thesis",
            "direction": "LONG",
            "magnitude": "LARGE",
            "raw_confidence": 1.5,  # Out of range — should be clamped to 1.0
            "time_horizon_days": 90,
            "evidence": [],
            "invalidation_conditions": [
                {
                    "description": "Price below 100",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 100.0,
                },
                {
                    "description": "Revenue below 50B",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "revenue",
                    "operator": "LT",
                    "threshold": 50_000_000_000,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "NONE",
            },
        }
    ]
}
