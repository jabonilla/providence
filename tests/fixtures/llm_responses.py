"""Sample LLM responses for mocking Research Agent tests.

Contains realistic JSON outputs that the Anthropic API would return,
as well as malformed responses for error-handling tests.

Covers: COGNIT-FUNDAMENTAL, COGNIT-MACRO, COGNIT-NARRATIVE, COGNIT-EVENT, COGNIT-CROSSSEC, REGIME-NARR, DECIDE-SYNTH, COGNIT-EXIT
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

# ---------------------------------------------------------------------------
# COGNIT-MACRO mock responses
# ---------------------------------------------------------------------------

# Valid macro response: yield curve inversion thesis
VALID_MACRO_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "MACRO-YIELDCURVE-20260212-INVERSION",
            "ticker": "SPY",
            "thesis_summary": "2-10Y inversion at -70bps signals elevated recession risk; favor defensive positioning via reduced equity beta",
            "direction": "SHORT",
            "magnitude": "MODERATE",
            "raw_confidence": 0.68,
            "time_horizon_days": 120,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaa01-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.spread_2s10s",
                    "observation": "2-10Y spread at -70bps, deepest inversion in 6 months",
                    "weight": 0.9,
                },
                {
                    "source_fragment_id": "aaaaaa02-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.spread_bps",
                    "observation": "IG CDS spreads widening +20bps in 5 days confirms risk-off shift",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "2-10Y spread normalizes above 0 (curve dis-inverts)",
                    "data_source_agent": "PERCEPT-MACRO",
                    "metric": "spread_2s10s_bps",
                    "operator": "GT",
                    "threshold": 0.0,
                },
                {
                    "description": "CPI falls below 2% easing inflation fears",
                    "data_source_agent": "PERCEPT-MACRO",
                    "metric": "cpi_yoy_pct",
                    "operator": "LT",
                    "threshold": 2.0,
                },
                {
                    "description": "SPY rallies above 600 invalidating bearish regime",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "spy_close",
                    "operator": "GT",
                    "threshold": 600.0,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Macro",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "MACRO",
            },
        }
    ]
}

# Multi-belief macro response: inversion + credit stress
MULTI_MACRO_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "MACRO-YIELDCURVE-20260212-DEFENSIVE",
            "ticker": "XLU",
            "thesis_summary": "Inverted yield curve favors defensive utilities sector",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.62,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaa01-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.spread_2s10s",
                    "observation": "2-10Y inversion supports defensive rotation",
                    "weight": 0.8,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "2-10Y spread steepens above +50bps",
                    "data_source_agent": "PERCEPT-MACRO",
                    "metric": "spread_2s10s_bps",
                    "operator": "GT",
                    "threshold": 50.0,
                },
                {
                    "description": "Utilities ETF breaks below $60",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "xlu_close",
                    "operator": "LT",
                    "threshold": 60.0,
                },
            ],
            "metadata": {
                "sector": "Utilities",
                "market_cap_bucket": "LARGE",
                "catalyst_type": "MACRO",
            },
        },
        {
            "thesis_id": "MACRO-CREDIT-20260212-WIDENING",
            "ticker": "HYG",
            "thesis_summary": "HY CDS widening signals credit stress; short high-yield exposure",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.55,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaa02-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.spread_bps",
                    "observation": "HY CDS spreads at 450bps, up 40bps in 10 days",
                    "weight": 0.9,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "HY CDS spreads compress below 350bps",
                    "data_source_agent": "PERCEPT-CDS",
                    "metric": "cds_hy_spread_bps",
                    "operator": "LT",
                    "threshold": 350.0,
                },
                {
                    "description": "HYG ETF rallies above $80",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "hyg_close",
                    "operator": "GT",
                    "threshold": 80.0,
                },
            ],
            "metadata": {
                "sector": "Macro",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "MACRO",
            },
        },
    ]
}

# Empty beliefs array (macro variant)
MACRO_EMPTY_BELIEFS_RESPONSE = {
    "beliefs": [],
}

# Macro response with vague invalidation conditions (< 2 valid → belief dropped)
MACRO_VAGUE_INVALIDATION_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "MACRO-MOMENTUM-20260212-GROWTH",
            "ticker": "QQQ",
            "thesis_summary": "Economic growth may slow",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.50,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": "aaaaaa03-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                    "field_path": "payload.value",
                    "observation": "GDP growth decelerating",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "If the economy recovers",
                    "metric": "gdp_growth_pct",
                    "threshold": 3.0,
                },
                {
                    "description": "If markets rally",
                    "operator": "GT",
                    "threshold": 500.0,
                },
                {
                    "description": "GDP growth exceeds 3%",
                    "data_source_agent": "PERCEPT-MACRO",
                    "metric": "gdp_growth_pct",
                    "operator": "GT",
                    "threshold": 3.0,
                },
            ],
            "metadata": {
                "sector": "Macro",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "MACRO",
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# COGNIT-NARRATIVE mock responses
# ---------------------------------------------------------------------------

# Valid narrative response: management tone shift
VALID_NARRATIVE_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "NARR-AAPL-20260212-TONE",
            "ticker": "AAPL",
            "thesis_summary": "CEO shifted from hedged language to definitive 'will exceed' on services targets; tone upgrade signals beat likelihood",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.62,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "cccccc01-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.tone_score",
                    "observation": "Tone positivity score rose from 0.52 to 0.74 quarter-over-quarter",
                    "weight": 0.9,
                },
                {
                    "source_fragment_id": "cccccc02-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.sentiment_score",
                    "observation": "News sentiment turned positive after earnings call; +0.65 vs prior -0.1",
                    "weight": 0.7,
                },
                {
                    "source_fragment_id": "cccccc03-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.event_type",
                    "observation": "8K confirms stock buyback authorization increase, aligning with bullish tone",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Sentiment reverses negative on next earnings or news cycle",
                    "data_source_agent": "PERCEPT-NEWS",
                    "metric": "sentiment_score",
                    "operator": "LT",
                    "threshold": -0.2,
                    "current_value": 0.65,
                },
                {
                    "description": "Stock price falls below $170, contradicting bullish narrative",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 170.0,
                    "current_value": 195.0,
                },
                {
                    "description": "Management tone reverts below 0.5 on next earnings call",
                    "data_source_agent": "PERCEPT-NEWS",
                    "metric": "tone_positivity_score",
                    "operator": "LT",
                    "threshold": 0.5,
                    "current_value": 0.74,
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

# Multi-belief narrative response: guidance divergence + peer divergence
MULTI_NARRATIVE_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "NARR-MSFT-20260212-GUIDANCE",
            "ticker": "MSFT",
            "thesis_summary": "Guidance maintained but hedging language increased 3x; implicit downside risk; SHORT 45 days",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.58,
            "time_horizon_days": 45,
            "evidence": [
                {
                    "source_fragment_id": "cccccc01-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.hedge_word_count",
                    "observation": "Hedging language ('may', 'expect', 'approximately') tripled vs prior quarter",
                    "weight": 0.85,
                },
                {
                    "source_fragment_id": "cccccc02-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.sentiment_score",
                    "observation": "Analyst sentiment neutral despite beat; skepticism about sustainability",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Guidance raised above $6.50 EPS (5% increase)",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "guidance_eps_delta_pct",
                    "operator": "GT",
                    "threshold": 5.0,
                    "current_value": 0.0,
                },
                {
                    "description": "Stock rallies above $440 invalidating bearish narrative",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "GT",
                    "threshold": 440.0,
                    "current_value": 420.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        },
        {
            "thesis_id": "NARR-MSFT-20260212-DIVERGENCE",
            "ticker": "MSFT",
            "thesis_summary": "MSFT narrative bullish while GOOG/META peers cautious on AI spend; positive divergence if correct",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.52,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": "cccccc01-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.tone_score",
                    "observation": "MSFT tone score 0.72 vs sector average 0.45; bullish outlier",
                    "weight": 0.8,
                },
                {
                    "source_fragment_id": "cccccc04-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.tone_score",
                    "observation": "GOOG peer tone score 0.38, below sector average; cautious positioning",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Peer sentiment spread closes; MSFT aligns with cautious sector",
                    "data_source_agent": "PERCEPT-NEWS",
                    "metric": "peer_sentiment_spread",
                    "operator": "LT",
                    "threshold": 0.1,
                    "current_value": 0.27,
                },
                {
                    "description": "MSFT stock underperforms sector by >5% over 30 days",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "relative_performance_30d_pct",
                    "operator": "LT",
                    "threshold": -5.0,
                    "current_value": 2.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        },
    ]
}

# Empty beliefs array (narrative variant)
NARRATIVE_EMPTY_BELIEFS_RESPONSE = {
    "beliefs": [],
}

# Narrative response with vague invalidation conditions (< 2 valid → belief dropped)
NARRATIVE_VAGUE_INVALIDATION_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "NARR-GOOG-20260212-DISCLOSURE",
            "ticker": "GOOG",
            "thesis_summary": "Unscheduled Friday evening 8K with new risk factor; disclosure timing suspicious",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.48,
            "time_horizon_days": 30,
            "evidence": [
                {
                    "source_fragment_id": "cccccc03-cccc-cccc-cccc-cccccccccccc",
                    "field_path": "payload.filing_date",
                    "observation": "8K filed Friday 6pm; atypical timing suggests management concern",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    # Has metric but missing operator — should be skipped
                    "description": "If risk factor is resolved",
                    "metric": "risk_factor_count",
                    "threshold": 0,
                },
                {
                    # Has operator but missing metric — should be skipped
                    "description": "If stock recovers",
                    "operator": "GT",
                    "threshold": 180.0,
                },
                {
                    # Valid condition — kept, but only 1 valid → belief dropped
                    "description": "Stock recovers above $175",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "GT",
                    "threshold": 175.0,
                    "current_value": 165.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EVENT",
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# COGNIT-EVENT mock responses
# ---------------------------------------------------------------------------

# Valid event response: M&A corporate action catalyst
VALID_EVENT_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "EVENT-AAPL-20260212-CORPACTION",
            "ticker": "AAPL",
            "thesis_summary": "Announced acquisition at 18% premium; arb spread 250bps; convergence expected over 21 days",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.72,
            "time_horizon_days": 21,
            "evidence": [
                {
                    "source_fragment_id": "bbbbbb01-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.deal_value",
                    "observation": "Acquisition priced at $185/share, 18% premium to closing",
                    "weight": 0.95,
                },
                {
                    "source_fragment_id": "bbbbbb03-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.close",
                    "observation": "Stock at $182, trading below deal price; arb opportunity",
                    "weight": 0.8,
                },
                {
                    "source_fragment_id": "bbbbbb02-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.sentiment_score",
                    "observation": "News sentiment positive; deal momentum confirmed",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Deal breaks; stock falls through $170",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 170.0,
                    "current_value": 182.0,
                },
                {
                    "description": "Regulatory approval unlikely; deal probability drops",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "deal_probability_pct",
                    "operator": "LT",
                    "threshold": 75.0,
                    "current_value": 95.0,
                },
                {
                    "description": "Arb spread widens beyond 400bps indicating deal risk",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "arb_spread_bps",
                    "operator": "GT",
                    "threshold": 400.0,
                    "current_value": 250.0,
                },
            ],
            "correlated_beliefs": [],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EVENT",
            },
        }
    ]
}

# Multi-belief event response: earnings miss SHORT + sentiment reversal LONG
MULTI_EVENT_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "EVENT-MSFT-20260212-EARNINGS",
            "ticker": "MSFT",
            "thesis_summary": "Earnings miss 8% below whisper; guidance cut 12%; short until stabilization in 5-10 days",
            "direction": "SHORT",
            "magnitude": "MODERATE",
            "raw_confidence": 0.68,
            "time_horizon_days": 8,
            "evidence": [
                {
                    "source_fragment_id": "bbbbbb01-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.eps_actual",
                    "observation": "EPS $2.85 vs whisper $3.10; miss by 8%",
                    "weight": 0.9,
                },
                {
                    "source_fragment_id": "bbbbbb02-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.sentiment_score",
                    "observation": "Sentiment collapsed from +0.6 to -0.4; analyst downgrades",
                    "weight": 0.85,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Stock rebounds to $415+ (2% recovery)",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "GT",
                    "threshold": 415.0,
                    "current_value": 405.0,
                },
                {
                    "description": "IV crush: implied vol falls below 0.22",
                    "data_source_agent": "PERCEPT-OPTIONS",
                    "metric": "implied_vol_atm",
                    "operator": "LT",
                    "threshold": 0.22,
                    "current_value": 0.35,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EVENT",
            },
        },
        {
            "thesis_id": "EVENT-MSFT-20260212-SENTIMENT",
            "ticker": "MSFT",
            "thesis_summary": "Sentiment reversal extreme; downgrades overdone; 3-5 day bounce expected",
            "direction": "LONG",
            "magnitude": "SMALL",
            "raw_confidence": 0.55,
            "time_horizon_days": 4,
            "evidence": [
                {
                    "source_fragment_id": "bbbbbb02-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.sentiment_score",
                    "observation": "Sentiment at -0.75, 5-year extreme; mean reversion likely",
                    "weight": 0.8,
                },
                {
                    "source_fragment_id": "bbbbbb04-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.put_call_ratio",
                    "observation": "Put/call ratio 2.1x; extreme fear vulnerable to short covering",
                    "weight": 0.75,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Further downside: stock breaks below $400",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 400.0,
                    "current_value": 405.0,
                },
                {
                    "description": "Sentiment remains depressed; no bounce within 5 days",
                    "data_source_agent": "PERCEPT-NEWS",
                    "metric": "sentiment_score",
                    "operator": "LT",
                    "threshold": -0.5,
                    "current_value": -0.75,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EVENT",
            },
        },
    ]
}

# Empty beliefs array (event variant)
EVENT_EMPTY_BELIEFS_RESPONSE = {
    "beliefs": [],
}

# Event response with vague invalidation conditions (< 2 valid → belief dropped)
EVENT_VAGUE_INVALIDATION_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "EVENT-GOOG-20260212-OPTIONS",
            "ticker": "GOOG",
            "thesis_summary": "Unusual options activity spike; IV at 3-month high; event risk imminent",
            "direction": "LONG",
            "magnitude": "SMALL",
            "raw_confidence": 0.50,
            "time_horizon_days": 5,
            "evidence": [
                {
                    "source_fragment_id": "bbbbbb04-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                    "field_path": "payload.implied_vol_atm",
                    "observation": "IV at 0.38; 3-month high signals event risk",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    # Has metric but missing operator — should be skipped
                    "description": "If IV drops",
                    "metric": "implied_vol_atm",
                    "threshold": 0.25,
                },
                {
                    # Has operator but missing metric — should be skipped
                    "description": "If stock rallies",
                    "operator": "GT",
                    "threshold": 200.0,
                },
                {
                    # Valid condition — kept, but only 1 valid → belief dropped
                    "description": "IV normalizes below 0.28",
                    "data_source_agent": "PERCEPT-OPTIONS",
                    "metric": "implied_vol_atm",
                    "operator": "LT",
                    "threshold": 0.28,
                    "current_value": 0.38,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EVENT",
            },
        }
    ]
}


# ---------------------------------------------------------------------------
# COGNIT-CROSSSEC mock responses
# ---------------------------------------------------------------------------

# Valid crosssec response: relative valuation dislocation
VALID_CROSSSEC_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "XSEC-AAPL-20260212-VALUATION",
            "ticker": "AAPL",
            "thesis_summary": "AAPL trading at 22x P/E vs sector avg 28x; 2.5 sigma discount despite superior margins; relative value opportunity",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.65,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "dddddd01-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.pe_ratio",
                    "observation": "AAPL P/E at 22x, 6 turns below sector average of 28x",
                    "weight": 0.9,
                },
                {
                    "source_fragment_id": "dddddd02-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.close",
                    "observation": "Stock underperforming peers by 8% over 30 days despite stronger fundamentals",
                    "weight": 0.75,
                },
                {
                    "source_fragment_id": "dddddd03-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.tone_score",
                    "observation": "Management tone more bullish than peer average; guidance language firmer",
                    "weight": 0.6,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Valuation discount closes; P/E gap narrows to within 1 turn",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "pe_ratio_vs_peer_avg",
                    "operator": "GT",
                    "threshold": -1.0,
                    "current_value": -6.0,
                },
                {
                    "description": "Stock underperforms peers by >15% in 30 days",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "relative_return_30d_pct",
                    "operator": "LT",
                    "threshold": -15.0,
                    "current_value": -8.0,
                },
                {
                    "description": "Operating margin drops below 28%, closing quality gap",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "operating_margin_pct",
                    "operator": "LT",
                    "threshold": 28.0,
                    "current_value": 32.5,
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

# Multi-belief crosssec response: fundamental divergence + mean reversion
MULTI_CROSSSEC_BELIEF_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "XSEC-MSFT-20260212-DIVERGENCE",
            "ticker": "MSFT",
            "thesis_summary": "MSFT operating margin expanding +200bps while GOOG/META contracting; widening quality gap not yet priced",
            "direction": "LONG",
            "magnitude": "MODERATE",
            "raw_confidence": 0.68,
            "time_horizon_days": 90,
            "evidence": [
                {
                    "source_fragment_id": "dddddd01-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.operating_margin",
                    "observation": "MSFT margin at 44% and rising, peers averaging 38% and falling",
                    "weight": 0.85,
                },
                {
                    "source_fragment_id": "dddddd03-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.tone_score",
                    "observation": "Management confident on margin sustainability; peers hedging",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Margin spread narrows below 200bps vs peers",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "margin_spread_bps",
                    "operator": "LT",
                    "threshold": 200.0,
                    "current_value": 600.0,
                },
                {
                    "description": "Stock drops below $380, contradicting thesis",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "close_price",
                    "operator": "LT",
                    "threshold": 380.0,
                    "current_value": 420.0,
                },
            ],
            "metadata": {
                "sector": "Technology",
                "market_cap_bucket": "MEGA",
                "catalyst_type": "EARNINGS",
            },
        },
        {
            "thesis_id": "XSEC-MSFT-20260212-MEANREV",
            "ticker": "MSFT",
            "thesis_summary": "MSFT/GOOG price ratio at 10-year extreme; 2.8 sigma; historical mean reversion within 60-90 days",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.52,
            "time_horizon_days": 75,
            "evidence": [
                {
                    "source_fragment_id": "dddddd02-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.close",
                    "observation": "MSFT/GOOG price ratio at 2.6x, 2.8 standard deviations above 10-year mean",
                    "weight": 0.8,
                },
            ],
            "invalidation_conditions": [
                {
                    "description": "Price ratio continues widening past 3.0x",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "price_ratio_vs_peer",
                    "operator": "GT",
                    "threshold": 3.0,
                    "current_value": 2.6,
                },
                {
                    "description": "Relative return exceeds +15%, momentum overrides mean reversion",
                    "data_source_agent": "PERCEPT-PRICE",
                    "metric": "relative_return_30d_pct",
                    "operator": "GT",
                    "threshold": 15.0,
                    "current_value": 5.0,
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

# Empty beliefs array (crosssec variant)
CROSSSEC_EMPTY_BELIEFS_RESPONSE = {
    "beliefs": [],
}

# Crosssec response with vague invalidation conditions (< 2 valid → belief dropped)
CROSSSEC_VAGUE_INVALIDATION_RESPONSE = {
    "beliefs": [
        {
            "thesis_id": "XSEC-GOOG-20260212-QUALITY",
            "ticker": "GOOG",
            "thesis_summary": "GOOG earnings quality deteriorating vs peers; cash conversion gap widening",
            "direction": "SHORT",
            "magnitude": "SMALL",
            "raw_confidence": 0.50,
            "time_horizon_days": 60,
            "evidence": [
                {
                    "source_fragment_id": "dddddd01-dddd-dddd-dddd-dddddddddddd",
                    "field_path": "payload.cash_conversion_ratio",
                    "observation": "Cash conversion 0.6x vs sector avg 0.9x; quality gap widening",
                    "weight": 0.7,
                },
            ],
            "invalidation_conditions": [
                {
                    # Has metric but missing operator — should be skipped
                    "description": "If quality improves",
                    "metric": "cash_conversion_ratio",
                    "threshold": 0.85,
                },
                {
                    # Has operator but missing metric — should be skipped
                    "description": "If stock rallies",
                    "operator": "GT",
                    "threshold": 190.0,
                },
                {
                    # Valid condition — kept, but only 1 valid → belief dropped
                    "description": "Cash conversion improves above 0.85x",
                    "data_source_agent": "PERCEPT-FILING",
                    "metric": "cash_conversion_ratio",
                    "operator": "GT",
                    "threshold": 0.85,
                    "current_value": 0.6,
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


# ---------------------------------------------------------------------------
# COGNIT-FUNDAMENTAL shared responses (continued)
# ---------------------------------------------------------------------------

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


# ===========================================================================
# REGIME-NARR Narrative Regime Fixtures
# ===========================================================================

# Valid narrative regime response — AI euphoria theme
VALID_NARRATIVE_REGIME_RESPONSE = {
    "narrative_label": "AI-driven tech euphoria with rotation risk",
    "narrative_confidence": 0.68,
    "key_signals": [
        "NVDA earnings beat by 15% driving semiconductor rally",
        "Enterprise AI spending accelerating per MSFT and GOOGL guidance",
        "Rate-sensitive sectors lagging despite dovish Fed commentary",
        "VIX at 14 despite concentration risk in Mag-7",
    ],
    "affected_sectors": ["Technology", "Communication Services", "Utilities"],
    "regime_alignment": "DIVERGES",
    "summary": (
        "Market narrative is dominated by AI infrastructure spending acceleration, "
        "with tech mega-caps driving indices to new highs. However, breadth is narrow "
        "and rate-sensitive sectors are not confirming the risk-on tone, creating "
        "a divergence with the statistical regime classifier."
    ),
    "statistical_regime_assessment": "LOW_VOL_TRENDING",
    "risk_mode_suggestion": "CAUTIOUS",
}

# Crisis narrative response — banking contagion
CRISIS_NARRATIVE_REGIME_RESPONSE = {
    "narrative_label": "Banking contagion fear and credit stress",
    "narrative_confidence": 0.75,
    "key_signals": [
        "Regional bank CRE exposure driving deposit flight fears",
        "CDS spreads widening +40bps in one week",
        "Flight to quality: Treasury yields falling sharply",
        "Fed emergency lending facility usage elevated",
        "Financials sector down 12% in 5 sessions",
    ],
    "affected_sectors": ["Financials", "Real Estate", "Consumer Discretionary"],
    "regime_alignment": "CONFIRMS",
    "summary": (
        "Credit stress narrative is intensifying as commercial real estate "
        "exposure at regional banks triggers deposit flight concerns. "
        "Cross-asset confirmation via CDS widening and Treasury rally "
        "supports the crisis classification."
    ),
    "statistical_regime_assessment": "CRISIS_DISLOCATION",
    "risk_mode_suggestion": "DEFENSIVE",
}

# Weak/ambiguous narrative — no dominant theme
WEAK_NARRATIVE_REGIME_RESPONSE = {
    "narrative_label": "Mixed signals with no dominant theme",
    "narrative_confidence": 0.28,
    "key_signals": [
        "Earnings broadly in-line with expectations",
        "Fed messaging intentionally ambiguous",
    ],
    "affected_sectors": [],
    "regime_alignment": "NEUTRAL",
    "summary": "No clear dominant narrative emerging from current data.",
    "statistical_regime_assessment": "TRANSITION_UNCERTAIN",
    "risk_mode_suggestion": "CAUTIOUS",
}

# Malformed response — missing required field
MALFORMED_NARRATIVE_REGIME_RESPONSE = {
    "key_signals": ["Some signal"],
    "narrative_confidence": 0.50,
    # Missing narrative_label — parser should return None
}


# ===========================================================================
# DECIDE-SYNTH Belief Synthesis Fixtures
# ===========================================================================

# Valid single-intent synthesis — AAPL LONG with contributing theses
VALID_SYNTHESIS_RESPONSE = {
    "position_intents": [
        {
            "ticker": "AAPL",
            "net_direction": "LONG",
            "synthesized_confidence": 0.62,
            "contributing_theses": [
                {
                    "thesis_id": "FUND-AAPL-2026Q1-MARGIN-EXPANSION",
                    "agent_id": "COGNIT-FUNDAMENTAL",
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "raw_confidence": 0.72,
                    "magnitude": "MODERATE",
                    "synthesis_weight": 0.35,
                },
                {
                    "thesis_id": "NARR-AAPL-20260212-TONE",
                    "agent_id": "COGNIT-NARRATIVE",
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "raw_confidence": 0.62,
                    "magnitude": "MODERATE",
                    "synthesis_weight": 0.20,
                },
            ],
            "conflicting_theses": [],
            "conflict_resolution": {
                "has_conflict": False,
                "conflict_type": "NONE",
                "resolution_method": "CONFIDENCE_WEIGHTED",
                "resolution_rationale": "All agents agree on LONG direction for AAPL",
                "net_conviction_delta": 0.0,
            },
            "time_horizon_days": 75,
            "regime_adjustment": -0.05,
            "active_invalidations": [
                {
                    "source_thesis_id": "FUND-AAPL-2026Q1-MARGIN-EXPANSION",
                    "source_agent_id": "COGNIT-FUNDAMENTAL",
                    "metric": "gross_margin_pct",
                    "operator": "LT",
                    "threshold": 44.0,
                    "description": "Gross margin drops below 44%",
                },
            ],
            "synthesis_rationale": "Strong fundamental and narrative alignment supports LONG position with moderate confidence.",
        }
    ],
    "total_beliefs_consumed": 4,
}

# Multi-intent synthesis — AAPL + MSFT with conflict on MSFT
MULTI_INTENT_SYNTHESIS_RESPONSE = {
    "position_intents": [
        {
            "ticker": "AAPL",
            "net_direction": "LONG",
            "synthesized_confidence": 0.58,
            "contributing_theses": [
                {
                    "thesis_id": "FUND-AAPL-2026Q1-MARGIN",
                    "agent_id": "COGNIT-FUNDAMENTAL",
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "raw_confidence": 0.72,
                    "magnitude": "MODERATE",
                    "synthesis_weight": 0.40,
                },
            ],
            "conflicting_theses": [],
            "conflict_resolution": {
                "has_conflict": False,
                "conflict_type": "NONE",
                "resolution_method": "CONFIDENCE_WEIGHTED",
                "resolution_rationale": "Consensus LONG",
                "net_conviction_delta": 0.0,
            },
            "time_horizon_days": 90,
            "regime_adjustment": -0.05,
            "active_invalidations": [],
            "synthesis_rationale": "Fundamental thesis drives AAPL LONG.",
        },
        {
            "ticker": "MSFT",
            "net_direction": "LONG",
            "synthesized_confidence": 0.45,
            "contributing_theses": [
                {
                    "thesis_id": "XSEC-MSFT-DIVERGENCE",
                    "agent_id": "COGNIT-CROSSSEC",
                    "ticker": "MSFT",
                    "direction": "LONG",
                    "raw_confidence": 0.68,
                    "magnitude": "MODERATE",
                    "synthesis_weight": 0.30,
                },
            ],
            "conflicting_theses": [
                {
                    "thesis_id": "NARR-MSFT-GUIDANCE",
                    "agent_id": "COGNIT-NARRATIVE",
                    "ticker": "MSFT",
                    "direction": "SHORT",
                    "raw_confidence": 0.58,
                    "magnitude": "SMALL",
                    "synthesis_weight": 0.15,
                },
            ],
            "conflict_resolution": {
                "has_conflict": True,
                "conflict_type": "DIRECTIONAL",
                "resolution_method": "CONFIDENCE_WEIGHTED",
                "resolution_rationale": "Cross-section margin divergence outweighs hedging language concern",
                "net_conviction_delta": -0.15,
            },
            "time_horizon_days": 60,
            "regime_adjustment": -0.10,
            "active_invalidations": [
                {
                    "source_thesis_id": "XSEC-MSFT-DIVERGENCE",
                    "source_agent_id": "COGNIT-CROSSSEC",
                    "metric": "margin_spread_bps",
                    "operator": "LT",
                    "threshold": 200.0,
                    "description": "Margin spread narrows below 200bps vs peers",
                },
            ],
            "synthesis_rationale": "Directional conflict resolved via confidence weighting; reduced conviction due to narrative headwind.",
        },
    ],
    "total_beliefs_consumed": 8,
}

# Crisis synthesis — defensive SHORT with high regime adjustment
CRISIS_SYNTHESIS_RESPONSE = {
    "position_intents": [
        {
            "ticker": "SPY",
            "net_direction": "SHORT",
            "synthesized_confidence": 0.70,
            "contributing_theses": [
                {
                    "thesis_id": "MACRO-YIELDCURVE-INVERSION",
                    "agent_id": "COGNIT-MACRO",
                    "ticker": "SPY",
                    "direction": "SHORT",
                    "raw_confidence": 0.68,
                    "magnitude": "MODERATE",
                    "synthesis_weight": 0.40,
                },
            ],
            "conflicting_theses": [],
            "conflict_resolution": {
                "has_conflict": False,
                "conflict_type": "NONE",
                "resolution_method": "REGIME_ADJUSTED",
                "resolution_rationale": "Crisis regime supports defensive positioning",
                "net_conviction_delta": 0.0,
            },
            "time_horizon_days": 30,
            "regime_adjustment": -0.20,
            "active_invalidations": [
                {
                    "source_thesis_id": "MACRO-YIELDCURVE-INVERSION",
                    "source_agent_id": "COGNIT-MACRO",
                    "metric": "spread_2s10s_bps",
                    "operator": "GT",
                    "threshold": 0.0,
                    "description": "Yield curve dis-inverts",
                },
            ],
            "synthesis_rationale": "Crisis regime with yield curve inversion supports SHORT with high conviction.",
        }
    ],
    "total_beliefs_consumed": 3,
}

# Malformed synthesis — missing position_intents
MALFORMED_SYNTHESIS_RESPONSE = {
    "analysis": "Markets look bearish",
    "recommendation": "Reduce exposure",
}

# Malformed synthesis — empty position_intents array
EMPTY_SYNTHESIS_RESPONSE = {
    "position_intents": [],
}

# Malformed synthesis — invalid direction
INVALID_DIRECTION_SYNTHESIS_RESPONSE = {
    "position_intents": [
        {
            "ticker": "AAPL",
            "net_direction": "BUY",  # Invalid — not LONG/SHORT/NEUTRAL
            "synthesized_confidence": 0.60,
        }
    ],
}

# Synthesis with over-confidence (should be clamped to 0.85)
OVERCCONFIDENT_SYNTHESIS_RESPONSE = {
    "position_intents": [
        {
            "ticker": "NVDA",
            "net_direction": "LONG",
            "synthesized_confidence": 0.95,  # Over cap — should clamp to 0.85
            "contributing_theses": [
                {
                    "thesis_id": "FUND-NVDA-AI-CAPEX",
                    "agent_id": "COGNIT-FUNDAMENTAL",
                    "ticker": "NVDA",
                    "direction": "LONG",
                    "raw_confidence": 0.82,
                    "magnitude": "LARGE",
                    "synthesis_weight": 0.35,
                },
            ],
            "conflicting_theses": [],
            "conflict_resolution": {
                "has_conflict": False,
                "conflict_type": "NONE",
                "resolution_method": "CONFIDENCE_WEIGHTED",
                "resolution_rationale": "Unanimous consensus",
                "net_conviction_delta": 0.0,
            },
            "time_horizon_days": 120,
            "regime_adjustment": 0.05,
            "active_invalidations": [],
            "synthesis_rationale": "Strong multi-agent consensus on NVDA AI thesis.",
        }
    ],
    "total_beliefs_consumed": 6,
}


# =============================================================================
# COGNIT-EXIT: Exit Assessment Responses
# =============================================================================

VALID_EXIT_RESPONSE = {
    "assessments": [
        {
            "ticker": "AAPL",
            "exit_action": "HOLD",
            "exit_confidence": 0.20,
            "regret_estimate_bps": 15.0,
            "regret_direction": "MISSED_UPSIDE",
            "thesis_health_score": 0.85,
            "rationale": "Thesis remains healthy with strong margin expansion trajectory. No exit signals.",
        },
    ],
}

MULTI_EXIT_RESPONSE = {
    "assessments": [
        {
            "ticker": "AAPL",
            "exit_action": "HOLD",
            "exit_confidence": 0.15,
            "regret_estimate_bps": 10.0,
            "regret_direction": "MISSED_UPSIDE",
            "thesis_health_score": 0.90,
            "rationale": "Strong thesis health, all conditions intact.",
        },
        {
            "ticker": "MSFT",
            "exit_action": "REDUCE",
            "exit_confidence": 0.65,
            "regret_estimate_bps": 45.0,
            "regret_direction": "SUFFERED_GIVEBACK",
            "thesis_health_score": 0.45,
            "rationale": "Two invalidation conditions triggered. Cloud growth decelerating.",
        },
        {
            "ticker": "TSLA",
            "exit_action": "EXIT",
            "exit_confidence": 0.88,
            "regret_estimate_bps": 5.0,
            "regret_direction": "SUFFERED_GIVEBACK",
            "thesis_health_score": 0.20,
            "rationale": "Multiple conditions breached. Thesis fundamentally invalidated.",
        },
    ],
}

MALFORMED_EXIT_RESPONSE = {
    "wrong_key": "no assessments here",
}

EMPTY_EXIT_RESPONSE = {
    "assessments": [],
}

INVALID_ACTION_EXIT_RESPONSE = {
    "assessments": [
        {
            "ticker": "AAPL",
            "exit_action": "BUY_MORE",
            "exit_confidence": 0.50,
            "rationale": "Invalid action should default to HOLD.",
        },
    ],
}

OVERCONFIDENT_EXIT_RESPONSE = {
    "assessments": [
        {
            "ticker": "AAPL",
            "exit_action": "EXIT",
            "exit_confidence": 1.5,
            "regret_estimate_bps": -10.0,
            "regret_direction": "WRONG_DIRECTION",
            "thesis_health_score": 1.5,
            "rationale": "Overconfident values should be clamped.",
        },
    ],
}
