"""Sample Polygon.io API responses for testing PERCEPT-OPTIONS.

These fixtures replicate the structure of real Polygon.io options chain
responses so that tests can run without making actual API calls.
"""


def options_chain_aapl() -> list[dict]:
    """Sample Polygon.io options chain response for AAPL.

    Returns 4 contracts: 2 calls and 2 puts at different strikes.
    """
    return [
        {
            "details": {
                "contract_type": "call",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 190.0,
                "ticker": "O:AAPL260320C00190000",
            },
            "greeks": {"delta": 0.55, "gamma": 0.03, "theta": -0.15, "vega": 0.25},
            "last_quote": {"bid": 5.20, "ask": 5.40, "bid_size": 10, "ask_size": 15},
            "last_trade": {"price": 5.30, "size": 5, "conditions": [209]},
            "open_interest": 12500,
            "implied_volatility": 0.28,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 3456},
        },
        {
            "details": {
                "contract_type": "call",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 195.0,
                "ticker": "O:AAPL260320C00195000",
            },
            "greeks": {"delta": 0.45, "gamma": 0.035, "theta": -0.12, "vega": 0.22},
            "last_quote": {"bid": 3.10, "ask": 3.30, "bid_size": 20, "ask_size": 25},
            "last_trade": {"price": 3.20, "size": 8, "conditions": [209]},
            "open_interest": 8900,
            "implied_volatility": 0.30,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 2345},
        },
        {
            "details": {
                "contract_type": "put",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 180.0,
                "ticker": "O:AAPL260320P00180000",
            },
            "greeks": {"delta": -0.40, "gamma": 0.032, "theta": -0.14, "vega": 0.24},
            "last_quote": {"bid": 2.80, "ask": 3.00, "bid_size": 12, "ask_size": 18},
            "last_trade": {"price": 2.90, "size": 6, "conditions": [209]},
            "open_interest": 9800,
            "implied_volatility": 0.27,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 2789},
        },
        {
            "details": {
                "contract_type": "put",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 185.0,
                "ticker": "O:AAPL260320P00185000",
            },
            "greeks": {"delta": -0.50, "gamma": 0.033, "theta": -0.13, "vega": 0.26},
            "last_quote": {"bid": 4.40, "ask": 4.60, "bid_size": 14, "ask_size": 20},
            "last_trade": {"price": 4.50, "size": 7, "conditions": [209]},
            "open_interest": 11200,
            "implied_volatility": 0.29,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 3567},
        },
    ]


def options_chain_empty() -> list[dict]:
    """Polygon response with no contracts.

    Simulates a case where no options data is available for the given filters.
    """
    return []


def options_chain_no_greeks() -> list[dict]:
    """Polygon response with contracts missing the 'greeks' field (partial data).

    This should trigger PARTIAL validation status since Greeks data is required
    for full contracts analysis.
    """
    return [
        {
            "details": {
                "contract_type": "call",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 190.0,
                "ticker": "O:AAPL260320C00190000",
            },
            # greeks field intentionally omitted
            "last_quote": {"bid": 5.20, "ask": 5.40, "bid_size": 10, "ask_size": 15},
            "last_trade": {"price": 5.30, "size": 5, "conditions": [209]},
            "open_interest": 12500,
            "implied_volatility": 0.28,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 3456},
        },
        {
            "details": {
                "contract_type": "put",
                "exercise_style": "american",
                "expiration_date": "2026-03-20",
                "shares_per_contract": 100,
                "strike_price": 180.0,
                "ticker": "O:AAPL260320P00180000",
            },
            # greeks field intentionally omitted
            "last_quote": {"bid": 2.80, "ask": 3.00, "bid_size": 12, "ask_size": 18},
            "last_trade": {"price": 2.90, "size": 6, "conditions": [209]},
            "open_interest": 9800,
            "implied_volatility": 0.27,
            "underlying_asset": {"price": 186.90, "ticker": "AAPL"},
            "day": {"volume": 2789},
        },
    ]
