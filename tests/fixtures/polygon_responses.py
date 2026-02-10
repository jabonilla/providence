"""Sample Polygon.io API responses for testing PERCEPT-PRICE.

These fixtures replicate the structure of real Polygon.io responses
so that tests can run without making actual API calls.
"""


def daily_bars_aapl() -> dict:
    """Sample Polygon.io daily bars response for AAPL."""
    return {
        "ticker": "AAPL",
        "queryCount": 1,
        "resultsCount": 1,
        "adjusted": True,
        "results": [
            {
                "v": 52_345_678,       # volume
                "vw": 186.15,          # vwap
                "o": 185.50,           # open
                "c": 186.90,           # close
                "h": 187.20,           # high
                "l": 184.80,           # low
                "t": 1739059200000,    # timestamp (2025-02-09 00:00 UTC ms)
                "n": 623_456,          # number of trades
            }
        ],
        "status": "OK",
        "request_id": "abc123",
        "count": 1,
    }


def daily_bars_nvda() -> dict:
    """Sample Polygon.io daily bars response for NVDA."""
    return {
        "ticker": "NVDA",
        "queryCount": 1,
        "resultsCount": 1,
        "adjusted": True,
        "results": [
            {
                "v": 38_901_234,
                "vw": 875.30,
                "o": 870.00,
                "c": 878.50,
                "h": 880.25,
                "l": 868.10,
                "t": 1739059200000,
                "n": 412_789,
            }
        ],
        "status": "OK",
        "request_id": "def456",
        "count": 1,
    }


def daily_bars_empty() -> dict:
    """Polygon response with no results (e.g., market holiday)."""
    return {
        "ticker": "AAPL",
        "queryCount": 0,
        "resultsCount": 0,
        "adjusted": True,
        "results": [],
        "status": "OK",
        "request_id": "empty123",
        "count": 0,
    }


def daily_bars_missing_fields() -> dict:
    """Polygon response with incomplete bar data (missing volume)."""
    return {
        "ticker": "AAPL",
        "queryCount": 1,
        "resultsCount": 1,
        "adjusted": True,
        "results": [
            {
                "o": 185.50,
                "c": 186.90,
                "h": 187.20,
                "l": 184.80,
                "t": 1739059200000,
                # missing: v (volume), vw (vwap), n (num_trades)
            }
        ],
        "status": "OK",
        "request_id": "partial123",
        "count": 1,
    }


def daily_bars_no_ohlcv() -> dict:
    """Polygon response with results but no OHLCV fields at all."""
    return {
        "ticker": "AAPL",
        "queryCount": 1,
        "resultsCount": 1,
        "adjusted": True,
        "results": [
            {
                "t": 1739059200000,
                "otc": True,
            }
        ],
        "status": "OK",
        "request_id": "noohlcv123",
        "count": 1,
    }


def intraday_bars_aapl() -> list[dict]:
    """Sample intraday (hourly) bars for AAPL."""
    return [
        {
            "v": 8_123_456,
            "vw": 185.80,
            "o": 185.50,
            "c": 186.10,
            "h": 186.30,
            "l": 185.40,
            "t": 1739073600000,
            "n": 98_765,
        },
        {
            "v": 6_789_012,
            "vw": 186.25,
            "o": 186.10,
            "c": 186.50,
            "h": 186.70,
            "l": 185.90,
            "t": 1739077200000,
            "n": 76_543,
        },
    ]
