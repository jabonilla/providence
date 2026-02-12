"""Sample FRED API responses for testing PERCEPT-MACRO and PERCEPT-CDS.

These fixtures replicate the structure of real FRED API responses
so that tests can run without making actual API calls.
"""


def treasury_yields_response() -> dict[str, float]:
    """Sample full Treasury yield curve response from FRED.

    Returns a complete yield curve with tenors from 1M to 30Y.
    """
    return {
        "1M": 5.32,
        "3M": 5.28,
        "6M": 5.15,
        "1Y": 4.85,
        "2Y": 4.45,
        "3Y": 4.20,
        "5Y": 4.05,
        "7Y": 4.00,
        "10Y": 3.95,
        "20Y": 4.10,
        "30Y": 4.12,
    }


def treasury_yields_partial() -> dict[str, float]:
    """Sample partial Treasury yield curve response from FRED.

    Simulates missing data — only a few tenors available.
    """
    return {
        "1Y": 4.85,
        "2Y": 4.45,
        "10Y": 3.95,
    }


def treasury_yields_empty() -> dict[str, float]:
    """Sample empty Treasury yield curve response from FRED.

    Represents case where no yield data is available.
    """
    return {}


def gdp_observation() -> list[dict[str, str]]:
    """Sample GDP observation from FRED series A191RA1Q225SBEA.

    Single observation with quarterly GDP data.
    """
    return [
        {
            "date": "2026-01-01",
            "value": "28500.5",
            "realtime_start": "2026-02-01",
            "realtime_end": "2026-02-09",
        }
    ]


def cpi_observation() -> list[dict[str, str]]:
    """Sample CPI observation from FRED series CPIAUCSL.

    Single observation with monthly CPI data.
    """
    return [
        {
            "date": "2026-01-01",
            "value": "3.1",
            "realtime_start": "2026-02-01",
            "realtime_end": "2026-02-09",
        }
    ]


def unemployment_observation() -> list[dict[str, str]]:
    """Sample unemployment rate observation from FRED series UNRATE.

    Single observation with monthly unemployment rate.
    """
    return [
        {
            "date": "2026-01-01",
            "value": "3.8",
            "realtime_start": "2026-02-01",
            "realtime_end": "2026-02-09",
        }
    ]


def missing_observation() -> list[dict[str, str]]:
    """Sample observation with missing value (represented as ".").

    FRED uses "." to represent missing or not-available data.
    """
    return [
        {
            "date": "2026-01-01",
            "value": ".",
            "realtime_start": "2026-02-01",
            "realtime_end": "2026-02-09",
        }
    ]


def cds_observations() -> list[dict[str, str]]:
    """Sample CDS spread observations from FRED series (e.g., BAMLC0A0CM).

    Two observations representing current and prior day CDS spreads.
    """
    return [
        {
            "date": "2026-02-09",
            "value": "125.5",
        },
        {
            "date": "2026-02-08",
            "value": "122.3",
        },
    ]


def cds_observations_single() -> list[dict[str, str]]:
    """Sample CDS spread observation with only one data point.

    Single observation for a CDS series (no prior day data).
    """
    return [
        {
            "date": "2026-02-09",
            "value": "125.5",
        }
    ]


def cds_observations_empty() -> list[dict[str, str]]:
    """Sample empty CDS observations response.

    Represents case where no CDS data is available.
    """
    return []


def cds_observations_missing_value() -> list[dict[str, str]]:
    """Sample CDS observations with missing value.

    FRED uses "." to represent missing or not-available data.
    """
    return [
        {
            "date": "2026-02-09",
            "value": ".",
        }
    ]
