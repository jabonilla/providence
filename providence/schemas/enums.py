"""Shared enumerations for Providence schemas.

All enums used across the Providence system are defined here to ensure
consistency and avoid circular imports.
"""

from enum import Enum


class Direction(str, Enum):
    """Investment direction for a thesis."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class Magnitude(str, Enum):
    """Expected magnitude of a thesis outcome."""
    SMALL = "SMALL"
    MODERATE = "MODERATE"
    LARGE = "LARGE"


class ValidationStatus(str, Enum):
    """Validation status for MarketStateFragments."""
    VALID = "VALID"
    QUARANTINED = "QUARANTINED"
    PARTIAL = "PARTIAL"


class ComparisonOperator(str, Enum):
    """Operators for invalidation condition evaluation."""
    GT = "GT"
    LT = "LT"
    EQ = "EQ"
    CROSSES_ABOVE = "CROSSES_ABOVE"
    CROSSES_BELOW = "CROSSES_BELOW"


class ConditionStatus(str, Enum):
    """Status of an invalidation condition."""
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    EXPIRED = "EXPIRED"


class DataType(str, Enum):
    """Data Type Registry — classifies MarketStateFragment payloads.

    Each data type corresponds to a specific payload schema and
    is produced by a specific Perception agent.
    """
    PRICE_OHLCV = "PRICE_OHLCV"
    FILING_10K = "FILING_10K"
    FILING_10Q = "FILING_10Q"
    FILING_8K = "FILING_8K"
    EARNINGS_CALL = "EARNINGS_CALL"
    SENTIMENT_NEWS = "SENTIMENT_NEWS"
    OPTIONS_CHAIN = "OPTIONS_CHAIN"
    MACRO_YIELD_CURVE = "MACRO_YIELD_CURVE"
    MACRO_CDS = "MACRO_CDS"
    MACRO_ECONOMIC = "MACRO_ECONOMIC"


class MarketCapBucket(str, Enum):
    """Market capitalization classification."""
    MEGA = "MEGA"
    LARGE = "LARGE"
    MID = "MID"
    SMALL = "SMALL"


class CatalystType(str, Enum):
    """Type of catalyst driving a thesis."""
    EARNINGS = "EARNINGS"
    MACRO = "MACRO"
    EVENT = "EVENT"
    NONE = "NONE"
