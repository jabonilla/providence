"""Kenneth French Data Library client for factor returns data.

Provides async access to Fama-French factor data (5-factor model + momentum)
with minimal rate limiting and error handling.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-FACTORS)

Usage:
    client = FamaFrenchClient()
    five_factors = await client.get_five_factors_daily("2026-02-01", "2026-02-09")
    momentum = await client.get_momentum_daily("2026-02-01", "2026-02-09")
"""

import asyncio
import time
from datetime import datetime
from typing import Any

import structlog

from providence.exceptions import DataIngestionError, ExternalAPIError

logger = structlog.get_logger()


class FamaFrenchClient:
    """Async client for Kenneth French Data Library.

    Uses pandas_datareader.data.DataReader to fetch public factor data.
    Handles sync-to-async conversion and error mapping.
    """

    MIN_REQUEST_INTERVAL = 1.0  # seconds between requests

    def __init__(self) -> None:
        """Initialize the Fama-French client.

        No API key needed — Kenneth French data is publicly available.
        """
        self._last_request_time: float = 0.0

    async def _rate_limit(self) -> None:
        """Enforce minimal rate limiting to avoid overwhelming the data source."""
        now = time.monotonic()
        elapsed = now - self._last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            await asyncio.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self._last_request_time = time.monotonic()

    async def get_five_factors_daily(
        self, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        """Fetch daily 5-factor Fama-French data.

        Fetches the Fama-French Research Data 5 Factors (2x3) daily dataset.
        Factors: Mkt-RF (market excess return), SMB (small-big),
        HML (high-low value), RMW (robust-weak profitability),
        CMA (conservative-aggressive investment), RF (risk-free rate).

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            List of dicts, each containing:
              - date: str (YYYY-MM-DD)
              - mkt_rf: float (market excess return %)
              - smb: float (small minus big %)
              - hml: float (high minus low %)
              - rmw: float (robust minus weak %)
              - cma: float (conservative minus aggressive %)
              - rf: float (risk-free rate %)

        Raises:
            ExternalAPIError: On data fetch failures.
            DataIngestionError: On parsing failures.
        """
        await self._rate_limit()
        logger.info(
            "Fetching 5-factor data",
            start_date=start_date,
            end_date=end_date,
        )

        try:
            import pandas_datareader.data as web
        except ImportError as e:
            raise ExternalAPIError(
                message="pandas_datareader required for Fama-French data",
                service="fama_french",
            ) from e

        try:
            # Run pandas_datareader in executor to avoid blocking
            df = await asyncio.to_thread(
                web.DataReader,
                "F-F_Research_Data_5_Factors_2x3_daily",
                "famafrench",
                start=start_date,
                end=end_date,
            )

            # DataReader returns a tuple (df, description); unpack it
            if isinstance(df, tuple):
                df = df[0]

            if df is None or df.empty:
                raise DataIngestionError(
                    message=f"No 5-factor data returned for range {start_date} to {end_date}"
                )

            # Convert to list of dicts
            result = []
            for date_idx, row in df.iterrows():
                # date_idx is a Timestamp; convert to YYYY-MM-DD
                if hasattr(date_idx, "strftime"):
                    date_str = date_idx.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_idx).split()[0]

                result.append({
                    "date": date_str,
                    "mkt_rf": float(row.get("Mkt-RF", 0.0)),
                    "smb": float(row.get("SMB", 0.0)),
                    "hml": float(row.get("HML", 0.0)),
                    "rmw": float(row.get("RMW", 0.0)),
                    "cma": float(row.get("CMA", 0.0)),
                    "rf": float(row.get("RF", 0.0)),
                })

            logger.info(
                "5-factor data loaded",
                count=len(result),
                start_date=start_date,
                end_date=end_date,
            )
            return result

        except DataIngestionError:
            raise
        except Exception as e:
            raise ExternalAPIError(
                message=f"Failed to fetch 5-factor data: {e}",
                service="fama_french",
            ) from e

    async def get_momentum_daily(self, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch daily momentum factor data.

        Fetches the Fama-French Momentum (UMD) Factor daily dataset.

        Args:
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            List of dicts, each containing:
              - date: str (YYYY-MM-DD)
              - mom: float (momentum factor %)

        Raises:
            ExternalAPIError: On data fetch failures.
            DataIngestionError: On parsing failures.
        """
        await self._rate_limit()
        logger.info(
            "Fetching momentum factor data",
            start_date=start_date,
            end_date=end_date,
        )

        try:
            import pandas_datareader.data as web
        except ImportError as e:
            raise ExternalAPIError(
                message="pandas_datareader required for Fama-French data",
                service="fama_french",
            ) from e

        try:
            # Run pandas_datareader in executor to avoid blocking
            df = await asyncio.to_thread(
                web.DataReader,
                "F-F_Momentum_Factor_daily",
                "famafrench",
                start=start_date,
                end=end_date,
            )

            # DataReader returns a tuple (df, description); unpack it
            if isinstance(df, tuple):
                df = df[0]

            if df is None or df.empty:
                raise DataIngestionError(
                    message=f"No momentum data returned for range {start_date} to {end_date}"
                )

            # Convert to list of dicts
            result = []
            for date_idx, row in df.iterrows():
                # date_idx is a Timestamp; convert to YYYY-MM-DD
                if hasattr(date_idx, "strftime"):
                    date_str = date_idx.strftime("%Y-%m-%d")
                else:
                    date_str = str(date_idx).split()[0]

                result.append({
                    "date": date_str,
                    "mom": float(row.get("Mom   ", 0.0)),  # Note: FF dataset has trailing spaces
                })

            logger.info(
                "Momentum factor data loaded",
                count=len(result),
                start_date=start_date,
                end_date=end_date,
            )
            return result

        except DataIngestionError:
            raise
        except Exception as e:
            raise ExternalAPIError(
                message=f"Failed to fetch momentum data: {e}",
                service="fama_french",
            ) from e
