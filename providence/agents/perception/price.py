"""PERCEPT-PRICE: Price data ingestion agent.

Ingests OHLCV price data from Polygon.io and produces MarketStateFragments
with data_type=PRICE_OHLCV.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Polygon.io REST API
  2. VALIDATE    → Check schema completeness and freshness
  3. NORMALIZE   → Convert to PricePayload
  4. VERSION     → Compute content hash, assign fragment_id
  5. STORE       → Return MarketStateFragment (Kafka is future work)
  6. ALERT       → If validation fails, set QUARANTINED and log
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.polygon_client import PolygonClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import PricePayload

logger = structlog.get_logger()


class PerceptPrice(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-PRICE agent — ingests OHLCV price data.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, polygon_client: PolygonClient) -> None:
        super().__init__(
            agent_id="PERCEPT-PRICE",
            agent_type="perception",
            version="1.0.0",
        )
        self._polygon = polygon_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process price data for tickers specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch
            - date: str — date in YYYY-MM-DD format
            - timeframe: str — "1D" (default), "1H", "1min"

        Returns:
            List of MarketStateFragments, one per ticker.
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        date: str = context.metadata.get("date", "")
        timeframe: str = context.metadata.get("timeframe", "1D")

        if not tickers:
            raise AgentProcessingError(
                message="No tickers specified in context metadata",
                agent_id=self.agent_id,
            )
        if not date:
            raise AgentProcessingError(
                message="No date specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []
        for ticker in tickers:
            try:
                fragment = await self._process_ticker(ticker, date, timeframe)
                fragments.append(fragment)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process ticker",
                    agent_id=self.agent_id,
                    ticker=ticker,
                    error=str(e),
                )
                # Produce a quarantined fragment so the failure is tracked
                fragment = self._create_quarantined_fragment(ticker, date, timeframe, str(e))
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_ticker(
        self, ticker: str, date: str, timeframe: str
    ) -> MarketStateFragment:
        """Run the full Perception loop for a single ticker.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch(ticker, date, timeframe)

        # Step 2: VALIDATE
        validation_status = self._validate(raw_data, ticker)

        # Step 3: NORMALIZE
        payload = self._normalize(raw_data, timeframe)

        # Step 4: VERSION — content hash computed by MarketStateFragment
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment (STORE is future Kafka work)
        source_ts = self._extract_source_timestamp(raw_data)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=ticker,
            data_type=DataType.PRICE_OHLCV,
            schema_version="1.0.0",
            source_hash=source_hash,
            validation_status=validation_status,
            payload=payload,
        )

        logger.info(
            "Fragment produced",
            agent_id=self.agent_id,
            ticker=ticker,
            validation_status=validation_status.value,
            fragment_id=str(fragment.fragment_id),
        )

        return fragment

    async def _fetch(
        self, ticker: str, date: str, timeframe: str
    ) -> dict[str, Any]:
        """Step 1: FETCH — Pull raw data from Polygon.io."""
        try:
            if timeframe == "1D":
                return await self._polygon.get_daily_bars(ticker, date)
            else:
                # Map timeframe to Polygon API params
                timespan_map = {"1H": ("1", "hour"), "1min": ("1", "minute")}
                tf, ts = timespan_map.get(timeframe, ("1", "hour"))
                bars = await self._polygon.get_intraday_bars(ticker, date, tf, ts)
                return {"results": bars, "ticker": ticker, "resultsCount": len(bars)}
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch {ticker} data from Polygon: {e}",
                agent_id=self.agent_id,
            ) from e

    def _validate(self, raw_data: dict[str, Any], ticker: str) -> ValidationStatus:
        """Step 2: VALIDATE — Check schema completeness and freshness.

        Returns VALID, PARTIAL, or QUARANTINED based on data quality.
        """
        results = raw_data.get("results")

        # No results at all → quarantine
        if not results or not isinstance(results, list) or len(results) == 0:
            logger.warning(
                "No results in API response — quarantining",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return ValidationStatus.QUARANTINED

        bar = results[0] if isinstance(results[0], dict) else {}

        # Required OHLCV fields from Polygon
        required_fields = {"o", "h", "l", "c", "v"}
        present_fields = set(bar.keys()) & required_fields
        missing_fields = required_fields - present_fields

        if missing_fields == required_fields:
            return ValidationStatus.QUARANTINED
        elif missing_fields:
            logger.warning(
                "Partial data — missing fields",
                agent_id=self.agent_id,
                ticker=ticker,
                missing=list(missing_fields),
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize(self, raw_data: dict[str, Any], timeframe: str) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to PricePayload dict.

        Maps Polygon's field names (o, h, l, c, v, vw, n) to our schema.
        """
        results = raw_data.get("results", [])
        if not results:
            return PricePayload(
                open=0.0, high=0.0, low=0.0, close=0.0,
                volume=0, timeframe=timeframe,
            ).model_dump()

        bar = results[0] if isinstance(results[0], dict) else {}

        payload = PricePayload(
            open=float(bar.get("o", 0.0)),
            high=float(bar.get("h", 0.0)),
            low=float(bar.get("l", 0.0)),
            close=float(bar.get("c", 0.0)),
            volume=int(bar.get("v", 0)),
            vwap=float(bar["vw"]) if "vw" in bar else None,
            num_trades=int(bar["n"]) if "n" in bar else None,
            timeframe=timeframe,
        )
        return payload.model_dump()

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the raw API response for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _extract_source_timestamp(self, raw_data: dict[str, Any]) -> datetime:
        """Extract the source timestamp from the API response.

        Polygon uses 't' field (Unix ms timestamp) in results.
        Falls back to current time if not available.
        """
        results = raw_data.get("results", [])
        if results and isinstance(results[0], dict) and "t" in results[0]:
            unix_ms = results[0]["t"]
            return datetime.fromtimestamp(unix_ms / 1000, tz=timezone.utc)
        return datetime.now(timezone.utc)

    def _create_quarantined_fragment(
        self, ticker: str, date: str, timeframe: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=DataType.PRICE_OHLCV,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "date": date, "timeframe": timeframe},
        )

    def get_health(self) -> HealthStatus:
        """Report current health status."""
        if self._error_count_24h > 10:
            status = AgentStatus.UNHEALTHY
        elif self._error_count_24h > 3:
            status = AgentStatus.DEGRADED
        else:
            status = AgentStatus.HEALTHY

        return HealthStatus(
            agent_id=self.agent_id,
            status=status,
            last_run=self._last_run,
            last_success=self._last_success,
            error_count_24h=self._error_count_24h,
            avg_latency_ms=0.0,
        )
