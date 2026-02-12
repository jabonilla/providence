"""PERCEPT-MACRO: Macroeconomic data ingestion agent.

Ingests macroeconomic data (yield curves and economic indicators) from FRED API
and produces MarketStateFragments with data_type=MACRO_YIELD_CURVE or MACRO_ECONOMIC.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from FRED REST API
  2. VALIDATE    → Check schema completeness and data quality
  3. NORMALIZE   → Convert to MacroYieldPayload or MacroEconomicPayload
  4. VERSION     → Compute content hash, assign fragment_id
  5. STORE       → Return MarketStateFragment (Kafka is future work)
  6. ALERT       → If validation fails, set QUARANTINED and log
"""

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.fred_client import FredClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.utils.redaction import redact_error_message
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import MacroYieldPayload, MacroEconomicPayload

logger = structlog.get_logger()


class PerceptMacro(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-MACRO agent — ingests macroeconomic data from FRED.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, fred_client: FredClient) -> None:
        super().__init__(
            agent_id="PERCEPT-MACRO",
            agent_type="perception",
            version="1.0.0",
        )
        self._fred = fred_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process macro data (yield curves and economic indicators).

        Expected context.metadata keys:
            - date: str — date in YYYY-MM-DD format (required)
            - yield_curve: bool — whether to fetch yield curve (default True)
            - indicators: list[dict] — list of {"series_id": "GDP", "name": "GDP"} (optional)

        Returns:
            List of MarketStateFragments for yield curve and indicators.
        """
        self._last_run = datetime.now(timezone.utc)

        date: str = context.metadata.get("date", "")
        yield_curve: bool = context.metadata.get("yield_curve", True)
        indicators: list[dict] = context.metadata.get("indicators", [])

        if not date:
            raise AgentProcessingError(
                message="No date specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []

        # Fetch yield curve if requested
        if yield_curve:
            try:
                fragment = await self._process_yield_curve(date)
                fragments.append(fragment)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process yield curve",
                    agent_id=self.agent_id,
                    date=date,
                    error=str(e),
                )
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(
                    "YIELD_CURVE", date, safe_error
                )
                fragments.append(fragment)

        # Fetch economic indicators
        for indicator in indicators:
            series_id = indicator.get("series_id")
            name = indicator.get("name")

            if not series_id or not name:
                logger.warning(
                    "Indicator missing series_id or name, skipping",
                    agent_id=self.agent_id,
                )
                continue

            try:
                fragment = await self._process_indicator(series_id, name, date)
                fragments.append(fragment)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process indicator",
                    agent_id=self.agent_id,
                    series_id=series_id,
                    name=name,
                    error=str(e),
                )
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(name, date, safe_error)
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_yield_curve(self, date: str) -> MarketStateFragment:
        """Run the full Perception loop for yield curve data.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch_yield_curve(date)

        # Step 2: VALIDATE
        validation_status = self._validate_yield_curve(raw_data)

        # Step 3: NORMALIZE
        payload = self._normalize_yield_curve(raw_data)

        # Step 4: VERSION
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment
        source_ts = self._extract_source_timestamp(date)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=None,  # Macro data, no specific entity
            data_type=DataType.MACRO_YIELD_CURVE,
            schema_version="1.0.0",
            source_hash=source_hash,
            validation_status=validation_status,
            payload=payload,
        )

        logger.info(
            "Fragment produced",
            agent_id=self.agent_id,
            data_type="MACRO_YIELD_CURVE",
            validation_status=validation_status.value,
            fragment_id=str(fragment.fragment_id),
        )

        return fragment

    async def _process_indicator(
        self, series_id: str, name: str, date: str
    ) -> MarketStateFragment:
        """Run the full Perception loop for an economic indicator.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch_indicator(series_id, date)

        # Step 2: VALIDATE
        validation_status = self._validate_indicator(raw_data)

        # Step 3: NORMALIZE
        payload = self._normalize_indicator(raw_data, name, series_id)

        # Step 4: VERSION
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment
        source_ts = self._extract_source_timestamp(date)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=None,  # Macro data, no specific entity
            data_type=DataType.MACRO_ECONOMIC,
            schema_version="1.0.0",
            source_hash=source_hash,
            validation_status=validation_status,
            payload=payload,
        )

        logger.info(
            "Fragment produced",
            agent_id=self.agent_id,
            data_type="MACRO_ECONOMIC",
            indicator=name,
            validation_status=validation_status.value,
            fragment_id=str(fragment.fragment_id),
        )

        return fragment

    async def _fetch_yield_curve(self, date: str) -> dict[str, Any]:
        """Step 1: FETCH — Pull yield curve data from FRED."""
        try:
            tenors = await self._fred.get_treasury_yields(date)
            return {"tenors": tenors, "date": date}
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch yield curve from FRED: {e}",
                agent_id=self.agent_id,
            ) from e

    async def _fetch_indicator(self, series_id: str, date: str) -> dict[str, Any]:
        """Step 1: FETCH — Pull economic indicator from FRED."""
        try:
            observation = await self._fred.get_latest_observation(series_id)
            return {"observation": observation, "series_id": series_id, "date": date}
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch indicator {series_id} from FRED: {e}",
                agent_id=self.agent_id,
            ) from e

    def _validate_yield_curve(self, raw_data: dict[str, Any]) -> ValidationStatus:
        """Step 2: VALIDATE — Check yield curve completeness.

        Returns VALID, PARTIAL, or QUARANTINED based on data quality.
        """
        tenors = raw_data.get("tenors", {})

        # No tenors at all → quarantine
        if not tenors or not isinstance(tenors, dict) or len(tenors) == 0:
            logger.warning(
                "No tenors in yield curve response — quarantining",
                agent_id=self.agent_id,
            )
            return ValidationStatus.QUARANTINED

        # Fewer than 6 tenors → partial
        if len(tenors) < 6:
            logger.warning(
                "Partial yield curve — fewer than 6 tenors",
                agent_id=self.agent_id,
                num_tenors=len(tenors),
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _validate_indicator(self, raw_data: dict[str, Any]) -> ValidationStatus:
        """Step 2: VALIDATE — Check indicator data completeness.

        Returns VALID or QUARANTINED based on data quality.
        """
        observation = raw_data.get("observation", {})

        # No observation → quarantine
        if not observation or not isinstance(observation, dict):
            logger.warning(
                "No observation in indicator response — quarantining",
                agent_id=self.agent_id,
            )
            return ValidationStatus.QUARANTINED

        value = observation.get("value")

        # Missing or "." value → quarantine
        if not value or value == ".":
            logger.warning(
                "Missing or null value in observation — quarantining",
                agent_id=self.agent_id,
                series_id=raw_data.get("series_id"),
            )
            return ValidationStatus.QUARANTINED

        return ValidationStatus.VALID

    def _normalize_yield_curve(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to MacroYieldPayload dict."""
        tenors = raw_data.get("tenors", {})
        date = raw_data.get("date", "")

        # Compute spreads
        spread_2s10s = None
        spread_3m10y = None

        if "2Y" in tenors and "10Y" in tenors:
            spread_2s10s = (tenors["10Y"] - tenors["2Y"]) * 100

        if "3M" in tenors and "10Y" in tenors:
            spread_3m10y = (tenors["10Y"] - tenors["3M"]) * 100

        payload = MacroYieldPayload(
            curve_date=date,
            tenors=tenors,
            spread_2s10s=spread_2s10s,
            spread_3m10y=spread_3m10y,
            curve_source="FRED",
        )
        return payload.model_dump()

    def _normalize_indicator(
        self, raw_data: dict[str, Any], name: str, series_id: str
    ) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to MacroEconomicPayload dict."""
        observation = raw_data.get("observation", {})

        value = float(observation.get("value", 0.0))
        obs_date = observation.get("date", "")

        # Format observation date as YYYY-MM-DD if needed
        if obs_date and len(obs_date) == 10:
            period = obs_date  # Already in YYYY-MM-DD format
        else:
            period = obs_date

        payload = MacroEconomicPayload(
            indicator=name,
            value=value,
            period=period,
            frequency="MONTHLY",  # Default, can be overridden via metadata
            unit="INDEX",  # Default, can be overridden via metadata
            source_series_id=series_id,
            observation_date=obs_date,
        )
        return payload.model_dump()

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the raw API response for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _extract_source_timestamp(self, date: str) -> datetime:
        """Extract or construct the source timestamp from the date string.

        Returns a datetime object representing the data date.
        """
        try:
            # Parse YYYY-MM-DD format
            dt = datetime.strptime(date, "%Y-%m-%d")
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    def _create_quarantined_fragment(
        self, entity: str, date: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=entity,
            data_type=DataType.MACRO_ECONOMIC,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "date": date},
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
