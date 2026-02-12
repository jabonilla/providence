"""PERCEPT-CDS: Credit Default Swap data ingestion agent.

Ingests CDS spread data from FRED API and produces MarketStateFragments
with data_type=MACRO_CDS.

Note: Real CDS data typically comes from specialized providers (Markit/IHS).
For our implementation, we use FRED series for investment-grade and high-yield
CDS indices as proxies.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from FRED REST API
  2. VALIDATE    → Check schema completeness and data quality
  3. NORMALIZE   → Convert to CdsPayload
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
from providence.schemas.payloads import CdsPayload

logger = structlog.get_logger()


class PerceptCds(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-CDS agent — ingests CDS spread data from FRED.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, fred_client: FredClient) -> None:
        super().__init__(
            agent_id="PERCEPT-CDS",
            agent_type="perception",
            version="1.0.0",
        )
        self._fred = fred_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process CDS spread data for entities specified in context metadata.

        Expected context.metadata keys:
            - date: str — date in YYYY-MM-DD format (required)
            - entities: list[dict] — list of CDS entities to fetch (required)
              Each entity dict should have:
                - name: str — reference entity name (e.g., "IG_CDX")
                - series_id: str — FRED series ID (e.g., "BAMLC0A0CM")
                - tenor: str — CDS tenor (default "5Y")

        Example entities:
            [
                {"name": "IG_CDX", "series_id": "BAMLC0A0CM", "tenor": "5Y"},
                {"name": "HY_CDX", "series_id": "BAMLH0A0HYM2", "tenor": "5Y"}
            ]

        Returns:
            List of MarketStateFragments, one per entity.
        """
        self._last_run = datetime.now(timezone.utc)

        date: str = context.metadata.get("date", "")
        entities: list[dict] = context.metadata.get("entities", [])

        if not date:
            raise AgentProcessingError(
                message="No date specified in context metadata",
                agent_id=self.agent_id,
            )

        if not entities:
            raise AgentProcessingError(
                message="No entities specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []

        for entity in entities:
            name = entity.get("name")
            series_id = entity.get("series_id")
            tenor = entity.get("tenor", "5Y")

            if not name or not series_id:
                logger.warning(
                    "Entity missing name or series_id, skipping",
                    agent_id=self.agent_id,
                )
                continue

            try:
                fragment = await self._process_entity(name, series_id, tenor, date)
                fragments.append(fragment)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process CDS entity",
                    agent_id=self.agent_id,
                    entity_name=name,
                    series_id=series_id,
                    error=str(e),
                )
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(name, date, safe_error)
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_entity(
        self, name: str, series_id: str, tenor: str, date: str
    ) -> MarketStateFragment:
        """Run the full Perception loop for a single CDS entity.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch_entity(series_id, date)

        # Step 2: VALIDATE
        validation_status = self._validate_entity(raw_data)

        # Step 3: NORMALIZE
        payload = self._normalize_entity(raw_data, name, tenor)

        # Step 4: VERSION
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment
        source_ts = self._extract_source_timestamp(date)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=name,
            data_type=DataType.MACRO_CDS,
            schema_version="1.0.0",
            source_hash=source_hash,
            validation_status=validation_status,
            payload=payload,
        )

        logger.info(
            "Fragment produced",
            agent_id=self.agent_id,
            entity=name,
            data_type="MACRO_CDS",
            validation_status=validation_status.value,
            fragment_id=str(fragment.fragment_id),
        )

        return fragment

    async def _fetch_entity(self, series_id: str, date: str) -> dict[str, Any]:
        """Step 1: FETCH — Pull CDS observations from FRED using 5-day lookback.

        Uses a 5-day lookback to get current and previous values for comparison.
        """
        try:
            # Calculate 5-day lookback date
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            date_minus_5 = (date_obj - timedelta(days=5)).strftime("%Y-%m-%d")

            # Fetch up to 2 observations (current and previous)
            observations = await self._fred.get_series_observations(
                series_id, start_date=date_minus_5, end_date=date, limit=2
            )

            return {
                "observations": observations,
                "series_id": series_id,
                "date": date,
            }
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch CDS series {series_id} from FRED: {e}",
                agent_id=self.agent_id,
            ) from e

    def _validate_entity(self, raw_data: dict[str, Any]) -> ValidationStatus:
        """Step 2: VALIDATE — Check CDS observations completeness.

        Returns VALID or QUARANTINED based on data quality.
        """
        observations = raw_data.get("observations", [])

        # No observations → quarantine
        if not observations or not isinstance(observations, list) or len(observations) == 0:
            logger.warning(
                "No observations in CDS response — quarantining",
                agent_id=self.agent_id,
                series_id=raw_data.get("series_id"),
            )
            return ValidationStatus.QUARANTINED

        # Check first (most recent) observation
        first_obs = observations[0] if isinstance(observations[0], dict) else {}
        value = first_obs.get("value")

        # Missing or "." value → quarantine
        if not value or value == ".":
            logger.warning(
                "Missing or null value in first observation — quarantining",
                agent_id=self.agent_id,
                series_id=raw_data.get("series_id"),
            )
            return ValidationStatus.QUARANTINED

        return ValidationStatus.VALID

    def _normalize_entity(
        self, raw_data: dict[str, Any], name: str, tenor: str
    ) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to CdsPayload dict."""
        observations = raw_data.get("observations", [])
        date = raw_data.get("date", "")

        if not observations:
            return CdsPayload(
                reference_entity=name,
                tenor=tenor,
                spread_bps=0.0,
                recovery_rate=0.4,
                currency="USD",
                observation_date=date,
            ).model_dump()

        # Get most recent (first) observation
        current_obs = observations[0] if isinstance(observations[0], dict) else {}
        current_spread = float(current_obs.get("value", 0.0))
        current_date = current_obs.get("date", date)

        payload_dict = {
            "reference_entity": name,
            "tenor": tenor,
            "spread_bps": current_spread,
            "recovery_rate": 0.4,
            "currency": "USD",
            "observation_date": current_date,
        }

        # If we have a second (previous) observation, add spread change
        if len(observations) >= 2:
            previous_obs = observations[1] if isinstance(observations[1], dict) else {}
            previous_spread = float(previous_obs.get("value", 0.0))
            spread_change = current_spread - previous_spread

            payload_dict["previous_spread_bps"] = previous_spread
            payload_dict["spread_change_bps"] = spread_change

        payload = CdsPayload(**payload_dict)
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
            data_type=DataType.MACRO_CDS,
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
