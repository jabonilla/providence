"""PERCEPT-FACTORS: Fama-French factor returns ingestion agent.

Ingests daily factor returns (5-factor model + momentum) from Kenneth French
Data Library and produces MarketStateFragments with data_type=FACTOR_RETURNS.

Spec Reference: Technical Spec v2.3, Section 4.1 (PERCEPT-FACTORS)

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Kenneth French Data Library via pandas_datareader
  2. VALIDATE    → Check data completeness and date range coverage
  3. NORMALIZE   → Convert to FactorReturnsPayload
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
from providence.exceptions import AgentProcessingError
from providence.infra.famafrench_client import FamaFrenchClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import FactorReturnsPayload
from providence.utils.redaction import redact_error_message

logger = structlog.get_logger()


class PerceptFactors(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-FACTORS agent — ingests Fama-French factor returns.

    FROZEN: No LLM calls. Pure data fetching and transformation.

    Unlike ticker-based agents, this agent produces one fragment per
    trading day for the entire market. Entity is always "MARKET".
    """

    def __init__(self, famafrench_client: FamaFrenchClient) -> None:
        super().__init__(
            agent_id="PERCEPT-FACTORS",
            agent_type="perception",
            version="1.0.0",
        )
        self._ff = famafrench_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process factor returns for the date range in context metadata.

        Expected context.metadata keys:
            - date: str — end date in YYYY-MM-DD format
            - history_days: int — calendar days of history (default 30)

        Returns:
            List of MarketStateFragments with FACTOR_RETURNS data.
            One fragment per trading day in the range.
        """
        self._last_run = datetime.now(timezone.utc)

        date_str: str = context.metadata.get("date", "")
        history_days: int = context.metadata.get("history_days", 30)

        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        end_date = date_str
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=history_days)
        ).strftime("%Y-%m-%d")

        fragments: list[MarketStateFragment] = []

        try:
            # Step 1: FETCH — 5-factor data
            five_factors = await self._ff.get_five_factors_daily(start_date, end_date)

            # Step 1b: FETCH — momentum data
            momentum_data: list[dict[str, Any]] = []
            try:
                momentum_data = await self._ff.get_momentum_daily(start_date, end_date)
            except Exception as e:
                logger.warning(
                    "Could not fetch momentum data — proceeding without it",
                    agent_id=self.agent_id,
                    error=str(e),
                )

            # Index momentum by date for merging
            mom_by_date: dict[str, float] = {}
            for m in momentum_data:
                mom_by_date[m["date"]] = m.get("mom", 0.0)

            # Step 2+3+4+5: VALIDATE + NORMALIZE + VERSION + CREATE per day
            for day_data in five_factors:
                factor_date = day_data.get("date", "")

                # Merge momentum
                day_data["mom"] = mom_by_date.get(factor_date)

                validation_status = self._validate(day_data)
                payload = self._normalize(day_data)
                source_hash = self._compute_source_hash(day_data)

                fragment = MarketStateFragment(
                    fragment_id=uuid4(),
                    agent_id=self.agent_id,
                    timestamp=datetime.now(timezone.utc),
                    source_timestamp=self._parse_date(factor_date),
                    entity="MARKET",
                    data_type=DataType.FACTOR_RETURNS,
                    schema_version="1.0.0",
                    source_hash=source_hash,
                    validation_status=validation_status,
                    payload=payload,
                )
                fragments.append(fragment)

            logger.info(
                "Factor fragments produced",
                agent_id=self.agent_id,
                count=len(fragments),
                start_date=start_date,
                end_date=end_date,
            )

            if fragments:
                self._last_success = datetime.now(timezone.utc)

        except Exception as e:
            self._error_count_24h += 1
            logger.error(
                "Failed to fetch factor data",
                agent_id=self.agent_id,
                error=str(e),
            )
            safe_error = redact_error_message(str(e))
            fragments.append(self._create_quarantined_fragment(date_str, safe_error))

        return fragments

    def _validate(self, day_data: dict[str, Any]) -> ValidationStatus:
        """Step 2: VALIDATE — Check factor data completeness."""
        required_keys = {"date", "mkt_rf", "smb", "hml", "rf"}
        present = set(day_data.keys()) & required_keys

        if not present or "mkt_rf" not in day_data:
            return ValidationStatus.QUARANTINED

        if present != required_keys:
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize(self, day_data: dict[str, Any]) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to FactorReturnsPayload dict."""
        payload = FactorReturnsPayload(
            date=day_data.get("date", ""),
            mkt_rf=float(day_data.get("mkt_rf", 0.0)),
            smb=float(day_data.get("smb", 0.0)),
            hml=float(day_data.get("hml", 0.0)),
            rmw=float(day_data.get("rmw", 0.0)) if day_data.get("rmw") is not None else None,
            cma=float(day_data.get("cma", 0.0)) if day_data.get("cma") is not None else None,
            rf=float(day_data.get("rf", 0.0)),
            mom=float(day_data["mom"]) if day_data.get("mom") is not None else None,
            dataset="F-F_Research_Data_5_Factors_2x3_daily",
        )
        return payload.model_dump()

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the raw data for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse YYYY-MM-DD string to UTC datetime."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    def _create_quarantined_fragment(
        self, date_str: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity="MARKET",
            data_type=DataType.FACTOR_RETURNS,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "date": date_str},
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
