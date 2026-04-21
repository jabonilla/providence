"""PERCEPT-ALPHAVANTAGE: Alpha Vantage earnings data ingestion agent.

Ingests quarterly earnings data from Alpha Vantage and produces
MarketStateFragments with data_type=ALPHAVANTAGE_EARNINGS.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Alpha Vantage REST API
  2. VALIDATE    → Check schema completeness and freshness
  3. NORMALIZE   → Convert to AlphaVantageEarningsPayload
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
from providence.exceptions import AgentProcessingError
from providence.infra.alphavantage_client import AlphaVantageClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import AlphaVantageEarningsPayload
from providence.utils.redaction import redact_error_message

logger = structlog.get_logger()


class PerceptAlphaVantage(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-ALPHAVANTAGE agent — ingests earnings data from Alpha Vantage.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, alphavantage_client: AlphaVantageClient) -> None:
        super().__init__(
            agent_id="PERCEPT-ALPHAVANTAGE",
            agent_type="perception",
            version="1.0.0",
        )
        self._av = alphavantage_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process earnings data for tickers specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch
            - date: str — observation date in YYYY-MM-DD format
            - max_quarters: int — max quarterly earnings to return (default 4)

        Returns:
            List of MarketStateFragments with ALPHAVANTAGE_EARNINGS data.
            One fragment per quarterly earnings period per ticker.
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        date_str: str = context.metadata.get("date", "")
        max_quarters: int = context.metadata.get("max_quarters", 4)

        if not tickers:
            raise AgentProcessingError(
                message="No tickers specified in context metadata",
                agent_id=self.agent_id,
            )
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        fragments: list[MarketStateFragment] = []
        for ticker in tickers:
            try:
                ticker_fragments = await self._process_ticker(
                    ticker, date_str, max_quarters
                )
                fragments.extend(ticker_fragments)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process ticker",
                    agent_id=self.agent_id,
                    ticker=ticker,
                    error=str(e),
                )
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(ticker, date_str, safe_error)
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_ticker(
        self, ticker: str, date_str: str, max_quarters: int
    ) -> list[MarketStateFragment]:
        """Run the full Perception loop for a single ticker.

        Fetches earnings data and also the income statement for supplemental
        financial metrics (revenue, gross profit, EBITDA, net income).
        """
        # Step 1: FETCH earnings
        earnings_data = await self._av.get_earnings(ticker)

        # Step 2: VALIDATE
        quarterly = earnings_data.get("quarterlyEarnings", [])
        if not quarterly:
            logger.warning(
                "No quarterly earnings data",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return [self._create_quarantined_fragment(
                ticker, date_str, "No quarterly earnings data returned"
            )]

        # Optionally fetch income statement for supplemental metrics
        income_data: dict[str, Any] = {}
        try:
            income_resp = await self._av.get_income_statement(ticker)
            quarterly_reports = income_resp.get("quarterlyReports", [])
            # Index by fiscalDateEnding for lookup
            for report in quarterly_reports:
                fde = report.get("fiscalDateEnding", "")
                if fde:
                    income_data[fde] = report
        except Exception as e:
            logger.warning(
                "Could not fetch income statement — proceeding with earnings only",
                agent_id=self.agent_id,
                ticker=ticker,
                error=str(e),
            )

        # Step 3+4+5: NORMALIZE + VERSION + CREATE fragments
        fragments: list[MarketStateFragment] = []
        for quarter in quarterly[:max_quarters]:
            validation_status = self._validate_quarter(quarter, ticker)
            payload = self._normalize_quarter(quarter, ticker, date_str, income_data)
            source_hash = self._compute_source_hash(quarter)

            fragment = MarketStateFragment(
                fragment_id=uuid4(),
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                source_timestamp=datetime.now(timezone.utc),
                entity=ticker,
                data_type=DataType.ALPHAVANTAGE_EARNINGS,
                schema_version="1.0.0",
                source_hash=source_hash,
                validation_status=validation_status,
                payload=payload,
            )
            fragments.append(fragment)

            logger.info(
                "Fragment produced",
                agent_id=self.agent_id,
                ticker=ticker,
                fiscal_date=quarter.get("fiscalDateEnding", "unknown"),
                validation_status=validation_status.value,
                fragment_id=str(fragment.fragment_id),
            )

        return fragments

    def _validate_quarter(
        self, quarter: dict[str, Any], ticker: str
    ) -> ValidationStatus:
        """Step 2: VALIDATE — Check quarterly earnings data completeness."""
        if not quarter or not isinstance(quarter, dict):
            return ValidationStatus.QUARANTINED

        required = {"fiscalDateEnding", "reportedEPS"}
        present = set(quarter.keys()) & required

        if not present:
            return ValidationStatus.QUARANTINED

        if present != required:
            logger.warning(
                "Partial earnings data",
                agent_id=self.agent_id,
                ticker=ticker,
                missing=list(required - present),
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize_quarter(
        self,
        quarter: dict[str, Any],
        ticker: str,
        date_str: str,
        income_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to AlphaVantageEarningsPayload dict."""
        fiscal_date = quarter.get("fiscalDateEnding", date_str)

        # Parse numeric fields safely
        def _safe_float(val: Any) -> float | None:
            if val is None or val == "None" or val == "":
                return None
            try:
                return float(val)
            except (ValueError, TypeError):
                return None

        reported_eps = _safe_float(quarter.get("reportedEPS"))
        estimated_eps = _safe_float(quarter.get("estimatedEPS"))
        surprise = _safe_float(quarter.get("surprise"))
        surprise_pct = _safe_float(quarter.get("surprisePercentage"))

        # Supplemental income statement data
        income_report = income_data.get(fiscal_date, {})
        revenue = _safe_float(income_report.get("totalRevenue"))
        gross_profit = _safe_float(income_report.get("grossProfit"))
        ebitda = _safe_float(income_report.get("ebitda"))
        net_income = _safe_float(income_report.get("netIncome"))

        payload = AlphaVantageEarningsPayload(
            ticker=ticker,
            fiscal_date_ending=fiscal_date,
            reported_eps=reported_eps,
            estimated_eps=estimated_eps,
            surprise=surprise,
            surprise_pct=surprise_pct,
            reported_date=quarter.get("reportedDate"),
            revenue=revenue,
            estimated_revenue=None,  # Not provided by Alpha Vantage
            revenue_surprise=None,
            gross_profit=gross_profit,
            ebitda=ebitda,
            net_income=net_income,
            observation_date=date_str,
        )
        return payload.model_dump()

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the raw API response for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _create_quarantined_fragment(
        self, ticker: str, date_str: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=DataType.ALPHAVANTAGE_EARNINGS,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "ticker": ticker, "date": date_str},
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
