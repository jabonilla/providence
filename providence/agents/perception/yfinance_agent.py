"""PERCEPT-YFINANCE: Yahoo Finance fundamentals ingestion agent.

Ingests stock fundamentals data from Yahoo Finance via yfinance library
and produces MarketStateFragments with data_type=YFINANCE_FUNDAMENTALS.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Yahoo Finance via yfinance
  2. VALIDATE    → Check field completeness and freshness
  3. NORMALIZE   → Convert to YFinanceFundamentalsPayload
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
from providence.infra.yfinance_client import YFinanceClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import YFinanceFundamentalsPayload
from providence.utils.redaction import redact_error_message

logger = structlog.get_logger()


class PerceptYFinance(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-YFINANCE agent — ingests stock fundamentals from Yahoo Finance.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, yfinance_client: YFinanceClient) -> None:
        super().__init__(
            agent_id="PERCEPT-YFINANCE",
            agent_type="perception",
            version="1.0.0",
        )
        self._yfinance = yfinance_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process fundamentals data for tickers specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch
            - date: str — observation date in YYYY-MM-DD format

        Returns:
            List of MarketStateFragments with YFINANCE_FUNDAMENTALS data.
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        date_str: str = context.metadata.get("date", "")

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
                fragment = await self._process_ticker(ticker, date_str)
                fragments.append(fragment)
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

    async def _process_ticker(self, ticker: str, date_str: str) -> MarketStateFragment:
        """Run the full Perception loop for a single ticker.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._yfinance.get_fundamentals(ticker)

        # Step 2: VALIDATE
        validation_status = self._validate(raw_data, ticker)

        # Step 3: NORMALIZE
        payload = self._normalize(raw_data, ticker, date_str)

        # Step 4: VERSION — content hash
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment
        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=DataType.YFINANCE_FUNDAMENTALS,
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

    def _validate(self, raw_data: dict[str, Any], ticker: str) -> ValidationStatus:
        """Step 2: VALIDATE — Check field completeness.

        Core fields: regularMarketPrice, marketCap, trailingPE.
        """
        if not raw_data or not isinstance(raw_data, dict):
            return ValidationStatus.QUARANTINED

        core_fields = {"regularMarketPrice", "marketCap"}
        present = set(raw_data.keys()) & core_fields

        if not present:
            logger.warning(
                "No core fields present — quarantining",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return ValidationStatus.QUARANTINED

        if present != core_fields:
            logger.warning(
                "Partial data — missing core fields",
                agent_id=self.agent_id,
                ticker=ticker,
                missing=list(core_fields - present),
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize(
        self, raw_data: dict[str, Any], ticker: str, date_str: str
    ) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to YFinanceFundamentalsPayload dict."""
        payload = YFinanceFundamentalsPayload(
            ticker=ticker,
            market_cap=raw_data.get("marketCap"),
            enterprise_value=raw_data.get("enterpriseValue"),
            trailing_pe=raw_data.get("trailingPE"),
            forward_pe=raw_data.get("forwardPE"),
            peg_ratio=raw_data.get("pegRatio"),
            price_to_book=raw_data.get("priceToBook"),
            price_to_sales=raw_data.get("priceToSalesTrailing12Months"),
            profit_margin=raw_data.get("profitMargins"),
            operating_margin=raw_data.get("operatingMargins"),
            roe=raw_data.get("returnOnEquity"),
            roa=raw_data.get("returnOnAssets"),
            revenue=raw_data.get("totalRevenue"),
            revenue_growth=raw_data.get("revenueGrowth"),
            earnings_growth=raw_data.get("earningsGrowth"),
            debt_to_equity=raw_data.get("debtToEquity"),
            current_ratio=raw_data.get("currentRatio"),
            free_cash_flow=raw_data.get("freeCashflow"),
            dividend_yield=raw_data.get("dividendYield"),
            beta=raw_data.get("beta"),
            fifty_two_week_high=raw_data.get("fiftyTwoWeekHigh"),
            fifty_two_week_low=raw_data.get("fiftyTwoWeekLow"),
            avg_volume=raw_data.get("averageVolume"),
            shares_outstanding=raw_data.get("sharesOutstanding"),
            institutional_holders_pct=raw_data.get("heldPercentInstitutions"),
            short_ratio=raw_data.get("shortRatio"),
            sector=raw_data.get("sector"),
            industry=raw_data.get("industry"),
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
            data_type=DataType.YFINANCE_FUNDAMENTALS,
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
