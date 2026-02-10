"""PERCEPT-FILING: SEC EDGAR filing ingestion agent.

Ingests 10-K, 10-Q, and 8-K filings from SEC EDGAR and produces
MarketStateFragments with data_type FILING_10K, FILING_10Q, or FILING_8K.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data extraction and transformation.
"""

import hashlib
import json
from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.perception.filing_parser import parse_event_filing, parse_financial_filing
from providence.exceptions import AgentProcessingError, DataIngestionError
from providence.infra.edgar_client import EdgarClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import FilingType

logger = structlog.get_logger()

# Map FilingType to DataType
_FILING_TYPE_TO_DATA_TYPE: dict[FilingType, DataType] = {
    FilingType.FORM_10K: DataType.FILING_10K,
    FilingType.FORM_10Q: DataType.FILING_10Q,
    FilingType.FORM_8K: DataType.FILING_8K,
}


class PerceptFiling(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-FILING agent — ingests SEC EDGAR filings.

    FROZEN: No LLM calls. Pure data fetching, parsing, and transformation.
    """

    def __init__(self, edgar_client: EdgarClient) -> None:
        super().__init__(
            agent_id="PERCEPT-FILING",
            agent_type="perception",
            version="1.0.0",
        )
        self._edgar = edgar_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process filings for tickers/types specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch filings for
            - filing_types: list[str] — e.g., ["10-K", "10-Q", "8-K"]
            - count: int — number of recent filings per type (default 1)
            - cik_map: dict[str, str] — ticker → CIK mapping
            - company_names: dict[str, str] — ticker → company name mapping
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        filing_types: list[str] = context.metadata.get("filing_types", ["10-K"])
        count: int = context.metadata.get("count", 1)
        cik_map: dict[str, str] = context.metadata.get("cik_map", {})
        company_names: dict[str, str] = context.metadata.get("company_names", {})

        if not tickers:
            raise AgentProcessingError(
                message="No tickers specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []

        for ticker in tickers:
            for filing_type_str in filing_types:
                try:
                    filing_type = FilingType(filing_type_str)
                    new_fragments = await self._process_filing(
                        ticker=ticker,
                        filing_type=filing_type,
                        count=count,
                        cik=cik_map.get(ticker, ""),
                        company_name=company_names.get(ticker, ticker),
                    )
                    fragments.extend(new_fragments)
                except Exception as e:
                    self._error_count_24h += 1
                    logger.error(
                        "Failed to process filing",
                        agent_id=self.agent_id,
                        ticker=ticker,
                        filing_type=filing_type_str,
                        error=str(e),
                    )
                    fragment = self._create_quarantined_fragment(
                        ticker, filing_type_str, str(e)
                    )
                    fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_filing(
        self,
        ticker: str,
        filing_type: FilingType,
        count: int,
        cik: str,
        company_name: str,
    ) -> list[MarketStateFragment]:
        """Process a specific filing type for a ticker."""
        # Step 1: FETCH — get filing list from EDGAR
        filings = await self._edgar.get_recent_filings(
            ticker=ticker,
            filing_type=filing_type.value,
            count=count,
        )

        if not filings:
            logger.warning(
                "No filings found",
                agent_id=self.agent_id,
                ticker=ticker,
                filing_type=filing_type.value,
            )
            return [self._create_quarantined_fragment(
                ticker, filing_type.value, "No filings found"
            )]

        fragments: list[MarketStateFragment] = []

        for filing_data in filings:
            # Step 2: VALIDATE
            validation_status = self._validate(filing_data, filing_type)

            # Step 3: NORMALIZE — parse into FilingPayload
            filed_date = self._parse_date(filing_data.get("filed_date", ""))
            period_date = self._parse_date(filing_data.get("period_of_report", ""))

            if filing_type == FilingType.FORM_8K:
                payload_obj = parse_event_filing(
                    event_data=filing_data,
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik or filing_data.get("cik", ""),
                    filed_date=filed_date,
                    period_of_report=period_date,
                    raw_text_excerpt=filing_data.get("raw_text_excerpt", ""),
                )
            else:
                xbrl_data = filing_data.get("xbrl_data", filing_data)
                payload_obj = parse_financial_filing(
                    xbrl_data=xbrl_data,
                    filing_type=filing_type,
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik or filing_data.get("cik", ""),
                    filed_date=filed_date,
                    period_of_report=period_date,
                    raw_text_excerpt=filing_data.get("raw_text_excerpt", ""),
                )

            # Step 4: VERSION — hash computed by MarketStateFragment
            source_hash = self._compute_source_hash(filing_data)
            data_type = _FILING_TYPE_TO_DATA_TYPE[filing_type]

            # Step 5: Create fragment
            fragment = MarketStateFragment(
                fragment_id=uuid4(),
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                source_timestamp=datetime.combine(filed_date, datetime.min.time(), tzinfo=timezone.utc),
                entity=ticker,
                data_type=data_type,
                schema_version="1.0.0",
                source_hash=source_hash,
                validation_status=validation_status,
                payload=payload_obj.model_dump(mode="json"),
            )

            logger.info(
                "Filing fragment produced",
                agent_id=self.agent_id,
                ticker=ticker,
                filing_type=filing_type.value,
                validation_status=validation_status.value,
            )
            fragments.append(fragment)

        return fragments

    def _validate(
        self, filing_data: dict[str, Any], filing_type: FilingType
    ) -> ValidationStatus:
        """Step 2: VALIDATE — check data completeness."""
        if filing_type in (FilingType.FORM_10K, FilingType.FORM_10Q):
            # Check for XBRL data presence
            xbrl = filing_data.get("xbrl_data", filing_data)
            facts = xbrl.get("facts", {}).get("us-gaap", {})

            if not facts:
                # No XBRL at all but filing exists
                if filing_data.get("filed_date"):
                    return ValidationStatus.PARTIAL
                return ValidationStatus.QUARANTINED

            return ValidationStatus.VALID

        elif filing_type == FilingType.FORM_8K:
            if filing_data.get("event_type"):
                return ValidationStatus.VALID
            elif filing_data.get("filed_date"):
                return ValidationStatus.PARTIAL
            return ValidationStatus.QUARANTINED

        return ValidationStatus.QUARANTINED

    def _parse_date(self, date_str: str) -> date:
        """Parse a date string, with fallback to today."""
        if not date_str:
            return date.today()
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            return date.today()

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of raw filing data."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _create_quarantined_fragment(
        self, ticker: str, filing_type: str, error_msg: str
    ) -> MarketStateFragment:
        """Create a quarantined fragment for failed ingestion."""
        data_type_map = {
            "10-K": DataType.FILING_10K,
            "10-Q": DataType.FILING_10Q,
            "8-K": DataType.FILING_8K,
        }
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=data_type_map.get(filing_type, DataType.FILING_10K),
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={"error": error_msg, "filing_type": filing_type},
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
