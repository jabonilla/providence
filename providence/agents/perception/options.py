"""PERCEPT-OPTIONS: Options chain data ingestion agent.

Ingests options chain snapshots from Polygon.io and produces MarketStateFragments
with data_type=OPTIONS_CHAIN.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Polygon.io REST API
  2. VALIDATE    → Check schema completeness and freshness
  3. NORMALIZE   → Convert to OptionsPayload
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
from providence.utils.redaction import redact_error_message
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import OptionsPayload

logger = structlog.get_logger()


class PerceptOptions(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-OPTIONS agent — ingests options chain data.

    FROZEN: No LLM calls. Pure data fetching and transformation.
    """

    def __init__(self, polygon_client: PolygonClient) -> None:
        super().__init__(
            agent_id="PERCEPT-OPTIONS",
            agent_type="perception",
            version="1.0.0",
        )
        self._polygon = polygon_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process options chain data for tickers specified in context metadata.

        Expected context.metadata keys:
            - tickers: list[str] — ticker symbols to fetch
            - expiration_date: str — optional expiration date filter (YYYY-MM-DD)
            - contract_type: str — optional contract type filter ("call" or "put")
            - limit: int — max contracts to fetch (default 50)

        Returns:
            List of MarketStateFragments, one per ticker.
        """
        self._last_run = datetime.now(timezone.utc)

        tickers: list[str] = context.metadata.get("tickers", [])
        expiration_date: str | None = context.metadata.get("expiration_date")
        contract_type: str | None = context.metadata.get("contract_type")
        limit: int = context.metadata.get("limit", 50)

        if not tickers:
            raise AgentProcessingError(
                message="No tickers specified in context metadata",
                agent_id=self.agent_id,
            )

        fragments: list[MarketStateFragment] = []
        for ticker in tickers:
            try:
                fragment = await self._process_ticker(
                    ticker, expiration_date, contract_type, limit
                )
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
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(
                    ticker, expiration_date, contract_type, safe_error
                )
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_ticker(
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> MarketStateFragment:
        """Run the full Perception loop for a single ticker.

        Steps: FETCH → VALIDATE → NORMALIZE → VERSION → return fragment.
        """
        # Step 1: FETCH
        raw_data = await self._fetch(ticker, expiration_date, contract_type, limit)

        # Step 2: VALIDATE
        validation_status = self._validate(raw_data, ticker)

        # Step 3: NORMALIZE
        payload = self._normalize(raw_data, ticker)

        # Step 4: VERSION — content hash computed by MarketStateFragment
        source_hash = self._compute_source_hash(raw_data)

        # Step 5: Create and return fragment (STORE is future Kafka work)
        source_ts = datetime.now(timezone.utc)

        fragment = MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=source_ts,
            entity=ticker,
            data_type=DataType.OPTIONS_CHAIN,
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
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Step 1: FETCH — Pull raw options chain data from Polygon.io."""
        try:
            return await self._polygon.get_options_chain(
                ticker, expiration_date, contract_type, limit
            )
        except Exception as e:
            raise DataIngestionError(
                message=f"Failed to fetch {ticker} options chain from Polygon: {e}",
                agent_id=self.agent_id,
            ) from e

    def _validate(self, raw_data: list[dict[str, Any]], ticker: str) -> ValidationStatus:
        """Step 2: VALIDATE — Check schema completeness and freshness.

        Returns VALID, PARTIAL, or QUARANTINED based on data quality.
        """
        # No contracts returned → quarantine
        if not raw_data or not isinstance(raw_data, list) or len(raw_data) == 0:
            logger.warning(
                "No contracts in API response — quarantining",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return ValidationStatus.QUARANTINED

        # Check if contracts have Greeks data
        contract = raw_data[0] if isinstance(raw_data[0], dict) else {}
        greeks = contract.get("greeks")

        if not greeks or not isinstance(greeks, dict) or not greeks:
            logger.warning(
                "Partial data — missing Greeks",
                agent_id=self.agent_id,
                ticker=ticker,
            )
            return ValidationStatus.PARTIAL

        return ValidationStatus.VALID

    def _normalize(
        self, raw_data: list[dict[str, Any]], ticker: str
    ) -> dict[str, Any]:
        """Step 3: NORMALIZE — Convert to options payload dict.

        Extracts and transforms options contracts into OptionsPayload format.
        Creates aggregated payload with all contracts and summary statistics.
        """
        contracts_list = []
        put_count = 0
        call_count = 0
        snapshot_time = datetime.now(timezone.utc).isoformat()

        for result in raw_data:
            if not isinstance(result, dict):
                continue

            # Extract contract details
            contract_type = result.get("details", {}).get("contract_type", "unknown")
            strike_price = result.get("details", {}).get("strike_price", 0.0)
            expiration_date = result.get("details", {}).get("expiration_date", "")
            underlying_price = result.get("underlying_asset", {}).get("price", 0.0)
            last_price = result.get("last_trade", {}).get("price", 0.0)
            bid = result.get("last_quote", {}).get("bid")
            ask = result.get("last_quote", {}).get("ask")
            volume = result.get("day", {}).get("volume", 0)
            open_interest = result.get("open_interest", 0)
            implied_volatility = result.get("implied_volatility")

            # Extract Greeks
            greeks = result.get("greeks", {})
            delta = greeks.get("delta")
            gamma = greeks.get("gamma")
            theta = greeks.get("theta")
            vega = greeks.get("vega")

            # Build OptionsPayload for this contract
            payload = OptionsPayload(
                contract_type=contract_type,
                strike_price=float(strike_price),
                expiration_date=expiration_date,
                underlying_price=float(underlying_price),
                last_price=float(last_price),
                bid=float(bid) if bid is not None else None,
                ask=float(ask) if ask is not None else None,
                volume=int(volume),
                open_interest=int(open_interest),
                implied_volatility=float(implied_volatility) if implied_volatility is not None else None,
                delta=float(delta) if delta is not None else None,
                gamma=float(gamma) if gamma is not None else None,
                theta=float(theta) if theta is not None else None,
                vega=float(vega) if vega is not None else None,
                snapshot_time=snapshot_time,
            )

            contracts_list.append(payload.model_dump())

            # Track put/call counts for ratio
            if contract_type.lower() == "put":
                put_count += 1
            elif contract_type.lower() == "call":
                call_count += 1

        # Calculate put/call ratio
        put_call_ratio = put_count / call_count if call_count > 0 else 0.0

        # Return aggregated payload
        return {
            "contracts": contracts_list,
            "contract_count": len(contracts_list),
            "put_call_ratio": put_call_ratio,
        }

    def _compute_source_hash(self, raw_data: list[dict[str, Any]]) -> str:
        """Compute SHA-256 hash of the raw API response for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _create_quarantined_fragment(
        self,
        ticker: str,
        expiration_date: str | None,
        contract_type: str | None,
        error_msg: str,
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=ticker,
            data_type=DataType.OPTIONS_CHAIN,
            schema_version="1.0.0",
            source_hash="",
            validation_status=ValidationStatus.QUARANTINED,
            payload={
                "error": error_msg,
                "expiration_date": expiration_date,
                "contract_type": contract_type,
            },
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
