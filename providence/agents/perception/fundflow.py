"""PERCEPT-FUNDFLOW: Plaid fund flow data ingestion agent.

Ingests investment transaction and fund flow data from Plaid API
and produces MarketStateFragments with data_type=FUND_FLOW.

Spec Reference: Technical Spec v2.3, Section 4.1

Classification: FROZEN — zero LLM calls. Pure data transformation.

Common Perception Agent Loop:
  1. FETCH       → Pull raw data from Plaid investment transactions API
  2. VALIDATE    → Check transaction data completeness
  3. NORMALIZE   → Aggregate into FundFlowPayload (net flows per account per day)
  4. VERSION     → Compute content hash, assign fragment_id
  5. STORE       → Return MarketStateFragment (Kafka is future work)
  6. ALERT       → If validation fails, set QUARANTINED and log
"""

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.exceptions import AgentProcessingError
from providence.infra.plaid_client import PlaidClient
from providence.schemas.enums import DataType, ValidationStatus
from providence.schemas.market_state import MarketStateFragment
from providence.schemas.payloads import FundFlowPayload
from providence.utils.redaction import redact_error_message

logger = structlog.get_logger()


class PerceptFundFlow(BaseAgent[list[MarketStateFragment]]):
    """PERCEPT-FUNDFLOW agent — ingests fund flow data from Plaid.

    FROZEN: No LLM calls. Pure data fetching and transformation.

    Aggregates individual investment transactions into daily net flow
    summaries per account. Tracks inflows, outflows, transaction counts,
    and top tickers by flow volume.
    """

    def __init__(self, plaid_client: PlaidClient) -> None:
        super().__init__(
            agent_id="PERCEPT-FUNDFLOW",
            agent_type="perception",
            version="1.0.0",
        )
        self._plaid = plaid_client
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> list[MarketStateFragment]:
        """Process fund flow data from Plaid access tokens.

        Expected context.metadata keys:
            - access_tokens: list[str] — Plaid access tokens for linked institutions
            - date: str — end date in YYYY-MM-DD format
            - history_days: int — calendar days of history (default 30)

        Returns:
            List of MarketStateFragments with FUND_FLOW data.
            One fragment per account per day.
        """
        self._last_run = datetime.now(timezone.utc)

        access_tokens: list[str] = context.metadata.get("access_tokens", [])
        date_str: str = context.metadata.get("date", "")
        history_days: int = context.metadata.get("history_days", 30)

        if not access_tokens:
            raise AgentProcessingError(
                message="No Plaid access_tokens specified in context metadata",
                agent_id=self.agent_id,
            )
        if not date_str:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        end_date = date_str
        start_date = (
            datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=history_days)
        ).strftime("%Y-%m-%d")

        fragments: list[MarketStateFragment] = []

        for access_token in access_tokens:
            try:
                token_fragments = await self._process_token(
                    access_token, start_date, end_date, date_str
                )
                fragments.extend(token_fragments)
            except Exception as e:
                self._error_count_24h += 1
                logger.error(
                    "Failed to process Plaid access token",
                    agent_id=self.agent_id,
                    error=str(e),
                )
                safe_error = redact_error_message(str(e))
                fragment = self._create_quarantined_fragment(
                    "UNKNOWN_ACCOUNT", date_str, safe_error
                )
                fragments.append(fragment)

        if any(f.validation_status == ValidationStatus.VALID for f in fragments):
            self._last_success = datetime.now(timezone.utc)

        return fragments

    async def _process_token(
        self,
        access_token: str,
        start_date: str,
        end_date: str,
        observation_date: str,
    ) -> list[MarketStateFragment]:
        """Process all investment transactions for a single Plaid access token.

        Fetches transactions with pagination, aggregates into daily flows
        per account, and produces one fragment per account per day.
        """
        # Step 1: FETCH — paginate through all investment transactions
        all_transactions: list[dict[str, Any]] = []
        accounts_info: dict[str, dict[str, Any]] = {}
        securities_info: dict[str, dict[str, Any]] = {}

        offset = 0
        total = None
        while total is None or offset < total:
            resp = await self._plaid.get_investment_transactions(
                access_token, start_date, end_date, count=500, offset=offset
            )

            txns = resp.get("investment_transactions", [])
            all_transactions.extend(txns)

            # Cache account and security metadata
            for acct in resp.get("accounts", []):
                acct_id = acct.get("account_id", "")
                if acct_id:
                    accounts_info[acct_id] = acct

            for sec in resp.get("securities", []):
                sec_id = sec.get("security_id", "")
                if sec_id:
                    securities_info[sec_id] = sec

            total = resp.get("total_investment_transactions", len(txns))
            offset += len(txns)

            if not txns:
                break

        if not all_transactions:
            logger.warning(
                "No investment transactions returned",
                agent_id=self.agent_id,
                start_date=start_date,
                end_date=end_date,
            )
            return [self._create_quarantined_fragment(
                "NO_TRANSACTIONS", observation_date, "No transactions in range"
            )]

        # Step 2+3: VALIDATE + NORMALIZE — aggregate by (account_id, date)
        # Build daily flow aggregates
        DayKey = tuple[str, str]  # (account_id, date)
        day_flows: dict[DayKey, dict[str, Any]] = defaultdict(lambda: {
            "inflows": 0.0,
            "outflows": 0.0,
            "count": 0,
            "tickers": defaultdict(float),
        })

        for txn in all_transactions:
            acct_id = txn.get("account_id", "UNKNOWN")
            txn_date = txn.get("date", observation_date)
            amount = float(txn.get("amount", 0.0))

            key: DayKey = (acct_id, txn_date)
            day_flows[key]["count"] += 1

            # Plaid: positive amount = money leaving account (buy),
            # negative amount = money entering account (sell/dividend)
            if amount > 0:
                day_flows[key]["outflows"] += amount
            else:
                day_flows[key]["inflows"] += abs(amount)

            # Track ticker volumes
            sec_id = txn.get("security_id", "")
            security = securities_info.get(sec_id, {})
            ticker_symbol = security.get("ticker_symbol", "")
            if ticker_symbol:
                day_flows[key]["tickers"][ticker_symbol] += abs(amount)

        # Step 4+5: VERSION + CREATE fragments
        fragments: list[MarketStateFragment] = []
        for (acct_id, flow_date), agg in day_flows.items():
            inflows = agg["inflows"]
            outflows = agg["outflows"]
            net_flow = inflows - outflows

            # Top tickers by flow volume (up to 10)
            sorted_tickers = sorted(
                agg["tickers"].items(), key=lambda x: x[1], reverse=True
            )
            top_tickers = [t[0] for t in sorted_tickers[:10]]

            # Account metadata
            acct_meta = accounts_info.get(acct_id, {})
            institution_name = acct_meta.get("name") or acct_meta.get("official_name")

            # Anonymize account_id (hash it)
            anon_acct_id = hashlib.sha256(acct_id.encode()).hexdigest()[:16]

            payload = FundFlowPayload(
                account_id=anon_acct_id,
                flow_date=flow_date,
                net_flow=net_flow,
                inflows=inflows,
                outflows=outflows,
                transaction_count=agg["count"],
                category="INVESTMENT",
                institution_name=institution_name,
                top_tickers=top_tickers,
                observation_date=observation_date,
            )

            source_hash = self._compute_source_hash({
                "account_id": anon_acct_id,
                "flow_date": flow_date,
                "net_flow": net_flow,
                "count": agg["count"],
            })

            validation_status = (
                ValidationStatus.VALID if agg["count"] > 0
                else ValidationStatus.PARTIAL
            )

            fragment = MarketStateFragment(
                fragment_id=uuid4(),
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                source_timestamp=self._parse_date(flow_date),
                entity="FUND_FLOW",
                data_type=DataType.FUND_FLOW,
                schema_version="1.0.0",
                source_hash=source_hash,
                validation_status=validation_status,
                payload=payload.model_dump(),
            )
            fragments.append(fragment)

        logger.info(
            "Fund flow fragments produced",
            agent_id=self.agent_id,
            count=len(fragments),
            total_transactions=len(all_transactions),
            start_date=start_date,
            end_date=end_date,
        )

        return fragments

    def _compute_source_hash(self, raw_data: dict[str, Any]) -> str:
        """Compute SHA-256 hash of the data for provenance."""
        raw_bytes = json.dumps(raw_data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()

    def _parse_date(self, date_str: str) -> datetime:
        """Parse YYYY-MM-DD string to UTC datetime."""
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            return datetime.now(timezone.utc)

    def _create_quarantined_fragment(
        self, entity: str, date_str: str, error_msg: str
    ) -> MarketStateFragment:
        """Step 6: ALERT — Create a quarantined fragment for failed ingestion."""
        return MarketStateFragment(
            fragment_id=uuid4(),
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            source_timestamp=datetime.now(timezone.utc),
            entity=entity,
            data_type=DataType.FUND_FLOW,
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
