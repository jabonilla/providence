"""COGNIT-KRONOS: Foundation Model Price Forecasting Agent.

Uses the Kronos foundation model (pre-trained on 12B+ K-line records from
45+ global exchanges) to generate directional price forecasts and convert
them into structured BeliefObjects.

Spec Reference: Providence Architecture — Kronos Integration

Classification: FROZEN — zero LLM calls. Pure model inference.

Thesis types:
  1. Directional forecast (predicted return over horizon)
  2. Trend continuation / reversal (candle trajectory analysis)

Time horizon: 5-60 days (configurable via forecast horizon)

Research Agent Common Loop (FROZEN variant):
  1. RECEIVE CONTEXT  → AgentContext from CONTEXT-SVC
  2. ANALYZE          → Extract OHLCV data from PRICE_OHLCV fragments
  3. HYPOTHESIZE      → Run Kronos model inference for price forecast
  4. EVIDENCE LINK    → Attach fragment_ids from price data
  5. SCORE            → Confidence from forecast sample agreement
  6. INVALIDATE       → Machine-evaluable conditions from predicted levels
  7. EMIT             → Return validated BeliefObject
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import UUID

import structlog

from providence.agents.base import AgentContext, BaseAgent, HealthStatus, AgentStatus
from providence.exceptions import AgentProcessingError
from providence.schemas.belief import (
    Belief,
    BeliefMetadata,
    BeliefObject,
    EvidenceRef,
    InvalidationCondition,
)
from providence.schemas.enums import (
    ComparisonOperator,
    ConditionStatus,
    DataType,
    Direction,
    Magnitude,
    MarketCapBucket,
)
from providence.schemas.market_state import MarketStateFragment

logger = structlog.get_logger()


class CognitKronos(BaseAgent[BeliefObject]):
    """Kronos Foundation Model Research Agent.

    Consumes PRICE_OHLCV MarketStateFragments, runs Kronos model inference,
    and produces BeliefObjects with directional price forecasts.

    FROZEN: No LLM calls. All forecasting is deterministic model inference.

    Args:
        kronos_service: Optional pre-initialized KronosService instance.
            If None, creates a new instance (model loads lazily on first use).
        forecast_horizon: Number of future candles to predict. Default: 20.
    """

    CONSUMED_DATA_TYPES = {DataType.PRICE_OHLCV}

    # Minimum price points for meaningful forecast
    MIN_PRICE_POINTS = 30

    def __init__(
        self,
        kronos_service: "KronosService | None" = None,
        forecast_horizon: int = 20,
    ) -> None:
        super().__init__(
            agent_id="COGNIT-KRONOS",
            agent_type="cognition",
            version="1.0.0",
        )
        self._kronos_service = kronos_service
        self._forecast_horizon = forecast_horizon
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    def _get_service(self) -> "KronosService":
        """Get or lazily create the KronosService."""
        if self._kronos_service is None:
            from providence.services.kronos_service import KronosService
            self._kronos_service = KronosService()
        return self._kronos_service

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop with Kronos forecasting.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Extract OHLCV series from PRICE_OHLCV fragments
          3. HYPOTHESIZE      → Run Kronos model for price forecast
          4. EVIDENCE LINK    → Attach fragment_ids from price data
          5. SCORE            → Confidence from forecast sample agreement
          6. INVALIDATE       → Machine-evaluable conditions from predicted levels
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with directional forecast theses.

        Raises:
            AgentProcessingError: If processing fails or no beliefs generated.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
                fragment_count=len(context.fragments),
            )
            log.info("Starting Kronos forecast analysis")

            # Step 2: ANALYZE — extract OHLCV series
            ohlcv_series = self._extract_ohlcv_series(context.fragments)

            if not ohlcv_series:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="No PRICE_OHLCV fragments found in context",
                    agent_id=self.agent_id,
                )

            log.info("Extracted OHLCV series", ticker_count=len(ohlcv_series))

            # Steps 3-6: HYPOTHESIZE + EVIDENCE + SCORE + INVALIDATE
            beliefs = []
            service = self._get_service()

            for ticker, (ohlcv_records, fragment_ids) in ohlcv_series.items():
                if len(ohlcv_records) < self.MIN_PRICE_POINTS:
                    log.warning(
                        "Insufficient data points for Kronos forecast",
                        ticker=ticker,
                        count=len(ohlcv_records),
                        min_required=self.MIN_PRICE_POINTS,
                    )
                    continue

                try:
                    # Build DataFrame for Kronos
                    import pandas as pd

                    df = pd.DataFrame(ohlcv_records)
                    forecast = await service.predict(
                        ohlcv_data=df,
                        horizon=self._forecast_horizon,
                        ticker=ticker,
                    )

                    # Convert forecast to belief
                    belief = self._forecast_to_belief(
                        ticker=ticker,
                        forecast=forecast,
                        fragment_ids=fragment_ids,
                        current_close=float(df["close"].iloc[-1]),
                    )
                    beliefs.append(belief)

                    log.info(
                        "Kronos forecast generated",
                        ticker=ticker,
                        direction=forecast.predicted_direction,
                        predicted_return=forecast.predicted_return,
                        confidence=forecast.confidence,
                    )

                except ImportError:
                    # Kronos not installed — generate fallback neutral belief
                    log.warning(
                        "Kronos model not installed, generating neutral belief",
                        ticker=ticker,
                    )
                    belief = self._neutral_fallback(ticker, fragment_ids, ohlcv_records)
                    beliefs.append(belief)

                except Exception as e:
                    log.error(
                        "Kronos forecast failed for ticker",
                        ticker=ticker,
                        error=str(e),
                    )
                    # Generate neutral fallback on error
                    belief = self._neutral_fallback(ticker, fragment_ids, ohlcv_records)
                    beliefs.append(belief)

            if not beliefs:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="No beliefs generated from Kronos forecasting",
                    agent_id=self.agent_id,
                )

            # Step 7: EMIT
            belief_object = BeliefObject(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                beliefs=beliefs,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Kronos analysis complete",
                belief_count=len(beliefs),
                tickers=[b.ticker for b in beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-KRONOS processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _extract_ohlcv_series(
        self,
        fragments: list[MarketStateFragment],
    ) -> dict[str, tuple[list[dict], list[UUID]]]:
        """Extract and group OHLCV data by ticker.

        Groups PRICE_OHLCV fragments by entity (ticker), sorts by timestamp,
        and extracts OHLCV records as dicts.

        Args:
            fragments: List of MarketStateFragments.

        Returns:
            Mapping of ticker -> (list of {open, high, low, close, volume}, fragment_ids).
        """
        price_fragments_by_ticker: dict[str, list[MarketStateFragment]] = {}

        for frag in fragments:
            if frag.data_type == DataType.PRICE_OHLCV and frag.entity:
                if frag.entity not in price_fragments_by_ticker:
                    price_fragments_by_ticker[frag.entity] = []
                price_fragments_by_ticker[frag.entity].append(frag)

        result: dict[str, tuple[list[dict], list[UUID]]] = {}

        for ticker, frags in price_fragments_by_ticker.items():
            sorted_frags = sorted(frags, key=lambda f: f.timestamp)

            ohlcv_records = []
            fragment_ids = []

            for frag in sorted_frags:
                payload = frag.payload
                if "close" in payload:
                    record = {
                        "open": float(payload.get("open", payload["close"])),
                        "high": float(payload.get("high", payload["close"])),
                        "low": float(payload.get("low", payload["close"])),
                        "close": float(payload["close"]),
                    }
                    if "volume" in payload:
                        record["volume"] = float(payload["volume"])
                    ohlcv_records.append(record)
                    fragment_ids.append(frag.fragment_id)

            if ohlcv_records:
                result[ticker] = (ohlcv_records, fragment_ids)

        return result

    def _forecast_to_belief(
        self,
        ticker: str,
        forecast: "ForecastResult",
        fragment_ids: list[UUID],
        current_close: float,
    ) -> Belief:
        """Convert a ForecastResult into a Belief.

        Args:
            ticker: Ticker symbol.
            forecast: Kronos ForecastResult.
            fragment_ids: Supporting fragment IDs.
            current_close: Current closing price for invalidation thresholds.

        Returns:
            Belief object with forecast-derived thesis.
        """
        # Direction
        if forecast.predicted_direction == "UP":
            direction = Direction.LONG
        elif forecast.predicted_direction == "DOWN":
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # Magnitude based on predicted return
        abs_return = abs(forecast.predicted_return)
        if abs_return > 0.05:
            magnitude = Magnitude.LARGE
        elif abs_return > 0.02:
            magnitude = Magnitude.MODERATE
        else:
            magnitude = Magnitude.SMALL

        # Thesis summary
        ret_pct = forecast.predicted_return * 100
        thesis_summary = (
            f"Kronos foundation model forecasts {ticker} "
            f"{forecast.predicted_direction} {abs(ret_pct):.1f}% over "
            f"{forecast.horizon} trading days "
            f"(confidence: {forecast.confidence:.0%})"
        )

        # Evidence
        evidence_refs = []
        weight = 1.0 / max(len(fragment_ids), 1)
        for frag_id in fragment_ids[:5]:
            evidence_refs.append(
                EvidenceRef(
                    source_fragment_id=frag_id,
                    field_path="payload",
                    observation=f"OHLCV data used in Kronos model inference",
                    weight=weight,
                )
            )

        # Invalidation conditions
        invalidation_conditions = self._create_invalidation_conditions(
            direction, current_close, forecast
        )

        # Metadata
        metadata = BeliefMetadata(
            sector="UNKNOWN",
            market_cap_bucket=MarketCapBucket.LARGE,
            catalyst_type=None,
        )

        thesis_id = (
            f"KRONOS-{ticker}-{direction.value}-"
            f"{forecast.predicted_return:+.3f}"
        )

        return Belief(
            thesis_id=thesis_id,
            ticker=ticker,
            thesis_summary=thesis_summary,
            direction=direction,
            magnitude=magnitude,
            raw_confidence=forecast.confidence,
            time_horizon_days=forecast.horizon,
            evidence=evidence_refs,
            invalidation_conditions=invalidation_conditions,
            correlated_beliefs=[],
            metadata=metadata,
        )

    def _create_invalidation_conditions(
        self,
        direction: Direction,
        current_close: float,
        forecast: "ForecastResult",
    ) -> list[InvalidationCondition]:
        """Create machine-evaluable invalidation conditions from forecast."""
        conditions = []

        if direction == Direction.LONG:
            # Price drops below entry minus 2x the predicted upside
            stop_level = current_close * (1.0 - abs(forecast.predicted_return) * 2)
            conditions.append(
                InvalidationCondition(
                    description=(
                        f"Price drops below {stop_level:.2f} "
                        f"(2x forecast magnitude below entry)"
                    ),
                    data_source_agent="PERCEPT-PRICE",
                    metric="close",
                    operator=ComparisonOperator.LT,
                    threshold=round(stop_level, 2),
                    status=ConditionStatus.ACTIVE,
                )
            )
            # Forecast confidence threshold
            conditions.append(
                InvalidationCondition(
                    description="Kronos forecast confidence drops below 0.35 on re-evaluation",
                    data_source_agent="COGNIT-KRONOS",
                    metric="forecast_confidence",
                    operator=ComparisonOperator.LT,
                    threshold=0.35,
                    status=ConditionStatus.ACTIVE,
                )
            )

        elif direction == Direction.SHORT:
            # Price rises above entry plus 2x the predicted downside
            stop_level = current_close * (1.0 + abs(forecast.predicted_return) * 2)
            conditions.append(
                InvalidationCondition(
                    description=(
                        f"Price rises above {stop_level:.2f} "
                        f"(2x forecast magnitude above entry)"
                    ),
                    data_source_agent="PERCEPT-PRICE",
                    metric="close",
                    operator=ComparisonOperator.GT,
                    threshold=round(stop_level, 2),
                    status=ConditionStatus.ACTIVE,
                )
            )
            conditions.append(
                InvalidationCondition(
                    description="Kronos forecast confidence drops below 0.35 on re-evaluation",
                    data_source_agent="COGNIT-KRONOS",
                    metric="forecast_confidence",
                    operator=ComparisonOperator.LT,
                    threshold=0.35,
                    status=ConditionStatus.ACTIVE,
                )
            )

        else:  # NEUTRAL
            # Any strong directional move invalidates neutral thesis
            conditions.append(
                InvalidationCondition(
                    description="Price moves >5% from current level",
                    data_source_agent="PERCEPT-PRICE",
                    metric="close",
                    operator=ComparisonOperator.GT,
                    threshold=round(current_close * 1.05, 2),
                    status=ConditionStatus.ACTIVE,
                )
            )
            conditions.append(
                InvalidationCondition(
                    description="Price moves <-5% from current level",
                    data_source_agent="PERCEPT-PRICE",
                    metric="close",
                    operator=ComparisonOperator.LT,
                    threshold=round(current_close * 0.95, 2),
                    status=ConditionStatus.ACTIVE,
                )
            )

        return conditions

    def _neutral_fallback(
        self,
        ticker: str,
        fragment_ids: list[UUID],
        ohlcv_records: list[dict],
    ) -> Belief:
        """Generate a neutral low-confidence belief when Kronos is unavailable."""
        current_close = ohlcv_records[-1]["close"] if ohlcv_records else 0.0

        evidence_refs = []
        for frag_id in fragment_ids[:3]:
            evidence_refs.append(
                EvidenceRef(
                    source_fragment_id=frag_id,
                    field_path="payload",
                    observation="OHLCV data (Kronos model unavailable)",
                    weight=0.33,
                )
            )

        return Belief(
            thesis_id=f"KRONOS-{ticker}-NEUTRAL-fallback",
            ticker=ticker,
            thesis_summary=(
                f"Kronos model unavailable for {ticker}; "
                f"neutral stance with low confidence pending model availability"
            ),
            direction=Direction.NEUTRAL,
            magnitude=Magnitude.SMALL,
            raw_confidence=0.2,
            time_horizon_days=20,
            evidence=evidence_refs,
            invalidation_conditions=[
                InvalidationCondition(
                    description="Kronos model becomes available for re-evaluation",
                    data_source_agent="COGNIT-KRONOS",
                    metric="model_available",
                    operator=ComparisonOperator.EQ,
                    threshold=1.0,
                    status=ConditionStatus.ACTIVE,
                ),
            ],
            correlated_beliefs=[],
            metadata=BeliefMetadata(
                sector="UNKNOWN",
                market_cap_bucket=MarketCapBucket.LARGE,
            ),
        )

    def get_health(self) -> HealthStatus:
        """Report health status."""
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
            message=(
                f"Kronos model: {'loaded' if self._kronos_service and self._kronos_service.is_loaded else 'not loaded'}"
            ),
        )
