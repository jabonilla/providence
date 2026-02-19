"""COGNIT-TECHNICAL: Technical Analysis Research Agent.

Performs technical analysis using computed indicators (SMA, RSI, MACD,
Bollinger Bands, momentum) to generate investment theses.

Spec Reference: Technical Spec v2.3, Section 4.2

Classification: FROZEN — zero LLM calls. Pure computation.

Thesis types:
  1. Trend following (SMA crossovers, momentum)
  2. Mean reversion (RSI overbought/oversold, Bollinger Band touches)
  3. Momentum divergence (MACD signal crossovers)
  4. Volume confirmation (volume spikes with price moves)

Time horizon: 5-60 days

Research Agent Common Loop (FROZEN variant):
  1. RECEIVE CONTEXT  → AgentContext from CONTEXT-SVC
  2. ANALYZE          → Extract price series from PRICE_OHLCV fragments
  3. HYPOTHESIZE      → Compute indicators, generate signal-based thesis
  4. EVIDENCE LINK    → Attach fragment_ids from price data
  5. SCORE            → Confidence based on signal agreement count
  6. INVALIDATE       → Machine-evaluable conditions from indicator thresholds
  7. EMIT             → Return validated BeliefObject
"""

from datetime import datetime, timezone
from uuid import UUID

import structlog

from providence.agents.base import AgentContext, BaseAgent, HealthStatus, AgentStatus
from providence.agents.cognition.technical_indicators import (
    TechnicalSignals,
    compute_all_signals,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.belief import Belief, BeliefMetadata, BeliefObject, EvidenceRef, InvalidationCondition
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


class CognitTechnical(BaseAgent[BeliefObject]):
    """Technical analysis Research Agent.

    Consumes PRICE_OHLCV and OPTIONS_CHAIN MarketStateFragments.
    Produces BeliefObjects with investment theses based on technical indicators.

    FROZEN: No LLM calls. All analysis is deterministic computation.
    """

    CONSUMED_DATA_TYPES = {DataType.PRICE_OHLCV, DataType.OPTIONS_CHAIN}

    # Minimum price points needed for meaningful analysis
    MIN_PRICE_POINTS = 20

    def __init__(self) -> None:
        super().__init__(
            agent_id="COGNIT-TECHNICAL",
            agent_type="cognition",
            version="1.0.0",
        )
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> BeliefObject:
        """Execute the Research Agent common loop.

        Steps:
          1. RECEIVE CONTEXT  → Validated by type
          2. ANALYZE          → Extract price series from PRICE_OHLCV fragments
          3. HYPOTHESIZE      → Compute indicators, generate signal-based thesis
          4. EVIDENCE LINK    → Attach fragment_ids from price data
          5. SCORE            → Confidence based on signal agreement count
          6. INVALIDATE       → Machine-evaluable conditions from indicator thresholds
          7. EMIT             → Return validated BeliefObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            BeliefObject with investment theses based on technical indicators.

        Raises:
            AgentProcessingError: If processing fails or no beliefs are generated.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            # Step 1: RECEIVE CONTEXT — already validated by caller
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
                fragment_count=len(context.fragments),
            )
            log.info("Starting technical analysis")

            # Step 2: ANALYZE — extract price series from PRICE_OHLCV fragments
            price_series = self._extract_price_series(context.fragments)

            if not price_series:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="No PRICE_OHLCV fragments found in context",
                    agent_id=self.agent_id,
                )

            log.info("Extracted price series", ticker_count=len(price_series))

            # Steps 3-6: HYPOTHESIZE + EVIDENCE LINK + SCORE + INVALIDATE
            beliefs = []
            for ticker, (close_prices, volumes, fragment_ids) in price_series.items():
                # Skip tickers with insufficient data
                if len(close_prices) < self.MIN_PRICE_POINTS:
                    log.warning(
                        "Insufficient price points for ticker",
                        ticker=ticker,
                        price_count=len(close_prices),
                        min_required=self.MIN_PRICE_POINTS,
                    )
                    continue

                # Compute all technical indicators
                signals = compute_all_signals(close_prices, volumes)

                # Generate beliefs from signals
                ticker_beliefs = self._signals_to_beliefs(ticker, signals, fragment_ids)
                beliefs.extend(ticker_beliefs)

                log.info(
                    "Generated beliefs for ticker",
                    ticker=ticker,
                    belief_count=len(ticker_beliefs),
                    net_signal=signals.net_signal,
                )

            if not beliefs:
                self._error_count_24h += 1
                raise AgentProcessingError(
                    message="No beliefs generated from technical analysis",
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
                "Technical analysis complete",
                belief_count=len(beliefs),
                tickers=[b.ticker for b in beliefs],
            )
            return belief_object

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"COGNIT-TECHNICAL processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _extract_price_series(
        self, fragments: list[MarketStateFragment]
    ) -> dict[str, tuple[list[float], list[float], list[UUID]]]:
        """Extract and group price series by ticker.

        Groups PRICE_OHLCV fragments by entity (ticker), sorts by timestamp,
        and extracts close prices, volumes, and fragment IDs.

        Args:
            fragments: List of MarketStateFragments.

        Returns:
            Mapping of ticker -> (close_prices, volumes, fragment_ids).
            Prices and volumes are in chronological order (oldest first).
        """
        price_fragments_by_ticker: dict[str, list[MarketStateFragment]] = {}

        # Filter and group PRICE_OHLCV fragments
        for frag in fragments:
            if frag.data_type == DataType.PRICE_OHLCV and frag.entity:
                if frag.entity not in price_fragments_by_ticker:
                    price_fragments_by_ticker[frag.entity] = []
                price_fragments_by_ticker[frag.entity].append(frag)

        # Extract and sort price series
        result: dict[str, tuple[list[float], list[float], list[UUID]]] = {}

        for ticker, frags in price_fragments_by_ticker.items():
            # Sort by timestamp (oldest first)
            sorted_frags = sorted(frags, key=lambda f: f.timestamp)

            close_prices = []
            volumes = []
            fragment_ids = []

            for frag in sorted_frags:
                payload = frag.payload
                # Extract close price and volume from PRICE_OHLCV payload
                if "close" in payload:
                    close_prices.append(float(payload["close"]))
                if "volume" in payload:
                    volumes.append(float(payload["volume"]))
                fragment_ids.append(frag.fragment_id)

            # Only include if we have valid price data
            if close_prices:
                result[ticker] = (close_prices, volumes if volumes else [], fragment_ids)

        return result

    def _signals_to_beliefs(
        self,
        ticker: str,
        signals: TechnicalSignals,
        fragment_ids: list[UUID],
    ) -> list[Belief]:
        """Generate investment beliefs from technical signals.

        Determines direction (LONG, SHORT, NEUTRAL), magnitude (SMALL, MODERATE, LARGE),
        confidence, time horizon, and invalidation conditions based on signal patterns.

        Args:
            ticker: Ticker symbol.
            signals: TechnicalSignals with computed indicators.
            fragment_ids: Fragment IDs supporting this analysis.

        Returns:
            List of Belief objects.
        """
        beliefs = []

        # Determine direction and magnitude based on net_signal
        if signals.net_signal >= 3:
            direction = Direction.LONG
            magnitude = Magnitude.LARGE
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        elif signals.net_signal == 2:
            direction = Direction.LONG
            magnitude = Magnitude.MODERATE
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        elif signals.net_signal == 1:
            direction = Direction.LONG
            magnitude = Magnitude.SMALL
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        elif signals.net_signal <= -3:
            direction = Direction.SHORT
            magnitude = Magnitude.LARGE
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        elif signals.net_signal == -2:
            direction = Direction.SHORT
            magnitude = Magnitude.MODERATE
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        elif signals.net_signal == -1:
            direction = Direction.SHORT
            magnitude = Magnitude.SMALL
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)
        else:  # net_signal == 0
            direction = Direction.NEUTRAL
            magnitude = Magnitude.SMALL
            raw_confidence = min(0.85, 0.35 + abs(signals.net_signal) * 0.12)

        # Build thesis summary
        signal_descriptions = []
        if signals.golden_cross:
            signal_descriptions.append("golden cross")
        if signals.death_cross:
            signal_descriptions.append("death cross")
        if signals.rsi_oversold:
            signal_descriptions.append(f"RSI oversold ({signals.rsi_14:.1f})")
        if signals.rsi_overbought:
            signal_descriptions.append(f"RSI overbought ({signals.rsi_14:.1f})")
        if signals.macd_bullish_crossover:
            signal_descriptions.append("MACD bullish crossover")
        if signals.macd_bearish_crossover:
            signal_descriptions.append("MACD bearish crossover")
        if signals.price_below_lower_bb:
            signal_descriptions.append("price below Bollinger lower band")
        if signals.price_above_upper_bb:
            signal_descriptions.append("price above Bollinger upper band")

        signal_str = ", ".join(signal_descriptions) if signal_descriptions else "mixed signals"
        thesis_summary = f"Technical signals {direction.value} {ticker}: {signal_str}"

        # Determine time horizon based on signals
        if signals.golden_cross or signals.death_cross:
            time_horizon_days = 30  # Trend-following
        elif signals.rsi_oversold or signals.rsi_overbought or signals.price_below_lower_bb or signals.price_above_upper_bb:
            time_horizon_days = 10  # Mean reversion
        else:
            time_horizon_days = 20  # Default

        # Create evidence references
        evidence_refs = []
        weight = 1.0 / len(fragment_ids) if fragment_ids else 1.0
        for frag_id in fragment_ids[:5]:  # Limit to first 5 fragments
            evidence_refs.append(
                EvidenceRef(
                    source_fragment_id=frag_id,
                    field_path="close",
                    observation=f"Price data used in technical analysis",
                    weight=weight,
                )
            )

        # Create invalidation conditions
        invalidation_conditions = self._create_invalidation_conditions(
            direction, signals
        )

        # Create BeliefMetadata
        metadata = BeliefMetadata(
            sector="UNKNOWN",  # Technical analysis is sector-agnostic
            market_cap_bucket=MarketCapBucket.LARGE,  # Default
            catalyst_type=None,
        )

        # Create thesis ID
        thesis_id = f"TECH-{ticker}-{direction.value}-{signals.net_signal:+d}"

        # Create Belief
        belief = Belief(
            thesis_id=thesis_id,
            ticker=ticker,
            thesis_summary=thesis_summary,
            direction=direction,
            magnitude=magnitude,
            raw_confidence=raw_confidence,
            time_horizon_days=time_horizon_days,
            evidence=evidence_refs,
            invalidation_conditions=invalidation_conditions,
            correlated_beliefs=[],
            metadata=metadata,
        )

        beliefs.append(belief)
        return beliefs

    def _create_invalidation_conditions(
        self, direction: Direction, signals: TechnicalSignals
    ) -> list[InvalidationCondition]:
        """Create machine-evaluable invalidation conditions based on direction.

        Args:
            direction: LONG, SHORT, or NEUTRAL.
            signals: TechnicalSignals with current indicator values.

        Returns:
            List of InvalidationCondition objects.
        """
        conditions = []

        if direction == Direction.LONG:
            # Price falls 5% below SMA200
            if signals.sma_200 is not None:
                conditions.append(
                    InvalidationCondition(
                        description="Price falls 5% below SMA200 (invalidates bullish trend)",
                        data_source_agent="PERC-PRICE",
                        metric="close",
                        operator=ComparisonOperator.LT,
                        threshold=signals.sma_200 * 0.95,
                        status=ConditionStatus.ACTIVE,
                    )
                )

            # RSI becomes overbought (signal exhaustion)
            conditions.append(
                InvalidationCondition(
                    description="RSI rises above 75 (overbought, signal exhaustion)",
                    data_source_agent="COGNIT-TECHNICAL",
                    metric="rsi_14",
                    operator=ComparisonOperator.GT,
                    threshold=75.0,
                    status=ConditionStatus.ACTIVE,
                )
            )

            # MACD histogram turns negative
            if signals.macd_histogram is not None:
                conditions.append(
                    InvalidationCondition(
                        description="MACD histogram turns negative (bearish crossover)",
                        data_source_agent="COGNIT-TECHNICAL",
                        metric="macd_histogram",
                        operator=ComparisonOperator.LT,
                        threshold=0.0,
                        status=ConditionStatus.ACTIVE,
                    )
                )

        elif direction == Direction.SHORT:
            # Price rises 5% above SMA200
            if signals.sma_200 is not None:
                conditions.append(
                    InvalidationCondition(
                        description="Price rises 5% above SMA200 (invalidates bearish trend)",
                        data_source_agent="PERC-PRICE",
                        metric="close",
                        operator=ComparisonOperator.GT,
                        threshold=signals.sma_200 * 1.05,
                        status=ConditionStatus.ACTIVE,
                    )
                )

            # RSI becomes oversold (signal exhaustion)
            conditions.append(
                InvalidationCondition(
                    description="RSI falls below 25 (oversold, signal exhaustion)",
                    data_source_agent="COGNIT-TECHNICAL",
                    metric="rsi_14",
                    operator=ComparisonOperator.LT,
                    threshold=25.0,
                    status=ConditionStatus.ACTIVE,
                )
            )

            # MACD histogram turns positive
            if signals.macd_histogram is not None:
                conditions.append(
                    InvalidationCondition(
                        description="MACD histogram turns positive (bullish crossover)",
                        data_source_agent="COGNIT-TECHNICAL",
                        metric="macd_histogram",
                        operator=ComparisonOperator.GT,
                        threshold=0.0,
                        status=ConditionStatus.ACTIVE,
                    )
                )

        else:  # NEUTRAL
            # RSI becomes overbought
            conditions.append(
                InvalidationCondition(
                    description="RSI rises above 70 (enters overbought territory)",
                    data_source_agent="COGNIT-TECHNICAL",
                    metric="rsi_14",
                    operator=ComparisonOperator.GT,
                    threshold=70.0,
                    status=ConditionStatus.ACTIVE,
                )
            )

            # RSI becomes oversold
            conditions.append(
                InvalidationCondition(
                    description="RSI falls below 30 (enters oversold territory)",
                    data_source_agent="COGNIT-TECHNICAL",
                    metric="rsi_14",
                    operator=ComparisonOperator.LT,
                    threshold=30.0,
                    status=ConditionStatus.ACTIVE,
                )
            )

        return conditions

    def get_health(self) -> HealthStatus:
        """Report health status.

        Returns:
            HealthStatus with current metrics and status.
        """
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
        )
