"""REGIME-STAT: Statistical Regime Classification Agent.

Uses a Hidden Markov Model to classify the current market regime
from price volatility, yield curve, credit spreads, implied
volatility, and macro economic data.

Spec Reference: Technical Spec v2.3, Section 4.3

Classification: FROZEN — zero LLM calls. Pure computation.

Regime States:
  1. LOW_VOL_TRENDING — calm trending markets, normal risk
  2. HIGH_VOL_MEAN_REVERTING — elevated volatility, mean-reverting dynamics
  3. CRISIS_DISLOCATION — severe stress, dislocated correlations
  4. TRANSITION_UNCERTAIN — regime boundaries, unclear dynamics

System Risk Modes (derived):
  LOW_VOL_TRENDING       → NORMAL
  HIGH_VOL_MEAN_REVERTING → CAUTIOUS
  CRISIS_DISLOCATION     → DEFENSIVE (HALTED if confidence > 0.9)
  TRANSITION_UNCERTAIN   → CAUTIOUS
"""

from datetime import datetime, timezone

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.regime.hmm_model import (
    HMMParameters,
    DEFAULT_HMM_PARAMS,
    classify_regime,
    derive_risk_mode,
    features_to_composite_score,
)
from providence.agents.regime.regime_features import (
    RegimeFeatures,
    extract_regime_features,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import DataType
from providence.schemas.regime import RegimeStateObject

logger = structlog.get_logger()


class RegimeStat(BaseAgent[RegimeStateObject]):
    """Statistical regime classification agent.

    Consumes PRICE_OHLCV, MACRO_YIELD_CURVE, MACRO_CDS,
    MACRO_ECONOMIC, and OPTIONS_CHAIN MarketStateFragments.

    Produces RegimeStateObjects with the classified market regime,
    confidence, state probabilities, and derived risk mode.

    FROZEN: No LLM calls. All analysis is deterministic HMM computation.
    """

    CONSUMED_DATA_TYPES = {
        DataType.PRICE_OHLCV,
        DataType.MACRO_YIELD_CURVE,
        DataType.MACRO_CDS,
        DataType.MACRO_ECONOMIC,
        DataType.OPTIONS_CHAIN,
    }

    def __init__(
        self,
        hmm_params: HMMParameters | None = None,
    ) -> None:
        super().__init__(
            agent_id="REGIME-STAT",
            agent_type="regime",
            version="1.0.0",
        )
        self._hmm_params = hmm_params or DEFAULT_HMM_PARAMS
        self._prior: tuple[float, ...] | None = None  # Running HMM prior
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RegimeStateObject:
        """Execute the regime classification pipeline.

        Steps:
          1. RECEIVE CONTEXT  → Validated by caller
          2. EXTRACT FEATURES → From PRICE_OHLCV, macro, options fragments
          3. CLASSIFY REGIME  → HMM forward algorithm
          4. DERIVE RISK MODE → Map regime → system risk mode
          5. EMIT             → Return validated RegimeStateObject

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            RegimeStateObject with classified regime and risk mode.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
                fragment_count=len(context.fragments),
            )
            log.info("Starting regime classification")

            # Step 2: EXTRACT FEATURES
            features = extract_regime_features(context.fragments)
            composite = features_to_composite_score(features)

            log.info(
                "Features extracted",
                composite_score=round(composite, 4),
                realized_vol=features.realized_vol_20d,
                yield_spread=features.yield_spread_2s10s,
            )

            # Step 3: CLASSIFY REGIME — HMM forward pass
            regime, confidence, prob_dict = classify_regime(
                features, self._hmm_params, self._prior
            )

            # Update running prior for next classification
            from providence.agents.regime.hmm_model import REGIME_STATES
            self._prior = tuple(prob_dict[s.value] for s in REGIME_STATES)

            # Step 4: DERIVE RISK MODE
            risk_mode = derive_risk_mode(regime, confidence)

            log.info(
                "Regime classified",
                regime=regime.value,
                confidence=round(confidence, 4),
                risk_mode=risk_mode.value,
            )

            # Step 5: EMIT — build RegimeStateObject
            features_dict = {}
            if features.realized_vol_20d is not None:
                features_dict["realized_vol_20d"] = features.realized_vol_20d
            if features.realized_vol_60d is not None:
                features_dict["realized_vol_60d"] = features.realized_vol_60d
            if features.vol_of_vol is not None:
                features_dict["vol_of_vol"] = features.vol_of_vol
            if features.yield_spread_2s10s is not None:
                features_dict["yield_spread_2s10s"] = features.yield_spread_2s10s
            if features.cds_ig_spread is not None:
                features_dict["cds_ig_spread"] = features.cds_ig_spread
            if features.vix_proxy is not None:
                features_dict["vix_proxy"] = features.vix_proxy
            if features.macro_momentum is not None:
                features_dict["macro_momentum"] = features.macro_momentum
            if features.price_drawdown_pct is not None:
                features_dict["price_drawdown_pct"] = features.price_drawdown_pct
            if features.price_momentum_20d is not None:
                features_dict["price_momentum_20d"] = features.price_momentum_20d
            features_dict["composite_score"] = composite

            regime_state = RegimeStateObject(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                statistical_regime=regime,
                regime_confidence=round(confidence, 6),
                regime_probabilities=prob_dict,
                system_risk_mode=risk_mode,
                features_used=features_dict,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info("Regime classification complete")
            return regime_state

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"REGIME-STAT processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

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
        )
