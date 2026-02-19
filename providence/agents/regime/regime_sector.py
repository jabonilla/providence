"""REGIME-SECTOR: Sector-Level Regime Overlay Agent.

Enriches REGIME-STAT's global regime with per-sector regime overlays.
Each GICS sector gets an independent HMM classification based on
sector-aggregated price data.

Spec Reference: Technical Spec v2.3, Section 4.3

Classification: FROZEN — zero LLM calls. Pure computation.

The global RegimeStateObject (from REGIME-STAT) is taken as input,
and this agent populates the sector_overlays field with per-sector
regime classifications, relative stress scores, and key driving signals.
"""

from datetime import datetime, timezone

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.regime.hmm_model import (
    DEFAULT_HMM_PARAMS,
    HMMParameters,
    classify_regime,
    features_to_composite_score,
)
from providence.agents.regime.regime_features import (
    RegimeFeatures,
    extract_regime_features,
)
from providence.agents.regime.sector_features import (
    compute_relative_stress,
    extract_sector_features,
    group_fragments_by_sector,
    identify_key_signals,
)
from providence.exceptions import AgentProcessingError
from providence.schemas.enums import DataType
from providence.schemas.regime import RegimeStateObject, SectorRegimeOverlay

logger = structlog.get_logger()


class RegimeSector(BaseAgent[RegimeStateObject]):
    """Sector-level regime overlay agent.

    Consumes PRICE_OHLCV fragments (ticker-level), groups them by
    GICS sector, and classifies each sector's independent regime.

    Produces a RegimeStateObject with populated sector_overlays.
    Designed to enrich the output of REGIME-STAT.

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
            agent_id="REGIME-SECTOR",
            agent_type="regime",
            version="1.0.0",
        )
        self._hmm_params = hmm_params or DEFAULT_HMM_PARAMS
        self._sector_priors: dict[str, tuple[float, ...]] = {}
        self._last_run: datetime | None = None
        self._last_success: datetime | None = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> RegimeStateObject:
        """Execute the sector regime classification pipeline.

        Steps:
          1. RECEIVE CONTEXT   → Validated by caller
          2. GLOBAL FEATURES   → Extract market-wide features for baseline
          3. GROUP BY SECTOR   → Group PRICE_OHLCV by GICS sector
          4. PER-SECTOR HMM    → Classify each sector independently
          5. RELATIVE STRESS   → Compute sector vs market stress delta
          6. EMIT              → Return RegimeStateObject with sector_overlays

        Args:
            context: AgentContext assembled by CONTEXT-SVC.

        Returns:
            RegimeStateObject with sector_overlays populated.

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
            log.info("Starting sector regime classification")

            # Step 2: GLOBAL FEATURES — compute market-wide baseline
            global_features = extract_regime_features(context.fragments)
            market_composite = features_to_composite_score(global_features)

            # Also classify global regime for the output
            global_regime, global_confidence, global_probs = classify_regime(
                global_features, self._hmm_params
            )

            from providence.agents.regime.hmm_model import derive_risk_mode
            global_risk_mode = derive_risk_mode(global_regime, global_confidence)

            log.info(
                "Global baseline computed",
                market_composite=round(market_composite, 4),
                global_regime=global_regime.value,
            )

            # Step 3: GROUP BY SECTOR
            sector_groups = group_fragments_by_sector(context.fragments)

            log.info(
                "Sectors identified",
                sector_count=len(sector_groups),
                sectors=list(sector_groups.keys()),
            )

            # Step 4 & 5: PER-SECTOR HMM + RELATIVE STRESS
            overlays: dict[str, SectorRegimeOverlay] = {}

            for sector_name, group in sorted(sector_groups.items()):
                # Extract sector-level features
                sector_features = extract_sector_features(group)
                sector_composite = features_to_composite_score(sector_features)

                # Classify sector regime via HMM
                sector_prior = self._sector_priors.get(sector_name)
                sector_regime, sector_conf, sector_probs = classify_regime(
                    sector_features, self._hmm_params, sector_prior
                )

                # Update running prior for this sector
                from providence.agents.regime.hmm_model import REGIME_STATES
                self._sector_priors[sector_name] = tuple(
                    sector_probs[s.value] for s in REGIME_STATES
                )

                # Compute relative stress
                rel_stress = compute_relative_stress(sector_composite, market_composite)

                # Identify key signals
                key_signals = identify_key_signals(
                    sector_features, sector_composite, market_composite
                )

                overlay = SectorRegimeOverlay(
                    sector=sector_name,
                    regime=sector_regime,
                    regime_confidence=round(sector_conf, 6),
                    regime_probabilities=sector_probs,
                    relative_stress=round(rel_stress, 4),
                    key_signals=key_signals,
                    ticker_count=group.ticker_count,
                )
                overlays[sector_name] = overlay

                log.debug(
                    "Sector classified",
                    sector=sector_name,
                    regime=sector_regime.value,
                    confidence=round(sector_conf, 4),
                    relative_stress=round(rel_stress, 4),
                    ticker_count=group.ticker_count,
                )

            # Step 6: EMIT — build RegimeStateObject with overlays
            features_dict: dict[str, float] = {}
            if global_features.realized_vol_20d is not None:
                features_dict["realized_vol_20d"] = global_features.realized_vol_20d
            if global_features.realized_vol_60d is not None:
                features_dict["realized_vol_60d"] = global_features.realized_vol_60d
            if global_features.vol_of_vol is not None:
                features_dict["vol_of_vol"] = global_features.vol_of_vol
            if global_features.yield_spread_2s10s is not None:
                features_dict["yield_spread_2s10s"] = global_features.yield_spread_2s10s
            if global_features.cds_ig_spread is not None:
                features_dict["cds_ig_spread"] = global_features.cds_ig_spread
            if global_features.vix_proxy is not None:
                features_dict["vix_proxy"] = global_features.vix_proxy
            if global_features.macro_momentum is not None:
                features_dict["macro_momentum"] = global_features.macro_momentum
            if global_features.price_drawdown_pct is not None:
                features_dict["price_drawdown_pct"] = global_features.price_drawdown_pct
            if global_features.price_momentum_20d is not None:
                features_dict["price_momentum_20d"] = global_features.price_momentum_20d
            features_dict["composite_score"] = market_composite

            regime_state = RegimeStateObject(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                statistical_regime=global_regime,
                regime_confidence=round(global_confidence, 6),
                regime_probabilities=global_probs,
                system_risk_mode=global_risk_mode,
                features_used=features_dict,
                sector_overlays=overlays,
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Sector regime classification complete",
                sectors_classified=len(overlays),
            )
            return regime_state

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"REGIME-SECTOR processing failed: {e}",
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
