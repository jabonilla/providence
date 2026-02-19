"""DECIDE-OPTIM: Portfolio Optimization Agent.

Converts SynthesizedPositionIntents into a portfolio of ProposedPositions
using a simplified Black-Litterman framework. Enforces exposure limits,
sector concentration caps, and position size constraints.

Spec Reference: Technical Spec v2.3, Section 4.5

Classification: FROZEN — zero LLM calls. Pure computation.

Input: AgentContext with position intents and regime data in metadata:
  - metadata["position_intents"]: list of serialized SynthesizedPositionIntent dicts
  - metadata["regime_state"]: serialized RegimeStateObject dict

Output: PositionProposal containing ProposedPositions + portfolio metadata.
"""

import math
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID, uuid4

import structlog

from providence.agents.base import AgentContext, AgentStatus, BaseAgent, HealthStatus
from providence.agents.regime.sector_features import get_sector
from providence.exceptions import AgentProcessingError, ConstraintViolationError
from providence.schemas.decision import (
    PortfolioMetadata,
    PositionProposal,
    ProposedPosition,
)
from providence.schemas.enums import Action, Direction

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Exposure limits by risk mode
# ---------------------------------------------------------------------------
RISK_MODE_LIMITS: dict[str, dict[str, float]] = {
    "NORMAL": {
        "max_gross_exposure": 1.60,
        "max_net_exposure": 0.80,
        "max_position_weight": 0.10,
        "max_sector_concentration": 0.35,
        "confidence_floor": 0.25,
    },
    "CAUTIOUS": {
        "max_gross_exposure": 1.20,
        "max_net_exposure": 0.60,
        "max_position_weight": 0.08,
        "max_sector_concentration": 0.30,
        "confidence_floor": 0.30,
    },
    "DEFENSIVE": {
        "max_gross_exposure": 0.80,
        "max_net_exposure": 0.40,
        "max_position_weight": 0.05,
        "max_sector_concentration": 0.25,
        "confidence_floor": 0.35,
    },
    "HALTED": {
        "max_gross_exposure": 0.0,
        "max_net_exposure": 0.0,
        "max_position_weight": 0.0,
        "max_sector_concentration": 0.0,
        "confidence_floor": 1.0,
    },
}


# ---------------------------------------------------------------------------
# Black-Litterman core functions (pure, stateless)
# ---------------------------------------------------------------------------

def compute_equilibrium_weights(n_assets: int) -> list[float]:
    """Compute equal-weighted prior (market cap proxy for MVP).

    In a full implementation this would use market cap data.
    For MVP we use equal weights as a neutral starting point.

    Args:
        n_assets: Number of assets in the universe.

    Returns:
        List of equal weights summing to 1.0.
    """
    if n_assets <= 0:
        return []
    w = 1.0 / n_assets
    return [round(w, 6) for _ in range(n_assets)]


def compute_view_confidence_matrix(
    confidences: list[float],
    tau: float = 0.05,
) -> list[float]:
    """Compute diagonal uncertainty for each view (Omega in BL).

    Lower confidence → higher uncertainty (larger Omega diagonal).
    tau scales the overall uncertainty level.

    Args:
        confidences: Per-view confidence values [0, 1].
        tau: Scaling factor for view uncertainty (default 0.05).

    Returns:
        List of diagonal Omega values (one per view).
    """
    omega = []
    for c in confidences:
        c_clamped = max(0.05, min(0.95, c))
        # Omega_ii = tau / c — lower confidence → higher uncertainty
        omega.append(round(tau / c_clamped, 6))
    return omega


def black_litterman_weights(
    prior_weights: list[float],
    view_directions: list[float],
    view_confidences: list[float],
    tau: float = 0.05,
) -> list[float]:
    """Simplified Black-Litterman posterior weights.

    Uses a scalar-variance simplification suitable for MVP:
    - Prior: equal weights (market cap proxy)
    - Views: direction * confidence as expected return proxy
    - Posterior: Bayesian blend of prior and views

    The full BL model uses covariance matrices. This MVP uses
    a diagonal approximation where each asset is independent.

    Args:
        prior_weights: Equilibrium weights (len = n_assets).
        view_directions: +1.0 for LONG, -1.0 for SHORT, 0.0 for NEUTRAL.
        view_confidences: Confidence for each view [0, 1].
        tau: Uncertainty scaling factor.

    Returns:
        Posterior portfolio weights (may be negative for short positions).
    """
    n = len(prior_weights)
    if n == 0:
        return []

    omega = compute_view_confidence_matrix(view_confidences, tau)
    posterior = []

    for i in range(n):
        # Prior return proxy: equal weight * risk-free-ish baseline
        prior_return = prior_weights[i] * 0.05  # 5% baseline return
        # View return proxy: direction * confidence
        view_return = view_directions[i] * view_confidences[i] * 0.10
        # BL blend: weight by inverse uncertainty
        prior_precision = 1.0 / tau
        view_precision = 1.0 / omega[i] if omega[i] > 0 else 0.0
        total_precision = prior_precision + view_precision

        if total_precision > 0:
            blended_return = (
                prior_precision * prior_return + view_precision * view_return
            ) / total_precision
        else:
            blended_return = prior_return

        # Posterior weight proportional to blended expected return
        posterior.append(blended_return)

    # Normalize to target gross exposure (absolute weights sum to ~1.0)
    abs_sum = sum(abs(w) for w in posterior)
    if abs_sum > 0:
        posterior = [w / abs_sum for w in posterior]

    return [round(w, 6) for w in posterior]


def apply_position_limits(
    weights: list[float],
    max_weight: float,
) -> list[float]:
    """Clamp individual position weights to max size.

    Args:
        weights: Raw portfolio weights (may be negative for shorts).
        max_weight: Maximum absolute weight per position.

    Returns:
        Clamped weights.
    """
    clamped = []
    for w in weights:
        if abs(w) > max_weight:
            clamped.append(max_weight if w > 0 else -max_weight)
        else:
            clamped.append(w)
    return [round(w, 6) for w in clamped]


def apply_exposure_limits(
    weights: list[float],
    max_gross: float,
    max_net: float,
) -> list[float]:
    """Scale weights to satisfy gross and net exposure limits.

    Args:
        weights: Portfolio weights (may be negative).
        max_gross: Maximum gross exposure (sum of abs weights).
        max_net: Maximum net exposure (abs of sum of weights).

    Returns:
        Scaled weights satisfying constraints.
    """
    if not weights:
        return weights

    gross = sum(abs(w) for w in weights)
    net = sum(weights)

    # Scale down for gross exposure
    if gross > max_gross and gross > 0:
        scale = max_gross / gross
        weights = [w * scale for w in weights]

    # Check net exposure — if violated, reduce the dominant side
    net = sum(weights)
    if abs(net) > max_net and abs(net) > 0:
        scale = max_net / abs(net)
        # Scale all positions proportionally
        weights = [w * scale for w in weights]

    return [round(w, 6) for w in weights]


def compute_sector_concentrations(
    tickers: list[str],
    weights: list[float],
) -> dict[str, float]:
    """Compute absolute weight concentration per GICS sector.

    Args:
        tickers: List of ticker symbols.
        weights: Corresponding portfolio weights.

    Returns:
        Dict of sector -> sum of absolute weights.
    """
    concentrations: dict[str, float] = {}
    for ticker, weight in zip(tickers, weights):
        sector = get_sector(ticker) or "Unknown"
        concentrations[sector] = concentrations.get(sector, 0.0) + abs(weight)
    return {k: round(v, 6) for k, v in sorted(concentrations.items())}


def enforce_sector_limits(
    tickers: list[str],
    weights: list[float],
    max_sector_concentration: float,
) -> list[float]:
    """Scale down positions in sectors exceeding concentration limit.

    Args:
        tickers: List of ticker symbols.
        weights: Portfolio weights.
        max_sector_concentration: Max absolute weight per sector.

    Returns:
        Adjusted weights with sector limits enforced.
    """
    # Compute current concentrations
    sector_weights: dict[str, float] = {}
    for ticker, weight in zip(tickers, weights):
        sector = get_sector(ticker) or "Unknown"
        sector_weights[sector] = sector_weights.get(sector, 0.0) + abs(weight)

    # Scale down over-concentrated sectors
    adjusted = list(weights)
    for sector, total in sector_weights.items():
        if total > max_sector_concentration and total > 0:
            scale = max_sector_concentration / total
            for i, ticker in enumerate(tickers):
                if (get_sector(ticker) or "Unknown") == sector:
                    adjusted[i] = adjusted[i] * scale

    return [round(w, 6) for w in adjusted]


def estimate_sharpe(
    weights: list[float],
    confidences: list[float],
    directions: list[float],
) -> float:
    """Estimate ex-ante Sharpe ratio from view alignment.

    Simplified: weighted average of (confidence * direction_alignment).
    Higher confidence with consistent direction → higher Sharpe estimate.

    Args:
        weights: Portfolio weights.
        confidences: View confidences.
        directions: View directions (+1/-1/0).

    Returns:
        Estimated Sharpe ratio (typically 0.0 to 2.0).
    """
    if not weights:
        return 0.0

    abs_sum = sum(abs(w) for w in weights)
    if abs_sum == 0:
        return 0.0

    weighted_signal = 0.0
    for w, c, d in zip(weights, confidences, directions):
        # Alignment: weight direction matches view direction
        alignment = 1.0 if (w > 0 and d > 0) or (w < 0 and d < 0) else -0.5
        weighted_signal += abs(w) * c * alignment

    raw_sharpe = weighted_signal / abs_sum
    # Scale to typical Sharpe range [0, 2.0]
    return round(max(0.0, min(2.0, raw_sharpe * 3.0)), 4)


def intent_to_action(direction: str, weight: float) -> Action:
    """Determine the position action from direction and weight.

    Args:
        direction: LONG, SHORT, or NEUTRAL.
        weight: Portfolio weight (may be negative).

    Returns:
        Action enum value.
    """
    if abs(weight) < 0.001:
        return Action.CLOSE
    if direction == "LONG" and weight > 0:
        return Action.OPEN_LONG
    if direction == "SHORT" and weight < 0:
        return Action.OPEN_SHORT
    if direction == "NEUTRAL":
        return Action.CLOSE
    # Direction/weight mismatch or adjustment
    return Action.ADJUST


# ---------------------------------------------------------------------------
# DecideOptim agent class
# ---------------------------------------------------------------------------

class DecideOptim(BaseAgent[PositionProposal]):
    """Portfolio optimization agent.

    Consumes SynthesizedPositionIntents and RegimeStateObject from
    DECIDE-SYNTH and the Regime subsystem. Produces a PositionProposal
    using Black-Litterman optimization with regime-aware constraints.

    FROZEN: Zero LLM calls. Pure computation.
    """

    CONSUMED_DATA_TYPES: set = set()

    def __init__(self) -> None:
        super().__init__(
            agent_id="DECIDE-OPTIM",
            agent_type="decision",
            version="1.0.0",
        )
        self._last_run: Optional[datetime] = None
        self._last_success: Optional[datetime] = None
        self._error_count_24h: int = 0

    async def process(self, context: AgentContext) -> PositionProposal:
        """Execute the portfolio optimization pipeline.

        Steps:
          1. RECEIVE CONTEXT    → Extract intents and regime from metadata
          2. FILTER INTENTS     → Drop low-confidence, drop NEUTRAL
          3. DETERMINE LIMITS   → Look up risk mode exposure limits
          4. OPTIMIZE           → Black-Litterman posterior weights
          5. CONSTRAIN          → Enforce position, sector, exposure limits
          6. BUILD PROPOSALS    → Construct ProposedPositions
          7. EMIT               → Return PositionProposal

        Args:
            context: AgentContext with intents and regime in metadata.

        Returns:
            PositionProposal with optimized positions.

        Raises:
            AgentProcessingError: If processing fails.
        """
        self._last_run = datetime.now(timezone.utc)

        try:
            log = logger.bind(
                agent_id=self.agent_id,
                context_hash=context.context_window_hash,
            )
            log.info("Starting portfolio optimization")

            # Step 1: EXTRACT UPSTREAM DATA
            raw_intents = context.metadata.get("position_intents", [])
            regime_state = context.metadata.get("regime_state", {})

            risk_mode = "NORMAL"
            if isinstance(regime_state, dict):
                risk_mode = regime_state.get("system_risk_mode", "NORMAL")

            limits = RISK_MODE_LIMITS.get(risk_mode, RISK_MODE_LIMITS["NORMAL"])

            log.info(
                "Upstream data extracted",
                intent_count=len(raw_intents) if isinstance(raw_intents, list) else 0,
                risk_mode=risk_mode,
            )

            # HALTED mode: return empty portfolio
            if risk_mode == "HALTED":
                log.warning("System HALTED — returning empty portfolio")
                return self._empty_proposal(context, risk_mode)

            # Step 2: FILTER INTENTS
            intents = self._parse_and_filter_intents(raw_intents, limits)

            if not intents:
                log.info("No actionable intents after filtering")
                return self._empty_proposal(context, risk_mode)

            log.info("Intents filtered", actionable_count=len(intents))

            # Extract parallel arrays for optimization
            tickers = [i["ticker"] for i in intents]
            directions = [
                1.0 if i["net_direction"] == "LONG"
                else -1.0 if i["net_direction"] == "SHORT"
                else 0.0
                for i in intents
            ]
            confidences = [i["synthesized_confidence"] for i in intents]

            # Step 3: OPTIMIZE — Black-Litterman
            n = len(tickers)
            prior_weights = compute_equilibrium_weights(n)
            raw_weights = black_litterman_weights(
                prior_weights, directions, confidences,
            )

            # Step 4: CONSTRAIN
            # 4a: Position limits
            weights = apply_position_limits(
                raw_weights, limits["max_position_weight"],
            )

            # 4b: Sector concentration limits
            weights = enforce_sector_limits(
                tickers, weights, limits["max_sector_concentration"],
            )

            # 4c: Gross and net exposure limits
            weights = apply_exposure_limits(
                weights, limits["max_gross_exposure"], limits["max_net_exposure"],
            )

            # Step 5: BUILD PROPOSALS
            proposals = []
            for i, (ticker, weight) in enumerate(zip(tickers, weights)):
                if abs(weight) < 0.001:
                    continue  # Skip negligible positions

                direction_str = intents[i]["net_direction"]
                direction = Direction(direction_str)
                action = intent_to_action(direction_str, weight)

                intent_id = intents[i].get("intent_id", str(uuid4()))
                try:
                    source_uuid = UUID(str(intent_id))
                except (ValueError, TypeError):
                    source_uuid = uuid4()

                proposals.append(ProposedPosition(
                    ticker=ticker,
                    action=action,
                    target_weight=abs(weight),
                    direction=direction,
                    confidence=confidences[i],
                    source_intent_id=source_uuid,
                    time_horizon_days=intents[i].get("time_horizon_days", 60),
                    regime_adjustment=intents[i].get("regime_adjustment", 0.0),
                    sector=get_sector(ticker) or "Unknown",
                ))

            # Step 6: COMPUTE PORTFOLIO METADATA
            final_weights = [
                p.target_weight * (1.0 if p.direction == Direction.LONG else -1.0)
                for p in proposals
            ]
            gross_exposure = sum(abs(w) for w in final_weights)
            net_exposure = sum(final_weights)
            sector_conc = compute_sector_concentrations(
                [p.ticker for p in proposals],
                [p.target_weight for p in proposals],
            )
            sharpe = estimate_sharpe(
                [w for w in final_weights],
                [p.confidence for p in proposals],
                [1.0 if p.direction == Direction.LONG else -1.0 for p in proposals],
            )

            metadata = PortfolioMetadata(
                gross_exposure=round(gross_exposure, 6),
                net_exposure=round(max(-1.0, min(1.0, net_exposure)), 6),
                sector_concentrations=sector_conc,
                estimated_sharpe=sharpe,
                position_count=len(proposals),
                risk_mode_applied=risk_mode,
            )

            # Step 7: EMIT
            proposal = PositionProposal(
                agent_id=self.agent_id,
                timestamp=datetime.now(timezone.utc),
                context_window_hash=context.context_window_hash,
                proposals=proposals,
                portfolio_metadata=metadata,
                total_intents_consumed=len(intents),
            )

            self._last_success = datetime.now(timezone.utc)
            log.info(
                "Portfolio optimization complete",
                position_count=len(proposals),
                gross_exposure=round(gross_exposure, 4),
                net_exposure=round(net_exposure, 4),
                sharpe=sharpe,
            )
            return proposal

        except AgentProcessingError:
            raise
        except Exception as e:
            self._error_count_24h += 1
            raise AgentProcessingError(
                message=f"DECIDE-OPTIM processing failed: {e}",
                agent_id=self.agent_id,
            ) from e

    def _parse_and_filter_intents(
        self,
        raw_intents: Any,
        limits: dict[str, float],
    ) -> list[dict]:
        """Parse raw intent dicts and filter by confidence floor.

        Args:
            raw_intents: List of serialized SynthesizedPositionIntent dicts.
            limits: Risk mode limits including confidence_floor.

        Returns:
            List of valid, actionable intent dicts.
        """
        if not isinstance(raw_intents, list):
            return []

        confidence_floor = limits.get("confidence_floor", 0.25)
        filtered = []

        for intent in raw_intents:
            if not isinstance(intent, dict):
                continue
            ticker = intent.get("ticker")
            if not ticker:
                continue
            direction = intent.get("net_direction", "NEUTRAL")
            if direction == "NEUTRAL":
                continue
            confidence = intent.get("synthesized_confidence", 0.0)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                continue
            if confidence < confidence_floor:
                continue

            filtered.append({
                "ticker": ticker,
                "net_direction": direction,
                "synthesized_confidence": confidence,
                "intent_id": intent.get("intent_id", ""),
                "time_horizon_days": int(intent.get("time_horizon_days", 60)),
                "regime_adjustment": float(intent.get("regime_adjustment", 0.0)),
            })

        return filtered

    def _empty_proposal(
        self,
        context: AgentContext,
        risk_mode: str,
    ) -> PositionProposal:
        """Create an empty PositionProposal (no positions).

        Args:
            context: Current agent context.
            risk_mode: Applied risk mode.

        Returns:
            Empty PositionProposal.
        """
        return PositionProposal(
            agent_id=self.agent_id,
            timestamp=datetime.now(timezone.utc),
            context_window_hash=context.context_window_hash,
            proposals=[],
            portfolio_metadata=PortfolioMetadata(
                gross_exposure=0.0,
                net_exposure=0.0,
                sector_concentrations={},
                estimated_sharpe=0.0,
                position_count=0,
                risk_mode_applied=risk_mode,
            ),
            total_intents_consumed=0,
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
        )
